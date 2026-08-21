"""
Normalize an existing OpenCost parquet export against a schema file.

This module downloads a parquet file from an S3 bucket, aligns every row to the schema
described in a JSON or YAML file, dropping the columns that are not declared and filling
the missing ones with their default value, and writes the result to a local file with the
same name as the object in the bucket.

By default it is a dry run and the bucket is left untouched. With --no-dry-run every
normalized file is uploaded back, replacing the original object.

The schema file uses the same format as OPENCOST_PARQUET_SCHEMA_FILE, therefore the
normalized file has exactly the same layout an export made with that schema would have.
"""

import argparse
import os
import sys
import tempfile

import boto3
import pandas as pd
from botocore.exceptions import BotoCoreError, ClientError

# The exporter lives in src/, add it to the path so that the schema handling is shared and
# a normalized file holds the same data an export made with the same schema would hold.
# The files are not byte identical: this tool writes with index=False, so the pandas index
# metadata of the footer differs. No query engine reads that metadata.
sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, 'src'))

# The path above is resolved at runtime, pylint cannot follow it.
# pylint: disable=C0413,E0401
from opencost_parquet_exporter import load_schema_file, apply_schema


def bucket_name(bucket):
    """
    Return the plain bucket name, accepting it with or without the 's3://' scheme.

    Parameters:
        bucket (str): Bucket, for example 'my-bucket' or 's3://my-bucket/'.

    Returns:
        str: The bucket name.
    """
    return bucket.removeprefix('s3://').strip('/')


def download_from_s3(bucket, key, destination_dir, client=None):
    """
    Download an object from an S3 bucket into a local directory.

    Parameters:
        bucket (str): Bucket name, with or without the 's3://' scheme.
        key (str): Path of the object inside the bucket.
        destination_dir (str): Local directory where the object is written.
        client: Optional boto3 S3 client to reuse, a new one is created if not given.

    Returns:
        str: The local path of the downloaded file.
    """
    object_key = key.lstrip('/')
    destination = os.path.join(destination_dir, os.path.basename(object_key))
    print(f"Downloading s3://{bucket_name(bucket)}/{object_key}")
    client = client or boto3.client('s3')
    client.download_file(bucket_name(bucket), object_key, destination)
    return destination


def upload_to_s3(bucket, key, local_path, client=None):
    """
    Upload a local file to an S3 bucket, replacing the object at the given key.

    Parameters:
        bucket (str): Bucket name, with or without the 's3://' scheme.
        key (str): Path of the object inside the bucket, it is overwritten.
        local_path (str): Local file to upload.
        client: Optional boto3 S3 client to reuse, a new one is created if not given.
    """
    object_key = key.lstrip('/')
    print(f"  Replacing s3://{bucket_name(bucket)}/{object_key}")
    client = client or boto3.client('s3')
    client.upload_file(local_path, bucket_name(bucket), object_key)


def list_parquet_keys(bucket, prefix, client=None):
    """
    List every parquet object under a prefix, recursively.

    Parameters:
        bucket (str): Bucket name, with or without the 's3://' scheme.
        prefix (str): Prefix, treated as a directory, so a trailing slash is added.
        client: Optional boto3 S3 client to reuse, a new one is created if not given.

    Returns:
        tuple: (list, int) the sorted parquet keys and the number of objects skipped
               because they are not parquet files.
    """
    client = client or boto3.client('s3')
    directory = prefix.strip('/')
    directory = f"{directory}/" if directory else ""
    keys, skipped = [], 0
    for page in client.get_paginator('list_objects_v2').paginate(
            Bucket=bucket_name(bucket), Prefix=directory):
        for entry in page.get('Contents', []):
            key = entry['Key']
            if key.endswith('/'):
                continue
            if key.lower().endswith('.parquet'):
                keys.append(key)
            else:
                skipped += 1
    return sorted(keys), skipped


def normalize_dataframe(data, schema):
    """
    Align the data to the schema: drop the columns that are not declared, create the
    missing ones using their default, fill the null values and cast every column to its
    declared type, keeping the order of the schema.

    Parameters:
        data (DataFrame): Data read from the parquet file.
        schema (dict): Schema as returned by load_schema_file.

    Returns:
        tuple: (DataFrame, dict) the normalized data and a summary of the changes, with
               the 'rows' count and the 'dropped', 'added' and 'filled' columns.

    Raises:
        ValueError: If a column cannot be cast to its declared type.
    """
    summary = {
        'dropped': [column for column in data.columns if column not in schema],
        'added': [column for column in schema if column not in data.columns],
        'filled': {},
    }
    for column, definition in schema.items():
        if column in data.columns and 'default' in definition:
            nulls = int(data[column].isna().sum())
            if nulls:
                summary['filled'][column] = nulls

    normalized = apply_schema(data, schema)
    summary['rows'] = len(normalized)
    return normalized, summary


def normalize_object(bucket, key, schema, output_path, download_dir, client=None,
                     replace=False):
    """
    Download a single object, normalize it, save it locally and, when asked, upload it
    back to the bucket replacing the original object.

    Parameters:
        bucket (str): Bucket name, with or without the 's3://' scheme.
        key (str): Path of the object inside the bucket.
        schema (dict): Schema as returned by load_schema_file.
        output_path (str): Local path of the normalized file.
        download_dir (str): Directory used for the temporary download.
        client: Optional boto3 S3 client to reuse.
        replace (bool): When True the object in the bucket is overwritten with the
                        normalized file. The local copy is always written first, so the
                        upload only happens on a file that was normalized successfully.

    Returns:
        dict: Summary as returned by normalize_dataframe.

    Raises:
        ClientError, BotoCoreError, OSError: If the object cannot be downloaded or uploaded.
        ValueError, pd.errors.ParserError: If the file cannot be read or normalized.
    """
    local_path = download_from_s3(bucket, key, download_dir, client=client)
    try:
        data = pd.read_parquet(local_path)
        normalized, summary = normalize_dataframe(data, schema)
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        normalized.to_parquet(output_path, index=False)
    finally:
        # Keep the temporary directory small when normalizing a whole prefix.
        if os.path.exists(local_path):
            os.remove(local_path)
    if replace:
        upload_to_s3(bucket, key, output_path, client=client)
    return summary


def summary_line(summary, output_path):
    """
    Return a one line description of what the normalization changed.

    Parameters:
        summary (dict): Summary as returned by normalize_dataframe.
        output_path (str): Local path of the normalized file.

    Returns:
        str: The compact summary.
    """
    return (f"{summary['rows']} rows, "
            f"-{len(summary['dropped'])} columns, "
            f"+{len(summary['added'])} defaulted columns, "
            f"{sum(summary['filled'].values())} nulls filled -> {output_path}")


def print_summary(summary, output_path):
    """
    Print what the normalization changed.

    Parameters:
        summary (dict): Summary as returned by normalize_dataframe.
        output_path (str): Local path where the normalized file was written.
    """
    print(f"Rows normalized: {summary['rows']}")
    print(f"Columns dropped ({len(summary['dropped'])}): "
          f"{', '.join(summary['dropped']) or 'none'}")
    print(f"Columns added from their default ({len(summary['added'])}): "
          f"{', '.join(summary['added']) or 'none'}")
    if summary['filled']:
        print(f"Null values replaced by their default in {len(summary['filled'])} columns:")
        for column, count in sorted(summary['filled'].items(), key=lambda item: -item[1]):
            print(f"  {column}: {count}")
    else:
        print("Null values replaced by their default: none")
    print(f"Normalized file saved at: {output_path}")


def parse_args(argv=None):
    """
    Parse the command line arguments.

    Parameters:
        argv (list): Arguments to parse, defaults to sys.argv[1:].

    Returns:
        Namespace: The parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Download a parquet file from S3 and normalize it against a schema file.",
        epilog="example: tools/normalize_parquet.py -b my-bucket "
               "-k year=2026/month=8/day=20/opencost_2026-08-20_10_00_00.parquet "
               "-s schema.yaml -o ./normalized")
    parser.add_argument('-b', '--bucket', required=True,
                        help="S3 bucket holding the parquet file, "
                             "with or without the 's3://' scheme")
    parser.add_argument('-k', '--key', required=True,
                        help="path of the parquet file inside the bucket, or of the "
                             "directory holding them when --recursive is used")
    parser.add_argument('-r', '--recursive', action='store_true',
                        help="treat the key as a directory and normalize every parquet "
                             "file under it, recursively. The directory structure below "
                             "the key is recreated inside the output directory")
    parser.add_argument('-s', '--schema', required=True,
                        help="path of the JSON or YAML schema file")
    parser.add_argument('-o', '--output-dir', default='.',
                        help="local directory for the normalized file, defaults to the "
                             "current directory. The file keeps the name it has in the bucket")
    parser.add_argument('--no-dry-run', action='store_true',
                        help="upload every normalized file back to the bucket, replacing "
                             "the original object. Without this flag the tool is a dry run "
                             "and only writes locally")
    return parser.parse_args(argv)


def replace_notice(args):
    """
    Return the message telling whether the bucket was modified or not.

    Parameters:
        args (Namespace): The parsed command line arguments.

    Returns:
        str: The message to print at the end of the run.
    """
    if args.no_dry_run:
        return f"Objects replaced in s3://{bucket_name(args.bucket)}"
    return ("Dry run: nothing was written to the bucket, "
            "pass --no-dry-run to replace the objects with the normalized files")


def normalize_single(args, schema, client):
    """
    Normalize the single object referenced by the key.

    Parameters:
        args (Namespace): The parsed command line arguments.
        schema (dict): Schema as returned by load_schema_file.
        client: boto3 S3 client.

    Returns:
        int: 0 on success, 1 on failure.
    """
    output_path = os.path.join(args.output_dir, os.path.basename(args.key.rstrip('/')))
    if os.path.exists(output_path):
        print(f"Warning: {output_path} already exists and will be overwritten")

    with tempfile.TemporaryDirectory() as download_dir:
        try:
            summary = normalize_object(
                args.bucket, args.key, schema, output_path, download_dir, client=client,
                replace=args.no_dry_run)
        except (ClientError, BotoCoreError, OSError, ValueError,
                pd.errors.ParserError) as err:
            print(f"Failed to normalize s3://{bucket_name(args.bucket)}/{args.key}: {err}")
            return 1

    print_summary(summary, output_path)
    print(replace_notice(args))
    return 0


def normalize_recursive(args, schema, client):
    """
    Normalize every parquet object under the key, recreating the directory structure
    below it inside the output directory.

    Parameters:
        args (Namespace): The parsed command line arguments.
        schema (dict): Schema as returned by load_schema_file.
        client: boto3 S3 client.

    Returns:
        int: 0 when every file was normalized, 1 if the prefix is empty or any file failed.
    """
    directory = args.key.strip('/')
    try:
        keys, skipped = list_parquet_keys(args.bucket, directory, client=client)
    except (ClientError, BotoCoreError) as err:
        print(f"Failed to list s3://{bucket_name(args.bucket)}/{directory}: {err}")
        return 1

    if skipped:
        print(f"Skipping {skipped} objects that are not parquet files")
    if not keys:
        print(f"No parquet file found under s3://{bucket_name(args.bucket)}/{directory}")
        return 1
    print(f"Found {len(keys)} parquet files to normalize")

    failed, rows = [], 0
    with tempfile.TemporaryDirectory() as download_dir:
        for position, key in enumerate(keys, start=1):
            relative_path = os.path.relpath(key, directory) if directory else key
            output_path = os.path.join(args.output_dir, relative_path)
            print(f"[{position}/{len(keys)}] {key}")
            try:
                summary = normalize_object(
                    args.bucket, key, schema, output_path, download_dir, client=client,
                    replace=args.no_dry_run)
            except (ClientError, BotoCoreError, OSError, ValueError,
                    pd.errors.ParserError) as err:
                print(f"  Failed: {err}")
                failed.append(key)
                continue
            rows += summary['rows']
            print(f"  {summary_line(summary, output_path)}")

    print(f"\nNormalized {len(keys) - len(failed)}/{len(keys)} files, {rows} rows in total")
    print(replace_notice(args))
    if failed:
        print(f"Failed files ({len(failed)}):")
        for key in failed:
            print(f"  {key}")
        return 1
    return 0


def main(argv=None):
    """
    Entry point: download, normalize and save the parquet file or the whole prefix.

    Parameters:
        argv (list): Arguments to parse, defaults to sys.argv[1:].

    Returns:
        int: 0 on success, 1 on failure.
    """
    args = parse_args(argv)

    if not args.recursive and args.key.rstrip('/') != args.key:
        print("The key ends with a slash, use --recursive to normalize a whole directory")
        return 1

    try:
        schema = load_schema_file(file_path=args.schema)
    except (OSError, ValueError) as err:
        print(f"Failed to load schema: {err}")
        return 1
    print(f"Schema loaded, {len(schema)} columns declared")
    if args.no_dry_run:
        print(f"Running with --no-dry-run: the normalized files will replace the objects "
              f"in s3://{bucket_name(args.bucket)}")
    else:
        print("Running as a dry run: the bucket will not be modified")

    client = boto3.client('s3')
    if args.recursive:
        return normalize_recursive(args, schema, client)
    return normalize_single(args, schema, client)


if __name__ == "__main__":
    sys.exit(main())
