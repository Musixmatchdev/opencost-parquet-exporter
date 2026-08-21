""" Test cases for the parquet normalization tool."""
import os
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import pandas as pd
from normalize_parquet import (
    download_from_s3,
    list_parquet_keys,
    normalize_dataframe,
    main,
)


class TestDownloadFromS3(unittest.TestCase):
    """Test cases for download_from_s3 method"""

    def test_strips_scheme_and_leading_slash(self):
        """ Test that the bucket scheme and the leading slash of the key are removed """
        with patch('normalize_parquet.boto3.client') as mocked_client:
            destination = download_from_s3(
                's3://my-bucket/', '/year=2026/opencost.parquet', '/tmp')
            mocked_client.return_value.download_file.assert_called_once_with(
                'my-bucket', 'year=2026/opencost.parquet', '/tmp/opencost.parquet')
            self.assertEqual(destination, '/tmp/opencost.parquet')


class TestListParquetKeys(unittest.TestCase):
    """Test cases for list_parquet_keys method"""

    def paginated_client(self, pages):
        """ Return an S3 client stub whose paginator yields the given pages """
        client = MagicMock()
        client.get_paginator.return_value.paginate.return_value = pages
        return client

    def test_lists_only_parquet_files_across_pages(self):
        """ Test that every page is consumed and non parquet objects are counted """
        client = self.paginated_client([
            {'Contents': [{'Key': 'p/day=19/b.parquet'}, {'Key': 'p/day=19/_SUCCESS'}]},
            {'Contents': [{'Key': 'p/day=20/a.parquet'}, {'Key': 'p/day=20/'}]},
        ])
        keys, skipped = list_parquet_keys('s3://my-bucket', 'p', client=client)

        self.assertEqual(keys, ['p/day=19/b.parquet', 'p/day=20/a.parquet'])
        self.assertEqual(skipped, 1)
        client.get_paginator.return_value.paginate.assert_called_once_with(
            Bucket='my-bucket', Prefix='p/')

    def test_empty_prefix_lists_the_whole_bucket(self):
        """ Test that an empty prefix is not turned into a lone slash """
        client = self.paginated_client([{'Contents': [{'Key': 'a.parquet'}]}])
        keys, _ = list_parquet_keys('my-bucket', '/', client=client)

        self.assertEqual(keys, ['a.parquet'])
        client.get_paginator.return_value.paginate.assert_called_once_with(
            Bucket='my-bucket', Prefix='')


class TestNormalizeDataframe(unittest.TestCase):
    """Test cases for normalize_dataframe method"""

    def setUp(self):
        self.schema = {
            'properties.namespace': {'type': 'string', 'default': 'unknown'},
            'totalCost': {'type': 'float', 'default': 0.0},
            'gpuCost': {'type': 'float', 'default': 0.0},
        }

    def test_drops_adds_and_fills(self):
        """ Test that extra columns are dropped, missing ones added and nulls filled """
        data = pd.DataFrame({
            'totalCost': [1.0, None],
            'properties.namespace': ['kube-system', None],
            'properties.annotations.checksum_cm': [None, 'abc'],
        })
        normalized, summary = normalize_dataframe(data, self.schema)

        self.assertEqual(list(normalized.columns), list(self.schema.keys()))
        self.assertEqual(summary['rows'], 2)
        self.assertEqual(summary['dropped'], ['properties.annotations.checksum_cm'])
        self.assertEqual(summary['added'], ['gpuCost'])
        self.assertEqual(summary['filled'], {'properties.namespace': 1, 'totalCost': 1})
        self.assertEqual(normalized['totalCost'].tolist(), [1.0, 0.0])
        self.assertEqual(
            normalized['properties.namespace'].tolist(), ['kube-system', 'unknown'])
        self.assertEqual(normalized['gpuCost'].tolist(), [0.0, 0.0])

    def test_already_normalized_data_is_unchanged(self):
        """ Test that a file already matching the schema reports no change """
        data = pd.DataFrame({
            'properties.namespace': ['kube-system'],
            'totalCost': [1.0],
            'gpuCost': [0.0],
        })
        normalized, summary = normalize_dataframe(data, self.schema)

        self.assertEqual(summary['dropped'], [])
        self.assertEqual(summary['added'], [])
        self.assertEqual(summary['filled'], {})
        self.assertEqual(normalized['totalCost'].tolist(), [1.0])


class TestMain(unittest.TestCase):
    """Test cases for the command line entry point"""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.source = os.path.join(self.test_dir, 'source.parquet')
        pd.DataFrame({
            'totalCost': [1.0, None],
            'dropMe': ['a', 'b'],
        }).to_parquet(self.source, index=False)
        self.schema_path = os.path.join(self.test_dir, 'schema.yaml')
        with open(self.schema_path, 'w', encoding='utf-8') as file:
            file.write('totalCost: { type: float, default: 0.0 }\n'
                       'gpuCost: { type: float, default: 0.0 }\n')

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def fake_client(self, *_args, **_kwargs):
        """ Return an S3 client stub that copies the local source file """
        source = self.source

        class FakeS3:
            """ Minimal S3 client stub """
            # pylint: disable=R0903
            def download_file(self, _bucket, _key, destination):
                """ Copy the local source file to the requested destination """
                shutil.copyfile(source, destination)

        return FakeS3()

    def test_main_writes_normalized_file_with_the_same_name(self):
        """ Test the whole flow, the output keeps the name of the object in the bucket """
        output_dir = os.path.join(self.test_dir, 'out')
        with patch('normalize_parquet.boto3.client', side_effect=self.fake_client):
            exit_code = main([
                '-b', 's3://my-bucket',
                '-k', 'year=2026/month=8/day=20/opencost_2026-08-20_10_00_00.parquet',
                '-s', self.schema_path,
                '-o', output_dir,
            ])

        self.assertEqual(exit_code, 0)
        output_path = os.path.join(
            output_dir, 'opencost_2026-08-20_10_00_00.parquet')
        self.assertTrue(os.path.exists(output_path))
        result = pd.read_parquet(output_path)
        self.assertEqual(list(result.columns), ['totalCost', 'gpuCost'])
        self.assertEqual(result['totalCost'].tolist(), [1.0, 0.0])

    def test_dry_run_is_the_default_and_uploads_nothing(self):
        """ Test that without --no-dry-run the bucket is never written to """
        client = MagicMock()
        client.download_file.side_effect = (
            lambda _bucket, _key, destination: shutil.copyfile(self.source, destination))
        with patch('normalize_parquet.boto3.client', return_value=client):
            exit_code = main(['-b', 'my-bucket', '-k', 'a/b.parquet',
                              '-s', self.schema_path, '-o', self.test_dir])

        self.assertEqual(exit_code, 0)
        client.upload_file.assert_not_called()

    def test_no_dry_run_replaces_the_object(self):
        """ Test that --no-dry-run uploads the normalized file over the original key """
        client = MagicMock()
        client.download_file.side_effect = (
            lambda _bucket, _key, destination: shutil.copyfile(self.source, destination))
        output_dir = os.path.join(self.test_dir, 'out')
        with patch('normalize_parquet.boto3.client', return_value=client):
            exit_code = main(['-b', 's3://my-bucket', '-k', '/a/b.parquet',
                              '-s', self.schema_path, '-o', output_dir, '--no-dry-run'])

        self.assertEqual(exit_code, 0)
        client.upload_file.assert_called_once_with(
            os.path.join(output_dir, 'b.parquet'), 'my-bucket', 'a/b.parquet')

    def test_no_dry_run_replaces_every_object_of_a_directory(self):
        """ Test that --no-dry-run uploads each normalized file over its own key """
        keys = ['p/day=19/a.parquet', 'p/day=20/b.parquet']
        client = self.recursive_client(keys)
        output_dir = os.path.join(self.test_dir, 'out')
        with patch('normalize_parquet.boto3.client', return_value=client):
            exit_code = main(['-b', 'my-bucket', '-r', '-k', 'p', '-s', self.schema_path,
                              '-o', output_dir, '--no-dry-run'])

        self.assertEqual(exit_code, 0)
        self.assertEqual(
            [call.args for call in client.upload_file.call_args_list],
            [(os.path.join(output_dir, 'day=19/a.parquet'), 'my-bucket', keys[0]),
             (os.path.join(output_dir, 'day=20/b.parquet'), 'my-bucket', keys[1])])

    def test_broken_file_is_not_uploaded(self):
        """ Test that a file that fails to normalize is never uploaded """
        broken = os.path.join(self.test_dir, 'broken.parquet')
        with open(broken, 'w', encoding='utf-8') as file:
            file.write('not a parquet')
        client = MagicMock()
        client.download_file.side_effect = (
            lambda _bucket, _key, destination: shutil.copyfile(broken, destination))
        with patch('normalize_parquet.boto3.client', return_value=client):
            exit_code = main(['-b', 'my-bucket', '-k', 'a/b.parquet', '-s', self.schema_path,
                              '-o', self.test_dir, '--no-dry-run'])

        self.assertEqual(exit_code, 1)
        client.upload_file.assert_not_called()

    def test_main_fails_on_missing_schema(self):
        """ Test that a missing schema file aborts with a non zero exit code """
        exit_code = main([
            '-b', 'my-bucket', '-k', 'a.parquet',
            '-s', os.path.join(self.test_dir, 'nonexistent.yaml'),
        ])
        self.assertEqual(exit_code, 1)

    def recursive_client(self, keys):
        """ Return an S3 client stub listing the given keys and serving the source file """
        client = MagicMock()
        client.get_paginator.return_value.paginate.return_value = [
            {'Contents': [{'Key': key} for key in keys]}]
        client.download_file.side_effect = (
            lambda _bucket, _key, destination: shutil.copyfile(self.source, destination))
        return client

    def test_recursive_recreates_the_directory_structure(self):
        """ Test that every parquet under the key is normalized keeping its partitions """
        keys = [
            'cluster=prod/year=2026/month=8/day=19/opencost_23_00_00.parquet',
            'cluster=prod/year=2026/month=8/day=20/opencost_09_00_00.parquet',
            'cluster=prod/year=2026/month=8/day=20/_SUCCESS',
        ]
        output_dir = os.path.join(self.test_dir, 'out')
        with patch('normalize_parquet.boto3.client',
                   return_value=self.recursive_client(keys)):
            exit_code = main(['-b', 'my-bucket', '--recursive', '-k', 'cluster=prod/',
                              '-s', self.schema_path, '-o', output_dir])

        self.assertEqual(exit_code, 0)
        for relative_path in ('year=2026/month=8/day=19/opencost_23_00_00.parquet',
                              'year=2026/month=8/day=20/opencost_09_00_00.parquet'):
            path = os.path.join(output_dir, relative_path)
            self.assertTrue(os.path.exists(path), f"{relative_path} was not written")
            self.assertEqual(
                list(pd.read_parquet(path).columns), ['totalCost', 'gpuCost'])
        self.assertFalse(os.path.exists(
            os.path.join(output_dir, 'year=2026/month=8/day=20/_SUCCESS')))

    def test_recursive_reports_failures_but_keeps_going(self):
        """ Test that a broken file does not stop the batch and is reported with exit 1 """
        broken = os.path.join(self.test_dir, 'broken.parquet')
        with open(broken, 'w', encoding='utf-8') as file:
            file.write('not a parquet')
        keys = ['p/broken.parquet', 'p/good.parquet']
        client = MagicMock()
        client.get_paginator.return_value.paginate.return_value = [
            {'Contents': [{'Key': key} for key in keys]}]
        client.download_file.side_effect = (
            lambda _bucket, key, destination: shutil.copyfile(
                broken if 'broken' in key else self.source, destination))

        output_dir = os.path.join(self.test_dir, 'out')
        with patch('normalize_parquet.boto3.client', return_value=client):
            exit_code = main(['-b', 'my-bucket', '-r', '-k', 'p',
                              '-s', self.schema_path, '-o', output_dir])

        self.assertEqual(exit_code, 1)
        self.assertTrue(os.path.exists(os.path.join(output_dir, 'good.parquet')))

    def test_recursive_fails_on_empty_prefix(self):
        """ Test that a prefix without parquet files aborts with a non zero exit code """
        client = MagicMock()
        client.get_paginator.return_value.paginate.return_value = [{}]
        with patch('normalize_parquet.boto3.client', return_value=client):
            exit_code = main(['-b', 'my-bucket', '-r', '-k', 'empty/',
                              '-s', self.schema_path, '-o', self.test_dir])
        self.assertEqual(exit_code, 1)

    def test_directory_key_without_recursive_is_rejected(self):
        """ Test that a key ending with a slash requires --recursive """
        exit_code = main(['-b', 'my-bucket', '-k', 'cluster=prod/',
                          '-s', self.schema_path, '-o', self.test_dir])
        self.assertEqual(exit_code, 1)

    def test_main_fails_on_download_error(self):
        """ Test that a download failure aborts with a non zero exit code """
        with patch('normalize_parquet.boto3.client') as mocked_client:
            mocked_client.return_value.download_file.side_effect = OSError("boom")
            exit_code = main([
                '-b', 'my-bucket', '-k', 'a.parquet', '-s', self.schema_path,
                '-o', self.test_dir,
            ])
        self.assertEqual(exit_code, 1)


if __name__ == '__main__':
    unittest.main()
