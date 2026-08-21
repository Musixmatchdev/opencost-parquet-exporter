""" Test cases for opencost-parquet-exporter."""
import unittest
from unittest.mock import patch, MagicMock, mock_open
import json
import os
from datetime import datetime
import requests
import pandas as pd
from freezegun import freeze_time
from opencost_parquet_exporter import (
    get_config,
    request_data,
    load_config_file,
    load_structured_file,
    load_schema_file,
    apply_schema,
    process_result,
)


class TestGetConfig(unittest.TestCase):
    """Test cases for get_config method"""

    def test_get_aws_config_with_env_vars(self):
        """Test get_config returns correct configurations based on environment variables."""
        with patch.dict(os.environ, {
                'OPENCOST_PARQUET_SVC_HOSTNAME': 'testhost',
                'OPENCOST_PARQUET_SVC_PORT': '8080',
                'OPENCOST_PARQUET_WINDOW_START': '2020-01-01T00:00:00Z',
                'OPENCOST_PARQUET_WINDOW_END': '2020-01-01T23:59:59Z',
                'OPENCOST_PARQUET_S3_BUCKET': 's3://test-bucket',
                'OPENCOST_PARQUET_FILE_KEY_PREFIX': 'test-prefix/',
                'OPENCOST_PARQUET_AGGREGATE': 'namespace',
                'OPENCOST_PARQUET_STEP': '1m',
                'OPENCOST_PARQUET_STORAGE_BACKEND': 'aws'}, clear=True):
            config = get_config()

            self.assertEqual(
                config['url'], 'http://testhost:8080/allocation/compute')
            self.assertEqual(config['params'][0][1],
                             '2020-01-01T00:00:00Z,2020-01-01T23:59:59Z')
            self.assertEqual(config['s3_bucket'], 's3://test-bucket')
            self.assertEqual(config['storage_backend'], 'aws')
            self.assertEqual(config['params'][1][1], 'false')
            self.assertEqual(config['params'][2][1], 'false')

    def test_get_azure_config_with_env_vars(self):
        """Test get_config returns correct configurations based on environment variables."""
        with patch.dict(os.environ, {
                'OPENCOST_PARQUET_SVC_HOSTNAME': 'testhost',
                'OPENCOST_PARQUET_SVC_PORT': '8080',
                'OPENCOST_PARQUET_WINDOW_START': '2020-01-01T00:00:00Z',
                'OPENCOST_PARQUET_WINDOW_END': '2020-01-01T23:59:59Z',
                'OPENCOST_PARQUET_S3_BUCKET': 's3://test-bucket',
                'OPENCOST_PARQUET_FILE_KEY_PREFIX': 'test-prefix/',
                'OPENCOST_PARQUET_AGGREGATE': 'namespace',
                'OPENCOST_PARQUET_STEP': '1m',
                'OPENCOST_PARQUET_STORAGE_BACKEND': 'azure',
                'OPENCOST_PARQUET_AZURE_STORAGE_ACCOUNT_NAME': 'testaccount',
                'OPENCOST_PARQUET_AZURE_CONTAINER_NAME': 'testcontainer',
                'OPENCOST_PARQUET_AZURE_TENANT': 'testtenant',
                'OPENCOST_PARQUET_AZURE_APPLICATION_ID': 'testid',
                'OPENCOST_PARQUET_AZURE_APPLICATION_SECRET': 'testsecret',
                'OPENCOST_PARQUET_IDLE_BY_NODE': 'true',
                'OPENCOST_PARQUET_INCLUDE_IDLE': 'true'}, clear=True):
            config = get_config()

            self.assertEqual(
                config['url'], 'http://testhost:8080/allocation/compute')
            self.assertEqual(config['params'][0][1],
                             '2020-01-01T00:00:00Z,2020-01-01T23:59:59Z')
            self.assertEqual(config['storage_backend'], 'azure')
            self.assertEqual(
                config['azure_storage_account_name'], 'testaccount')
            self.assertEqual(config['azure_container_name'], 'testcontainer')
            self.assertEqual(config['azure_tenant'], 'testtenant')
            self.assertEqual(config['azure_application_id'], 'testid')
            self.assertEqual(config['azure_application_secret'], 'testsecret')
            self.assertEqual(config['params'][1][1], 'true')
            self.assertEqual(config['params'][2][1], 'true')

    def test_get_gcp_config_with_env_vars(self):
        """Test get_config returns correct configurations based on environment variables."""
        with patch.dict(os.environ, {
            'OPENCOST_PARQUET_SVC_HOSTNAME': 'testhost',
            'OPENCOST_PARQUET_SVC_PORT': '8080',
            'OPENCOST_PARQUET_WINDOW_START': '2020-01-01T00:00:00Z',
            'OPENCOST_PARQUET_WINDOW_END': '2020-01-01T23:59:59Z',
            'OPENCOST_PARQUET_S3_BUCKET': 's3://test-bucket',
            'OPENCOST_PARQUET_FILE_KEY_PREFIX': 'test-prefix/',
            'OPENCOST_PARQUET_AGGREGATE': 'namespace',
            'OPENCOST_PARQUET_STEP': '1m',
            'OPENCOST_PARQUET_STORAGE_BACKEND': 'gcp',
            'OPENCOST_PARQUET_GCP_BUCKET_NAME': 'testbucket',
            'OPENCOST_PARQUET_GCP_CREDENTIALS_JSON': '{"type": "service_account"}',
            'OPENCOST_PARQUET_IDLE_BY_NODE': 'true',
                'OPENCOST_PARQUET_INCLUDE_IDLE': 'true'}, clear=True):
            config = get_config()

            self.assertEqual(
                config['url'], 'http://testhost:8080/allocation/compute')
            self.assertEqual(config['params'][0][1],
                             '2020-01-01T00:00:00Z,2020-01-01T23:59:59Z')
            self.assertEqual(config['storage_backend'], 'gcp')
            self.assertEqual(
                config['gcp_bucket_name'], 'testbucket')
            self.assertEqual(config['gcp_credentials'], {
                             'type': 'service_account'})
            self.assertEqual(config['params'][1][1], 'true')
            self.assertEqual(config['params'][2][1], 'true')

    @freeze_time("2024-01-31")
    def test_get_config_defaults_last_day_of_month(self):
        """Test get_config returns correct defaults when no env vars are set."""
        with patch.dict(os.environ, {}, clear=True):
            yesterday = '2024-01-30'
            window_start = yesterday+'T00:00:00Z'
            window_end = yesterday+'T23:59:59Z'
            window = f"{window_start},{window_end}"

            config = get_config()
            self.assertEqual(
                config['url'], 'http://localhost:9003/allocation/compute')
            self.assertTrue(config['file_key_prefix'], '/tmp/')
            self.assertNotIn('s3_bucket', config)
            self.assertEqual(config['params'][0][1], window)

    @freeze_time("2024-02-01")
    def test_get_config_defaults_first_day_of_month(self):
        """Test get_config returns correct defaults when no env vars are set."""
        with patch.dict(os.environ, {}, clear=True):
            yesterday = '2024-01-31'
            window_start = yesterday+'T00:00:00Z'
            window_end = yesterday+'T23:59:59Z'
            window = f"{window_start},{window_end}"

            config = get_config()
            self.assertEqual(
                config['url'], 'http://localhost:9003/allocation/compute')
            self.assertTrue(config['file_key_prefix'], '/tmp/')
            self.assertNotIn('s3_bucket', config)
            self.assertEqual(config['params'][0][1], window)

    @freeze_time("2024-01-11")
    def test_get_config_no_window_start(self):
        """Test get_config returns correct defaults when window start
        is not set. It should set window to yesterday"""
        with patch.dict(os.environ, {
                'OPENCOST_PARQUET_WINDOW_END': '2020-01-01T23:59:59Z'
        }, clear=True):
            yesterday = '2024-01-10'
            window_start = yesterday+'T00:00:00Z'
            window_end = yesterday+'T23:59:59Z'
            window = f"{window_start},{window_end}"

            config = get_config()
            self.assertEqual(
                config['url'], 'http://localhost:9003/allocation/compute')
            self.assertTrue(config['file_key_prefix'], '/tmp/')
            self.assertNotIn('s3_bucket', config)
            self.assertEqual(config['params'][0][1], window)

    @freeze_time('2024-12-20')
    def test_get_config_no_window_end(self):
        """Test get_config returns correct defaults when window end
        is not set. It should set window to yesterday"""
        with patch.dict(os.environ, {
                'OPENCOST_PARQUET_WINDOW_START': '2020-01-01T00:00:00Z'
        }, clear=True):
            yesterday = '2024-12-19'
            window_start = yesterday+'T00:00:00Z'
            window_end = yesterday+'T23:59:59Z'
            window = f"{window_start},{window_end}"

            config = get_config()
            self.assertEqual(
                config['url'], 'http://localhost:9003/allocation/compute')
            self.assertTrue(config['file_key_prefix'], '/tmp/')
            self.assertNotIn('s3_bucket', config)
            self.assertEqual(config['params'][0][1], window)

    def test_get_config_with_idle_env_vars(self):
        """Test get_config overrides parquet prefix when env var is set."""
        with patch.dict(os.environ, {
            'OPENCOST_PARQUET_INCLUDE_IDLE': 'true',
            'OPENCOST_PARQUET_IDLE_BY_NODE': 'true'}, clear=True):
            config = get_config()
            self.assertEqual(config['params'][2][0], 'includeIdle')
            self.assertEqual(config['params'][2][1], 'true')

            self.assertEqual(config['params'][3][0], 'idleByNode')
            self.assertEqual(config['params'][3][1], 'true')

    @freeze_time('2024-06-29')
    def test_get_config_with_parquet_prefix_env_var(self):
        yesterday = '2024-06-28'
        window_start = yesterday+'T01:00:00Z'
        window = datetime.strptime(window_start, "%Y-%m-%dT%H:%M:%SZ")

        """Test get_config overrides parquet prefix when env var is set."""
        with patch.dict(os.environ, {
            'OPENCOST_PARQUET_FILE_KEY_PREFIX': 'test-prefix/',
            'OPENCOST_PARQUET_PREFIX': f"year={window.year}/month={window.month}/day={window.day}/hour={window.hour}",
            'OPENCOST_PARQUET_STEP': '1m'}, clear=True):
            config = get_config()
            self.assertEqual(config['parquet_prefix'], 'year=2024/month=6/day=28/hour=1')

    @freeze_time('2024-06-29')
    def test_get_config_with_default_parquet_prefix_env_var(self):
        """Test get_config with default parquet prefix when env var is not set."""
        config = get_config()
        self.assertEqual(config['parquet_prefix'], '/tmp//year=2024/month=6/day=28')

class TestRequestData(unittest.TestCase):
    """ Test request_data method """
    @patch('opencost_parquet_exporter.requests.get')
    def test_request_data_success(self, mock_get):
        """Test request_data successfully retrieves data when response is OK."""
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.headers = {'content-type': 'application/json'}
        mock_response.json.return_value = {'data': [{'key': 'value'}]}
        mock_get.return_value = mock_response

        config = {
            'url': 'http://testurl',
            'params': (
                ('sample_param',  'value')
            )
        }
        data = request_data(config)
        self.assertEqual(data, [{'key': 'value'}])

    @patch('opencost_parquet_exporter.requests.get')
    def test_request_data_wrong_content_type(self, mock_get):
        """Test request_data successfully retrieves data when response is OK."""
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.headers = {'content-type': 'text/html'}
        mock_response.json.return_value = {'data': ['something']}
        mock_get.return_value = mock_response

        config = {
            'url': 'http://testurl',
            'params': (
                ('sample_param',  'value')
            )
        }
        data = request_data(config)
        self.assertEqual(data, None)

    @patch('opencost_parquet_exporter.requests.get')
    def test_request_data_failure(self, mock_get):
        """Test request_data returns None when there is a RequestException."""
        mock_get.side_effect = requests.RequestException

        config = {
            'url': 'http://testurl',
            'params': (
                ('sample_param',  'value')
            )
        }
        data = request_data(config)
        self.assertIsNone(data)


class TestLoadConfigMaps(unittest.TestCase):
    """Test cases for load_config_file method"""

    def setUp(self):
        # Setup a temporary directory and example config data for testing
        self.test_dir = 'test_configs'
        os.makedirs(self.test_dir, exist_ok=True)
        self.valid_json_path = os.path.join(self.test_dir, 'valid_config.json')
        self.invalid_json_path = os.path.join(
            self.test_dir, 'invalid_config.json')
        self.empty_json_path = os.path.join(self.test_dir, 'empty.json')

        # Create a valid JSON file
        with open(self.valid_json_path, 'w', encoding='utf-8') as f:
            json.dump({"key": "value"}, f)

        # Create an invalid JSON file
        with open(self.invalid_json_path, 'w', encoding='utf-8') as f:
            f.write('{"key": "value",}')

        # Create an empty JSON file
        with open(self.empty_json_path, 'w', encoding='utf-8') as f:
            pass

    def tearDown(self):
        # Remove the directory after tests
        for file in os.listdir(self.test_dir):
            os.remove(os.path.join(self.test_dir, file))
        os.rmdir(self.test_dir)

    def test_successful_load(self):
        """ Test loading a valid JSON file """
        result = load_config_file(self.valid_json_path)
        self.assertEqual(result, {"key": "value"})

    def test_file_not_found(self):
        """ Test the response when the JSON file does not exist """
        with self.assertRaises(FileNotFoundError):
            load_config_file('nonexistent.json')

    def test_permission_error(self):
        """ Test the response to inadequate file permissions """
        # Simulate permission error by patching os.open
        with patch('builtins.open', mock_open()) as mocked_file:
            mocked_file.side_effect = PermissionError("Permission denied")
            with self.assertRaises(PermissionError):
                load_config_file(self.valid_json_path)

    def test_invalid_json_format(self):
        """ Test how it handles a file with invalid JSON """
        with self.assertRaises(json.JSONDecodeError):
            load_config_file(self.invalid_json_path)

    def test_empty_file(self):
        """ Test the function's response to an empty JSON file """
        with self.assertRaises(json.JSONDecodeError):
            load_config_file(self.empty_json_path)


class TestLoadSchemaFile(unittest.TestCase):
    """Test cases for load_structured_file and load_schema_file methods"""

    def setUp(self):
        self.test_dir = 'test_schemas'
        os.makedirs(self.test_dir, exist_ok=True)

    def tearDown(self):
        for file in os.listdir(self.test_dir):
            os.remove(os.path.join(self.test_dir, file))
        os.rmdir(self.test_dir)

    def write_file(self, name, content):
        """ Write content to a file inside the test directory and return its path """
        path = os.path.join(self.test_dir, name)
        with open(path, 'w', encoding='utf-8') as file:
            file.write(content)
        return path

    def test_load_structured_file_json(self):
        """ Test loading a JSON file """
        path = self.write_file('schema.json', '{"cpuCost": "float"}')
        self.assertEqual(load_structured_file(path), {"cpuCost": "float"})

    def test_load_structured_file_yaml(self):
        """ Test loading a YAML file """
        path = self.write_file(
            'schema.yaml', 'cpuCost:\n  type: float\n  default: 0.0\n')
        self.assertEqual(
            load_structured_file(path),
            {"cpuCost": {"type": "float", "default": 0.0}})

    def test_load_structured_file_unknown_extension(self):
        """ Test that an unknown extension falls back to JSON then YAML """
        json_path = self.write_file('schema.txt', '{"cpuCost": "float"}')
        self.assertEqual(load_structured_file(json_path), {"cpuCost": "float"})
        yaml_path = self.write_file('other.txt', 'cpuCost: float\n')
        self.assertEqual(load_structured_file(yaml_path), {"cpuCost": "float"})

    def test_load_structured_file_invalid_json(self):
        """ Test that invalid JSON raises a ValueError """
        path = self.write_file('schema.json', '{"cpuCost": "float",}')
        with self.assertRaises(ValueError):
            load_structured_file(path)

    def test_load_schema_file_normalizes_definitions(self):
        """ Test that plain type strings and objects are both accepted """
        path = self.write_file('schema.yaml', '\n'.join([
            'properties.namespace:',
            '  type: string',
            '  default: ""',
            'totalCost: float',
            'label.team:',
            '  type: string',
        ]))
        schema = load_schema_file(path)
        self.assertEqual(schema, {
            'properties.namespace': {'type': 'string', 'default': ''},
            'totalCost': {'type': 'float'},
            'label.team': {'type': 'string'},
        })
        # The declaration order is preserved, it drives the parquet column order.
        self.assertEqual(
            list(schema.keys()),
            ['properties.namespace', 'totalCost', 'label.team'])

    def test_load_schema_file_empty(self):
        """ Test that an empty schema raises a ValueError """
        path = self.write_file('schema.json', '{}')
        with self.assertRaises(ValueError):
            load_schema_file(path)

    def test_load_schema_file_missing_type(self):
        """ Test that a column definition without a type raises a ValueError """
        path = self.write_file('schema.yaml', 'cpuCost:\n  default: 0.0\n')
        with self.assertRaises(ValueError):
            load_schema_file(path)

    def test_load_schema_file_unsupported_key(self):
        """ Test that an unsupported key in a column definition raises a ValueError """
        path = self.write_file(
            'schema.yaml', 'cpuCost:\n  type: float\n  fallback: 0.0\n')
        with self.assertRaises(ValueError):
            load_schema_file(path)

    def test_load_schema_file_not_found(self):
        """ Test the response when the schema file does not exist """
        with self.assertRaises(FileNotFoundError):
            load_schema_file(os.path.join(self.test_dir, 'nonexistent.yaml'))


class TestApplySchema(unittest.TestCase):
    """Test cases for apply_schema method"""

    def test_projects_orders_and_casts_columns(self):
        """ Test that only the declared columns are kept, in the declared order """
        data = pd.DataFrame({
            'totalCost': ['1.5'],
            'dropMe': ['whatever'],
            'properties.namespace': ['kube-system'],
        })
        schema = {
            'properties.namespace': {'type': 'string', 'default': ''},
            'totalCost': {'type': 'float', 'default': 0.0},
        }
        result = apply_schema(data, schema)
        self.assertEqual(
            list(result.columns), ['properties.namespace', 'totalCost'])
        self.assertEqual(result['totalCost'].tolist(), [1.5])

    def test_creates_missing_columns_with_default(self):
        """ Test that columns missing from the data are created with their default """
        data = pd.DataFrame({'totalCost': [2.0]})
        schema = {
            'totalCost': {'type': 'float', 'default': 0.0},
            'label.team': {'type': 'string', 'default': 'unknown'},
        }
        result = apply_schema(data, schema)
        self.assertEqual(result['label.team'].tolist(), ['unknown'])

    def test_fills_null_values_with_default(self):
        """ Test that null values of existing columns are replaced by the default """
        data = pd.DataFrame({'totalCost': [None, 3.0]})
        result = apply_schema(data, {'totalCost': {'type': 'float', 'default': 0.0}})
        self.assertEqual(result['totalCost'].tolist(), [0.0, 3.0])

    def test_missing_column_without_default_stays_null(self):
        """ Test that a column without a default is created as null """
        data = pd.DataFrame({'totalCost': [1.0]})
        result = apply_schema(data, {
            'totalCost': {'type': 'float'},
            'gpuCost': {'type': 'float'},
        })
        self.assertTrue(result['gpuCost'].isna().all())

    def test_invalid_cast_raises_value_error(self):
        """ Test that a column that cannot be cast raises a ValueError """
        data = pd.DataFrame({'totalCost': ['not-a-number']})
        with self.assertRaises(ValueError):
            apply_schema(data, {'totalCost': {'type': 'float'}})


class TestProcessResultWithSchema(unittest.TestCase):
    """Test cases for process_result when a parquet schema is provided"""

    def setUp(self):
        self.result = [{
            'kube-system/pod-1/container-1': {
                'name': 'kube-system/pod-1/container-1',
                'minutes': 60,
                'totalCost': 1.25,
                'pvs': {'some': 'value'},
                'properties': {'namespace': 'kube-system', 'pod': 'pod-1'},
            },
        }]

    def test_schema_replaces_data_types(self):
        """ Test that the schema drives the exported columns instead of data_types """
        schema = {
            'properties.namespace': {'type': 'string', 'default': ''},
            'running_minutes': {'type': 'float', 'default': 0.0},
            'gpuCost': {'type': 'float', 'default': 0.0},
        }
        processed = process_result(
            result=self.result,
            ignored_alloc_keys=['pvs'],
            rename_cols={'minutes': 'running_minutes'},
            data_types={'this': 'would-fail'},
            schema=schema,
        )
        self.assertEqual(
            list(processed.columns),
            ['properties.namespace', 'running_minutes', 'gpuCost'])
        self.assertEqual(processed['running_minutes'].tolist(), [60.0])
        self.assertEqual(processed['gpuCost'].tolist(), [0.0])

    def test_without_schema_data_types_are_used(self):
        """ Test that the previous behaviour is preserved when no schema is given """
        processed = process_result(
            result=self.result,
            ignored_alloc_keys=['pvs'],
            rename_cols={'minutes': 'running_minutes'},
            data_types={'running_minutes': 'float', 'totalCost': 'float'},
        )
        self.assertIn('properties.pod', processed.columns)
        self.assertEqual(processed['running_minutes'].tolist(), [60.0])


if __name__ == '__main__':
    unittest.main()
