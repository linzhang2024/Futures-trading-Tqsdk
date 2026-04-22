import os
import sys
import pytest
import tempfile
import yaml
from unittest.mock import Mock, patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.connection import TqConnector


class TestTqConnector:
    @pytest.fixture
    def temp_config_file(self):
        config_data = {
            "tq_sdk": {
                "account": "${TQ_ACCOUNT:test_account}",
                "password": "${TQ_PASSWORD:test_password}"
            },
            "trading": {
                "default_contract": "SHFE.rb2410",
                "contracts": ["SHFE.rb2410", "SHFE.hc2410"]
            },
            "logging": {
                "level": "DEBUG",
                "file": "logs/test.log"
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False, encoding='utf-8') as f:
            yaml.dump(config_data, f)
            temp_path = f.name
        
        yield temp_path
        
        if os.path.exists(temp_path):
            os.unlink(temp_path)

    def test_init_with_valid_config(self, temp_config_file):
        connector = TqConnector(config_path=temp_config_file)
        assert connector.config_path == temp_config_file
        assert connector.api is None
        assert connector.is_connected() is False

    def test_init_with_invalid_config_path(self):
        with pytest.raises(FileNotFoundError):
            TqConnector(config_path="/nonexistent/path.yaml")

    def test_load_config(self, temp_config_file):
        connector = TqConnector(config_path=temp_config_file)
        assert connector.config['tq_sdk']['account'] == 'test_account'
        assert connector.config['tq_sdk']['password'] == 'test_password'
        assert connector.config['trading']['default_contract'] == 'SHFE.rb2410'

    def test_resolve_env_vars_with_env_var_set(self, temp_config_file):
        with patch.dict(os.environ, {'TQ_ACCOUNT': 'env_account', 'TQ_PASSWORD': 'env_password'}):
            connector = TqConnector(config_path=temp_config_file)
            assert connector.config['tq_sdk']['account'] == 'env_account'
            assert connector.config['tq_sdk']['password'] == 'env_password'

    def test_get_default_contract(self, temp_config_file):
        connector = TqConnector(config_path=temp_config_file)
        assert connector.get_default_contract() == 'SHFE.rb2410'

    def test_get_contracts(self, temp_config_file):
        connector = TqConnector(config_path=temp_config_file)
        contracts = connector.get_contracts()
        assert len(contracts) == 2
        assert "SHFE.rb2410" in contracts
        assert "SHFE.hc2410" in contracts

    @patch('core.connection.TqAuth')
    @patch('core.connection.TqApi')
    def test_connect_success(self, mock_tqapi, mock_tqauth, temp_config_file):
        mock_api = Mock()
        mock_api.is_closed.return_value = False
        mock_tqapi.return_value = mock_api
        
        connector = TqConnector(config_path=temp_config_file)
        api = connector.connect()
        
        assert mock_tqauth.called
        mock_tqauth.assert_called_once_with('test_account', 'test_password')
        
        assert mock_tqapi.called
        assert connector.is_connected() is True
        assert api == mock_api

    @patch('core.connection.TqAuth')
    @patch('core.connection.TqApi')
    def test_connect_already_connected(self, mock_tqapi, mock_tqauth, temp_config_file):
        mock_api = Mock()
        mock_api.is_closed.return_value = False
        mock_tqapi.return_value = mock_api
        
        connector = TqConnector(config_path=temp_config_file)
        api1 = connector.connect()
        api2 = connector.connect()
        
        assert mock_tqapi.call_count == 1
        assert api1 == api2

    @patch('core.connection.TqAuth')
    @patch('core.connection.TqApi')
    def test_connect_with_exception(self, mock_tqapi, mock_tqauth, temp_config_file):
        mock_tqapi.side_effect = Exception("Connection failed")
        
        connector = TqConnector(config_path=temp_config_file)
        
        with pytest.raises(Exception) as exc_info:
            connector.connect()
        
        assert "Connection failed" in str(exc_info.value)
        assert connector.is_connected() is False

    @patch('core.connection.TqAuth')
    @patch('core.connection.TqApi')
    def test_disconnect(self, mock_tqapi, mock_tqauth, temp_config_file):
        mock_api = Mock()
        mock_api.is_closed.return_value = False
        mock_tqapi.return_value = mock_api
        
        connector = TqConnector(config_path=temp_config_file)
        connector.connect()
        
        assert connector.is_connected() is True
        
        connector.disconnect()
        
        mock_api.close.assert_called_once()
        assert connector.is_connected() is False
        assert connector.api is None

    @patch('core.connection.TqAuth')
    @patch('core.connection.TqApi')
    def test_context_manager(self, mock_tqapi, mock_tqauth, temp_config_file):
        mock_api = Mock()
        mock_api.is_closed.return_value = False
        mock_tqapi.return_value = mock_api
        
        with TqConnector(config_path=temp_config_file) as connector:
            assert connector.is_connected() is True
            assert connector.get_api() == mock_api
        
        mock_api.close.assert_called_once()
        assert connector.is_connected() is False

    def test_resolve_env_vars_nested_structure(self):
        config = {
            "level1": {
                "level2": "${TEST_VAR:default}",
                "list": ["${TEST_VAR:default}", "static"]
            }
        }
        
        with patch.dict(os.environ, {'TEST_VAR': 'env_value'}):
            connector = TqConnector.__new__(TqConnector)
            connector.config = {}
            result = connector._resolve_env_vars(config)
            
            assert result['level1']['level2'] == 'env_value'
            assert result['level1']['list'][0] == 'env_value'
            assert result['level1']['list'][1] == 'static'

    def test_resolve_env_vars_no_match(self):
        config = {
            "key": "static_value",
            "number": 123,
            "bool": True,
            "none": None
        }
        
        connector = TqConnector.__new__(TqConnector)
        connector.config = {}
        result = connector._resolve_env_vars(config)
        
        assert result == config
