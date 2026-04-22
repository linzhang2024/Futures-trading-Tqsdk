import os
import sys
import pytest
import tempfile
import threading
import time
import yaml
import logging
from unittest.mock import Mock, patch, MagicMock, call

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.connection import TqConnector

USE_REAL_CONNECTION = os.environ.get('USE_REAL_CONNECTION', 'false').lower() == 'true'
TEST_TQ_ACCOUNT = os.environ.get('TEST_TQ_ACCOUNT', '')
TEST_TQ_PASSWORD = os.environ.get('TEST_TQ_PASSWORD', '')


class TestTqConnector:
    @pytest.fixture
    def temp_config_file(self):
        config_data = {
            "tq_sdk": {
                "account": "${TQ_ACCOUNT:test_account}",
                "password": "${TQ_PASSWORD:test_password}"
            },
            "env": {
                "mode": "sim",
                "sim": {
                    "init_balance": 1000000.0,
                    "account_type": "tqsim"
                },
                "real": {
                    "broker_id": "test_broker",
                    "futures_account": "test_futures_account",
                    "futures_password": "test_futures_password"
                }
            },
            "trading": {
                "default_contract": "SHFE.rb2410",
                "contracts": ["SHFE.rb2410", "SHFE.hc2410"]
            },
            "connection": {
                "retry_times": 3,
                "initial_retry_delay": 2,
                "max_retry_delay": 30
            },
            "logging": {
                "level": "DEBUG",
                "file": "logs/test.log",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False, encoding='utf-8') as f:
            yaml.dump(config_data, f)
            temp_path = f.name
        
        yield temp_path
        
        if os.path.exists(temp_path):
            os.unlink(temp_path)

    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        TqConnector.reset_instance()
        TqConnector._logger_initialized = False
        for handler in logging.getLogger('TqConnector').handlers[:]:
            logging.getLogger('TqConnector').removeHandler(handler)
        yield
        TqConnector.reset_instance()
        TqConnector._logger_initialized = False
        for handler in logging.getLogger('TqConnector').handlers[:]:
            logging.getLogger('TqConnector').removeHandler(handler)

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

    @pytest.mark.skipif(USE_REAL_CONNECTION, reason="Mock测试被跳过，使用真实连接")
    @patch('core.connection.TqAuth')
    @patch('core.connection.TqSim')
    @patch('core.connection.TqApi')
    def test_connect_success_mock_sim_mode(self, mock_tqapi, mock_tqsim, mock_tqauth, temp_config_file):
        mock_api = Mock()
        mock_api.is_closed.return_value = False
        mock_tqapi.return_value = mock_api
        
        mock_sim = Mock()
        mock_tqsim.return_value = mock_sim
        
        connector = TqConnector(config_path=temp_config_file)
        api = connector.connect()
        
        assert mock_tqauth.called
        mock_tqauth.assert_called_once_with('test_account', 'test_password')
        
        assert mock_tqsim.called
        mock_tqsim.assert_called_once_with(init_balance=1000000.0)
        
        assert mock_tqapi.called
        mock_tqapi.assert_called_once_with(account=mock_sim, auth=mock_tqauth.return_value)
        
        assert connector.is_connected() is True
        assert api == mock_api

    @pytest.mark.skipif(USE_REAL_CONNECTION, reason="Mock测试被跳过，使用真实连接")
    @patch('core.connection.TqAuth')
    @patch('core.connection.TqKq')
    @patch('core.connection.TqApi')
    def test_connect_success_mock_tqkq_mode(self, mock_tqapi, mock_tqkq, mock_tqauth):
        config_data = {
            "tq_sdk": {"account": "test", "password": "test"},
            "env": {"mode": "sim", "sim": {"init_balance": 1000000.0, "account_type": "tqkq"}},
            "trading": {"default_contract": "SHFE.rb2410", "contracts": []},
            "connection": {"retry_times": 3, "initial_retry_delay": 2, "max_retry_delay": 30},
            "logging": {"level": "DEBUG", "file": "logs/test.log"}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False, encoding='utf-8') as f:
            yaml.dump(config_data, f)
            temp_path = f.name
        
        try:
            mock_api = Mock()
            mock_api.is_closed.return_value = False
            mock_tqapi.return_value = mock_api
            
            mock_kq = Mock()
            mock_tqkq.return_value = mock_kq
            
            connector = TqConnector(config_path=temp_path)
            api = connector.connect()
            
            assert mock_tqkq.called
            assert connector.is_connected() is True
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    @pytest.mark.skipif(USE_REAL_CONNECTION, reason="Mock测试被跳过，使用真实连接")
    @patch('core.connection.TqAuth')
    @patch('core.connection.TqAccount')
    @patch('core.connection.TqApi')
    def test_connect_success_mock_real_mode(self, mock_tqapi, mock_tqaccount, mock_tqauth):
        config_data = {
            "tq_sdk": {"account": "test", "password": "test"},
            "env": {
                "mode": "real",
                "sim": {"init_balance": 1000000.0},
                "real": {
                    "broker_id": "H海通期货",
                    "futures_account": "022631",
                    "futures_password": "123456"
                }
            },
            "trading": {"default_contract": "SHFE.rb2410", "contracts": []},
            "connection": {"retry_times": 3, "initial_retry_delay": 2, "max_retry_delay": 30},
            "logging": {"level": "DEBUG", "file": "logs/test.log"}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False, encoding='utf-8') as f:
            yaml.dump(config_data, f)
            temp_path = f.name
        
        try:
            mock_api = Mock()
            mock_api.is_closed.return_value = False
            mock_tqapi.return_value = mock_api
            
            mock_account = Mock()
            mock_tqaccount.return_value = mock_account
            
            connector = TqConnector(config_path=temp_path)
            api = connector.connect()
            
            assert mock_tqaccount.called
            mock_tqaccount.assert_called_once_with("H海通期货", "022631", "123456")
            assert connector.is_connected() is True
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    @pytest.mark.skipif(USE_REAL_CONNECTION, reason="Mock测试被跳过，使用真实连接")
    @patch('core.connection.TqAuth')
    @patch('core.connection.TqSim')
    @patch('core.connection.TqApi')
    def test_connect_already_connected_mock(self, mock_tqapi, mock_tqsim, mock_tqauth, temp_config_file):
        mock_api = Mock()
        mock_api.is_closed.return_value = False
        mock_tqapi.return_value = mock_api
        
        connector = TqConnector(config_path=temp_config_file)
        api1 = connector.connect()
        api2 = connector.connect()
        
        assert mock_tqapi.call_count == 1
        assert api1 == api2

    @pytest.mark.skipif(USE_REAL_CONNECTION, reason="Mock测试被跳过，使用真实连接")
    @patch('core.connection.TqAuth')
    @patch('core.connection.TqSim')
    @patch('core.connection.TqApi')
    def test_disconnect_mock(self, mock_tqapi, mock_tqsim, mock_tqauth, temp_config_file):
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

    @pytest.mark.skipif(USE_REAL_CONNECTION, reason="Mock测试被跳过，使用真实连接")
    @patch('core.connection.TqAuth')
    @patch('core.connection.TqSim')
    @patch('core.connection.TqApi')
    def test_context_manager_mock(self, mock_tqapi, mock_tqsim, mock_tqauth, temp_config_file):
        mock_api = Mock()
        mock_api.is_closed.return_value = False
        mock_tqapi.return_value = mock_api
        
        with TqConnector(config_path=temp_config_file) as connector:
            assert connector.is_connected() is True
            assert connector.get_api() == mock_api
        
        mock_api.close.assert_called_once()
        assert connector.is_connected() is False

    def test_resolve_env_vars_nested_structure(self, temp_config_file):
        config = {
            "level1": {
                "level2": "${TEST_VAR:default}",
                "list": ["${TEST_VAR:default}", "static"]
            }
        }
        
        with patch.dict(os.environ, {'TEST_VAR': 'env_value'}):
            connector = TqConnector.__new__(TqConnector)
            connector.config = {}
            connector.api = None
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
        connector.api = None
        result = connector._resolve_env_vars(config)
        
        assert result == config

    def test_calculate_exponential_backoff(self, temp_config_file):
        connector = TqConnector(config_path=temp_config_file)
        
        assert connector._calculate_exponential_backoff(1) == 2
        assert connector._calculate_exponential_backoff(2) == 4
        assert connector._calculate_exponential_backoff(3) == 8
        assert connector._calculate_exponential_backoff(4) == 16
        assert connector._calculate_exponential_backoff(5) == 30

    @pytest.mark.skipif(USE_REAL_CONNECTION, reason="Mock测试被跳过，使用真实连接")
    @patch('core.connection.TqAuth')
    @patch('core.connection.TqSim')
    @patch('core.connection.TqApi')
    def test_singleton_pattern_mock(self, mock_tqapi, mock_tqsim, mock_tqauth, temp_config_file):
        mock_api = Mock()
        mock_api.is_closed.return_value = False
        mock_tqapi.return_value = mock_api
        
        connector1 = TqConnector(config_path=temp_config_file)
        connector2 = TqConnector(config_path=temp_config_file)
        
        assert connector1 is connector2
        assert id(connector1) == id(connector2)
        
        TqConnector.reset_instance()
        connector3 = TqConnector(config_path=temp_config_file)
        assert connector1 is not connector3

    @pytest.mark.skipif(USE_REAL_CONNECTION, reason="Mock测试被跳过，使用真实连接")
    def test_connect_with_missing_credentials_mock(self):
        config_data = {
            "tq_sdk": {
                "account": "",
                "password": ""
            },
            "env": {"mode": "sim", "sim": {"init_balance": 1000000.0}},
            "trading": {"default_contract": "SHFE.rb2410", "contracts": []},
            "connection": {"retry_times": 3, "initial_retry_delay": 2, "max_retry_delay": 30},
            "logging": {"level": "DEBUG", "file": "logs/test.log"}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False, encoding='utf-8') as f:
            yaml.dump(config_data, f)
            temp_path = f.name
        
        try:
            connector = TqConnector(config_path=temp_path)
            with pytest.raises(ValueError, match="缺少天勤账号或密码配置"):
                connector.connect()
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_thread_safe_singleton_creation(self, temp_config_file):
        instances = []
        errors = []
        
        def create_connector():
            try:
                connector = TqConnector(config_path=temp_config_file)
                instances.append(connector)
            except Exception as e:
                errors.append(e)
        
        threads = []
        for _ in range(10):
            t = threading.Thread(target=create_connector)
            threads.append(t)
        
        for t in threads:
            t.start()
        
        for t in threads:
            t.join()
        
        assert len(errors) == 0, f"发生了 {len(errors)} 个错误: {errors}"
        
        first_instance = instances[0]
        for inst in instances:
            assert inst is first_instance, "多线程环境下应该返回同一个实例"
        
        assert len(set(id(inst) for inst in instances)) == 1, "所有实例应该是同一个对象"

    def test_thread_safe_singleton_with_slow_initialization(self, temp_config_file):
        instances = []
        init_count = [0]
        
        original_new = TqConnector.__new__
        
        def slow_new(cls, config_path=None):
            init_count[0] += 1
            time.sleep(0.01)
            return original_new(cls, config_path)
        
        TqConnector.__new__ = classmethod(slow_new).__func__
        
        try:
            def create_connector():
                connector = TqConnector(config_path=temp_config_file)
                instances.append(connector)
            
            threads = []
            for _ in range(20):
                t = threading.Thread(target=create_connector)
                threads.append(t)
            
            for t in threads:
                t.start()
            
            for t in threads:
                t.join()
            
            first_instance = instances[0]
            for inst in instances:
                assert inst is first_instance, "多线程环境下应该返回同一个实例"
            
            assert all(inst._initialized for inst in instances), "所有实例都应该已初始化"
        finally:
            TqConnector.__new__ = original_new

    def test_reset_instance_thread_safe(self, temp_config_file):
        results = []
        
        def create_and_reset():
            try:
                connector = TqConnector(config_path=temp_config_file)
                TqConnector.reset_instance()
                results.append(True)
            except Exception as e:
                results.append(e)
        
        threads = []
        for _ in range(15):
            t = threading.Thread(target=create_and_reset)
            threads.append(t)
        
        for t in threads:
            t.start()
        
        for t in threads:
            t.join()
        
        for result in results:
            if isinstance(result, Exception):
                pytest.fail(f"线程安全测试失败: {result}")

    def test_lock_exists(self):
        assert hasattr(TqConnector, '_lock')
        lock = TqConnector._lock
        assert hasattr(lock, 'acquire')
        assert hasattr(lock, 'release')

    def test_get_env_mode(self, temp_config_file):
        connector = TqConnector(config_path=temp_config_file)
        assert connector.get_env_mode() == 'sim'

    def test_get_retry_config(self, temp_config_file):
        connector = TqConnector(config_path=temp_config_file)
        retry_config = connector.get_retry_config()
        assert retry_config['retry_times'] == 3
        assert retry_config['initial_retry_delay'] == 2
        assert retry_config['max_retry_delay'] == 30

    @pytest.mark.skipif(USE_REAL_CONNECTION, reason="Mock测试被跳过，使用真实连接")
    @patch('time.sleep')
    @patch('core.connection.TqAuth')
    @patch('core.connection.TqSim')
    @patch('core.connection.TqApi')
    def test_exponential_backoff_retry(self, mock_tqapi, mock_tqsim, mock_tqauth, mock_sleep, temp_config_file):
        mock_api = Mock()
        mock_api.is_closed.return_value = False
        
        mock_tqapi.side_effect = [
            Exception("Network error 1"),
            Exception("Network error 2"),
            mock_api
        ]
        
        connector = TqConnector(config_path=temp_config_file)
        api = connector.connect()
        
        assert mock_tqapi.call_count == 3
        assert mock_sleep.call_count == 2
        
        mock_sleep.assert_has_calls([call(2), call(4)])
        
        assert connector.is_connected() is True
        assert api == mock_api

    @pytest.mark.skipif(USE_REAL_CONNECTION, reason="Mock测试被跳过，使用真实连接")
    @patch('time.sleep')
    @patch('core.connection.TqAuth')
    @patch('core.connection.TqSim')
    @patch('core.connection.TqApi')
    def test_auto_retry_exhausted(self, mock_tqapi, mock_tqsim, mock_tqauth, mock_sleep, temp_config_file):
        mock_tqapi.side_effect = [
            Exception("Network error 1"),
            Exception("Network error 2"),
            Exception("Network error 3")
        ]
        
        connector = TqConnector(config_path=temp_config_file)
        
        with pytest.raises(ConnectionError) as exc_info:
            connector.connect()
        
        assert "Network error" in str(exc_info.value)
        assert mock_tqapi.call_count == 3
        assert mock_sleep.call_count == 2
        mock_sleep.assert_has_calls([call(2), call(4)])
        assert connector.is_connected() is False

    @pytest.mark.skipif(USE_REAL_CONNECTION, reason="Mock测试被跳过，使用真实连接")
    @patch('time.sleep')
    @patch('core.connection.TqAuth')
    @patch('core.connection.TqSim')
    @patch('core.connection.TqApi')
    def test_logging_during_retry_process(self, mock_tqapi, mock_tqsim, mock_tqauth, mock_sleep, temp_config_file):
        mock_api = Mock()
        mock_api.is_closed.return_value = False
        mock_tqapi.side_effect = [Exception("Network error"), mock_api]
        
        log_calls = []
        
        connector = TqConnector(config_path=temp_config_file)
        logger = connector.logger
        
        original_info = logger.info
        original_error = logger.error
        original_critical = logger.critical
        
        def capture_info(msg, *args, **kwargs):
            log_calls.append(('info', str(msg)))
            return original_info(msg, *args, **kwargs)
        
        def capture_error(msg, *args, **kwargs):
            log_calls.append(('error', str(msg)))
            return original_error(msg, *args, **kwargs)
        
        def capture_critical(msg, *args, **kwargs):
            log_calls.append(('critical', str(msg)))
            return original_critical(msg, *args, **kwargs)
        
        logger.info = capture_info
        logger.error = capture_error
        logger.critical = capture_critical
        
        try:
            connector.connect()
            
            info_messages = [msg for method, msg in log_calls if method == 'info']
            error_messages = [msg for method, msg in log_calls if method == 'error']
            
            assert any('正在连接' in msg for msg in info_messages)
            assert any('连接成功' in msg for msg in info_messages)
            assert any('指数退避等待' in msg for msg in info_messages)
            assert any('连接失败' in msg for msg in error_messages)
        finally:
            logger.info = original_info
            logger.error = original_error
            logger.critical = original_critical

    def test_logging_setup(self, temp_config_file):
        connector = TqConnector(config_path=temp_config_file)
        
        assert connector.logger is not None
        assert connector.logger.name == 'TqConnector'
        assert len(connector.logger.handlers) > 0

    def test_invalid_env_mode(self):
        config_data = {
            "tq_sdk": {"account": "test", "password": "test"},
            "env": {"mode": "invalid_mode"},
            "trading": {"default_contract": "SHFE.rb2410", "contracts": []},
            "connection": {"retry_times": 3, "initial_retry_delay": 2, "max_retry_delay": 30},
            "logging": {"level": "DEBUG", "file": "logs/test.log"}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False, encoding='utf-8') as f:
            yaml.dump(config_data, f)
            temp_path = f.name
        
        try:
            connector = TqConnector(config_path=temp_path)
            with pytest.raises(ValueError, match="不支持的环境模式"):
                connector._create_account()
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_real_mode_missing_config(self):
        config_data = {
            "tq_sdk": {"account": "test", "password": "test"},
            "env": {
                "mode": "real",
                "real": {
                    "broker_id": "",
                    "futures_account": "",
                    "futures_password": ""
                }
            },
            "trading": {"default_contract": "SHFE.rb2410", "contracts": []},
            "connection": {"retry_times": 3, "initial_retry_delay": 2, "max_retry_delay": 30},
            "logging": {"level": "DEBUG", "file": "logs/test.log"}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False, encoding='utf-8') as f:
            yaml.dump(config_data, f)
            temp_path = f.name
        
        try:
            connector = TqConnector(config_path=temp_path)
            with pytest.raises(ValueError, match="实盘模式需要配置"):
                connector._create_account()
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_env_mode_via_config(self):
        config_data = {
            "tq_sdk": {"account": "test", "password": "test"},
            "env": {"mode": "real", "sim": {"init_balance": 1000000.0}, "real": {"broker_id": "H海通期货", "futures_account": "022631", "futures_password": "123456"}},
            "trading": {"default_contract": "SHFE.rb2410", "contracts": []},
            "connection": {"retry_times": 3, "initial_retry_delay": 2, "max_retry_delay": 30},
            "logging": {"level": "DEBUG", "file": "logs/test.log"}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False, encoding='utf-8') as f:
            yaml.dump(config_data, f)
            temp_path = f.name
        
        try:
            connector = TqConnector(config_path=temp_path)
            assert connector.get_env_mode() == 'real'
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    @pytest.mark.skipif(USE_REAL_CONNECTION, reason="Mock测试被跳过，使用真实连接")
    @patch('time.sleep')
    @patch('core.connection.TqAuth')
    @patch('core.connection.TqSim')
    @patch('core.connection.TqApi')
    def test_connection_error_raised_after_retries(self, mock_tqapi, mock_tqsim, mock_tqauth, mock_sleep, temp_config_file):
        mock_tqapi.side_effect = Exception("Network failure")
        
        connector = TqConnector(config_path=temp_config_file)
        
        with pytest.raises(ConnectionError) as exc_info:
            connector.connect()
        
        assert "已尝试 3 次" in str(exc_info.value)
        assert "Network failure" in str(exc_info.value)

    @pytest.mark.skipif(not USE_REAL_CONNECTION, reason="需要设置 USE_REAL_CONNECTION=true")
    @pytest.mark.skipif(not TEST_TQ_ACCOUNT or not TEST_TQ_PASSWORD, 
                        reason="需要设置 TEST_TQ_ACCOUNT 和 TEST_TQ_PASSWORD 环境变量")
    def test_connect_success_real(self, temp_config_file):
        with patch.dict(os.environ, {
            'TQ_ACCOUNT': TEST_TQ_ACCOUNT,
            'TQ_PASSWORD': TEST_TQ_PASSWORD
        }):
            connector = TqConnector(config_path=temp_config_file)
            try:
                api = connector.connect()
                assert api is not None
                assert connector.is_connected() is True
            finally:
                connector.disconnect()

    @pytest.mark.skipif(not USE_REAL_CONNECTION, reason="需要设置 USE_REAL_CONNECTION=true")
    @pytest.mark.skipif(not TEST_TQ_ACCOUNT or not TEST_TQ_PASSWORD, 
                        reason="需要设置 TEST_TQ_ACCOUNT 和 TEST_TQ_PASSWORD 环境变量")
    def test_singleton_pattern_real(self, temp_config_file):
        with patch.dict(os.environ, {
            'TQ_ACCOUNT': TEST_TQ_ACCOUNT,
            'TQ_PASSWORD': TEST_TQ_PASSWORD
        }):
            try:
                connector1 = TqConnector(config_path=temp_config_file)
                connector2 = TqConnector(config_path=temp_config_file)
                
                assert connector1 is connector2
                assert id(connector1) == id(connector2)
            finally:
                TqConnector.reset_instance()

    @pytest.mark.skipif(not USE_REAL_CONNECTION, reason="需要设置 USE_REAL_CONNECTION=true")
    def test_connect_with_invalid_credentials_real(self, temp_config_file):
        with patch.dict(os.environ, {
            'TQ_ACCOUNT': 'invalid_account',
            'TQ_PASSWORD': 'invalid_password'
        }):
            connector = TqConnector(config_path=temp_config_file)
            
            with pytest.raises(Exception) as exc_info:
                connector.connect()
            
            assert connector.is_connected() is False
            print(f"登录失败异常: {str(exc_info.value)}")
