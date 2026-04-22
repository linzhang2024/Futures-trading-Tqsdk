import os
import re
import yaml
import logging
import threading
import time
from typing import Optional, Dict, Any
from tqsdk import TqApi, TqAuth, TqSim, TqKq, TqAccount


class TqConnector:
    _instance: Optional['TqConnector'] = None
    _lock: threading.Lock = threading.Lock()
    _logger_initialized: bool = False
    
    def __new__(cls, config_path: str = None):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
                    cls._instance.api = None
                    cls._instance.config = {}
                    cls._instance.logger = None
        return cls._instance
    
    def __init__(self, config_path: str = None):
        if self._initialized:
            return
        
        with self._lock:
            if self._initialized:
                return
            
            self.config_path = config_path or os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "config",
                "settings.yaml"
            )
            self.config: Dict[str, Any] = {}
            self.api: Optional[TqApi] = None
            self._retry_times: int = 3
            self._initial_retry_delay: int = 2
            self._max_retry_delay: int = 30
            self._env_mode: str = "sim"
            
            self._load_config()
            self._setup_logging()
            self._initialized = True
    
    def _load_config(self) -> None:
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"配置文件不存在: {self.config_path}")
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            raw_config = yaml.safe_load(f)
        
        self.config = self._resolve_env_vars(raw_config)
        
        conn_config = self.config.get('connection', {})
        self._retry_times = conn_config.get('retry_times', 3)
        self._initial_retry_delay = conn_config.get('initial_retry_delay', 2)
        self._max_retry_delay = conn_config.get('max_retry_delay', 30)
        
        env_config = self.config.get('env', {})
        self._env_mode = env_config.get('mode', 'sim').lower()

    def _setup_logging(self) -> None:
        if TqConnector._logger_initialized:
            self.logger = logging.getLogger('TqConnector')
            return
        
        log_config = self.config.get('logging', {})
        log_level_str = log_config.get('level', 'INFO')
        log_file = log_config.get('file', 'logs/app.log')
        log_format = log_config.get('format', "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        log_datefmt = log_config.get('datefmt', "%Y-%m-%d %H:%M:%S")
        
        log_level = getattr(logging, log_level_str.upper(), logging.INFO)
        
        self.logger = logging.getLogger('TqConnector')
        self.logger.setLevel(log_level)
        
        if not self.logger.handlers:
            log_dir = os.path.dirname(log_file)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True)
            
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setLevel(log_level)
            file_formatter = logging.Formatter(log_format, datefmt=log_datefmt)
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)
            
            console_handler = logging.StreamHandler()
            console_handler.setLevel(log_level)
            console_formatter = logging.Formatter(log_format, datefmt=log_datefmt)
            console_handler.setFormatter(console_formatter)
            self.logger.addHandler(console_handler)
        
        TqConnector._logger_initialized = True

    def _resolve_env_vars(self, config: Any) -> Any:
        if isinstance(config, str):
            pattern = r'\$\{(\w+)(?::([^}]+))?\}'
            matches = re.findall(pattern, config)
            
            for env_var, default_value in matches:
                actual_value = os.environ.get(env_var, default_value)
                config = config.replace(f'${{{env_var}:{default_value}}}', actual_value)
                config = config.replace(f'${{{env_var}}}', actual_value)
            
            return config
        elif isinstance(config, dict):
            return {k: self._resolve_env_vars(v) for k, v in config.items()}
        elif isinstance(config, list):
            return [self._resolve_env_vars(item) for item in config]
        else:
            return config

    def _create_account(self) -> Any:
        env_config = self.config.get('env', {})
        
        if self._env_mode == 'sim':
            sim_config = env_config.get('sim', {})
            account_type = sim_config.get('account_type', 'tqsim').lower()
            init_balance = sim_config.get('init_balance', 10000000.0)
            
            if account_type == 'tqkq':
                self.logger.info("使用快期模拟账户模式 (TqKq)")
                return TqKq()
            else:
                self.logger.info(f"使用本地模拟账户模式 (TqSim)，初始资金: {init_balance}")
                return TqSim(init_balance=init_balance)
        
        elif self._env_mode == 'real':
            real_config = env_config.get('real', {})
            broker_id = real_config.get('broker_id')
            futures_account = real_config.get('futures_account')
            futures_password = real_config.get('futures_password')
            front_broker = real_config.get('front_broker')
            front_url = real_config.get('front_url')
            
            if not broker_id or not futures_account or not futures_password:
                raise ValueError("实盘模式需要配置 broker_id、futures_account 和 futures_password")
            
            self.logger.info(f"使用实盘模式 (TqAccount)，期货公司: {broker_id}")
            
            kwargs = {}
            if front_broker:
                kwargs['front_broker'] = front_broker
            if front_url:
                kwargs['front_url'] = front_url
            
            return TqAccount(broker_id, futures_account, futures_password, **kwargs)
        
        else:
            raise ValueError(f"不支持的环境模式: {self._env_mode}，请使用 'sim' 或 'real'")

    def _calculate_exponential_backoff(self, attempt: int) -> float:
        delay = self._initial_retry_delay * (2 ** (attempt - 1))
        return min(delay, self._max_retry_delay)

    def connect(self) -> TqApi:
        if self.api and not self.api.is_closed():
            self.logger.info("API 已连接，无需重新连接")
            return self.api
        
        tq_config = self.config.get('tq_sdk', {})
        account = tq_config.get('account')
        password = tq_config.get('password')
        
        if not account or not password:
            self.logger.error("缺少天勤账号或密码配置")
            raise ValueError("缺少天勤账号或密码配置")
        
        if account == 'your_account' or password == 'your_password':
            self.logger.warning("检测到使用默认占位符凭证，请设置环境变量 TQ_ACCOUNT 和 TQ_PASSWORD")
        
        last_exception = None
        
        for attempt in range(1, self._retry_times + 1):
            try:
                self.logger.info(f"[{attempt}/{self._retry_times}] 正在连接天勤 API，账号: {account}，环境模式: {self._env_mode}")
                
                auth = TqAuth(account, password)
                trading_account = self._create_account()
                
                self.api = TqApi(account=trading_account, auth=auth)
                self.logger.info(f"[{attempt}/{self._retry_times}] 天勤 API 连接成功！环境: {self._env_mode}")
                return self.api
                
            except Exception as e:
                last_exception = e
                self.logger.error(f"[{attempt}/{self._retry_times}] 连接失败: {str(e)}", exc_info=True)
                
                if attempt < self._retry_times:
                    delay = self._calculate_exponential_backoff(attempt)
                    self.logger.info(f"指数退避等待 {delay} 秒后进行第 {attempt + 1} 次重试...")
                    time.sleep(delay)
        
        final_msg = f"连接失败，已尝试 {self._retry_times} 次，最后错误: {str(last_exception)}"
        self.logger.critical(final_msg)
        raise ConnectionError(final_msg) from last_exception

    def disconnect(self) -> None:
        api = getattr(self, 'api', None)
        if api is not None and not api.is_closed():
            self.logger.info("正在断开天勤 API 连接...")
            api.close()
            self.logger.info("天勤 API 已断开连接")
        self.api = None

    def is_connected(self) -> bool:
        api = getattr(self, 'api', None)
        return api is not None and not api.is_closed()

    def get_api(self) -> Optional[TqApi]:
        return self.api

    def get_env_mode(self) -> str:
        return self._env_mode

    def get_default_contract(self) -> str:
        return self.config.get('trading', {}).get('default_contract', '')

    def get_contracts(self) -> list:
        return self.config.get('trading', {}).get('contracts', [])

    def get_retry_config(self) -> Dict[str, Any]:
        return {
            'retry_times': self._retry_times,
            'initial_retry_delay': self._initial_retry_delay,
            'max_retry_delay': self._max_retry_delay
        }

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()
        return False

    @classmethod
    def reset_instance(cls):
        with cls._lock:
            if cls._instance:
                cls._instance.disconnect()
            cls._instance = None
