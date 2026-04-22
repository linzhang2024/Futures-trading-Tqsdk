import os
import re
import yaml
import logging
from typing import Optional, Dict, Any
from tqsdk import TqApi, TqAuth


class TqConnector:
    _instance: Optional['TqConnector'] = None
    
    def __new__(cls, config_path: str = None):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, config_path: str = None):
        if self._initialized:
            return
        
        self.config_path = config_path or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "config",
            "settings.yaml"
        )
        self.config: Dict[str, Any] = {}
        self.api: Optional[TqApi] = None
        self.logger = logging.getLogger(__name__)
        self._load_config()
        self._initialized = True
    
    def _load_config(self) -> None:
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"配置文件不存在: {self.config_path}")
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            raw_config = yaml.safe_load(f)
        
        self.config = self._resolve_env_vars(raw_config)
        
        log_level = self.config.get('logging', {}).get('level', 'INFO')
        self.logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

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

    def connect(self) -> TqApi:
        if self.api and not self.api.is_closed():
            self.logger.info("API 已连接，无需重新连接")
            return self.api
        
        tq_config = self.config.get('tq_sdk', {})
        account = tq_config.get('account')
        password = tq_config.get('password')
        
        if not account or not password:
            raise ValueError("缺少账号或密码配置")
        
        if account == 'your_account' or password == 'your_password':
            self.logger.warning("检测到使用默认占位符凭证，请设置环境变量 TQ_ACCOUNT 和 TQ_PASSWORD")
        
        try:
            self.logger.info(f"正在连接天勤 API，账号: {account}")
            auth = TqAuth(account, password)
            self.api = TqApi(auth=auth)
            self.logger.info("天勤 API 连接成功")
            return self.api
        except Exception as e:
            self.logger.error(f"连接失败: {str(e)}")
            raise

    def disconnect(self) -> None:
        if self.api and not self.api.is_closed():
            self.api.close()
            self.logger.info("天勤 API 已断开连接")
        self.api = None

    def is_connected(self) -> bool:
        return self.api is not None and not self.api.is_closed()

    def get_api(self) -> Optional[TqApi]:
        return self.api

    def get_default_contract(self) -> str:
        return self.config.get('trading', {}).get('default_contract', '')

    def get_contracts(self) -> list:
        return self.config.get('trading', {}).get('contracts', [])

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()
        return False

    @classmethod
    def reset_instance(cls):
        if cls._instance:
            cls._instance.disconnect()
        cls._instance = None
