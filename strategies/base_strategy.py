import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from enum import Enum


class SignalType(Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class StrategyBase(ABC):
    def __init__(self, connector: Any = None):
        self.connector = connector
        self.api = None
        self.logger = logging.getLogger(self.__class__.__name__)
        
        if connector:
            self.api = connector.get_api()
        
        self._initialized = False
        self._signal: SignalType = SignalType.HOLD

    @property
    def signal(self) -> SignalType:
        return self._signal

    @signal.setter
    def signal(self, value: SignalType):
        self._signal = value

    def set_connector(self, connector: Any):
        if connector is None:
            self.logger.error("Connector 不能为 None")
            raise ValueError("Connector 不能为 None")
        
        self.connector = connector
        self.api = connector.get_api()
        self.logger.info("Connector 已设置")

    def get_api(self) -> Optional[Any]:
        return self.api

    @abstractmethod
    def on_bar(self, bar_data: Dict[str, Any]):
        pass

    @abstractmethod
    def subscribe(self):
        pass

    def initialize(self):
        if self._initialized:
            self.logger.info("策略已初始化，跳过重复初始化")
            return
        
        try:
            self._validate_initialize()
            self._initialized = True
            self.logger.info("策略初始化完成")
        except Exception as e:
            self.logger.error(f"策略初始化失败: {str(e)}", exc_info=True)
            raise

    def _validate_initialize(self):
        if self.connector is None:
            raise RuntimeError("Connector 未设置，请先调用 set_connector() 或在初始化时传入")
        
        if self.api is None:
            raise RuntimeError("API 未初始化，请确保 Connector 已连接")
        
        self.subscribe()

    def run(self):
        if not self._initialized:
            self.logger.info("策略未初始化，正在执行初始化...")
            self.initialize()
        
        self.logger.info("策略开始运行")
        
        try:
            self._run_loop()
        except KeyboardInterrupt:
            self.logger.info("用户中断，停止策略")
        except Exception as e:
            self.logger.error(f"策略运行出错: {str(e)}", exc_info=True)
            raise

    def _run_loop(self):
        if self.api is None:
            raise RuntimeError("API 未初始化")
        
        while True:
            self.api.wait_update()
            self._on_update()

    @abstractmethod
    def _on_update(self):
        pass

    def stop(self):
        self.logger.info("策略停止")
        self._on_stop()

    def _on_stop(self):
        pass
