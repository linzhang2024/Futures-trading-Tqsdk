import logging
from typing import Dict, Any, List, Optional, Type
from collections import defaultdict

from strategies.base_strategy import StrategyBase, SignalType


class StrategyManager:
    _strategy_classes: Dict[str, Type[StrategyBase]] = {}
    
    def __init__(self, connector: Any = None):
        self.connector = connector
        self.api = None
        self.logger = logging.getLogger(self.__class__.__name__)
        
        if connector:
            self.api = connector.get_api()
        
        self._strategies: Dict[str, StrategyBase] = {}
        self._contract_to_strategies: Dict[str, List[str]] = defaultdict(list)
        self._initialized: bool = False
        self._running: bool = False
        
        self._register_default_strategies()
    
    @classmethod
    def _register_default_strategies(cls):
        if not cls._strategy_classes:
            try:
                from strategies.double_ma_strategy import DoubleMAStrategy
                cls._strategy_classes['DoubleMAStrategy'] = DoubleMAStrategy
            except ImportError:
                pass
    
    @classmethod
    def register_strategy_class(cls, name: str, strategy_class: Type[StrategyBase]):
        if not issubclass(strategy_class, StrategyBase):
            raise ValueError(f"策略类 {name} 必须继承自 StrategyBase")
        
        cls._strategy_classes[name] = strategy_class
        logging.getLogger('StrategyManager').info(f"已注册策略类: {name}")
    
    @classmethod
    def get_strategy_class(cls, name: str) -> Optional[Type[StrategyBase]]:
        return cls._strategy_classes.get(name)
    
    def set_connector(self, connector: Any):
        if connector is None:
            self.logger.error("Connector 不能为 None")
            raise ValueError("Connector 不能为 None")
        
        self.connector = connector
        self.api = connector.get_api()
        self.logger.info("Connector 已设置")
    
    def register_strategy(self, name: str, strategy: StrategyBase) -> None:
        if name in self._strategies:
            self.logger.warning(f"策略 {name} 已存在，将覆盖旧策略")
            self.unregister_strategy(name)
        
        if not isinstance(strategy, StrategyBase):
            raise ValueError(f"策略 {name} 必须继承自 StrategyBase")
        
        if hasattr(strategy, 'contract'):
            contract = strategy.contract
            self._contract_to_strategies[contract].append(name)
        
        self._strategies[name] = strategy
        self.logger.info(f"已注册策略: {name}")
    
    def unregister_strategy(self, name: str) -> bool:
        if name not in self._strategies:
            self.logger.warning(f"策略 {name} 不存在")
            return False
        
        strategy = self._strategies[name]
        
        if hasattr(strategy, 'contract'):
            contract = strategy.contract
            if name in self._contract_to_strategies[contract]:
                self._contract_to_strategies[contract].remove(name)
        
        strategy.stop()
        del self._strategies[name]
        
        self.logger.info(f"已注销策略: {name}")
        return True
    
    def get_strategy(self, name: str) -> Optional[StrategyBase]:
        return self._strategies.get(name)
    
    def get_all_strategies(self) -> Dict[str, StrategyBase]:
        return self._strategies.copy()
    
    def get_strategies_by_contract(self, contract: str) -> List[str]:
        return self._contract_to_strategies.get(contract, [])
    
    def _create_strategy_from_config(self, config: Dict[str, Any]) -> StrategyBase:
        strategy_class_name = config.get('class')
        if not strategy_class_name:
            raise ValueError("策略配置缺少 'class' 字段")
        
        strategy_class = self.get_strategy_class(strategy_class_name)
        if not strategy_class:
            raise ValueError(f"未找到策略类: {strategy_class_name}")
        
        params = config.get('params', {})
        
        params['connector'] = self.connector
        
        param_mapping = {
            'fast': 'short_period',
            'slow': 'long_period',
            'period': 'kline_duration',
        }
        
        mapped_params = {}
        for key, value in params.items():
            mapped_key = param_mapping.get(key, key)
            mapped_params[mapped_key] = value
        
        try:
            strategy = strategy_class(**mapped_params)
            return strategy
        except TypeError as e:
            self.logger.error(f"创建策略 {strategy_class_name} 时参数错误: {e}")
            raise
    
    def load_strategies_from_config(self, config: Dict[str, Any]) -> None:
        strategies_config = config.get('strategies', [])
        
        if not strategies_config:
            self.logger.warning("配置中未找到策略列表")
            return
        
        for strategy_config in strategies_config:
            name = strategy_config.get('name')
            if not name:
                self.logger.warning("策略配置缺少 'name' 字段，跳过")
                continue
            
            try:
                strategy = self._create_strategy_from_config(strategy_config)
                self.register_strategy(name, strategy)
            except Exception as e:
                self.logger.error(f"创建策略 {name} 失败: {e}")
                raise
    
    def initialize(self) -> None:
        if self._initialized:
            self.logger.info("策略管理器已初始化，跳过重复初始化")
            return
        
        if self.connector is None:
            raise RuntimeError("Connector 未设置，请先调用 set_connector()")
        
        if self.api is None:
            raise RuntimeError("API 未初始化，请确保 Connector 已连接")
        
        self.logger.info("正在初始化所有策略...")
        
        for name, strategy in self._strategies.items():
            try:
                self.logger.info(f"初始化策略: {name}")
                strategy.initialize()
                self.logger.info(f"策略 {name} 初始化完成")
            except Exception as e:
                self.logger.error(f"策略 {name} 初始化失败: {e}")
                raise
        
        self._initialized = True
        self.logger.info("所有策略初始化完成")
    
    def _distribute_bar(self, contract: str, bar_data: Dict[str, Any]) -> None:
        strategy_names = self._contract_to_strategies.get(contract, [])
        
        if not strategy_names:
            return
        
        for name in strategy_names:
            strategy = self._strategies.get(name)
            if strategy:
                try:
                    strategy.on_bar(bar_data)
                except Exception as e:
                    self.logger.error(f"策略 {name} 处理 K 线数据时出错: {e}")
    
    def _distribute_bar_to_all(self, bar_data: Dict[str, Any]) -> None:
        for name, strategy in self._strategies.items():
            try:
                strategy.on_bar(bar_data)
            except Exception as e:
                self.logger.error(f"策略 {name} 处理 K 线数据时出错: {e}")
    
    def _get_latest_kline(self, klines: Any) -> Optional[Dict[str, Any]]:
        try:
            if klines is None or len(klines) == 0:
                return None
            
            latest_kline = klines.iloc[-1]
            return latest_kline.to_dict()
        except Exception as e:
            self.logger.error(f"获取最新 K 线数据失败: {e}")
            return None
    
    def _run_loop(self) -> None:
        if self.api is None:
            raise RuntimeError("API 未初始化")
        
        self.logger.info("开始运行策略管理器主循环...")
        self._running = True
        
        while self._running:
            try:
                self.api.wait_update()
                
                for name, strategy in self._strategies.items():
                    try:
                        if hasattr(strategy, '_on_update'):
                            strategy._on_update()
                    except Exception as e:
                        self.logger.error(f"策略 {name} 更新时出错: {e}")
                        
            except KeyboardInterrupt:
                self.logger.info("用户中断，停止策略管理器")
                self._running = False
                break
            except Exception as e:
                self.logger.error(f"主循环出错: {e}")
                self._running = False
                raise
    
    def run_all(self) -> None:
        if not self._initialized:
            self.logger.info("策略管理器未初始化，正在执行初始化...")
            self.initialize()
        
        self.logger.info("=" * 60)
        self.logger.info("策略管理器开始运行")
        self.logger.info(f"已注册策略数量: {len(self._strategies)}")
        for name in self._strategies.keys():
            self.logger.info(f"  - {name}")
        self.logger.info("=" * 60)
        
        try:
            self._run_loop()
        except KeyboardInterrupt:
            self.logger.info("用户中断，停止所有策略")
        except Exception as e:
            self.logger.error(f"策略管理器运行出错: {e}")
            raise
        finally:
            self.stop_all()
    
    def stop_all(self) -> None:
        self._running = False
        
        self.logger.info("正在停止所有策略...")
        
        for name, strategy in self._strategies.items():
            try:
                self.logger.info(f"停止策略: {name}")
                strategy.stop()
                self.logger.info(f"策略 {name} 已停止")
            except Exception as e:
                self.logger.error(f"停止策略 {name} 时出错: {e}")
        
        self.logger.info("所有策略已停止")
    
    def get_all_signals(self) -> Dict[str, SignalType]:
        signals = {}
        for name, strategy in self._strategies.items():
            if hasattr(strategy, 'get_signal'):
                signals[name] = strategy.get_signal()
            elif hasattr(strategy, 'signal'):
                signals[name] = strategy.signal
        return signals
    
    def get_strategy_state(self, name: str) -> Optional[Dict[str, Any]]:
        strategy = self.get_strategy(name)
        if not strategy:
            return None
        
        state = {
            'name': name,
            'class': strategy.__class__.__name__,
            'signal': strategy.signal if hasattr(strategy, 'signal') else None,
            'is_ready': strategy.is_ready() if hasattr(strategy, 'is_ready') else False,
        }
        
        if hasattr(strategy, 'get_ma_values'):
            state['ma_values'] = strategy.get_ma_values()
        
        if hasattr(strategy, 'short_period'):
            state['short_period'] = strategy.short_period
        if hasattr(strategy, 'long_period'):
            state['long_period'] = strategy.long_period
        if hasattr(strategy, 'contract'):
            state['contract'] = strategy.contract
        if hasattr(strategy, 'kline_duration'):
            state['kline_duration'] = strategy.kline_duration
        
        return state
    
    def get_all_states(self) -> Dict[str, Dict[str, Any]]:
        states = {}
        for name in self._strategies.keys():
            state = self.get_strategy_state(name)
            if state:
                states[name] = state
        return states
