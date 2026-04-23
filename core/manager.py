import logging
import time
import json
import os
from typing import Dict, Any, List, Optional, Type, Callable
from collections import defaultdict
from datetime import datetime, timedelta
from enum import Enum

from strategies.base_strategy import StrategyBase, SignalType
from core.risk_manager import (
    RiskManager,
    RiskLevel,
    RiskEvent,
    RiskEventType,
    PositionInfo,
    AccountSnapshot,
)


class StrategyHealthStatus(Enum):
    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    FAILED = "FAILED"


class StrategyHealth:
    def __init__(self, strategy_name: str):
        self.name = strategy_name
        self.status = StrategyHealthStatus.HEALTHY
        self.error_count = 0
        self.consecutive_errors = 0
        self.last_error_time: Optional[datetime] = None
        self.last_error_message: Optional[str] = None
        self.total_updates = 0
        self.successful_updates = 0
        self.max_consecutive_errors = 5
        self.degraded_threshold = 2

    def record_success(self):
        self.total_updates += 1
        self.successful_updates += 1
        self.consecutive_errors = 0
        
        if self.status == StrategyHealthStatus.DEGRADED:
            self.status = StrategyHealthStatus.HEALTHY

    def record_error(self, error_message: str):
        self.total_updates += 1
        self.error_count += 1
        self.consecutive_errors += 1
        self.last_error_time = datetime.now()
        self.last_error_message = error_message
        
        if self.consecutive_errors >= self.max_consecutive_errors:
            self.status = StrategyHealthStatus.FAILED
        elif self.consecutive_errors >= self.degraded_threshold:
            self.status = StrategyHealthStatus.DEGRADED

    def reset(self):
        self.status = StrategyHealthStatus.HEALTHY
        self.error_count = 0
        self.consecutive_errors = 0
        self.last_error_time = None
        self.last_error_message = None

    def can_run(self) -> bool:
        return self.status != StrategyHealthStatus.FAILED

    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'status': self.status.value,
            'error_count': self.error_count,
            'consecutive_errors': self.consecutive_errors,
            'last_error_time': self.last_error_time.isoformat() if self.last_error_time else None,
            'last_error_message': self.last_error_message,
            'total_updates': self.total_updates,
            'successful_updates': self.successful_updates,
            'success_rate': self.successful_updates / self.total_updates if self.total_updates > 0 else 1.0,
        }


class StrategyManager:
    _strategy_classes: Dict[str, Type[StrategyBase]] = {}
    
    def __init__(
        self,
        connector: Any = None,
        state_dir: str = None,
        risk_manager: RiskManager = None,
        on_risk_event: Callable[[RiskEvent], None] = None,
        on_risk_frozen: Callable[[], None] = None,
    ):
        self.connector = connector
        self.api = None
        self.logger = logging.getLogger(self.__class__.__name__)
        
        if connector:
            self.api = connector.get_api()
        
        self._strategies: Dict[str, StrategyBase] = {}
        self._strategy_health: Dict[str, StrategyHealth] = {}
        self._contract_to_strategies: Dict[str, List[str]] = defaultdict(list)
        self._initialized: bool = False
        self._running: bool = False
        
        self._state_dir = state_dir or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'data',
            'states'
        )
        
        self._status_report_interval = 60
        self._last_status_report_time = 0.0
        
        self._cycle_count = 0
        self._start_time = 0.0
        
        self._risk_manager = risk_manager
        self._on_risk_event = on_risk_event
        self._on_risk_frozen = on_risk_frozen
        self._risk_check_interval = 1
        self._last_risk_check_time = 0.0
        
        self._register_default_strategies()
    
    def _init_risk_manager(self) -> None:
        if self._risk_manager is not None:
            return
        
        self._risk_manager = RiskManager(
            connector=self.connector,
            on_risk_event=self._on_risk_event,
            on_frozen=self._on_risk_frozen,
        )
        self.logger.info("RiskManager 已初始化")
    
    def get_risk_manager(self) -> Optional[RiskManager]:
        return self._risk_manager
    
    def set_risk_manager(self, risk_manager: RiskManager) -> None:
        self._risk_manager = risk_manager
        self.logger.info("RiskManager 已设置")
    
    def configure_risk_from_dict(self, config: Dict[str, Any]) -> None:
        if not config:
            return
        
        risk_config = config.get('risk', {})
        
        if self._risk_manager is None:
            self._init_risk_manager()
        
        max_drawdown = risk_config.get('max_drawdown_percent')
        if max_drawdown is not None:
            self._risk_manager.max_drawdown_percent = float(max_drawdown)
            self.logger.info(f"最大回撤阈值已设置为: {max_drawdown}%")
        
        max_strategy_margin = risk_config.get('max_strategy_margin_percent')
        if max_strategy_margin is not None:
            self._risk_manager.max_strategy_margin_percent = float(max_strategy_margin)
            self.logger.info(f"单策略最大保证金比例已设置为: {max_strategy_margin}%")
        
        max_total_margin = risk_config.get('max_total_margin_percent')
        if max_total_margin is not None:
            self._risk_manager.max_total_margin_percent = float(max_total_margin)
            self.logger.info(f"总最大保证金比例已设置为: {max_total_margin}%")
        
        risk_check_interval = risk_config.get('risk_check_interval')
        if risk_check_interval is not None and risk_check_interval > 0:
            self._risk_check_interval = risk_check_interval
            self.logger.info(f"风控检查间隔已设置为: {self._risk_check_interval} 秒")
    
    def _should_run_risk_check(self) -> bool:
        if self._risk_manager is None:
            return False
        
        if self._risk_check_interval <= 0:
            return True
        
        current_time = time.time()
        if current_time - self._last_risk_check_time >= self._risk_check_interval:
            self._last_risk_check_time = current_time
            return True
        
        return False
    
    def run_risk_checks(self) -> Dict[str, RiskLevel]:
        if self._risk_manager is None:
            return {}
        
        return self._risk_manager.run_risk_checks()
    
    def is_risk_frozen(self) -> bool:
        if self._risk_manager is None:
            return False
        return self._risk_manager.is_frozen()
    
    def get_risk_info(self) -> Dict[str, Any]:
        if self._risk_manager is None:
            return {'risk_enabled': False}
        
        info = self._risk_manager.get_total_risk_info()
        info['risk_enabled'] = True
        return info
    
    def emergency_stop(self, reason: str) -> Dict[str, Any]:
        if self._risk_manager is None:
            self.logger.warning("RiskManager 未初始化，无法执行紧急停止")
            return {'status': 'error', 'message': 'RiskManager 未初始化'}
        
        result = self._risk_manager.emergency_stop(reason)
        
        self._running = False
        
        return result
    
    def configure_from_dict(self, config: Dict[str, Any]) -> None:
        if not config:
            return
        
        manager_config = config.get('manager', {})
        
        status_report_interval = manager_config.get('status_report_interval')
        if status_report_interval is not None and status_report_interval > 0:
            self._status_report_interval = status_report_interval
            self.logger.info(f"状态报告间隔已设置为: {self._status_report_interval} 秒")
        
        state_dir = manager_config.get('state_dir')
        if state_dir:
            if not os.path.isabs(state_dir):
                state_dir = os.path.join(
                    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    state_dir
                )
            self._state_dir = state_dir
            self.logger.info(f"状态存储目录已设置为: {self._state_dir}")
        
        self._auto_save_states = manager_config.get('auto_save_states', True)
        
        self.configure_risk_from_dict(config)
    
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
        self._strategy_health[name] = StrategyHealth(name)
        
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
        del self._strategy_health[name]
        
        self.logger.info(f"已注销策略: {name}")
        return True
    
    def get_strategy(self, name: str) -> Optional[StrategyBase]:
        return self._strategies.get(name)
    
    def get_all_strategies(self) -> Dict[str, StrategyBase]:
        return self._strategies.copy()
    
    def get_strategies_by_contract(self, contract: str) -> List[str]:
        return self._contract_to_strategies.get(contract, [])
    
    def get_strategy_health(self, name: str) -> Optional[StrategyHealth]:
        return self._strategy_health.get(name)
    
    def get_all_health_status(self) -> Dict[str, Dict[str, Any]]:
        return {name: health.to_dict() for name, health in self._strategy_health.items()}
    
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
    
    def _get_state_file_path(self, strategy_name: str) -> str:
        safe_name = "".join(c if c.isalnum() or c in "._-" else "_" for c in strategy_name)
        return os.path.join(self._state_dir, f"{safe_name}.json")
    
    def save_strategy_state(self, strategy_name: str) -> bool:
        strategy = self.get_strategy(strategy_name)
        if not strategy:
            self.logger.warning(f"策略 {strategy_name} 不存在，无法保存状态")
            return False
        
        try:
            os.makedirs(self._state_dir, exist_ok=True)
            state_file = self._get_state_file_path(strategy_name)
            
            state = {}
            
            if hasattr(strategy, 'save_state'):
                try:
                    custom_state = strategy.save_state()
                    if custom_state:
                        state.update(custom_state)
                except Exception as e:
                    self.logger.warning(f"策略 {strategy_name} save_state 方法出错: {e}")
            
            if hasattr(strategy, '_all_prices'):
                state['_all_prices'] = list(strategy._all_prices) if strategy._all_prices else []
            
            if hasattr(strategy, 'short_prices'):
                state['short_prices'] = list(strategy.short_prices) if strategy.short_prices else []
            
            if hasattr(strategy, 'long_prices'):
                state['long_prices'] = list(strategy.long_prices) if strategy.long_prices else []
            
            if hasattr(strategy, 'short_ma'):
                state['short_ma'] = strategy.short_ma
            
            if hasattr(strategy, 'long_ma'):
                state['long_ma'] = strategy.long_ma
            
            if hasattr(strategy, 'prev_short_ma'):
                state['prev_short_ma'] = strategy.prev_short_ma
            
            if hasattr(strategy, 'prev_long_ma'):
                state['prev_long_ma'] = strategy.prev_long_ma
            
            if hasattr(strategy, 'signal'):
                state['signal'] = strategy.signal.value if hasattr(strategy.signal, 'value') else str(strategy.signal)
            
            if hasattr(strategy, 'prev_signal'):
                state['prev_signal'] = strategy.prev_signal.value if hasattr(strategy.prev_signal, 'value') else str(strategy.prev_signal)
            
            state['saved_at'] = datetime.now().isoformat()
            state['strategy_name'] = strategy_name
            state['strategy_class'] = strategy.__class__.__name__
            
            with open(state_file, 'w', encoding='utf-8') as f:
                json.dump(state, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"策略 {strategy_name} 状态已保存到: {state_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"保存策略 {strategy_name} 状态失败: {e}")
            return False
    
    def load_strategy_state(self, strategy_name: str) -> bool:
        strategy = self.get_strategy(strategy_name)
        if not strategy:
            self.logger.warning(f"策略 {strategy_name} 不存在，无法加载状态")
            return False
        
        try:
            state_file = self._get_state_file_path(strategy_name)
            
            if not os.path.exists(state_file):
                self.logger.info(f"策略 {strategy_name} 状态文件不存在，跳过加载: {state_file}")
                return False
            
            with open(state_file, 'r', encoding='utf-8') as f:
                state = json.load(f)
            
            self.logger.info(f"正在加载策略 {strategy_name} 的状态...")
            
            if hasattr(strategy, 'load_state'):
                try:
                    strategy.load_state(state)
                except Exception as e:
                    self.logger.warning(f"策略 {strategy_name} load_state 方法出错: {e}")
            
            if '_all_prices' in state and hasattr(strategy, '_all_prices'):
                prices = state['_all_prices']
                if prices:
                    strategy._all_prices = [float(p) for p in prices]
                    self.logger.info(f"  - 恢复价格历史: {len(strategy._all_prices)} 条记录")
            
            if 'short_prices' in state and hasattr(strategy, 'short_prices'):
                from collections import deque
                prices = state['short_prices']
                if prices:
                    maxlen = strategy.short_prices.maxlen if hasattr(strategy.short_prices, 'maxlen') else None
                    strategy.short_prices = deque([float(p) for p in prices], maxlen=maxlen)
            
            if 'long_prices' in state and hasattr(strategy, 'long_prices'):
                from collections import deque
                prices = state['long_prices']
                if prices:
                    maxlen = strategy.long_prices.maxlen if hasattr(strategy.long_prices, 'maxlen') else None
                    strategy.long_prices = deque([float(p) for p in prices], maxlen=maxlen)
            
            if 'short_ma' in state and hasattr(strategy, 'short_ma'):
                strategy.short_ma = float(state['short_ma']) if state['short_ma'] is not None else None
            
            if 'long_ma' in state and hasattr(strategy, 'long_ma'):
                strategy.long_ma = float(state['long_ma']) if state['long_ma'] is not None else None
            
            if 'prev_short_ma' in state and hasattr(strategy, 'prev_short_ma'):
                strategy.prev_short_ma = float(state['prev_short_ma']) if state['prev_short_ma'] is not None else None
            
            if 'prev_long_ma' in state and hasattr(strategy, 'prev_long_ma'):
                strategy.prev_long_ma = float(state['prev_long_ma']) if state['prev_long_ma'] is not None else None
            
            if 'signal' in state and hasattr(strategy, 'signal'):
                try:
                    signal_value = state['signal']
                    if isinstance(signal_value, str):
                        for sig_type in SignalType:
                            if sig_type.value == signal_value:
                                strategy.signal = sig_type
                                break
                except Exception as e:
                    self.logger.warning(f"恢复信号状态失败: {e}")
            
            if 'prev_signal' in state and hasattr(strategy, 'prev_signal'):
                try:
                    signal_value = state['prev_signal']
                    if isinstance(signal_value, str):
                        for sig_type in SignalType:
                            if sig_type.value == signal_value:
                                strategy.prev_signal = sig_type
                                break
                except Exception as e:
                    self.logger.warning(f"恢复历史信号状态失败: {e}")
            
            saved_at = state.get('saved_at', '未知')
            self.logger.info(f"策略 {strategy_name} 状态已从 {saved_at} 恢复")
            
            return True
            
        except Exception as e:
            self.logger.error(f"加载策略 {strategy_name} 状态失败: {e}")
            return False
    
    def save_all_states(self) -> int:
        count = 0
        for name in self._strategies.keys():
            if self.save_strategy_state(name):
                count += 1
        return count
    
    def load_all_states(self) -> int:
        count = 0
        for name in self._strategies.keys():
            if self.load_strategy_state(name):
                count += 1
        return count
    
    def initialize(self, load_saved_states: bool = True) -> None:
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
                health = self._strategy_health.get(name)
                if health:
                    health.record_error(str(e))
        
        if load_saved_states:
            self.logger.info("正在加载保存的策略状态...")
            loaded_count = self.load_all_states()
            if loaded_count > 0:
                self.logger.info(f"已加载 {loaded_count} 个策略的状态")
            else:
                self.logger.info("没有找到已保存的状态，将从头开始")
        
        self._initialized = True
        self.logger.info("所有策略初始化完成")
    
    def _distribute_bar(self, contract: str, bar_data: Dict[str, Any]) -> None:
        strategy_names = self._contract_to_strategies.get(contract, [])
        
        if not strategy_names:
            return
        
        for name in strategy_names:
            strategy = self._strategies.get(name)
            health = self._strategy_health.get(name)
            
            if not strategy or not health:
                continue
            
            if not health.can_run():
                self.logger.warning(f"策略 {name} 已标记为 FAILED，跳过处理")
                continue
            
            try:
                strategy.on_bar(bar_data)
                health.record_success()
            except Exception as e:
                error_msg = f"策略 {name} 处理 K 线数据时出错: {e}"
                self.logger.error(error_msg)
                health.record_error(str(e))
                
                if not health.can_run():
                    self.logger.critical(f"策略 {name} 连续错误次数过多，已标记为 FAILED，需要手动重置")
    
    def _distribute_bar_to_all(self, bar_data: Dict[str, Any]) -> None:
        for name, strategy in self._strategies.items():
            health = self._strategy_health.get(name)
            
            if not health:
                continue
            
            if not health.can_run():
                self.logger.warning(f"策略 {name} 已标记为 FAILED，跳过处理")
                continue
            
            try:
                strategy.on_bar(bar_data)
                health.record_success()
            except Exception as e:
                error_msg = f"策略 {name} 处理 K 线数据时出错: {e}"
                self.logger.error(error_msg)
                health.record_error(str(e))
                
                if not health.can_run():
                    self.logger.critical(f"策略 {name} 连续错误次数过多，已标记为 FAILED，需要手动重置")
    
    def _get_latest_kline(self, klines: Any) -> Optional[Dict[str, Any]]:
        try:
            if klines is None or len(klines) == 0:
                return None
            
            latest_kline = klines.iloc[-1]
            return latest_kline.to_dict()
        except Exception as e:
            self.logger.error(f"获取最新 K 线数据失败: {e}")
            return None
    
    def _format_status_report(self) -> str:
        lines = []
        lines.append("")
        lines.append("=" * 80)
        lines.append(f"策略状态报告 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"运行周期: {self._cycle_count} 次 | 运行时间: {timedelta(seconds=int(time.time() - self._start_time))}")
        lines.append("-" * 80)
        
        for name, strategy in self._strategies.items():
            health = self._strategy_health.get(name)
            state = self.get_strategy_state(name)
            
            lines.append(f"【{name}】")
            lines.append(f"  状态: {health.status.value if health else 'UNKNOWN'}")
            
            if state:
                is_ready = state.get('is_ready', False)
                signal = state.get('signal', SignalType.HOLD)
                
                if is_ready:
                    ma_values = state.get('ma_values', {})
                    short_period = state.get('short_period', 'N/A')
                    long_period = state.get('long_period', 'N/A')
                    
                    short_ma = ma_values.get(f'ma_{short_period}') if ma_values else None
                    long_ma = ma_values.get(f'ma_{long_period}') if ma_values else None
                    
                    lines.append(f"  均线: MA{short_period} = {short_ma:.2f}" if short_ma else f"  均线: MA{short_period} = N/A")
                    lines.append(f"  均线: MA{long_period} = {long_ma:.2f}" if long_ma else f"  均线: MA{long_period} = N/A")
                    
                    signal_color = "🟢" if signal == SignalType.BUY else "🔴" if signal == SignalType.SELL else "⚪"
                    lines.append(f"  信号: {signal_color} {signal.value}")
                    
                    if signal == SignalType.BUY:
                        lines.append(f"  建议: 建仓/持有多单")
                    elif signal == SignalType.SELL:
                        lines.append(f"  建议: 建仓/持有空单")
                    else:
                        lines.append(f"  建议: 观望/保持当前仓位")
                else:
                    lines.append(f"  状态: ⏳ 数据收集中，暂未就绪")
                    
                    if hasattr(strategy, '_all_prices'):
                        price_count = len(strategy._all_prices)
                        lines.append(f"  进度: 已收集 {price_count} 条价格数据")
            
            if health:
                health_dict = health.to_dict()
                success_rate = health_dict.get('success_rate', 1.0) * 100
                lines.append(f"  健康: 成功率 {success_rate:.1f}% | 总错误 {health_dict.get('error_count', 0)} 次")
                
                if health.status != StrategyHealthStatus.HEALTHY:
                    lines.append(f"  ⚠️  最后错误: {health_dict.get('last_error_message', 'N/A')}")
            
            lines.append("-" * 80)
        
        lines.append("=" * 80)
        lines.append("")
        
        return "\n".join(lines)
    
    def _should_report_status(self) -> bool:
        current_time = time.time()
        
        if self._cycle_count == 0:
            self._last_status_report_time = current_time
            return False
        
        if current_time - self._last_status_report_time >= self._status_report_interval:
            self._last_status_report_time = current_time
            return True
        
        return False
    
    def _run_loop(self) -> None:
        if self.api is None:
            raise RuntimeError("API 未初始化")
        
        if self._risk_manager is not None and not self._risk_manager._initialized:
            self._risk_manager.initialize()
        
        self.logger.info("开始运行策略管理器主循环...")
        self._running = True
        self._start_time = time.time()
        self._cycle_count = 0
        
        self.logger.info(self._format_status_report())
        
        while self._running:
            try:
                if self.is_risk_frozen():
                    self.logger.critical("系统已被风控冻结，停止运行")
                    self._running = False
                    break
                
                self.api.wait_update()
                self._cycle_count += 1
                
                if self._should_run_risk_check():
                    risk_results = self.run_risk_checks()
                    
                    if self.is_risk_frozen():
                        frozen_reason = self._risk_manager.get_frozen_reason() if self._risk_manager else "未知原因"
                        self.logger.critical(f"风控检查触发冻结: {frozen_reason}")
                        self._running = False
                        break
                
                for name, strategy in self._strategies.items():
                    health = self._strategy_health.get(name)
                    
                    if not health:
                        continue
                    
                    if not health.can_run():
                        continue
                    
                    if self.is_risk_frozen():
                        break
                    
                    try:
                        if hasattr(strategy, '_on_update'):
                            strategy._on_update()
                        health.record_success()
                    except Exception as e:
                        error_msg = f"策略 {name} 更新时出错: {e}"
                        self.logger.error(error_msg)
                        health.record_error(str(e))
                        
                        if not health.can_run():
                            self.logger.critical(f"策略 {name} 连续错误次数过多，已标记为 FAILED，需要手动重置")
                
                if self._should_report_status():
                    self.logger.info(self._format_status_report())
                    
                    if self._risk_manager is not None:
                        risk_info = self.get_risk_info()
                        drawdown_info = risk_info.get('drawdown_info', {})
                        self.logger.info(
                            f"[风控状态] 权益: {risk_info.get('equity', 0):.2f}, "
                            f"回撤: {drawdown_info.get('current_drawdown_percent', 0):.2f}%, "
                            f"冻结: {risk_info.get('is_frozen', False)}"
                        )
                    
                    self.logger.info("自动保存所有策略状态...")
                    saved_count = self.save_all_states()
                    self.logger.info(f"已保存 {saved_count} 个策略的状态")
                    
            except KeyboardInterrupt:
                self.logger.info("用户中断，停止策略管理器")
                self._running = False
                break
            except Exception as e:
                self.logger.error(f"主循环出错: {e}")
                self._running = False
                raise
    
    def run_all(self, load_saved_states: bool = True) -> None:
        if not self._initialized:
            self.logger.info("策略管理器未初始化，正在执行初始化...")
            self.initialize(load_saved_states=load_saved_states)
        
        self.logger.info("=" * 80)
        self.logger.info("策略管理器开始运行")
        self.logger.info(f"已注册策略数量: {len(self._strategies)}")
        for name in self._strategies.keys():
            strategy = self._strategies[name]
            health = self._strategy_health.get(name)
            status_line = f"  - {name}: {strategy.__class__.__name__}"
            if hasattr(strategy, 'short_period') and hasattr(strategy, 'long_period'):
                status_line += f" (MA{strategy.short_period}/MA{strategy.long_period})"
            self.logger.info(status_line)
        self.logger.info("=" * 80)
        self.logger.info(f"状态报告间隔: {self._status_report_interval} 秒")
        self.logger.info(f"状态存储目录: {self._state_dir}")
        self.logger.info("提示: 按 Ctrl+C 停止所有策略")
        self.logger.info("=" * 80)
        
        try:
            self._run_loop()
        except KeyboardInterrupt:
            self.logger.info("用户中断，停止所有策略")
        except Exception as e:
            self.logger.error(f"策略管理器运行出错: {e}")
            raise
        finally:
            self.logger.info("正在保存所有策略状态...")
            saved_count = self.save_all_states()
            self.logger.info(f"已保存 {saved_count} 个策略的状态")
            
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
    
    def reset_strategy_health(self, name: str) -> bool:
        health = self._strategy_health.get(name)
        if not health:
            self.logger.warning(f"策略 {name} 不存在")
            return False
        
        health.reset()
        self.logger.info(f"策略 {name} 健康状态已重置")
        return True
    
    def reset_all_health(self) -> int:
        count = 0
        for name in self._strategy_health.keys():
            if self.reset_strategy_health(name):
                count += 1
        return count
    
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
