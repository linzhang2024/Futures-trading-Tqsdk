import os
import sys
import logging
from typing import Dict, Any, List, Optional, Type, Callable
from dataclasses import dataclass, field, asdict
from collections import defaultdict
from datetime import datetime
from enum import Enum

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, base_dir)

from strategies.base_strategy import StrategyBase, SignalType
from core.manager import StrategyManager, StrategyHealth, StrategyHealthStatus


class ContractStatus(Enum):
    ACTIVE = "ACTIVE"
    PAUSED = "PAUSED"
    ERROR = "ERROR"


@dataclass
class ContractState:
    contract: str
    status: ContractStatus = ContractStatus.ACTIVE
    strategy_name: Optional[str] = None
    strategy_instance: Optional[StrategyBase] = None
    strategy_health: Optional[StrategyHealth] = None
    
    position_direction: str = "FLAT"
    position_volume: int = 0
    entry_price: float = 0.0
    current_price: float = 0.0
    float_profit: float = 0.0
    margin_used: float = 0.0
    
    atr: Optional[float] = None
    volatility_ratio: float = 1.0
    
    signal: SignalType = SignalType.HOLD
    last_signal_time: Optional[datetime] = None
    
    total_trades: int = 0
    win_trades: int = 0
    total_profit: float = 0.0
    
    allocated_capital: float = 0.0
    capital_weight: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'contract': self.contract,
            'status': self.status.value,
            'strategy_name': self.strategy_name,
            'position_direction': self.position_direction,
            'position_volume': self.position_volume,
            'entry_price': self.entry_price,
            'current_price': self.current_price,
            'float_profit': self.float_profit,
            'margin_used': self.margin_used,
            'atr': self.atr,
            'volatility_ratio': self.volatility_ratio,
            'signal': self.signal.value,
            'last_signal_time': self.last_signal_time.isoformat() if self.last_signal_time else None,
            'total_trades': self.total_trades,
            'win_trades': self.win_trades,
            'total_profit': self.total_profit,
            'allocated_capital': self.allocated_capital,
            'capital_weight': self.capital_weight,
            'health': self.strategy_health.to_dict() if self.strategy_health else None,
        }


@dataclass
class CapitalPoolConfig:
    total_capital: float = 1000000.0
    min_weight_per_contract: float = 0.05
    max_weight_per_contract: float = 0.4
    volatility_lookback_period: int = 20
    weight_calculation_method: str = "inverse_volatility"


class CapitalPoolManager:
    
    def __init__(self, config: CapitalPoolConfig = None):
        self.config = config or CapitalPoolConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
        
        self._contract_volatilities: Dict[str, List[float]] = defaultdict(list)
        self._contract_weights: Dict[str, float] = {}
        self._allocated_capitals: Dict[str, float] = {}
        
        self._total_capital = self.config.total_capital
        
        self.logger.info(
            f"CapitalPoolManager 初始化: "
            f"总资金={self._total_capital:,.0f}, "
            f"最小权重={self.config.min_weight_per_contract*100:.1f}%, "
            f"最大权重={self.config.max_weight_per_contract*100:.1f}%"
        )
    
    def set_total_capital(self, capital: float):
        if capital <= 0:
            raise ValueError("总资金必须大于0")
        
        self._total_capital = capital
        self.logger.info(f"总资金已更新为: {capital:,.0f}")
        self._reallocate_capital()
    
    def get_total_capital(self) -> float:
        return self._total_capital
    
    def record_contract_volatility(self, contract: str, atr: float, price: float):
        if atr is None or atr <= 0 or price is None or price <= 0:
            return
        
        volatility = atr / price
        
        if contract not in self._contract_volatilities:
            self._contract_volatilities[contract] = []
        
        self._contract_volatilities[contract].append(volatility)
        
        if len(self._contract_volatilities[contract]) > self.config.volatility_lookback_period:
            self._contract_volatilities[contract] = self._contract_volatilities[contract][-self.config.volatility_lookback_period:]
        
        self.logger.debug(
            f"记录波动率: {contract}, "
            f"ATR={atr:.2f}, "
            f"价格={price:.2f}, "
            f"波动率={volatility*100:.4f}%"
        )
    
    def get_contract_avg_volatility(self, contract: str) -> float:
        volatilities = self._contract_volatilities.get(contract, [])
        if not volatilities:
            return 0.01
        
        return sum(volatilities) / len(volatilities)
    
    def get_contract_volatility_ratio(self, contract: str) -> float:
        avg_vol = self.get_contract_avg_volatility(contract)
        
        all_vols = []
        for c, vols in self._contract_volatilities.items():
            if vols:
                all_vols.append(sum(vols) / len(vols))
        
        if not all_vols:
            return 1.0
        
        avg_market_vol = sum(all_vols) / len(all_vols)
        
        if avg_market_vol <= 0:
            return 1.0
        
        return avg_vol / avg_market_vol
    
    def calculate_weights(self, contracts: List[str]) -> Dict[str, float]:
        if not contracts:
            return {}
        
        if len(contracts) == 1:
            return {contracts[0]: 1.0}
        
        contract_vols = {}
        for contract in contracts:
            contract_vols[contract] = self.get_contract_avg_volatility(contract)
        
        if self.config.weight_calculation_method == "inverse_volatility":
            return self._calculate_inverse_volatility_weights(contract_vols)
        else:
            return self._calculate_equal_weights(contracts)
    
    def _calculate_equal_weights(self, contracts: List[str]) -> Dict[str, float]:
        if not contracts:
            return {}
        
        weight = 1.0 / len(contracts)
        return {contract: weight for contract in contracts}
    
    def _calculate_inverse_volatility_weights(
        self, 
        contract_vols: Dict[str, float]
    ) -> Dict[str, float]:
        if not contract_vols:
            return {}
        
        inverse_vols = {}
        for contract, vol in contract_vols.items():
            if vol <= 0:
                inverse_vols[contract] = 1.0
            else:
                inverse_vols[contract] = 1.0 / vol
        
        total_inverse_vol = sum(inverse_vols.values())
        
        if total_inverse_vol <= 0:
            return self._calculate_equal_weights(list(contract_vols.keys()))
        
        weights = {}
        for contract, inv_vol in inverse_vols.items():
            weight = inv_vol / total_inverse_vol
            
            weight = max(self.config.min_weight_per_contract, min(self.config.max_weight_per_contract, weight))
            
            weights[contract] = weight
        
        total_weight = sum(weights.values())
        if total_weight > 0:
            for contract in weights:
                weights[contract] = weights[contract] / total_weight
        
        return weights
    
    def allocate_capital(self, contracts: List[str]) -> Dict[str, float]:
        weights = self.calculate_weights(contracts)
        
        self._contract_weights = weights
        self._allocated_capitals = {}
        
        for contract, weight in weights.items():
            allocated = self._total_capital * weight
            self._allocated_capitals[contract] = allocated
            
            self.logger.info(
                f"资金分配: {contract}, "
                f"权重={weight*100:.2f}%, "
                f"分配资金={allocated:,.0f}"
            )
        
        return self._allocated_capitals
    
    def _reallocate_capital(self):
        contracts = list(self._contract_weights.keys())
        if contracts:
            self.allocate_capital(contracts)
    
    def get_allocated_capital(self, contract: str) -> float:
        return self._allocated_capitals.get(contract, 0.0)
    
    def get_weight(self, contract: str) -> float:
        return self._contract_weights.get(contract, 0.0)
    
    def get_allocation_summary(self) -> Dict[str, Any]:
        return {
            'total_capital': self._total_capital,
            'contract_weights': self._contract_weights.copy(),
            'allocated_capitals': self._allocated_capitals.copy(),
            'contract_volatilities': {
                c: self.get_contract_avg_volatility(c) 
                for c in self._contract_volatilities
            },
        }


class MultiContractRouter:
    
    def __init__(
        self,
        connector: Any = None,
        capital_pool_config: CapitalPoolConfig = None,
        strategy_manager: StrategyManager = None,
    ):
        self.connector = connector
        self.api = None
        self.logger = logging.getLogger(self.__class__.__name__)
        
        if connector:
            self.api = connector.get_api()
        
        self._contract_states: Dict[str, ContractState] = {}
        self._contract_to_strategy: Dict[str, str] = {}
        self._strategy_to_contract: Dict[str, str] = {}
        
        self._capital_pool = CapitalPoolManager(capital_pool_config)
        self._strategy_manager = strategy_manager or StrategyManager(connector=connector)
        
        self._initialized = False
        self._running = False
        
        self.logger.info("MultiContractRouter 初始化完成")
    
    def set_connector(self, connector: Any):
        if connector is None:
            self.logger.error("Connector 不能为 None")
            raise ValueError("Connector 不能为 None")
        
        self.connector = connector
        self.api = connector.get_api()
        self._strategy_manager.set_connector(connector)
        self.logger.info("Connector 已设置")
    
    def get_capital_pool(self) -> CapitalPoolManager:
        return self._capital_pool
    
    def register_contract_strategy(
        self,
        contract: str,
        strategy_class: Type[StrategyBase],
        strategy_params: Dict[str, Any] = None,
        strategy_name: str = None,
    ) -> str:
        if contract in self._contract_states:
            self.logger.warning(f"合约 {contract} 已注册，将覆盖旧配置")
            self.unregister_contract(contract)
        
        strategy_name = strategy_name or f"{strategy_class.__name__}_{contract}"
        strategy_params = strategy_params or {}
        
        strategy_params['connector'] = self.connector
        strategy_params['contract'] = contract
        
        try:
            strategy = strategy_class(**strategy_params)
        except Exception as e:
            self.logger.error(f"创建策略实例失败 [{contract}]: {e}")
            raise
        
        self._strategy_manager.register_strategy(strategy_name, strategy)
        
        contract_state = ContractState(
            contract=contract,
            strategy_name=strategy_name,
            strategy_instance=strategy,
            strategy_health=self._strategy_manager.get_strategy_health(strategy_name),
        )
        
        self._contract_states[contract] = contract_state
        self._contract_to_strategy[contract] = strategy_name
        self._strategy_to_contract[strategy_name] = contract
        
        self.logger.info(f"已注册合约策略: {contract} -> {strategy_name}")
        
        return strategy_name
    
    def unregister_contract(self, contract: str) -> bool:
        if contract not in self._contract_states:
            self.logger.warning(f"合约 {contract} 未注册")
            return False
        
        strategy_name = self._contract_to_strategy.get(contract)
        
        if strategy_name:
            self._strategy_manager.unregister_strategy(strategy_name)
            del self._strategy_to_contract[strategy_name]
        
        del self._contract_states[contract]
        del self._contract_to_strategy[contract]
        
        self.logger.info(f"已注销合约: {contract}")
        return True
    
    def get_registered_contracts(self) -> List[str]:
        return list(self._contract_states.keys())
    
    def get_contract_state(self, contract: str) -> Optional[ContractState]:
        return self._contract_states.get(contract)
    
    def get_all_contract_states(self) -> Dict[str, ContractState]:
        return self._contract_states.copy()
    
    def initialize(self):
        if self._initialized:
            self.logger.info("MultiContractRouter 已初始化，跳过重复初始化")
            return
        
        if self.api is None:
            raise RuntimeError("API 未初始化，请确保 Connector 已连接")
        
        contracts = self.get_registered_contracts()
        if not contracts:
            self.logger.warning("没有注册任何合约策略")
        
        self._strategy_manager.initialize(load_saved_states=False)
        
        if contracts:
            self._capital_pool.allocate_capital(contracts)
            
            for contract in contracts:
                state = self._contract_states[contract]
                state.allocated_capital = self._capital_pool.get_allocated_capital(contract)
                state.capital_weight = self._capital_pool.get_weight(contract)
        
        self._initialized = True
        self.logger.info("MultiContractRouter 初始化完成")
    
    def update_contract_volatility(self, contract: str, atr: float, price: float):
        self._capital_pool.record_contract_volatility(contract, atr, price)
        
        if contract in self._contract_states:
            state = self._contract_states[contract]
            state.atr = atr
            state.volatility_ratio = self._capital_pool.get_contract_volatility_ratio(contract)
    
    def rebalance_capital(self):
        contracts = self.get_registered_contracts()
        if not contracts:
            return
        
        old_weights = self._capital_pool.get_allocation_summary()['contract_weights'].copy()
        
        self._capital_pool.allocate_capital(contracts)
        
        new_weights = self._capital_pool.get_allocation_summary()['contract_weights']
        
        for contract in contracts:
            if contract in self._contract_states:
                state = self._contract_states[contract]
                state.allocated_capital = self._capital_pool.get_allocated_capital(contract)
                state.capital_weight = self._capital_pool.get_weight(contract)
                
                old_weight = old_weights.get(contract, 0)
                new_weight = new_weights.get(contract, 0)
                
                if abs(old_weight - new_weight) > 0.01:
                    self.logger.info(
                        f"资金再平衡: {contract}, "
                        f"旧权重={old_weight*100:.2f}%, "
                        f"新权重={new_weight*100:.2f}%"
                    )
    
    def update_contract_position(
        self,
        contract: str,
        direction: str,
        volume: int,
        entry_price: float = 0.0,
        current_price: float = 0.0,
        float_profit: float = 0.0,
        margin_used: float = 0.0,
    ):
        if contract not in self._contract_states:
            return
        
        state = self._contract_states[contract]
        state.position_direction = direction
        state.position_volume = volume
        state.entry_price = entry_price
        state.current_price = current_price
        state.float_profit = float_profit
        state.margin_used = margin_used
    
    def update_contract_signal(
        self,
        contract: str,
        signal: SignalType,
    ):
        if contract not in self._contract_states:
            return
        
        state = self._contract_states[contract]
        
        if signal != SignalType.HOLD and state.signal != signal:
            state.total_trades += 1
            state.last_signal_time = datetime.now()
            
            if signal == SignalType.BUY:
                self.logger.info(f"[{contract}] 买入信号")
            elif signal == SignalType.SELL:
                self.logger.info(f"[{contract}] 卖出信号")
        
        state.signal = signal
    
    def get_multi_contract_status(self) -> Dict[str, Any]:
        contract_statuses = {}
        
        for contract, state in self._contract_states.items():
            strategy = state.strategy_instance
            health = state.strategy_health
            
            status = state.to_dict()
            
            if strategy:
                if hasattr(strategy, 'short_ma'):
                    status['short_ma'] = strategy.short_ma
                if hasattr(strategy, 'long_ma'):
                    status['long_ma'] = strategy.long_ma
                if hasattr(strategy, 'rsi'):
                    status['rsi'] = strategy.rsi
                if hasattr(strategy, 'short_period'):
                    status['short_period'] = strategy.short_period
                if hasattr(strategy, 'long_period'):
                    status['long_period'] = strategy.long_period
            
            contract_statuses[contract] = status
        
        return {
            'generated_at': datetime.now().isoformat(),
            'total_contracts': len(self._contract_states),
            'contracts': contract_statuses,
            'capital_pool': self._capital_pool.get_allocation_summary(),
        }
    
    def get_dashboard_positions(self) -> List[Dict[str, Any]]:
        positions = []
        
        for contract, state in self._contract_states.items():
            if state.position_volume > 0:
                direction_display = "多单" if state.position_direction == "LONG" else "空单"
                
                positions.append({
                    'contract': contract,
                    'direction': direction_display,
                    'volume': state.position_volume,
                    'open_price': state.entry_price,
                    'current_price': state.current_price,
                    'float_profit': state.float_profit,
                    'margin': state.margin_used,
                })
        
        return positions
    
    def get_dashboard_strategies(self) -> List[Dict[str, Any]]:
        strategies = []
        
        for contract, state in self._contract_states.items():
            strategy = state.strategy_instance
            health = state.strategy_health
            
            if strategy is None:
                continue
            
            short_ma = getattr(strategy, 'short_ma', None)
            long_ma = getattr(strategy, 'long_ma', None)
            short_period = getattr(strategy, 'short_period', 5)
            long_period = getattr(strategy, 'long_period', 20)
            signal = state.signal.value
            is_ready = getattr(strategy, 'is_ready', lambda: False)()
            status = health.status.value if health else "UNKNOWN"
            error_count = health.error_count if health else 0
            
            strategies.append({
                'name': state.strategy_name or strategy.__class__.__name__,
                'contract': contract,
                'short_ma': short_ma,
                'long_ma': long_ma,
                'short_period': short_period,
                'long_period': long_period,
                'signal': signal,
                'is_ready': is_ready,
                'status': status,
                'error_count': error_count,
            })
        
        return strategies
    
    def run_all(self):
        if not self._initialized:
            self.logger.info("MultiContractRouter 未初始化，正在执行初始化...")
            self.initialize()
        
        self._running = True
        self.logger.info("MultiContractRouter 开始运行")
        
        try:
            self._strategy_manager.run_all(load_saved_states=False)
        except KeyboardInterrupt:
            self.logger.info("用户中断，停止 MultiContractRouter")
        except Exception as e:
            self.logger.error(f"MultiContractRouter 运行出错: {e}")
            raise
        finally:
            self._running = False
    
    def stop_all(self):
        self._running = False
        self._strategy_manager.stop_all()
        self.logger.info("MultiContractRouter 已停止")
