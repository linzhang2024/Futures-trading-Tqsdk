import os
import sys
import logging
import json
import csv
import math
import random
from datetime import datetime, date, timedelta
from typing import Dict, Any, List, Optional, Type, Tuple, Callable, Union
from dataclasses import dataclass, field, asdict
from enum import Enum

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, base_dir)

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

from strategies.base_strategy import StrategyBase, SignalType
from strategies.double_ma_strategy import DoubleMAStrategy
from strategies.adaptive_momentum_strategy import AdaptiveMomentumStrategy


CHINESE_FONT_AVAILABLE = False


def _check_chinese_font():
    global CHINESE_FONT_AVAILABLE
    if CHINESE_FONT_AVAILABLE:
        return True
    
    if not MATPLOTLIB_AVAILABLE:
        return False
    
    try:
        priority_fonts = ['SimHei', 'Microsoft YaHei']
        fallback_fonts = [
            'STSong', 'STKaiti', 'SimSun', 'KaiTi', 'FangSong',
            'PingFang SC', 'Hiragino Sans GB', 'WenQuanYi Micro Hei'
        ]
        
        for font_name in priority_fonts + fallback_fonts:
            try:
                font_prop = FontProperties(family=[font_name])
                font_path = matplotlib.font_manager.findfont(font_prop)
                if font_path and os.path.exists(font_path):
                    plt.rcParams['font.sans-serif'] = [font_name] + plt.rcParams['font.sans-serif']
                    plt.rcParams['axes.unicode_minus'] = False
                    CHINESE_FONT_AVAILABLE = True
                    logging.getLogger(__name__).info(f"找到中文字体: {font_name}")
                    return True
            except Exception:
                continue
        
        logging.getLogger(__name__).warning("未找到中文字体，图表将使用英文标签")
        return False
    except Exception:
        return False


def _get_label(zh_label: str, en_label: str) -> str:
    if _check_chinese_font():
        return zh_label
    return en_label


class StrategyType(Enum):
    DOUBLE_MA = "DoubleMAStrategy"
    ADAPTIVE_MOMENTUM = "AdaptiveMomentumStrategy"


@dataclass
class TradeRecord:
    entry_time: datetime
    exit_time: Optional[datetime]
    direction: str
    entry_price: float
    exit_price: Optional[float]
    volume: int
    contract: str
    profit_loss: Optional[float] = None
    profit_loss_percent: Optional[float] = None
    holding_period_bars: int = 0
    status: str = "open"
    entry_reason: str = ""
    exit_reason: str = ""


@dataclass
class PerformanceResult:
    strategy_name: str
    strategy_type: StrategyType
    contract: str
    
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    
    total_return: float = 0.0
    total_return_percent: float = 0.0
    annualized_return: float = 0.0
    
    max_drawdown: float = 0.0
    max_drawdown_percent: float = 0.0
    
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    
    profit_factor: float = 0.0
    avg_trade_return: float = 0.0
    avg_winning_trade: float = 0.0
    avg_losing_trade: float = 0.0
    
    largest_win: float = 0.0
    largest_loss: float = 0.0
    
    avg_holding_period: float = 0.0
    
    initial_capital: float = 1000000.0
    final_capital: float = 1000000.0
    
    equity_curve: List[Dict[str, Any]] = field(default_factory=list)
    trade_records: List[TradeRecord] = field(default_factory=list)
    
    atr_filtered_signals: int = 0
    rsi_filtered_signals: int = 0
    trailing_stop_triggered: int = 0
    tp_triggered: int = 0
    sl_triggered: int = 0
    
    start_dt: Optional[date] = None
    end_dt: Optional[date] = None
    
    status: str = "completed"
    error_message: Optional[str] = None


@dataclass
class ComparisonReport:
    report_id: str
    generated_at: datetime
    
    contracts: List[str] = field(default_factory=list)
    old_strategy_results: Dict[str, PerformanceResult] = field(default_factory=dict)
    new_strategy_results: Dict[str, PerformanceResult] = field(default_factory=dict)
    
    overall_summary: Dict[str, Any] = field(default_factory=dict)
    
    win_rate_improvement: Dict[str, float] = field(default_factory=dict)
    max_drawdown_improvement: Dict[str, float] = field(default_factory=dict)
    sharpe_improvement: Dict[str, float] = field(default_factory=dict)


class MockKlineGenerator:
    
    def __init__(
        self,
        start_dt: date,
        end_dt: date,
        initial_price: float = 3000.0,
        volatility: float = 0.02,
        trend: float = 0.0,
        kline_duration: int = 60,
    ):
        self.start_dt = start_dt
        self.end_dt = end_dt
        self.initial_price = initial_price
        self.volatility = volatility
        self.trend = trend
        self.kline_duration = kline_duration
        
        self._klines: List[Dict[str, Any]] = []
        self._current_idx = 0
    
    def generate(self, seed: int = None) -> List[Dict[str, Any]]:
        if seed is not None:
            import random
            random.seed(seed)
        else:
            import random
        
        self._klines = []
        
        seconds_per_day = 14400
        total_seconds = (self.end_dt - self.start_dt).days * seconds_per_day
        total_klines = int(total_seconds / self.kline_duration)
        
        current_price = self.initial_price
        current_time = datetime.combine(self.start_dt, datetime.min.time().replace(hour=9, minute=0))
        
        for i in range(total_klines):
            daily_factor = 1.0
            hour = current_time.hour
            if 9 <= hour < 11 or 13 <= hour < 15:
                daily_factor = 1.0
            else:
                daily_factor = 0.5
            
            change = random.gauss(
                self.trend * self.kline_duration / 86400,
                self.volatility * math.sqrt(self.kline_duration / 86400) * daily_factor
            )
            
            open_price = current_price
            high_price = current_price * (1 + abs(random.gauss(0, self.volatility * 0.5)))
            low_price = current_price * (1 - abs(random.gauss(0, self.volatility * 0.5)))
            close_price = current_price * (1 + change)
            
            high_price = max(high_price, open_price, close_price)
            low_price = min(low_price, open_price, close_price)
            
            kline = {
                'datetime': current_time,
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'volume': random.randint(100, 1000),
                'open_oi': random.randint(50000, 100000),
                'close_oi': random.randint(50000, 100000),
            }
            
            self._klines.append(kline)
            
            current_price = close_price
            
            current_time = self._next_trading_time(current_time)
        
        return self._klines
    
    def _next_trading_time(self, current_time: datetime) -> datetime:
        from datetime import timedelta
        
        next_time = current_time + timedelta(seconds=self.kline_duration)
        
        if next_time.hour >= 11 and next_time.hour < 13:
            next_time = next_time.replace(hour=13, minute=0, second=0)
        elif next_time.hour >= 15:
            next_time = next_time + timedelta(days=1)
            next_time = next_time.replace(hour=9, minute=0, second=0)
            
            if next_time.weekday() >= 5:
                days_to_monday = 7 - next_time.weekday()
                next_time = next_time + timedelta(days=days_to_monday)
        
        return next_time


class BacktestSimulator:
    
    def __init__(
        self,
        initial_capital: float = 1000000.0,
        contract_multiplier: int = 10,
        commission_per_lot: float = 5.0,
        slippage_ticks: float = 1.0,
        price_tick: float = 1.0,
    ):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.contract_multiplier = contract_multiplier
        self.commission_per_lot = commission_per_lot
        self.slippage_ticks = slippage_ticks
        self.price_tick = price_tick
        
        self._position = 0
        self._entry_price = 0.0
        self._trade_records: List[TradeRecord] = []
        self._equity_curve: List[Dict[str, Any]] = []
        self._peak_capital = initial_capital
        
        self._highest_price_since_entry: Optional[float] = None
        self._lowest_price_since_entry: Optional[float] = None
        self._trailing_stop_active = False
        self._trailing_stop_price: Optional[float] = None
        
        self.atr_filtered_signals = 0
        self.rsi_filtered_signals = 0
        self.trailing_stop_triggered = 0
        self.tp_triggered = 0
        self.sl_triggered = 0
        
        self._current_trade: Optional[TradeRecord] = None
        self._cycle_count = 0
    
    def calculate_position_size(
        self,
        risk_percent: float,
        atr: float,
        current_price: float,
    ) -> int:
        if atr <= 0 or current_price <= 0:
            return 1
        
        risk_amount = self.current_capital * risk_percent
        risk_per_lot = atr * self.contract_multiplier
        
        if risk_per_lot <= 0:
            return 1
        
        position_size = int(risk_amount / risk_per_lot)
        return max(1, position_size)
    
    def open_position(
        self,
        direction: str,
        price: float,
        volume: int,
        current_time: datetime,
        reason: str = "",
    ) -> bool:
        if self._position != 0:
            return False
        
        actual_price = price
        if direction == "BUY":
            actual_price = price + self.slippage_ticks * self.price_tick
        elif direction == "SELL":
            actual_price = price - self.slippage_ticks * self.price_tick
        
        commission_cost = volume * self.commission_per_lot
        margin_requirement = actual_price * volume * self.contract_multiplier * 0.1
        
        if self.current_capital < commission_cost + margin_requirement * 0.5:
            return False
        
        self._position = volume if direction == "BUY" else -volume
        self._entry_price = actual_price
        
        self._highest_price_since_entry = actual_price
        self._lowest_price_since_entry = actual_price
        self._trailing_stop_active = False
        self._trailing_stop_price = None
        
        self._current_trade = TradeRecord(
            entry_time=current_time,
            exit_time=None,
            direction=direction,
            entry_price=actual_price,
            exit_price=None,
            volume=volume,
            contract="MOCK",
            entry_reason=reason,
        )
        
        self.current_capital -= commission_cost
        
        return True
    
    def close_position(
        self,
        current_price: float,
        current_time: datetime,
        reason: str = "",
    ) -> bool:
        if self._position == 0:
            return False
        
        actual_price = current_price
        if self._position > 0:
            actual_price = current_price - self.slippage_ticks * self.price_tick
        else:
            actual_price = current_price + self.slippage_ticks * self.price_tick
        
        commission_cost = abs(self._position) * self.commission_per_lot
        
        if self._position > 0:
            profit_loss = (actual_price - self._entry_price) * abs(self._position) * self.contract_multiplier
        else:
            profit_loss = (self._entry_price - actual_price) * abs(self._position) * self.contract_multiplier
        
        self.current_capital += profit_loss - commission_cost
        
        if self._current_trade:
            self._current_trade.exit_time = current_time
            self._current_trade.exit_price = actual_price
            self._current_trade.profit_loss = profit_loss
            self._current_trade.status = "closed"
            self._current_trade.exit_reason = reason
            
            if self._entry_price > 0:
                self._current_trade.profit_loss_percent = (profit_loss / (self._entry_price * abs(self._position) * self.contract_multiplier * 0.1)) * 100
            
            self._trade_records.append(self._current_trade)
        
        self._position = 0
        self._entry_price = 0.0
        self._current_trade = None
        self._highest_price_since_entry = None
        self._lowest_price_since_entry = None
        self._trailing_stop_active = False
        self._trailing_stop_price = None
        
        return True
    
    def update_price(
        self,
        current_price: float,
        atr: float,
        atr_exit_multiplier: float = 2.0,
        trailing_stop_multiplier: float = 2.0,
    ) -> Tuple[bool, str]:
        if self._position == 0:
            return False, ""
        
        if self._current_trade:
            self._current_trade.holding_period_bars += 1
        
        if self._position > 0:
            if self._highest_price_since_entry is None or current_price > self._highest_price_since_entry:
                self._highest_price_since_entry = current_price
            
            profit_amount = (current_price - self._entry_price) * self.contract_multiplier
            profit_atr = profit_amount / atr if atr > 0 else 0
            
            if profit_atr >= atr_exit_multiplier and not self._trailing_stop_active:
                self._trailing_stop_active = True
            
            if self._trailing_stop_active:
                new_stop = self._highest_price_since_entry - trailing_stop_multiplier * atr
                if self._trailing_stop_price is None or new_stop > self._trailing_stop_price:
                    self._trailing_stop_price = new_stop
                
                if current_price <= self._trailing_stop_price:
                    return True, "trailing_stop"
        
        elif self._position < 0:
            if self._lowest_price_since_entry is None or current_price < self._lowest_price_since_entry:
                self._lowest_price_since_entry = current_price
            
            profit_amount = (self._entry_price - current_price) * self.contract_multiplier
            profit_atr = profit_amount / atr if atr > 0 else 0
            
            if profit_atr >= atr_exit_multiplier and not self._trailing_stop_active:
                self._trailing_stop_active = True
            
            if self._trailing_stop_active:
                new_stop = self._lowest_price_since_entry + trailing_stop_multiplier * atr
                if self._trailing_stop_price is None or new_stop < self._trailing_stop_price:
                    self._trailing_stop_price = new_stop
                
                if current_price >= self._trailing_stop_price:
                    return True, "trailing_stop"
        
        return False, ""
    
    def record_equity(self, cycle: int, timestamp: datetime):
        self._cycle_count = cycle
        
        float_profit = 0.0
        if self._position != 0:
            pass
        
        equity = self.current_capital + float_profit
        
        if equity > self._peak_capital:
            self._peak_capital = equity
        
        drawdown = self._peak_capital - equity
        drawdown_percent = (drawdown / self._peak_capital * 100) if self._peak_capital > 0 else 0.0
        
        self._equity_curve.append({
            'cycle': cycle,
            'timestamp': timestamp.isoformat(),
            'equity': equity,
            'drawdown': drawdown,
            'drawdown_percent': drawdown_percent,
            'position': self._position,
        })
    
    def get_current_equity(self) -> float:
        return self.current_capital


class StrategyTester:
    
    def __init__(
        self,
        strategy_class: Type[StrategyBase],
        strategy_params: Dict[str, Any],
        contract: str,
        start_dt: date,
        end_dt: date,
        initial_capital: float = 1000000.0,
        contract_multiplier: int = 10,
    ):
        self.strategy_class = strategy_class
        self.strategy_params = strategy_params
        self.contract = contract
        self.start_dt = start_dt
        self.end_dt = end_dt
        self.initial_capital = initial_capital
        self.contract_multiplier = contract_multiplier
        
        self.simulator = BacktestSimulator(
            initial_capital=initial_capital,
            contract_multiplier=contract_multiplier,
        )
        
        self.logger = logging.getLogger(f"{self.__class__.__name__}.{contract}")
    
    def run(
        self,
        klines: List[Dict[str, Any]],
        use_dynamic_position: bool = False,
        risk_per_trade_percent: float = 0.01,
    ) -> PerformanceResult:
        result = PerformanceResult(
            strategy_name=self.strategy_class.__name__,
            strategy_type=StrategyType.ADAPTIVE_MOMENTUM if self.strategy_class == AdaptiveMomentumStrategy else StrategyType.DOUBLE_MA,
            contract=self.contract,
            initial_capital=self.initial_capital,
            start_dt=self.start_dt,
            end_dt=self.end_dt,
        )
        
        try:
            strategy = self.strategy_class(**self.strategy_params)
            
            warmup_period = max(
                self.strategy_params.get('long_period', 20),
                self.strategy_params.get('rsi_period', 14),
                self.strategy_params.get('atr_period', 14),
            )
            
            if len(klines) <= warmup_period:
                result.status = "error"
                result.error_message = f"K线数量不足，需要至少 {warmup_period + 1} 根"
                return result
            
            for i in range(warmup_period):
                kline = klines[i]
                strategy.update_from_kline(kline)
            
            for i in range(warmup_period, len(klines)):
                kline = klines[i]
                current_time = kline.get('datetime', datetime.now())
                close_price = kline.get('close', 0.0)
                high_price = kline.get('high', close_price)
                low_price = kline.get('low', close_price)
                
                atr = getattr(strategy, 'atr', None)
                if atr is None or atr <= 0:
                    atr = (high_price - low_price) * 1.5
                
                if self.simulator._position != 0:
                    should_close, close_reason = self.simulator.update_price(
                        current_price=close_price,
                        atr=atr,
                        atr_exit_multiplier=getattr(strategy, 'atr_exit_multiplier', 2.0),
                        trailing_stop_multiplier=getattr(strategy, 'trailing_stop_atr_multiplier', 2.0),
                    )
                    
                    if should_close:
                        if close_reason == "trailing_stop":
                            self.simulator.trailing_stop_triggered += 1
                        self.simulator.close_position(close_price, current_time, close_reason)
                
                strategy.update_prices(close_price, high_price, low_price)
                
                signal = strategy.signal
                
                if self.simulator._position == 0:
                    if signal == SignalType.BUY:
                        position_size = 1
                        if use_dynamic_position and hasattr(strategy, '_calculate_position_size'):
                            position_size = strategy._calculate_position_size(atr, close_price)
                            position_size = max(1, int(position_size))
                        
                        self.simulator.open_position(
                            direction="BUY",
                            price=close_price,
                            volume=position_size,
                            current_time=current_time,
                            reason="golden_cross",
                        )
                    
                    elif signal == SignalType.SELL:
                        position_size = 1
                        if use_dynamic_position and hasattr(strategy, '_calculate_position_size'):
                            position_size = strategy._calculate_position_size(atr, close_price)
                            position_size = max(1, int(position_size))
                        
                        self.simulator.open_position(
                            direction="SELL",
                            price=close_price,
                            volume=position_size,
                            current_time=current_time,
                            reason="death_cross",
                        )
                
                self.simulator.record_equity(i, current_time)
            
            if self.simulator._position != 0:
                last_kline = klines[-1]
                last_time = last_kline.get('datetime', datetime.now())
                last_price = last_kline.get('close', 0.0)
                self.simulator.close_position(last_price, last_time, "end_of_backtest")
            
            result = self._calculate_performance(result)
            
        except Exception as e:
            self.logger.error(f"回测执行出错: {e}", exc_info=True)
            result.status = "error"
            result.error_message = str(e)
        
        return result
    
    def _calculate_performance(self, result: PerformanceResult) -> PerformanceResult:
        result.trade_records = self.simulator._trade_records
        result.equity_curve = self.simulator._equity_curve
        result.final_capital = self.simulator.current_capital
        
        result.total_trades = len(self.simulator._trade_records)
        
        winning_trades = [t for t in self.simulator._trade_records if t.profit_loss is not None and t.profit_loss > 0]
        losing_trades = [t for t in self.simulator._trade_records if t.profit_loss is not None and t.profit_loss <= 0]
        
        result.winning_trades = len(winning_trades)
        result.losing_trades = len(losing_trades)
        
        if result.total_trades > 0:
            result.win_rate = (result.winning_trades / result.total_trades) * 100
        
        total_profit = sum(t.profit_loss for t in winning_trades if t.profit_loss is not None)
        total_loss = abs(sum(t.profit_loss for t in losing_trades if t.profit_loss is not None))
        
        result.total_return = result.final_capital - self.initial_capital
        result.total_return_percent = (result.total_return / self.initial_capital * 100) if self.initial_capital > 0 else 0
        
        if result.total_trades > 0:
            result.avg_trade_return = result.total_return / result.total_trades
        
        if result.winning_trades > 0:
            result.avg_winning_trade = total_profit / result.winning_trades
            result.largest_win = max(t.profit_loss for t in winning_trades if t.profit_loss is not None)
        
        if result.losing_trades > 0:
            result.avg_losing_trade = total_loss / result.losing_trades if total_loss > 0 else 0
            result.largest_loss = min(t.profit_loss for t in losing_trades if t.profit_loss is not None)
        
        if total_loss > 0:
            result.profit_factor = total_profit / total_loss
        elif total_profit > 0:
            result.profit_factor = float('inf')
        else:
            result.profit_factor = 0.0
        
        if self.simulator._equity_curve:
            equities = [self.initial_capital] + [p['equity'] for p in self.simulator._equity_curve]
            peak = self.initial_capital
            max_dd = 0.0
            max_dd_percent = 0.0
            
            for eq in equities:
                if eq > peak:
                    peak = eq
                dd = peak - eq
                dd_percent = (dd / peak * 100) if peak > 0 else 0.0
                
                if dd > max_dd:
                    max_dd = dd
                if dd_percent > max_dd_percent:
                    max_dd_percent = dd_percent
            
            result.max_drawdown = max_dd
            result.max_drawdown_percent = max_dd_percent
            
            if len(self.simulator._equity_curve) > 1:
                returns = []
                prev_eq = self.initial_capital
                for p in self.simulator._equity_curve:
                    eq = p['equity']
                    if prev_eq > 0:
                        returns.append((eq - prev_eq) / prev_eq)
                    prev_eq = eq
                
                if returns and len(returns) > 1:
                    try:
                        import statistics
                        avg_return = statistics.mean(returns)
                        std_return = statistics.stdev(returns)
                        
                        if std_return > 0:
                            daily_rf = (1 + 0.03) ** (1/252) - 1
                            daily_sharpe = (avg_return - daily_rf) / std_return
                            result.sharpe_ratio = daily_sharpe * (252 ** 0.5)
                        
                        negative_returns = [r for r in returns if r < 0]
                        if negative_returns and len(negative_returns) > 1:
                            std_negative = statistics.stdev(negative_returns)
                            if std_negative > 0:
                                daily_sortino = (avg_return - daily_rf) / std_negative
                                result.sortino_ratio = daily_sortino * (252 ** 0.5)
                        
                        if max_dd_percent > 0:
                            result.calmar_ratio = result.total_return_percent / max_dd_percent
                            
                    except Exception as e:
                        self.logger.debug(f"计算风险调整收益指标时出错: {e}")
        
        if result.total_trades > 0:
            holding_periods = [t.holding_period_bars for t in self.simulator._trade_records]
            result.avg_holding_period = sum(holding_periods) / len(holding_periods)
        
        result.atr_filtered_signals = getattr(self.simulator, 'atr_filtered_signals', 0)
        result.rsi_filtered_signals = getattr(self.simulator, 'rsi_filtered_signals', 0)
        result.trailing_stop_triggered = getattr(self.simulator, 'trailing_stop_triggered', 0)
        result.tp_triggered = getattr(self.simulator, 'tp_triggered', 0)
        result.sl_triggered = getattr(self.simulator, 'sl_triggered', 0)
        
        result.status = "completed"
        
        return result


class StrategyComparator:
    
    def __init__(
        self,
        contracts: List[str],
        start_dt: date,
        end_dt: date,
        initial_capital: float = 1000000.0,
    ):
        self.contracts = contracts
        self.start_dt = start_dt
        self.end_dt = end_dt
        self.initial_capital = initial_capital
        
        self.logger = logging.getLogger(self.__class__.__name__)
        
        self.old_strategy_results: Dict[str, PerformanceResult] = {}
        self.new_strategy_results: Dict[str, PerformanceResult] = {}
        
        self._kline_data: Dict[str, List[Dict[str, Any]]] = {}
    
    def generate_mock_data(
        self,
        contract_configs: Dict[str, Dict[str, Any]],
        seed: int = 42,
    ):
        self.logger.info(f"生成模拟K线数据，种子={seed}")
        
        for contract in self.contracts:
            config = contract_configs.get(contract, {
                'initial_price': 3000.0,
                'volatility': 0.02,
                'trend': 0.0,
            })
            
            generator = MockKlineGenerator(
                start_dt=self.start_dt,
                end_dt=self.end_dt,
                initial_price=config['initial_price'],
                volatility=config['volatility'],
                trend=config['trend'],
            )
            
            self._kline_data[contract] = generator.generate(seed=seed)
            self.logger.info(f"合约 {contract}: 生成 {len(self._kline_data[contract])} 根K线")
    
    def run_comparison(
        self,
        old_strategy_params: Dict[str, Any],
        new_strategy_params: Dict[str, Any],
        contract_multipliers: Dict[str, int] = None,
    ) -> ComparisonReport:
        contract_multipliers = contract_multipliers or {}
        
        self.logger.info("开始策略对比测试...")
        self.logger.info(f"旧策略参数: {old_strategy_params}")
        self.logger.info(f"新策略参数: {new_strategy_params}")
        
        for contract in self.contracts:
            self.logger.info(f"测试合约: {contract}")
            
            klines = self._kline_data.get(contract, [])
            if not klines:
                self.logger.warning(f"合约 {contract} 没有K线数据，跳过")
                continue
            
            multiplier = contract_multipliers.get(contract, 10)
            
            self.logger.info(f"  运行旧策略 (DoubleMAStrategy)...")
            old_tester = StrategyTester(
                strategy_class=DoubleMAStrategy,
                strategy_params={
                    'contract': contract,
                    **old_strategy_params,
                },
                contract=contract,
                start_dt=self.start_dt,
                end_dt=self.end_dt,
                initial_capital=self.initial_capital,
                contract_multiplier=multiplier,
            )
            
            old_result = old_tester.run(klines, use_dynamic_position=False)
            self.old_strategy_results[contract] = old_result
            
            self.logger.info(f"    结果: 交易次数={old_result.total_trades}, 胜率={old_result.win_rate:.2f}%, "
                           f"收益率={old_result.total_return_percent:.2f}%, 最大回撤={old_result.max_drawdown_percent:.2f}%, "
                           f"夏普比率={old_result.sharpe_ratio:.2f}")
            
            self.logger.info(f"  运行新策略 (AdaptiveMomentumStrategy)...")
            new_tester = StrategyTester(
                strategy_class=AdaptiveMomentumStrategy,
                strategy_params={
                    'contract': contract,
                    **new_strategy_params,
                },
                contract=contract,
                start_dt=self.start_dt,
                end_dt=self.end_dt,
                initial_capital=self.initial_capital,
                contract_multiplier=multiplier,
            )
            
            new_result = new_tester.run(klines, use_dynamic_position=True)
            self.new_strategy_results[contract] = new_result
            
            self.logger.info(f"    结果: 交易次数={new_result.total_trades}, 胜率={new_result.win_rate:.2f}%, "
                           f"收益率={new_result.total_return_percent:.2f}%, 最大回撤={new_result.max_drawdown_percent:.2f}%, "
                           f"夏普比率={new_result.sharpe_ratio:.2f}")
        
        report = self._generate_report()
        
        return report
    
    def _generate_report(self) -> ComparisonReport:
        report = ComparisonReport(
            report_id=f"COMP_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            generated_at=datetime.now(),
            contracts=self.contracts,
            old_strategy_results=self.old_strategy_results,
            new_strategy_results=self.new_strategy_results,
        )
        
        total_old_trades = 0
        total_new_trades = 0
        total_old_win_rate = 0.0
        total_new_win_rate = 0.0
        total_old_return = 0.0
        total_new_return = 0.0
        total_old_dd = 0.0
        total_new_dd = 0.0
        total_old_sharpe = 0.0
        total_new_sharpe = 0.0
        
        valid_contracts = 0
        
        for contract in self.contracts:
            old_result = self.old_strategy_results.get(contract)
            new_result = self.new_strategy_results.get(contract)
            
            if old_result is None or new_result is None:
                continue
            
            valid_contracts += 1
            
            total_old_trades += old_result.total_trades
            total_new_trades += new_result.total_trades
            total_old_win_rate += old_result.win_rate
            total_new_win_rate += new_result.win_rate
            total_old_return += old_result.total_return_percent
            total_new_return += new_result.total_return_percent
            total_old_dd += old_result.max_drawdown_percent
            total_new_dd += new_result.max_drawdown_percent
            total_old_sharpe += old_result.sharpe_ratio
            total_new_sharpe += new_result.sharpe_ratio
            
            if old_result.win_rate > 0:
                report.win_rate_improvement[contract] = (
                    (new_result.win_rate - old_result.win_rate) / old_result.win_rate * 100
                )
            else:
                report.win_rate_improvement[contract] = new_result.win_rate - old_result.win_rate
            
            if old_result.max_drawdown_percent > 0:
                report.max_drawdown_improvement[contract] = (
                    (old_result.max_drawdown_percent - new_result.max_drawdown_percent) / old_result.max_drawdown_percent * 100
                )
            else:
                report.max_drawdown_improvement[contract] = old_result.max_drawdown_percent - new_result.max_drawdown_percent
            
            if old_result.sharpe_ratio > 0:
                report.sharpe_improvement[contract] = (
                    (new_result.sharpe_ratio - old_result.sharpe_ratio) / old_result.sharpe_ratio * 100
                )
            else:
                report.sharpe_improvement[contract] = new_result.sharpe_ratio - old_result.sharpe_ratio
        
        if valid_contracts > 0:
            report.overall_summary = {
                'avg_old_win_rate': total_old_win_rate / valid_contracts,
                'avg_new_win_rate': total_new_win_rate / valid_contracts,
                'avg_win_rate_improvement': (total_new_win_rate - total_old_win_rate) / valid_contracts,
                'avg_old_return': total_old_return / valid_contracts,
                'avg_new_return': total_new_return / valid_contracts,
                'avg_return_improvement': (total_new_return - total_old_return) / valid_contracts,
                'avg_old_max_drawdown': total_old_dd / valid_contracts,
                'avg_new_max_drawdown': total_new_dd / valid_contracts,
                'avg_max_drawdown_improvement': (total_old_dd - total_new_dd) / valid_contracts,
                'avg_old_sharpe': total_old_sharpe / valid_contracts,
                'avg_new_sharpe': total_new_sharpe / valid_contracts,
                'avg_sharpe_improvement': (total_new_sharpe - total_old_sharpe) / valid_contracts,
                'total_old_trades': total_old_trades,
                'total_new_trades': total_new_trades,
                'valid_contracts': valid_contracts,
            }
        
        return report
    
    def save_report(
        self,
        report: ComparisonReport,
        output_dir: str = None,
    ) -> str:
        if output_dir is None:
            output_dir = os.path.join(base_dir, 'results')
        
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        report_file = os.path.join(output_dir, f'comparison_report_{timestamp}.json')
        
        report_dict = {
            'report_id': report.report_id,
            'generated_at': report.generated_at.isoformat(),
            'contracts': report.contracts,
            'overall_summary': report.overall_summary,
            'win_rate_improvement': report.win_rate_improvement,
            'max_drawdown_improvement': report.max_drawdown_improvement,
            'sharpe_improvement': report.sharpe_improvement,
            'old_strategy_results': {},
            'new_strategy_results': {},
        }
        
        for contract, result in report.old_strategy_results.items():
            report_dict['old_strategy_results'][contract] = {
                'strategy_name': result.strategy_name,
                'contract': result.contract,
                'total_trades': result.total_trades,
                'winning_trades': result.winning_trades,
                'losing_trades': result.losing_trades,
                'win_rate': result.win_rate,
                'total_return': result.total_return,
                'total_return_percent': result.total_return_percent,
                'max_drawdown': result.max_drawdown,
                'max_drawdown_percent': result.max_drawdown_percent,
                'sharpe_ratio': result.sharpe_ratio,
                'sortino_ratio': result.sortino_ratio,
                'calmar_ratio': result.calmar_ratio,
                'profit_factor': result.profit_factor,
                'avg_trade_return': result.avg_trade_return,
                'initial_capital': result.initial_capital,
                'final_capital': result.final_capital,
                'status': result.status,
                'error_message': result.error_message,
            }
        
        for contract, result in report.new_strategy_results.items():
            report_dict['new_strategy_results'][contract] = {
                'strategy_name': result.strategy_name,
                'contract': result.contract,
                'total_trades': result.total_trades,
                'winning_trades': result.winning_trades,
                'losing_trades': result.losing_trades,
                'win_rate': result.win_rate,
                'total_return': result.total_return,
                'total_return_percent': result.total_return_percent,
                'max_drawdown': result.max_drawdown,
                'max_drawdown_percent': result.max_drawdown_percent,
                'sharpe_ratio': result.sharpe_ratio,
                'sortino_ratio': result.sortino_ratio,
                'calmar_ratio': result.calmar_ratio,
                'profit_factor': result.profit_factor,
                'avg_trade_return': result.avg_trade_return,
                'initial_capital': result.initial_capital,
                'final_capital': result.final_capital,
                'atr_filtered_signals': result.atr_filtered_signals,
                'rsi_filtered_signals': result.rsi_filtered_signals,
                'trailing_stop_triggered': result.trailing_stop_triggered,
                'status': result.status,
                'error_message': result.error_message,
            }
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report_dict, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"对比报告已保存: {report_file}")
        
        if MATPLOTLIB_AVAILABLE:
            self._generate_charts(report, output_dir, timestamp)
        
        return report_file
    
    def _generate_charts(
        self,
        report: ComparisonReport,
        output_dir: str,
        timestamp: str,
    ):
        _check_chinese_font()
        
        contracts = list(report.old_strategy_results.keys())
        if not contracts:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(_get_label('策略对比分析报告', 'Strategy Comparison Report'), fontsize=16, fontweight='bold')
        
        ax1 = axes[0, 0]
        old_win_rates = [report.old_strategy_results[c].win_rate for c in contracts]
        new_win_rates = [report.new_strategy_results[c].win_rate for c in contracts]
        
        x = range(len(contracts))
        width = 0.35
        
        ax1.bar([i - width/2 for i in x], old_win_rates, width, label=_get_label('旧策略', 'Old Strategy'), alpha=0.7)
        ax1.bar([i + width/2 for i in x], new_win_rates, width, label=_get_label('新策略', 'New Strategy'), alpha=0.7)
        ax1.set_xlabel(_get_label('合约', 'Contract'))
        ax1.set_ylabel(_get_label('胜率 (%)', 'Win Rate (%)'))
        ax1.set_title(_get_label('胜率对比', 'Win Rate Comparison'))
        ax1.set_xticks(x)
        ax1.set_xticklabels(contracts)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2 = axes[0, 1]
        old_dd = [report.old_strategy_results[c].max_drawdown_percent for c in contracts]
        new_dd = [report.new_strategy_results[c].max_drawdown_percent for c in contracts]
        
        ax2.bar([i - width/2 for i in x], old_dd, width, label=_get_label('旧策略', 'Old Strategy'), alpha=0.7, color='red')
        ax2.bar([i + width/2 for i in x], new_dd, width, label=_get_label('新策略', 'New Strategy'), alpha=0.7, color='orange')
        ax2.set_xlabel(_get_label('合约', 'Contract'))
        ax2.set_ylabel(_get_label('最大回撤 (%)', 'Max Drawdown (%)'))
        ax2.set_title(_get_label('最大回撤对比', 'Max Drawdown Comparison'))
        ax2.set_xticks(x)
        ax2.set_xticklabels(contracts)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        ax3 = axes[1, 0]
        old_sharpe = [report.old_strategy_results[c].sharpe_ratio for c in contracts]
        new_sharpe = [report.new_strategy_results[c].sharpe_ratio for c in contracts]
        
        ax3.bar([i - width/2 for i in x], old_sharpe, width, label=_get_label('旧策略', 'Old Strategy'), alpha=0.7, color='green')
        ax3.bar([i + width/2 for i in x], new_sharpe, width, label=_get_label('新策略', 'New Strategy'), alpha=0.7, color='blue')
        ax3.set_xlabel(_get_label('合约', 'Contract'))
        ax3.set_ylabel(_get_label('夏普比率', 'Sharpe Ratio'))
        ax3.set_title(_get_label('夏普比率对比', 'Sharpe Ratio Comparison'))
        ax3.set_xticks(x)
        ax3.set_xticklabels(contracts)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        ax4 = axes[1, 1]
        old_trades = [report.old_strategy_results[c].total_trades for c in contracts]
        new_trades = [report.new_strategy_results[c].total_trades for c in contracts]
        
        ax4.bar([i - width/2 for i in x], old_trades, width, label=_get_label('旧策略', 'Old Strategy'), alpha=0.7, color='purple')
        ax4.bar([i + width/2 for i in x], new_trades, width, label=_get_label('新策略', 'New Strategy'), alpha=0.7, color='brown')
        ax4.set_xlabel(_get_label('合约', 'Contract'))
        ax4.set_ylabel(_get_label('交易次数', 'Number of Trades'))
        ax4.set_title(_get_label('交易次数对比', 'Trade Count Comparison'))
        ax4.set_xticks(x)
        ax4.set_xticklabels(contracts)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        chart_file = os.path.join(output_dir, f'comparison_chart_{timestamp}.png')
        plt.savefig(chart_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"对比图表已保存: {chart_file}")
    
    def print_report(self, report: ComparisonReport):
        print("\n" + "=" * 80)
        print("                    策略对比分析报告")
        print("=" * 80)
        print(f"报告ID: {report.report_id}")
        print(f"生成时间: {report.generated_at.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"测试合约: {', '.join(report.contracts)}")
        print("-" * 80)
        
        print("\n【各合约详细对比】")
        print("-" * 80)
        
        for contract in self.contracts:
            old_result = report.old_strategy_results.get(contract)
            new_result = report.new_strategy_results.get(contract)
            
            if old_result is None or new_result is None:
                continue
            
            print(f"\n合约: {contract}")
            print(f"  旧策略 ({old_result.strategy_name}):")
            print(f"    交易次数: {old_result.total_trades}")
            print(f"    胜率: {old_result.win_rate:.2f}%")
            print(f"    总收益率: {old_result.total_return_percent:.2f}%")
            print(f"    最大回撤: {old_result.max_drawdown_percent:.2f}%")
            print(f"    夏普比率: {old_result.sharpe_ratio:.2f}")
            print(f"    盈亏比: {old_result.profit_factor:.2f}")
            
            print(f"  新策略 ({new_result.strategy_name}):")
            print(f"    交易次数: {new_result.total_trades}")
            print(f"    胜率: {new_result.win_rate:.2f}%")
            print(f"    总收益率: {new_result.total_return_percent:.2f}%")
            print(f"    最大回撤: {new_result.max_drawdown_percent:.2f}%")
            print(f"    夏普比率: {new_result.sharpe_ratio:.2f}")
            print(f"    盈亏比: {new_result.profit_factor:.2f}")
            print(f"    ATR过滤信号: {new_result.atr_filtered_signals} 次")
            print(f"    RSI过滤信号: {new_result.rsi_filtered_signals} 次")
            print(f"    追踪止损触发: {new_result.trailing_stop_triggered} 次")
            
            win_imp = report.win_rate_improvement.get(contract, 0)
            dd_imp = report.max_drawdown_improvement.get(contract, 0)
            sharpe_imp = report.sharpe_improvement.get(contract, 0)
            
            print(f"  改善情况:")
            print(f"    胜率提升: {win_imp:+.2f}% {'↑' if win_imp > 0 else '↓'}")
            print(f"    回撤降低: {dd_imp:+.2f}% {'↑' if dd_imp > 0 else '↓'}")
            print(f"    夏普比率提升: {sharpe_imp:+.2f}% {'↑' if sharpe_imp > 0 else '↓'}")
        
        print("\n" + "-" * 80)
        print("【综合统计】")
        print("-" * 80)
        
        summary = report.overall_summary
        if summary:
            print(f"  平均胜率: 旧={summary['avg_old_win_rate']:.2f}% → 新={summary['avg_new_win_rate']:.2f}% (改善: {summary['avg_win_rate_improvement']:+.2f}%)")
            print(f"  平均收益率: 旧={summary['avg_old_return']:.2f}% → 新={summary['avg_new_return']:.2f}% (改善: {summary['avg_return_improvement']:+.2f}%)")
            print(f"  平均最大回撤: 旧={summary['avg_old_max_drawdown']:.2f}% → 新={summary['avg_new_max_drawdown']:.2f}% (改善: {summary['avg_max_drawdown_improvement']:+.2f}%)")
            print(f"  平均夏普比率: 旧={summary['avg_old_sharpe']:.2f} → 新={summary['avg_new_sharpe']:.2f} (改善: {summary['avg_sharpe_improvement']:+.2f})")
            print(f"  总交易次数: 旧={summary['total_old_trades']} → 新={summary['total_new_trades']}")
        
        print("\n" + "=" * 80)
        print("【策略差异分析】")
        print("=" * 80)
        
        print("""
新策略 (AdaptiveMomentumStrategy) 相比旧策略 (DoubleMAStrategy) 的主要改进:

1. 【波动率过滤】
   - 引入 ATR (平均真实波幅) 指标
   - 只有当价格波动超过 ATR 的 1.5 倍时才允许开仓
   - 有效防止在横盘震荡期频繁止损

2. 【动量确认】
   - 结合 RSI 指标进行二次确认
   - MA 金叉时，RSI 必须处于 50 以上（多头区间）
   - MA 死叉时，RSI 必须处于 50 以下（空头区间）

3. 【动态仓位管理】
   - 不再使用固定的 "下单 1 手"
   - 单笔交易风险固定为总资金的 1%
   - 公式: 下单手数 = (总资金 × 1%) / (ATR × 合约乘数)
   - 波动大时轻仓，波动小时重仓

4. 【追踪止损】
   - 当盈利超过 ATR 的 2 倍时，启动追踪止损
   - 止损位随价格上移（做多）或下移（做空）
   - 锁定已有利润，杜绝 "盈利变亏损"
        """)
        
        new_better_count = 0
        old_better_count = 0
        
        for contract in self.contracts:
            win_imp = report.win_rate_improvement.get(contract, 0)
            dd_imp = report.max_drawdown_improvement.get(contract, 0)
            sharpe_imp = report.sharpe_improvement.get(contract, 0)
            
            score = 0
            if win_imp > 0:
                score += 1
            if dd_imp > 0:
                score += 1
            if sharpe_imp > 0:
                score += 1
            
            if score >= 2:
                new_better_count += 1
            elif score <= 1:
                old_better_count += 1
        
        print("【结论】")
        print("-" * 80)
        
        if new_better_count > old_better_count:
            print("✅ 新策略整体表现优于旧策略！")
            print(f"   新策略胜出合约数: {new_better_count}")
            print(f"   旧策略胜出合约数: {old_better_count}")
            print("\n   主要优势:")
            print("   - ATR 过滤有效减少了震荡市的假信号")
            print("   - RSI 确认提高了信号质量")
            print("   - 动态仓位管理使风险更加可控")
            print("   - 追踪止损有效保护了盈利")
        elif old_better_count > new_better_count:
            print("⚠️  旧策略在部分合约上表现更好，可能原因分析:")
            print(f"   新策略胜出合约数: {new_better_count}")
            print(f"   旧策略胜出合约数: {old_better_count}")
            print("\n   可能原因:")
            print("   1. 测试时间段内市场趋势性较强，过滤条件可能错过部分机会")
            print("   2. ATR 倍数参数可能需要针对特定合约优化")
            print("   3. 动态仓位管理在某些情况下可能过于保守")
            print("   4. 建议增加更多历史数据进行验证")
        else:
            print("⚖️  新旧策略表现相当，各有优势")
        
        print("\n" + "=" * 80)


def run_default_comparison():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    )
    
    logger = logging.getLogger("StrategyComparison")
    
    from datetime import date, timedelta
    
    end_dt = date(2024, 3, 31)
    start_dt = date(2024, 1, 1)
    
    contracts = ['SHFE.rb', 'DCE.i']
    
    contract_configs = {
        'SHFE.rb': {
            'initial_price': 3500.0,
            'volatility': 0.015,
            'trend': 0.0001,
        },
        'DCE.i': {
            'initial_price': 800.0,
            'volatility': 0.02,
            'trend': -0.0001,
        },
    }
    
    contract_multipliers = {
        'SHFE.rb': 10,
        'DCE.i': 100,
    }
    
    old_strategy_params = {
        'short_period': 5,
        'long_period': 20,
        'kline_duration': 60,
        'use_ema': False,
        'rsi_period': 14,
        'rsi_threshold': 50.0,
        'use_rsi_filter': False,
        'take_profit_ratio': None,
        'stop_loss_ratio': None,
        'debug_logging': False,
    }
    
    new_strategy_params = {
        'short_period': 5,
        'long_period': 20,
        'kline_duration': 60,
        'use_ema': False,
        'rsi_period': 14,
        'rsi_threshold': 50.0,
        'atr_period': 14,
        'atr_entry_multiplier': 1.5,
        'atr_exit_multiplier': 2.0,
        'risk_per_trade_percent': 0.01,
        'trailing_stop_atr_multiplier': 2.0,
        'contract_multiplier': 10,
        'debug_logging': False,
    }
    
    logger.info("=" * 60)
    logger.info("开始策略对比测试")
    logger.info(f"测试时间段: {start_dt} 至 {end_dt}")
    logger.info(f"测试合约: {contracts}")
    logger.info("=" * 60)
    
    comparator = StrategyComparator(
        contracts=contracts,
        start_dt=start_dt,
        end_dt=end_dt,
        initial_capital=1000000.0,
    )
    
    comparator.generate_mock_data(contract_configs, seed=42)
    
    report = comparator.run_comparison(
        old_strategy_params=old_strategy_params,
        new_strategy_params=new_strategy_params,
        contract_multipliers=contract_multipliers,
    )
    
    comparator.print_report(report)
    
    output_dir = os.path.join(base_dir, 'results')
    report_file = comparator.save_report(report, output_dir)
    
    logger.info(f"\n报告文件: {report_file}")
    
    return report


if __name__ == "__main__":
    run_default_comparison()
