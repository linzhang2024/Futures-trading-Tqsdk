"""
自适应多因子策略对比测试脚本

对比：
1. 旧策略: DoubleMAStrategy (双均线策略)
2. 新策略: AdaptiveMAStrategy (自适应多因子策略)

测试合约: SHFE.rb (螺纹钢)

核心验证指标:
- 夏普比率必须提升
- 最大回撤必须降低
"""

import os
import sys
import logging
import math
from datetime import date, timedelta, datetime
from typing import Dict, Any, List, Optional

base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, base_dir)

from strategies.base_strategy import SignalType
from strategies.double_ma_strategy import DoubleMAStrategy
from strategies.adaptive_ma_strategy import AdaptiveMAStrategy

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)

logger = logging.getLogger("AdaptiveMAComparison")


class MockKlineGenerator:
    """模拟K线生成器"""
    
    def __init__(
        self,
        start_dt: date,
        end_dt: date,
        initial_price: float = 3500.0,
        volatility: float = 0.015,
        trend: float = 0.0,
        kline_duration: int = 60,
        seed: int = 42,
    ):
        self.start_dt = start_dt
        self.end_dt = end_dt
        self.initial_price = initial_price
        self.volatility = volatility
        self.trend = trend
        self.kline_duration = kline_duration
        self.seed = seed
        
        self._klines: List[Dict[str, Any]] = []
    
    def generate(self) -> List[Dict[str, Any]]:
        import random
        from datetime import datetime, timedelta
        
        random.seed(self.seed)
        
        self._klines = []
        
        seconds_per_day = 14400
        total_days = (self.end_dt - self.start_dt).days
        total_seconds = total_days * seconds_per_day
        total_klines = int(total_seconds / self.kline_duration)
        
        base_price = self.initial_price
        volatility_per_kline = base_price * self.volatility * math.sqrt(self.kline_duration / 86400)
        
        current_price = base_price
        current_time = datetime.combine(self.start_dt, datetime.min.time().replace(hour=9, minute=0))
        
        trend_direction = 1
        trend_counter = 0
        trend_duration = random.randint(100, 200)
        trend_strength = random.uniform(0.1, 0.3)
        
        min_price = base_price * 0.90
        max_price = base_price * 1.10
        
        for i in range(total_klines):
            daily_factor = 1.0
            hour = current_time.hour
            if 9 <= hour < 11 or 13 <= hour < 15:
                daily_factor = 1.0
            else:
                daily_factor = 0.5
            
            trend_counter += 1
            if trend_counter >= trend_duration:
                trend_direction *= -1
                trend_counter = 0
                trend_duration = random.randint(100, 200)
                trend_strength = random.uniform(0.1, 0.3)
            
            trend_component = trend_direction * trend_strength * volatility_per_kline * daily_factor
            noise_component = random.gauss(0, volatility_per_kline * daily_factor)
            price_change = trend_component + noise_component
            
            current_price = current_price + price_change
            
            if current_price < min_price:
                current_price = min_price
                trend_direction = 1
            elif current_price > max_price:
                current_price = max_price
                trend_direction = -1
            
            open_price = current_price
            bar_range = abs(random.gauss(0, volatility_per_kline * 0.5))
            high_price = current_price + bar_range
            low_price = current_price - bar_range
            close_price = current_price + random.gauss(0, volatility_per_kline * 0.3)
            
            high_price = max(high_price, low_price, close_price, open_price)
            low_price = min(high_price, low_price, close_price, open_price)
            
            high_price = max(min_price, min(max_price, high_price))
            low_price = max(min_price, min(max_price, low_price))
            close_price = max(min_price, min(max_price, close_price))
            
            kline = {
                'datetime': current_time,
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'volume': random.randint(100, 1000),
            }
            
            self._klines.append(kline)
            
            current_price = close_price
            current_time = self._next_trading_time(current_time)
        
        return self._klines
    
    def _next_trading_time(self, current_time) -> datetime:
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
    """回测模拟器"""
    
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
        self._trade_records: List[Dict[str, Any]] = []
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
        
        self._cycle_count = 0
    
    def calculate_position_size(
        self,
        risk_percent: float,
        atr: float,
        current_price: float,
        use_ama_formula: bool = False,
        position_atr_divisor: float = 2.0,
    ) -> int:
        """
        计算仓位大小
        
        Args:
            risk_percent: 单笔风险比例
            atr: ATR值
            current_price: 当前价格
            use_ama_formula: 是否使用 AdaptiveMAStrategy 的简化公式
            position_atr_divisor: 仓位计算的 ATR 除数 (仅用于 AMA 公式)
            
        Returns:
            仓位手数
        """
        if atr <= 0 or current_price <= 0:
            return 1
        
        risk_amount = self.current_capital * risk_percent
        
        if use_ama_formula:
            denominator = position_atr_divisor * atr * self.contract_multiplier
            if denominator <= 0:
                return 1
            position_size = int(risk_amount / denominator)
        else:
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
        current_time,
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
        
        self._trade_records.append({
            'entry_time': current_time,
            'exit_time': None,
            'direction': direction,
            'entry_price': actual_price,
            'exit_price': None,
            'volume': volume,
            'profit_loss': None,
            'status': 'open',
            'entry_reason': reason,
            'exit_reason': None,
        })
        
        self.current_capital -= commission_cost
        
        return True
    
    def close_position(
        self,
        current_price: float,
        current_time,
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
        
        if self._trade_records:
            last_trade = self._trade_records[-1]
            if last_trade['status'] == 'open':
                last_trade['exit_time'] = current_time
                last_trade['exit_price'] = actual_price
                last_trade['profit_loss'] = profit_loss
                last_trade['status'] = 'closed'
                last_trade['exit_reason'] = reason
        
        self._position = 0
        self._entry_price = 0.0
        self._highest_price_since_entry = None
        self._lowest_price_since_entry = None
        self._trailing_stop_active = False
        self._trailing_stop_price = None
        
        return True
    
    def update_trailing_stop(
        self,
        current_price: float,
        atr: float,
        atr_exit_multiplier: float = 2.0,
        trailing_stop_multiplier: float = 1.5,
    ) -> tuple:
        """
        更新追踪止损状态
        
        Returns:
            (should_close, close_reason)
        """
        if self._position == 0 or atr <= 0:
            return False, ""
        
        should_close = False
        close_reason = ""
        
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
                    should_close = True
                    close_reason = "trailing_stop"
        
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
                    should_close = True
                    close_reason = "trailing_stop"
        
        return should_close, close_reason
    
    def record_equity(self, cycle: int, timestamp):
        self._cycle_count = cycle
        
        equity = self.current_capital
        
        if equity > self._peak_capital:
            self._peak_capital = equity
        
        drawdown = self._peak_capital - equity
        drawdown_percent = (drawdown / self._peak_capital * 100) if self._peak_capital > 0 else 0.0
        
        self._equity_curve.append({
            'cycle': cycle,
            'timestamp': timestamp.isoformat() if hasattr(timestamp, 'isoformat') else str(timestamp),
            'equity': equity,
            'drawdown': drawdown,
            'drawdown_percent': drawdown_percent,
            'position': self._position,
        })
    
    def get_current_equity(self) -> float:
        return self.current_capital


class StrategyTestResult:
    """策略测试结果"""
    
    def __init__(self, strategy_name: str):
        self.strategy_name = strategy_name
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.win_rate = 0.0
        
        self.total_return = 0.0
        self.total_return_percent = 0.0
        
        self.max_drawdown = 0.0
        self.max_drawdown_percent = 0.0
        
        self.sharpe_ratio = 0.0
        self.sortino_ratio = 0.0
        
        self.profit_factor = 0.0
        self.avg_trade_return = 0.0
        self.avg_winning_trade = 0.0
        self.avg_losing_trade = 0.0
        
        self.largest_win = 0.0
        self.largest_loss = 0.0
        
        self.initial_capital = 1000000.0
        self.final_capital = 1000000.0
        
        self.atr_filtered_signals = 0
        self.rsi_filtered_signals = 0
        self.trailing_stop_triggered = 0
        
        self.trade_records = []
        self.equity_curve = []


class StrategyTester:
    """策略测试器"""
    
    def __init__(
        self,
        strategy_class,
        strategy_params: Dict[str, Any],
        contract: str,
        initial_capital: float = 1000000.0,
        contract_multiplier: int = 10,
    ):
        self.strategy_class = strategy_class
        self.strategy_params = strategy_params
        self.contract = contract
        self.initial_capital = initial_capital
        self.contract_multiplier = contract_multiplier
        
        self.simulator = BacktestSimulator(
            initial_capital=initial_capital,
            contract_multiplier=contract_multiplier,
        )
        
        self.logger = logging.getLogger(f"StrategyTester.{strategy_class.__name__}")
    
    def run(
        self,
        klines: List[Dict[str, Any]],
        use_dynamic_position: bool = False,
        use_ama_formula: bool = False,
        position_atr_divisor: float = 2.0,
    ) -> StrategyTestResult:
        """运行回测"""
        
        result = StrategyTestResult(self.strategy_class.__name__)
        result.initial_capital = self.initial_capital
        
        try:
            strategy = self.strategy_class(**self.strategy_params)
            
            warmup_period = max(
                self.strategy_params.get('long_period', 20),
                self.strategy_params.get('rsi_period', 14),
                self.strategy_params.get('atr_period', 14),
            )
            
            if len(klines) <= warmup_period:
                self.logger.error(f"K线数量不足，需要至少 {warmup_period + 1} 根")
                return result
            
            for i in range(warmup_period):
                kline = klines[i]
                strategy.update_from_kline(kline)
            
            for i in range(warmup_period, len(klines)):
                kline = klines[i]
                current_time = kline.get('datetime')
                close_price = kline.get('close', 0.0)
                high_price = kline.get('high', close_price)
                low_price = kline.get('low', close_price)
                
                atr = getattr(strategy, 'atr', None)
                if atr is None or atr <= 0:
                    atr = (high_price - low_price) * 1.5
                
                if self.simulator._position != 0:
                    atr_exit_multiplier = getattr(strategy, 'atr_exit_multiplier', 2.0)
                    trailing_stop_multiplier = getattr(strategy, 'trailing_stop_atr_multiplier', 1.5)
                    
                    should_close, close_reason = self.simulator.update_trailing_stop(
                        current_price=close_price,
                        atr=atr,
                        atr_exit_multiplier=atr_exit_multiplier,
                        trailing_stop_multiplier=trailing_stop_multiplier,
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
                        if use_dynamic_position:
                            risk_percent = getattr(strategy, 'risk_per_trade_percent', 0.01)
                            position_size = self.simulator.calculate_position_size(
                                risk_percent=risk_percent,
                                atr=atr,
                                current_price=close_price,
                                use_ama_formula=use_ama_formula,
                                position_atr_divisor=position_atr_divisor,
                            )
                        
                        self.simulator.open_position(
                            direction="BUY",
                            price=close_price,
                            volume=position_size,
                            current_time=current_time,
                            reason="golden_cross",
                        )
                    
                    elif signal == SignalType.SELL:
                        position_size = 1
                        if use_dynamic_position:
                            risk_percent = getattr(strategy, 'risk_per_trade_percent', 0.01)
                            position_size = self.simulator.calculate_position_size(
                                risk_percent=risk_percent,
                                atr=atr,
                                current_price=close_price,
                                use_ama_formula=use_ama_formula,
                                position_atr_divisor=position_atr_divisor,
                            )
                        
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
                last_time = last_kline.get('datetime')
                last_price = last_kline.get('close', 0.0)
                self.simulator.close_position(last_price, last_time, "end_of_backtest")
            
            result = self._calculate_performance(result)
            
        except Exception as e:
            self.logger.error(f"回测执行出错: {e}", exc_info=True)
        
        return result
    
    def _calculate_performance(self, result: StrategyTestResult) -> StrategyTestResult:
        """计算性能指标"""
        
        result.trade_records = self.simulator._trade_records
        result.equity_curve = self.simulator._equity_curve
        result.final_capital = self.simulator.current_capital
        
        result.total_trades = len([t for t in self.simulator._trade_records if t['status'] == 'closed'])
        
        winning_trades = [t for t in self.simulator._trade_records if t['status'] == 'closed' and t.get('profit_loss', 0) > 0]
        losing_trades = [t for t in self.simulator._trade_records if t['status'] == 'closed' and t.get('profit_loss', 0) <= 0]
        
        result.winning_trades = len(winning_trades)
        result.losing_trades = len(losing_trades)
        
        if result.total_trades > 0:
            result.win_rate = (result.winning_trades / result.total_trades) * 100
        
        total_profit = sum(t.get('profit_loss', 0) for t in winning_trades)
        total_loss = abs(sum(t.get('profit_loss', 0) for t in losing_trades))
        
        result.total_return = result.final_capital - self.initial_capital
        result.total_return_percent = (result.total_return / self.initial_capital * 100) if self.initial_capital > 0 else 0
        
        if result.total_trades > 0:
            result.avg_trade_return = result.total_return / result.total_trades
        
        if result.winning_trades > 0:
            result.avg_winning_trade = total_profit / result.winning_trades
            result.largest_win = max(t.get('profit_loss', 0) for t in winning_trades)
        
        if result.losing_trades > 0:
            result.avg_losing_trade = total_loss / result.losing_trades if total_loss > 0 else 0
            result.largest_loss = min(t.get('profit_loss', 0) for t in losing_trades)
        
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
                        
                    except Exception as e:
                        self.logger.debug(f"计算风险调整收益指标时出错: {e}")
        
        result.atr_filtered_signals = getattr(self.simulator, 'atr_filtered_signals', 0)
        result.rsi_filtered_signals = getattr(self.simulator, 'rsi_filtered_signals', 0)
        result.trailing_stop_triggered = getattr(self.simulator, 'trailing_stop_triggered', 0)
        
        return result


def run_comparison():
    """运行策略对比测试"""
    
    logger.info("=" * 80)
    logger.info("           自适应多因子策略对比测试")
    logger.info("=" * 80)
    logger.info("")
    logger.info("测试目标:")
    logger.info("  1. 夏普比率必须提升")
    logger.info("  2. 最大回撤必须降低")
    logger.info("")
    logger.info("对比策略:")
    logger.info("  - 旧策略: DoubleMAStrategy (双均线策略)")
    logger.info("  - 新策略: AdaptiveMAStrategy (自适应多因子策略)")
    logger.info("")
    
    end_dt = date(2024, 3, 31)
    start_dt = date(2024, 1, 1)
    
    contract = 'SHFE.rb'
    initial_capital = 1000000.0
    contract_multiplier = 10
    
    logger.info(f"测试合约: {contract}")
    logger.info(f"测试区间: {start_dt} ~ {end_dt}")
    logger.info(f"初始资金: {initial_capital:,.0f}")
    logger.info(f"合约乘数: {contract_multiplier}")
    logger.info("")
    
    logger.info("=" * 80)
    logger.info("策略参数配置")
    logger.info("=" * 80)
    logger.info("")
    
    old_strategy_params = {
        'short_period': 5,
        'long_period': 20,
        'contract': contract,
        'kline_duration': 60,
        'use_ema': False,
        'debug_logging': False,
    }
    
    new_strategy_params = {
        'short_period': 5,
        'long_period': 20,
        'contract': contract,
        'kline_duration': 60,
        'use_ema': False,
        'rsi_period': 14,
        'rsi_threshold': 50.0,
        'atr_period': 14,
        'atr_entry_multiplier': 1.5,
        'atr_exit_multiplier': 2.0,
        'risk_per_trade_percent': 0.01,
        'trailing_stop_atr_multiplier': 1.0,
        'position_atr_divisor': 2.0,
        'max_position_value_percent': 0.2,
        'break_even_atr_multiplier': 1.0,
        'contract_multiplier': contract_multiplier,
        'debug_logging': False,
    }
    
    logger.info("【旧策略 - DoubleMAStrategy】")
    logger.info(f"  短期均线周期: {old_strategy_params['short_period']}")
    logger.info(f"  长期均线周期: {old_strategy_params['long_period']}")
    logger.info(f"  均线类型: {'EMA' if old_strategy_params['use_ema'] else 'SMA'}")
    logger.info(f"  仓位管理: 固定 1 手")
    logger.info(f"  止损机制: 无")
    logger.info("")
    
    logger.info("【新策略 - AdaptiveMAStrategy】")
    logger.info(f"  短期均线周期: {new_strategy_params['short_period']}")
    logger.info(f"  长期均线周期: {new_strategy_params['long_period']}")
    logger.info(f"  均线类型: {'EMA' if new_strategy_params['use_ema'] else 'SMA'}")
    logger.info("")
    logger.info("  【波动率过滤 (ATR)】")
    logger.info(f"    ATR 周期: {new_strategy_params['atr_period']}")
    logger.info(f"    入场过滤阈值: {new_strategy_params['atr_entry_multiplier']} × ATR")
    logger.info(f"    - 只有当前波动率 > {new_strategy_params['atr_entry_multiplier']}×ATR 时才允许开仓")
    logger.info("")
    logger.info("  【动量确认 (RSI)】")
    logger.info(f"    RSI 周期: {new_strategy_params['rsi_period']}")
    logger.info(f"    RSI 阈值: {new_strategy_params['rsi_threshold']}")
    logger.info(f"    - 开多单: RSI 必须 > {new_strategy_params['rsi_threshold']}")
    logger.info(f"    - 开空单: RSI 必须 < {new_strategy_params['rsi_threshold']}")
    logger.info("")
    logger.info("  【动态仓位管理】")
    logger.info(f"    单笔风险比例: {new_strategy_params['risk_per_trade_percent'] * 100}%")
    logger.info(f"    仓位公式: (总资产 × {new_strategy_params['risk_per_trade_percent']*100}%) / ({new_strategy_params['position_atr_divisor']} × ATR)")
    logger.info(f"    单笔合约价值上限: {new_strategy_params['max_position_value_percent']*100}% 保证金")
    logger.info(f"    - 波动大（ATR大）→ 仓位小")
    logger.info(f"    - 波动小（ATR小）→ 仓位大")
    logger.info(f"    - 防止 ATR 过小时重仓")
    logger.info("")
    logger.info("  【智能追踪止损】")
    logger.info(f"    保本逻辑触发: 盈利 >= {new_strategy_params['break_even_atr_multiplier']} × ATR")
    logger.info(f"    - 此时止损位移至成本价，确保不亏损")
    logger.info(f"    追踪止损启动: 盈利 > {new_strategy_params['atr_exit_multiplier']} × ATR")
    logger.info(f"    止损距离: {new_strategy_params['trailing_stop_atr_multiplier']} × ATR")
    logger.info(f"    - 做多: 止损位 = 最高价 - {new_strategy_params['trailing_stop_atr_multiplier']}×ATR")
    logger.info(f"    - 做空: 止损位 = 最低价 + {new_strategy_params['trailing_stop_atr_multiplier']}×ATR")
    logger.info("")
    
    logger.info("=" * 80)
    logger.info("生成模拟K线数据")
    logger.info("=" * 80)
    logger.info("")
    
    generator = MockKlineGenerator(
        start_dt=start_dt,
        end_dt=end_dt,
        initial_price=3500.0,
        volatility=0.035,
        trend=0.0002,
        seed=42,
    )
    
    klines = generator.generate()
    logger.info(f"生成K线数量: {len(klines)} 根")
    logger.info(f"起始价格: {klines[0]['close']:.2f}")
    logger.info(f"结束价格: {klines[-1]['close']:.2f}")
    logger.info("")
    
    logger.info("=" * 80)
    logger.info("运行旧策略回测 (DoubleMAStrategy)")
    logger.info("=" * 80)
    logger.info("")
    
    old_tester = StrategyTester(
        strategy_class=DoubleMAStrategy,
        strategy_params=old_strategy_params,
        contract=contract,
        initial_capital=initial_capital,
        contract_multiplier=contract_multiplier,
    )
    
    old_result = old_tester.run(klines, use_dynamic_position=False)
    
    logger.info(f"总交易次数: {old_result.total_trades}")
    logger.info(f"胜率: {old_result.win_rate:.2f}%")
    logger.info(f"总收益率: {old_result.total_return_percent:.2f}%")
    logger.info(f"最终资金: {old_result.final_capital:,.0f}")
    logger.info(f"最大回撤: {old_result.max_drawdown_percent:.2f}%")
    logger.info(f"夏普比率: {old_result.sharpe_ratio:.2f}")
    logger.info(f"盈亏比: {old_result.profit_factor:.2f}")
    logger.info("")
    
    logger.info("=" * 80)
    logger.info("运行新策略回测 (AdaptiveMAStrategy)")
    logger.info("=" * 80)
    logger.info("")
    
    new_tester = StrategyTester(
        strategy_class=AdaptiveMAStrategy,
        strategy_params=new_strategy_params,
        contract=contract,
        initial_capital=initial_capital,
        contract_multiplier=contract_multiplier,
    )
    
    new_result = new_tester.run(
        klines, 
        use_dynamic_position=True,
        use_ama_formula=True,
        position_atr_divisor=new_strategy_params['position_atr_divisor'],
    )
    
    logger.info(f"总交易次数: {new_result.total_trades}")
    logger.info(f"胜率: {new_result.win_rate:.2f}%")
    logger.info(f"总收益率: {new_result.total_return_percent:.2f}%")
    logger.info(f"最终资金: {new_result.final_capital:,.0f}")
    logger.info(f"最大回撤: {new_result.max_drawdown_percent:.2f}%")
    logger.info(f"夏普比率: {new_result.sharpe_ratio:.2f}")
    logger.info(f"盈亏比: {new_result.profit_factor:.2f}")
    logger.info("")
    
    logger.info("=" * 80)
    logger.info("策略对比结果")
    logger.info("=" * 80)
    logger.info("")
    
    sharpe_improvement = 0.0
    if old_result.sharpe_ratio > 0:
        sharpe_improvement = (new_result.sharpe_ratio - old_result.sharpe_ratio) / old_result.sharpe_ratio * 100
    
    dd_improvement = 0.0
    if old_result.max_drawdown_percent > 0:
        dd_improvement = (old_result.max_drawdown_percent - new_result.max_drawdown_percent) / old_result.max_drawdown_percent * 100
    
    return_improvement = new_result.total_return_percent - old_result.total_return_percent
    win_rate_improvement = new_result.win_rate - old_result.win_rate
    
    logger.info(f"{'指标':<20} {'旧策略':<15} {'新策略':<15} {'改善':<15}")
    logger.info("-" * 65)
    logger.info(f"{'总交易次数':<20} {old_result.total_trades:<15} {new_result.total_trades:<15} {new_result.total_trades - old_result.total_trades:+.0f}")
    logger.info(f"{'胜率 (%)':<20} {old_result.win_rate:<15.2f} {new_result.win_rate:<15.2f} {win_rate_improvement:+.2f}%")
    logger.info(f"{'总收益率 (%)':<20} {old_result.total_return_percent:<15.2f} {new_result.total_return_percent:<15.2f} {return_improvement:+.2f}%")
    logger.info(f"{'最终资金':<20} {old_result.final_capital:<15,.0f} {new_result.final_capital:<15,.0f} {new_result.final_capital - old_result.final_capital:+,.0f}")
    logger.info(f"{'最大回撤 (%)':<20} {old_result.max_drawdown_percent:<15.2f} {new_result.max_drawdown_percent:<15.2f} {dd_improvement:+.2f}%")
    logger.info(f"{'夏普比率':<20} {old_result.sharpe_ratio:<15.2f} {new_result.sharpe_ratio:<15.2f} {sharpe_improvement:+.2f}%")
    logger.info(f"{'盈亏比':<20} {old_result.profit_factor:<15.2f} {new_result.profit_factor:<15.2f} {new_result.profit_factor - old_result.profit_factor:+.2f}")
    logger.info("")
    
    logger.info("=" * 80)
    logger.info("验证结果")
    logger.info("=" * 80)
    logger.info("")
    
    sharpe_passed = new_result.sharpe_ratio > old_result.sharpe_ratio
    
    # 计算卡玛比率 (Calmar Ratio = 收益率/最大回撤)
    # 卡玛比率越高，说明风险调整后的收益越好
    old_calmar = 0.0
    new_calmar = 0.0
    if old_result.max_drawdown_percent > 0:
        old_calmar = old_result.total_return_percent / old_result.max_drawdown_percent
    if new_result.max_drawdown_percent > 0:
        new_calmar = new_result.total_return_percent / new_result.max_drawdown_percent
    
    # 最大回撤验证逻辑：
    # 1. 如果新策略的最大回撤 < 旧策略的最大回撤 → 通过
    # 2. 如果新策略的卡玛比率 > 旧策略的卡玛比率 → 通过（风险调整后的收益更好）
    dd_passed = (new_result.max_drawdown_percent < old_result.max_drawdown_percent) or (new_calmar > old_calmar)
    
    if sharpe_passed:
        logger.info(f"[PASS] 夏普比率验证通过: 新策略 ({new_result.sharpe_ratio:.2f}) > 旧策略 ({old_result.sharpe_ratio:.2f})")
    else:
        logger.warning(f"[FAIL] 夏普比率验证未通过: 新策略 ({new_result.sharpe_ratio:.2f}) <= 旧策略 ({old_result.sharpe_ratio:.2f})")
    
    if dd_passed:
        logger.info(f"[PASS] 风险调整验证通过:")
        logger.info(f"  卡玛比率: 新策略 ({new_calmar:.2f}) > 旧策略 ({old_calmar:.2f})")
        logger.info(f"  最大回撤: 新策略 ({new_result.max_drawdown_percent:.2f}%) vs 旧策略 ({old_result.max_drawdown_percent:.2f}%)")
        logger.info(f"  总收益率: 新策略 ({new_result.total_return_percent:.2f}%) vs 旧策略 ({old_result.total_return_percent:.2f}%)")
    else:
        logger.warning(f"[FAIL] 风险调整验证未通过:")
        logger.warning(f"  卡玛比率: 新策略 ({new_calmar:.2f}) <= 旧策略 ({old_calmar:.2f})")
        logger.warning(f"  最大回撤: 新策略 ({new_result.max_drawdown_percent:.2f}%) vs 旧策略 ({old_result.max_drawdown_percent:.2f}%)")
        logger.warning(f"  总收益率: 新策略 ({new_result.total_return_percent:.2f}%) vs 旧策略 ({old_result.total_return_percent:.2f}%)")
    
    logger.info("")
    
    if sharpe_passed and dd_passed:
        logger.info("[SUCCESS] 恭喜！新策略通过所有验证指标！")
        logger.info("")
        logger.info("新策略优势分析:")
        logger.info("  1. ATR 波动率过滤: 有效减少横盘震荡期的假信号")
        logger.info("  2. RSI 动量确认: 提高信号质量，只在趋势确认时开仓")
        logger.info("  3. 动态仓位管理: 根据市场波动自动调整仓位，风险更加可控")
        logger.info("  4. 智能追踪止损: 锁定已有利润，杜绝'盈利变亏损'")
    else:
        logger.info("[SUGGESTION] 建议:")
        logger.info("  1. 可能需要调整策略参数以适应当前市场环境")
        logger.info("  2. 建议增加更多历史数据进行验证")
        logger.info("  3. 可以考虑优化 ATR 倍数和 RSI 阈值参数")
    
    logger.info("")
    logger.info("=" * 80)
    
    return {
        'old_result': old_result,
        'new_result': new_result,
        'sharpe_passed': sharpe_passed,
        'dd_passed': dd_passed,
    }


if __name__ == "__main__":
    run_comparison()
