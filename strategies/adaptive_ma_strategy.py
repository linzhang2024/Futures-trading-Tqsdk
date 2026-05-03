import os
import sys
import logging
from typing import Optional, List, Dict, Any, Tuple
from collections import deque
from datetime import datetime

from strategies.base_strategy import StrategyBase, SignalType
from strategies.adaptive_momentum_strategy import AdaptiveMomentumStrategy

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEBUG_SIGNAL_LOGGER = None
DEBUG_SIGNAL_FILE = None

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    logging.getLogger(__name__).warning("pandas 库未安装，将使用手动计算技术指标")

try:
    import talib
    import numpy as np
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False


def _init_debug_logger():
    global DEBUG_SIGNAL_LOGGER, DEBUG_SIGNAL_FILE
    
    if DEBUG_SIGNAL_LOGGER is not None:
        return DEBUG_SIGNAL_LOGGER
    
    logs_dir = os.path.join(base_dir, 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    DEBUG_SIGNAL_FILE = os.path.join(logs_dir, f'debug_signal_{timestamp}.log')
    
    logger = logging.getLogger('debug_signal_ama')
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    
    file_handler = logging.FileHandler(DEBUG_SIGNAL_FILE, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
    file_handler.setFormatter(file_formatter)
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(asctime)s [DEBUG_SIGNAL_AMA] %(message)s')
    console_handler.setFormatter(console_formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    logger.info("=" * 80)
    logger.info("信号调试日志已初始化 - 自适应多因子策略 (AdaptiveMAStrategy)")
    logger.info(f"日志文件: {DEBUG_SIGNAL_FILE}")
    logger.info("=" * 80)
    
    DEBUG_SIGNAL_LOGGER = logger
    return logger


def get_debug_logger():
    return _init_debug_logger()


def log_signal_debug(contract: str, cycle: int, data: Dict[str, Any]):
    logger = get_debug_logger()
    msg_parts = [f"[{contract}] cycle={cycle}"]
    
    if 'close_price' in data:
        msg_parts.append(f"close={data['close_price']:.2f}")
    
    if 'atr' in data and data['atr'] is not None:
        msg_parts.append(f"ATR={data['atr']:.2f}")
    
    if 'volatility_ratio' in data:
        msg_parts.append(f"vol_ratio={data['volatility_ratio']:.2f}")
    
    if 'short_ma' in data and data['short_ma'] is not None:
        msg_parts.append(f"MA{data.get('short_period', 5)}={data['short_ma']:.2f}")
    if 'long_ma' in data and data['long_ma'] is not None:
        msg_parts.append(f"MA{data.get('long_period', 20)}={data['long_ma']:.2f}")
    
    if 'rsi' in data and data['rsi'] is not None:
        msg_parts.append(f"RSI={data['rsi']:.2f}")
    
    if 'position_size' in data and data['position_size'] is not None:
        msg_parts.append(f"size={data['position_size']:.1f}手")
    
    if 'trailing_stop' in data and data['trailing_stop'] is not None:
        msg_parts.append(f"trail_stop={data['trailing_stop']:.2f}")
    
    if 'signal' in data:
        msg_parts.append(f"signal={data['signal']}")
    
    if 'position' in data:
        msg_parts.append(f"position={data['position']}")
    
    msg = " | ".join(msg_parts)
    logger.info(msg)


class AdaptiveMAStrategy(AdaptiveMomentumStrategy):
    """
    自适应多因子策略 (Adaptive Multi-Factor Strategy)
    
    核心逻辑：
    1. ATR 波动率过滤：只有当前价格波动 > 1.2倍 ATR 时才允许触发金叉/死叉信号
    2. 动量确认：开多单时 RSI 必须 > 50；开空单时 RSI 必须 < 50
    3. 动态仓位：下单手数 = (总资产 * 1%) / (2 * ATR)
       - 行情波动大（ATR大）就少买点
       - 行情波动小（ATR小）就多买点
    4. 智能追踪止损：
       - 当盈利超过 2倍 ATR 时，启动追踪止损
       - 止损位设在：最高价 - 1.5 * ATR（做多）或 最低价 + 1.5 * ATR（做空）
    """
    
    def __init__(
        self,
        connector: Any = None,
        short_period: int = 5,
        long_period: int = 20,
        contract: str = "SHFE.rb2410",
        kline_duration: int = 60,
        use_ema: bool = False,
        rsi_period: int = 14,
        rsi_threshold: float = 50.0,
        atr_period: int = 14,
        atr_entry_multiplier: float = 1.5,
        atr_exit_multiplier: float = 2.0,
        risk_per_trade_percent: float = 0.01,
        trailing_stop_atr_multiplier: float = 1.0,
        position_atr_divisor: float = 2.0,
        max_position_value_percent: float = 0.2,
        break_even_atr_multiplier: float = 1.0,
        initial_data_days: int = 5,
        force_trade_test: bool = False,
        debug_logging: bool = True,
        slippage_ticks: float = 1.0,
        contract_multiplier: int = 10,
        **kwargs,
    ):
        super().__init__(
            connector=connector,
            short_period=short_period,
            long_period=long_period,
            contract=contract,
            kline_duration=kline_duration,
            use_ema=use_ema,
            rsi_period=rsi_period,
            rsi_threshold=rsi_threshold,
            atr_period=atr_period,
            atr_entry_multiplier=atr_entry_multiplier,
            atr_exit_multiplier=atr_exit_multiplier,
            risk_per_trade_percent=risk_per_trade_percent,
            atr_slippage_multiplier=1.0,
            trailing_stop_atr_multiplier=trailing_stop_atr_multiplier,
            initial_data_days=initial_data_days,
            force_trade_test=force_trade_test,
            debug_logging=debug_logging,
            slippage_ticks=slippage_ticks,
            contract_multiplier=contract_multiplier,
            **kwargs,
        )
        
        self.position_atr_divisor = position_atr_divisor
        self.max_position_value_percent = max_position_value_percent
        self.break_even_atr_multiplier = break_even_atr_multiplier
        self._is_break_even_active = False
        
        if position_atr_divisor <= 0:
            raise ValueError("仓位计算 ATR 除数必须大于 0")
        
        if max_position_value_percent <= 0 or max_position_value_percent > 1:
            raise ValueError("单笔合约价值最大比例必须在 0-1 之间")
        
        self.logger.info(
            f"自适应多因子策略初始化: "
            f"ATR入场过滤倍数={atr_entry_multiplier}, "
            f"追踪止损启动倍数={atr_exit_multiplier}, "
            f"追踪止损距离={trailing_stop_atr_multiplier}×ATR, "
            f"仓位公式: (总资产×{risk_per_trade_percent*100}%) / ({position_atr_divisor}×ATR), "
            f"单笔合约价值上限={max_position_value_percent*100}%保证金, "
            f"保本逻辑触发={break_even_atr_multiplier}×ATR"
        )
    
    def _calculate_position_size(self, atr: float, current_price: float) -> float:
        """
        动态仓位计算（带仓位上限限制）
        
        用户需求公式（简化版）：
        下单手数 = (总资产 * 1%) / (2 * ATR)
        
        实际实现（考虑合约乘数）：
        下单手数 = (总资产 * 1%) / (2 * ATR * 合约乘数)
        
        新增限制：
        - 单笔合约价值严禁超过保证金可用额度的 20%
        - 合约价值 = 手数 * 当前价格 * 合约乘数
        
        逻辑说明：
        - 行情波动大（ATR大）→ 分母大 → 仓位小
        - 行情波动小（ATR小）→ 分母小 → 仓位大
        
        注意：
        - 必须考虑合约乘数，因为盈亏计算会乘以合约乘数
        - 如果不考虑合约乘数，实际风险会被放大 合约乘数 倍
        
        Args:
            atr: 当前 ATR 值
            current_price: 当前价格（用于计算合约价值上限）
            
        Returns:
            计算出的仓位手数（至少1手，不超过上限）
        """
        if atr is None or atr <= 0:
            self.logger.warning("[动态仓位] ATR 无效，使用默认仓位 1 手")
            return 1.0
        
        if current_price is None or current_price <= 0:
            self.logger.warning("[动态仓位] 当前价格无效，使用默认仓位 1 手")
            return 1.0
        
        risk_amount = self._total_capital * self.risk_per_trade_percent
        
        denominator = self.position_atr_divisor * atr * self.contract_multiplier
        
        if denominator <= 0:
            self.logger.warning("[动态仓位] 分母无效，使用默认仓位 1 手")
            return 1.0
        
        position_size = risk_amount / denominator
        
        max_contract_value = self._total_capital * self.max_position_value_percent
        max_position_by_value = max_contract_value / (current_price * self.contract_multiplier)
        
        original_position_size = position_size
        position_size = min(position_size, max_position_by_value)
        
        position_size = max(1.0, position_size)
        
        if position_size < original_position_size:
            self.logger.info(
                f"[动态仓位] 仓位上限触发: "
                f"原计算仓位={original_position_size:.2f}手, "
                f"合约价值上限={self.max_position_value_percent*100}%保证金, "
                f"最大允许仓位={max_position_by_value:.2f}手, "
                f"调整后仓位={position_size:.2f}手"
            )
        
        self.logger.debug(
            f"[动态仓位] 计算: 总资产={self._total_capital:,.0f}, "
            f"风险比例={self.risk_per_trade_percent*100}%, "
            f"风险金额={risk_amount:,.0f}, "
            f"ATR={atr:.2f}, "
            f"合约乘数={self.contract_multiplier}, "
            f"分母={self.position_atr_divisor}×{atr:.2f}×{self.contract_multiplier}={denominator:.2f}, "
            f"计算仓位={position_size:.2f}手"
        )
        
        return position_size
    
    def _detect_signal(self, close_price: float = 0):
        """
        信号检测逻辑（重写父类方法）
        
        核心逻辑：
        1. 检测 MA 金叉/死叉
        2. ATR 波动率过滤：只有当前价格波动 > 1.2倍 ATR 时才允许触发
        3. 动量确认：
           - 开多单时 RSI 必须 > 50
           - 开空单时 RSI 必须 < 50
        """
        self.prev_signal = self.signal
        
        if self.short_ma is None or self.long_ma is None:
            self.signal = SignalType.HOLD
            if self.debug_logging:
                log_signal_debug(self.contract, self._cycle_count, {
                    'signal': 'HOLD_MA_NOT_READY',
                    'short_ma': self.short_ma,
                    'long_ma': self.long_ma,
                })
            return
        
        if self.prev_short_ma is None or self.prev_long_ma is None:
            self.signal = SignalType.HOLD
            rsi_status = f", RSI={self.rsi:.2f}" if self.rsi is not None else ""
            self.logger.debug(f"均线数据准备中... 短期均线={self.short_ma:.2f}, 长期均线={self.long_ma:.2f}{rsi_status}")
            if self.debug_logging:
                log_signal_debug(self.contract, self._cycle_count, {
                    'signal': 'HOLD_PREV_MA_NOT_READY',
                    'short_ma': self.short_ma,
                    'long_ma': self.long_ma,
                    'prev_short_ma': self.prev_short_ma,
                    'prev_long_ma': self.prev_long_ma,
                })
            return
        
        short_above_prev = self.prev_short_ma > self.prev_long_ma
        short_above_current = self.short_ma > self.long_ma
        
        short_below_prev = self.prev_short_ma < self.prev_long_ma
        short_below_current = self.short_ma < self.long_ma
        
        ma_signal: SignalType = SignalType.HOLD
        signal_reason = "HOLD_NO_CROSS"
        
        if short_below_prev and short_above_current:
            ma_signal = SignalType.BUY
            signal_reason = "GOLDEN_CROSS"
            self.logger.info(f"[SIGNAL] cycle={self._cycle_count} 金叉信号: 短期均线({self.short_ma:.2f}) 上穿 长期均线({self.long_ma:.2f})")
        elif short_above_prev and short_below_current:
            ma_signal = SignalType.SELL
            signal_reason = "DEATH_CROSS"
            self.logger.info(f"[SIGNAL] cycle={self._cycle_count} 死叉信号: 短期均线({self.short_ma:.2f}) 下穿 长期均线({self.long_ma:.2f})")
        else:
            ma_signal = SignalType.HOLD
            signal_reason = "HOLD_NO_CROSS"
            
            if self.debug_logging:
                if short_above_prev and short_above_current:
                    signal_reason = "HOLD_SHORT_ABOVE_LONG"
                elif short_below_prev and short_below_current:
                    signal_reason = "HOLD_SHORT_BELOW_LONG"
        
        if self.debug_logging:
            log_signal_debug(self.contract, self._cycle_count, {
                'signal': signal_reason,
                'ma_signal': ma_signal.value,
                'short_ma': self.short_ma,
                'long_ma': self.long_ma,
                'prev_short_ma': self.prev_short_ma,
                'prev_long_ma': self.prev_long_ma,
                'prev_above': short_above_prev,
                'curr_above': short_above_current,
                'prev_below': short_below_prev,
                'curr_below': short_below_current,
            })
        
        if ma_signal == SignalType.HOLD:
            self.signal = SignalType.HOLD
            return
        
        if self.atr is None or self.atr <= 0:
            self.signal = SignalType.HOLD
            self.logger.debug(f"[SIGNAL] cycle={self._cycle_count} ATR 数据未准备好或为0，跳过信号: {ma_signal.value}")
            if self.debug_logging:
                log_signal_debug(self.contract, self._cycle_count, {
                    'signal': 'HOLD_ATR_NOT_READY',
                    'ma_signal': ma_signal.value,
                    'atr': self.atr,
                })
            return
        
        atr_filter_threshold = self.atr * self.atr_entry_multiplier
        
        current_idx = len(self._high_list) - 1
        if current_idx >= 0 and len(self._high_list) == len(self._low_list):
            current_high = self._high_list[current_idx]
            current_low = self._low_list[current_idx]
            price_change = current_high - current_low
            volatility_source = "high_low"
        else:
            price_change = abs(close_price - self._price_list[-2]) if len(self._price_list) >= 2 else 0
            volatility_source = "close_change"
        
        volatility_ratio = price_change / self.atr if self.atr > 0 else 0
        
        if self.debug_logging:
            log_signal_debug(self.contract, self._cycle_count, {
                'close_price': close_price,
                'atr': self.atr,
                'volatility_ratio': volatility_ratio,
                'atr_filter_threshold': atr_filter_threshold,
                'price_change': price_change,
                'volatility_source': volatility_source,
            })
        
        if volatility_ratio < self.atr_entry_multiplier:
            self._atr_filtered_signals += 1
            self.signal = SignalType.HOLD
            self.logger.info(
                f"[ATR_FILTER] cycle={self._cycle_count} 原信号={ma_signal.value}, "
                f"波动率={volatility_ratio:.2f}×ATR < 阈值={self.atr_entry_multiplier}×ATR, "
                f"波动值={price_change:.2f}, ATR={self.atr:.2f}, 来源={volatility_source}，信号被过滤（横盘震荡）"
            )
            if self.debug_logging:
                log_signal_debug(self.contract, self._cycle_count, {
                    'signal': 'ATR_FILTERED',
                    'ma_signal': ma_signal.value,
                    'atr': self.atr,
                    'volatility_ratio': volatility_ratio,
                    'atr_entry_multiplier': self.atr_entry_multiplier,
                    'price_change': price_change,
                    'atr_filter_threshold': atr_filter_threshold,
                })
            return
        
        self.logger.info(
            f"[ATR_FILTER] cycle={self._cycle_count} 波动率={volatility_ratio:.2f}×ATR >= {self.atr_entry_multiplier}×ATR, "
            f"波动值={price_change:.2f}, ATR={self.atr:.2f}, 来源={volatility_source}，通过过滤"
        )
        
        if self.rsi is None:
            self.signal = SignalType.HOLD
            self.logger.debug(f"[SIGNAL] cycle={self._cycle_count} RSI 数据未准备好，跳过信号: {ma_signal.value}")
            if self.debug_logging:
                log_signal_debug(self.contract, self._cycle_count, {
                    'signal': 'HOLD_RSI_NOT_READY',
                    'ma_signal': ma_signal.value,
                })
            return
        
        rsi_status = ""
        final_signal = SignalType.HOLD
        
        if ma_signal == SignalType.BUY:
            if self.rsi > self.rsi_threshold:
                final_signal = SignalType.BUY
                rsi_status = f"RSI={self.rsi:.2f} > 阈值={self.rsi_threshold:.2f}，信号确认（多头区间）"
            else:
                self._rsi_filtered_signals += 1
                rsi_status = f"RSI={self.rsi:.2f} <= 阈值={self.rsi_threshold:.2f}，信号被过滤（非多头区间）"
        
        elif ma_signal == SignalType.SELL:
            if self.rsi < self.rsi_threshold:
                final_signal = SignalType.SELL
                rsi_status = f"RSI={self.rsi:.2f} < 阈值={self.rsi_threshold:.2f}，信号确认（空头区间）"
            else:
                self._rsi_filtered_signals += 1
                rsi_status = f"RSI={self.rsi:.2f} >= 阈值={self.rsi_threshold:.2f}，信号被过滤（非空头区间）"
        
        self.signal = final_signal
        
        if final_signal == SignalType.HOLD and ma_signal != SignalType.HOLD:
            self.logger.info(f"[RSI_FILTER] cycle={self._cycle_count} 原信号={ma_signal.value}, {rsi_status}")
            if self.debug_logging:
                log_signal_debug(self.contract, self._cycle_count, {
                    'signal': 'RSI_FILTERED',
                    'ma_signal': ma_signal.value,
                    'rsi': self.rsi,
                    'rsi_threshold': self.rsi_threshold,
                })
        elif final_signal != SignalType.HOLD:
            self.logger.info(f"[EXECUTE] cycle={self._cycle_count} 执行交易: 信号={final_signal.value}, {rsi_status}")
        
        if self.signal != SignalType.HOLD:
            self._execute_trade(self.signal, close_price)
    
    def _update_trailing_stop(self, current_price: float):
        """
        追踪止损逻辑（带保本逻辑）
        
        用户需求：
        1. 盈利达 1.0*ATR 后，立即将止损位移至成本价（保本逻辑）
        2. 盈利超过 2 倍 ATR，启动追踪止损，止损位设在 最高价 - 1.0 * ATR
        
        实现逻辑：
        1. 持续更新持仓期间的最高价/最低价
        2. 当盈利 >= break_even_atr_multiplier (默认1.0) 倍 ATR 时，将止损位移至成本价
        3. 当盈利 >= atr_exit_multiplier (默认2.0) 倍 ATR 时，启动追踪止损
        4. 追踪止损位 = 最高价 - trailing_stop_atr_multiplier (默认1.0) * ATR（做多）
           或 = 最低价 + trailing_stop_atr_multiplier (默认1.0) * ATR（做空）
        5. 止损位只向有利方向移动（上移/下移）
        """
        if self._position == 0 or self.atr is None:
            return
        
        if self._position > 0:
            if self._highest_price_since_entry is None or current_price > self._highest_price_since_entry:
                self._highest_price_since_entry = current_price
                self.logger.debug(
                    f"[追踪止损] 多单最高价更新: {self._highest_price_since_entry:.2f}"
                )
            
            if self._entry_price is not None:
                profit_amount = (current_price - self._entry_price) * self.contract_multiplier
                profit_atr = profit_amount / self.atr if self.atr > 0 else 0
                
                if not self._is_break_even_active and not self._trailing_stop_active:
                    self.logger.debug(
                        f"[追踪止损] 多单状态: 入场价={self._entry_price:.2f}, 当前价={current_price:.2f}, "
                        f"盈利={profit_atr:.2f}×ATR, 保本阈值={self.break_even_atr_multiplier}×ATR, "
                        f"追踪止损启动阈值={self.atr_exit_multiplier}×ATR"
                    )
                
                if profit_atr >= self.break_even_atr_multiplier and not self._is_break_even_active and not self._trailing_stop_active:
                    self._is_break_even_active = True
                    self._trailing_stop_price = self._entry_price
                    self._trailing_stop_active = True
                    self.logger.info(
                        f"[保本逻辑] 触发保本: 盈利={profit_atr:.2f}×ATR >= {self.break_even_atr_multiplier}×ATR, "
                        f"止损位设置为成本价={self._entry_price:.2f}"
                    )
                
                if self._trailing_stop_active:
                    if profit_atr >= self.atr_exit_multiplier:
                        new_stop = self._highest_price_since_entry - self.trailing_stop_atr_multiplier * self.atr
                        
                        if self._trailing_stop_price is None or new_stop > self._trailing_stop_price:
                            self._trailing_stop_price = new_stop
                            if self._is_break_even_active and new_stop > self._entry_price:
                                self.logger.info(
                                    f"[追踪止损] 多单止损位上移（从保本模式切换到追踪止损）: "
                                    f"旧={self._entry_price:.2f}（成本价）, "
                                    f"新={new_stop:.2f}, "
                                    f"最高价={self._highest_price_since_entry:.2f}, "
                                    f"止损距离={self.trailing_stop_atr_multiplier}×ATR={self.trailing_stop_atr_multiplier*self.atr:.2f}"
                                )
                            else:
                                self.logger.info(
                                    f"[追踪止损] 多单止损位上移: "
                                    f"旧={self._trailing_stop_price:.2f if self._trailing_stop_price else 'N/A'}, "
                                    f"新={new_stop:.2f}, "
                                    f"最高价={self._highest_price_since_entry:.2f}, "
                                    f"止损距离={self.trailing_stop_atr_multiplier}×ATR={self.trailing_stop_atr_multiplier*self.atr:.2f}"
                                )
        
        elif self._position < 0:
            if self._lowest_price_since_entry is None or current_price < self._lowest_price_since_entry:
                self._lowest_price_since_entry = current_price
                self.logger.debug(
                    f"[追踪止损] 空单最低价更新: {self._lowest_price_since_entry:.2f}"
                )
            
            if self._entry_price is not None:
                profit_amount = (self._entry_price - current_price) * self.contract_multiplier
                profit_atr = profit_amount / self.atr if self.atr > 0 else 0
                
                if not self._is_break_even_active and not self._trailing_stop_active:
                    self.logger.debug(
                        f"[追踪止损] 空单状态: 入场价={self._entry_price:.2f}, 当前价={current_price:.2f}, "
                        f"盈利={profit_atr:.2f}×ATR, 保本阈值={self.break_even_atr_multiplier}×ATR, "
                        f"追踪止损启动阈值={self.atr_exit_multiplier}×ATR"
                    )
                
                if profit_atr >= self.break_even_atr_multiplier and not self._is_break_even_active and not self._trailing_stop_active:
                    self._is_break_even_active = True
                    self._trailing_stop_price = self._entry_price
                    self._trailing_stop_active = True
                    self.logger.info(
                        f"[保本逻辑] 触发保本: 盈利={profit_atr:.2f}×ATR >= {self.break_even_atr_multiplier}×ATR, "
                        f"止损位设置为成本价={self._entry_price:.2f}"
                    )
                
                if self._trailing_stop_active:
                    if profit_atr >= self.atr_exit_multiplier:
                        new_stop = self._lowest_price_since_entry + self.trailing_stop_atr_multiplier * self.atr
                        
                        if self._trailing_stop_price is None or new_stop < self._trailing_stop_price:
                            self._trailing_stop_price = new_stop
                            if self._is_break_even_active and new_stop < self._entry_price:
                                self.logger.info(
                                    f"[追踪止损] 空单止损位下移（从保本模式切换到追踪止损）: "
                                    f"旧={self._entry_price:.2f}（成本价）, "
                                    f"新={new_stop:.2f}, "
                                    f"最低价={self._lowest_price_since_entry:.2f}, "
                                    f"止损距离={self.trailing_stop_atr_multiplier}×ATR={self.trailing_stop_atr_multiplier*self.atr:.2f}"
                                )
                            else:
                                self.logger.info(
                                    f"[追踪止损] 空单止损位下移: "
                                    f"旧={self._trailing_stop_price:.2f if self._trailing_stop_price else 'N/A'}, "
                                    f"新={new_stop:.2f}, "
                                    f"最低价={self._lowest_price_since_entry:.2f}, "
                                    f"止损距离={self.trailing_stop_atr_multiplier}×ATR={self.trailing_stop_atr_multiplier*self.atr:.2f}"
                                )
    
    def _close_position(self, current_price: float):
        """
        平仓（重写父类方法，添加保本逻辑标志重置）
        """
        if self._position == 0:
            return
        
        if self.api is None:
            self.logger.warning("API 未初始化，无法执行平仓")
            self._position = 0
            self._entry_price = None
            self._highest_price_since_entry = None
            self._lowest_price_since_entry = None
            self._trailing_stop_active = False
            self._trailing_stop_price = None
            self._is_break_even_active = False
            return
        
        try:
            quote = self.api.get_quote(self.contract)
            
            if self._position > 0:
                self._place_order(quote, direction="SELL", offset="CLOSE", volume=abs(self._position), limit_price=current_price)
                self.logger.info(f"平仓: 平多单 {self._position} 手, 当前价格={current_price:.2f}")
            elif self._position < 0:
                self._place_order(quote, direction="BUY", offset="CLOSE", volume=abs(self._position), limit_price=current_price)
                self.logger.info(f"平仓: 平空单 {abs(self._position)} 手, 当前价格={current_price:.2f}")
            
            self._position = 0
            self._entry_price = None
            self._highest_price_since_entry = None
            self._lowest_price_since_entry = None
            self._trailing_stop_active = False
            self._trailing_stop_price = None
            self._is_break_even_active = False
        except Exception as e:
            self.logger.error(f"平仓失败: {e}")
    
    def get_strategy_config(self) -> Dict[str, Any]:
        """获取策略配置参数"""
        return {
            'strategy_name': 'AdaptiveMAStrategy',
            'short_period': self.short_period,
            'long_period': self.long_period,
            'contract': self.contract,
            'kline_duration': self.kline_duration,
            'use_ema': self.use_ema,
            'rsi_period': self.rsi_period,
            'rsi_threshold': self.rsi_threshold,
            'atr_period': self.atr_period,
            'atr_entry_multiplier': self.atr_entry_multiplier,
            'atr_exit_multiplier': self.atr_exit_multiplier,
            'trailing_stop_atr_multiplier': self.trailing_stop_atr_multiplier,
            'risk_per_trade_percent': self.risk_per_trade_percent,
            'position_atr_divisor': self.position_atr_divisor,
            'max_position_value_percent': self.max_position_value_percent,
            'break_even_atr_multiplier': self.break_even_atr_multiplier,
            'contract_multiplier': self.contract_multiplier,
            'position_formula': f"(总资产 × {self.risk_per_trade_percent*100}%) / ({self.position_atr_divisor} × ATR)",
        }
    
    def run(self):
        if not self._initialized:
            self.logger.info("策略未初始化，正在执行初始化...")
            self.initialize()
        
        self.logger.info("=" * 80)
        self.logger.info("自适应多因子策略 (AdaptiveMAStrategy) 开始运行")
        self.logger.info("=" * 80)
        self.logger.info(f"【均线参数】")
        self.logger.info(f"  短期均线周期: {self.short_period}")
        self.logger.info(f"  长期均线周期: {self.long_period}")
        self.logger.info(f"  均线类型: {'EMA' if self.use_ema else 'SMA'}")
        self.logger.info(f"【动量确认 (RSI)】")
        self.logger.info(f"  RSI 周期: {self.rsi_period}")
        self.logger.info(f"  RSI 阈值: {self.rsi_threshold}")
        self.logger.info(f"  - 开多单: RSI 必须 > {self.rsi_threshold}")
        self.logger.info(f"  - 开空单: RSI 必须 < {self.rsi_threshold}")
        self.logger.info(f"【波动率过滤 (ATR)】")
        self.logger.info(f"  ATR 周期: {self.atr_period}")
        self.logger.info(f"  入场过滤阈值: {self.atr_entry_multiplier} × ATR")
        self.logger.info(f"  - 只有当前波动率 > {self.atr_entry_multiplier}×ATR 时才允许开仓")
        self.logger.info(f"【动态仓位管理】")
        self.logger.info(f"  单笔风险比例: {self.risk_per_trade_percent * 100}%")
        self.logger.info(f"  仓位公式: (总资产 × {self.risk_per_trade_percent*100}%) / ({self.position_atr_divisor} × ATR)")
        self.logger.info(f"  - 波动大（ATR大）→ 仓位小")
        self.logger.info(f"  - 波动小（ATR小）→ 仓位大")
        self.logger.info(f"【智能追踪止损】")
        self.logger.info(f"  启动条件: 盈利 > {self.atr_exit_multiplier} × ATR")
        self.logger.info(f"  止损距离: {self.trailing_stop_atr_multiplier} × ATR")
        self.logger.info(f"  - 做多: 止损位 = 最高价 - {self.trailing_stop_atr_multiplier}×ATR")
        self.logger.info(f"  - 做空: 止损位 = 最低价 + {self.trailing_stop_atr_multiplier}×ATR")
        self.logger.info(f"【交易合约】")
        self.logger.info(f"  合约: {self.contract}")
        self.logger.info(f"  K线周期: {self.kline_duration} 秒")
        self.logger.info(f"  合约乘数: {self.contract_multiplier}")
        self.logger.info("=" * 80)
        self.logger.info(f"pandas可用: {PANDAS_AVAILABLE}")
        self.logger.info(f"talib可用: {TALIB_AVAILABLE}")
        self.logger.info("=" * 80)
        
        try:
            self._run_loop()
        except KeyboardInterrupt:
            self.logger.info("用户中断，停止策略")
        except Exception as e:
            self.logger.error(f"策略运行出错: {str(e)}", exc_info=True)
            raise
