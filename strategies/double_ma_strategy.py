import os
import sys
import logging
from typing import Optional, List, Dict, Any, Tuple
from collections import deque
from datetime import datetime

from strategies.base_strategy import StrategyBase, SignalType

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
    logging.getLogger(__name__).info("talib 库已加载，将使用 talib 计算技术指标")
except ImportError:
    TALIB_AVAILABLE = False
    logging.getLogger(__name__).warning("talib 库未安装，将使用 pandas 或手动计算技术指标")


def _init_debug_logger():
    global DEBUG_SIGNAL_LOGGER, DEBUG_SIGNAL_FILE
    
    if DEBUG_SIGNAL_LOGGER is not None:
        return DEBUG_SIGNAL_LOGGER
    
    logs_dir = os.path.join(base_dir, 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    DEBUG_SIGNAL_FILE = os.path.join(logs_dir, f'debug_signal_{timestamp}.log')
    
    logger = logging.getLogger('debug_signal')
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    
    file_handler = logging.FileHandler(DEBUG_SIGNAL_FILE, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
    file_handler.setFormatter(file_formatter)
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(asctime)s [DEBUG_SIGNAL] %(message)s')
    console_handler.setFormatter(console_formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    logger.info("=" * 80)
    logger.info("信号调试日志已初始化 - 向量化计算模式")
    logger.info(f"日志文件: {DEBUG_SIGNAL_FILE}")
    logger.info(f"pandas可用: {PANDAS_AVAILABLE}, talib可用: {TALIB_AVAILABLE}")
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
    
    if 'short_ma' in data and data['short_ma'] is not None:
        msg_parts.append(f"MA{data.get('short_period', 5)}={data['short_ma']:.2f}")
    if 'prev_short_ma' in data and data['prev_short_ma'] is not None:
        msg_parts.append(f"prev_MA{data.get('short_period', 5)}={data['prev_short_ma']:.2f}")
    
    if 'long_ma' in data and data['long_ma'] is not None:
        msg_parts.append(f"MA{data.get('long_period', 20)}={data['long_ma']:.2f}")
    if 'prev_long_ma' in data and data['prev_long_ma'] is not None:
        msg_parts.append(f"prev_MA{data.get('long_period', 20)}={data['prev_long_ma']:.2f}")
    
    if 'rsi' in data and data['rsi'] is not None:
        msg_parts.append(f"RSI={data['rsi']:.2f}")
    
    if 'data_warmed_up' in data:
        msg_parts.append(f"warmed_up={data['data_warmed_up']}")
    if 'is_ready' in data:
        msg_parts.append(f"is_ready={data['is_ready']}")
    
    if 'prev_above' in data:
        msg_parts.append(f"prev_above={data['prev_above']}")
    if 'curr_above' in data:
        msg_parts.append(f"curr_above={data['curr_above']}")
    if 'prev_below' in data:
        msg_parts.append(f"prev_below={data['prev_below']}")
    if 'curr_below' in data:
        msg_parts.append(f"curr_below={data['curr_below']}")
    
    if 'signal' in data:
        msg_parts.append(f"signal={data['signal']}")
    
    if 'force_trade' in data:
        msg_parts.append(f"[FORCE_TRADE] action={data['force_trade']}")
    
    if 'position' in data:
        msg_parts.append(f"position={data['position']}")
    
    if 'action' in data:
        msg_parts.append(f"action={data['action']}")
    
    if 'order_price' in data:
        msg_parts.append(f"order_price={data['order_price']:.2f}")
    
    msg = " | ".join(msg_parts)
    logger.info(msg)


class VectorizedMAStrategy(StrategyBase):
    
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
        use_rsi_filter: bool = False,
        take_profit_ratio: Optional[float] = None,
        stop_loss_ratio: Optional[float] = None,
        initial_data_days: int = 5,
        force_trade_test: bool = False,
        debug_logging: bool = True,
        slippage_ticks: float = 1.0,
    ):
        super().__init__(connector)
        
        if short_period <= 0 or long_period <= 0:
            raise ValueError("均线周期必须大于 0")
        
        if short_period >= long_period:
            raise ValueError("短期均线周期必须小于长期均线周期")
        
        if rsi_period <= 1:
            raise ValueError("RSI 周期必须大于 1")
        
        if rsi_threshold < 0 or rsi_threshold > 100:
            raise ValueError("RSI 阈值必须在 0-100 之间")
        
        if take_profit_ratio is not None and take_profit_ratio <= 0:
            raise ValueError("止盈比例必须大于 0")
        
        if stop_loss_ratio is not None and stop_loss_ratio <= 0:
            raise ValueError("止损比例必须大于 0")
        
        self.short_period = short_period
        self.long_period = long_period
        self.contract = contract
        self.kline_duration = kline_duration
        self.use_ema = use_ema
        self.rsi_period = rsi_period
        self.rsi_threshold = rsi_threshold
        self.use_rsi_filter = use_rsi_filter
        self.take_profit_ratio = take_profit_ratio
        self.stop_loss_ratio = stop_loss_ratio
        
        self.slippage_ticks = slippage_ticks
        self._price_tick: float = 1.0
        
        self.initial_data_days = initial_data_days
        self._data_warmed_up = False
        
        klines_per_day = self._calculate_klines_per_day(kline_duration)
        self._required_warmup_klines = max(
            long_period * 3,
            rsi_period + 10,
            int(initial_data_days * klines_per_day),
        )
        
        self.logger.info(
            f"向量化策略初始化: initial_data_days={initial_data_days}天, "
            f"需要预热K线数={self._required_warmup_klines}根"
        )
        
        self._price_df: Optional[pd.DataFrame] = None
        self._price_list: List[float] = []
        
        self.short_ma: Optional[float] = None
        self.long_ma: Optional[float] = None
        self.prev_short_ma: Optional[float] = None
        self.prev_long_ma: Optional[float] = None
        
        self.short_ma_series: List[Optional[float]] = []
        self.long_ma_series: List[Optional[float]] = []
        
        self.rsi: Optional[float] = None
        self.prev_rsi: Optional[float] = None
        
        self.prev_signal: SignalType = SignalType.HOLD
        self.klines = None
        
        self._position = 0
        self._entry_price = None
        self._trade_count = 0
        
        self._rsi_filtered_signals = 0
        
        self._tp_triggered = 0
        self._sl_triggered = 0
        
        self.force_trade_test = force_trade_test
        self.debug_logging = debug_logging
        self._cycle_count = 0
        
        if self.force_trade_test:
            self.logger.warning("[FORCE_TRADE] 暴力开仓测试模式已开启！每根K线都会尝试交易")
        
        if self.debug_logging:
            _init_debug_logger()
            self.logger.info(f"调试日志已启用，将记录每根K线的信号判定")
        
        rsi_info = f", RSI周期={rsi_period}, RSI阈值={rsi_threshold}, RSI过滤={'开启' if use_rsi_filter else '关闭'}"
        tp_sl_info = ""
        if take_profit_ratio is not None:
            tp_sl_info += f", 止盈比例={take_profit_ratio*100:.1f}%"
        if stop_loss_ratio is not None:
            tp_sl_info += f", 止损比例={stop_loss_ratio*100:.1f}%"
        
        self.logger.info(f"向量化双均线策略初始化: 短期周期={short_period}, 长期周期={long_period}, 合约={contract}, K线周期={kline_duration}秒, 均线类型={'EMA' if use_ema else 'SMA'}{rsi_info}{tp_sl_info}")
    
    def subscribe(self):
        if self.api is None:
            raise RuntimeError("API 未设置，请先调用 set_connector() 或在初始化时传入 connector")
        
        self.logger.info(f"订阅合约: {self.contract}, K线周期: {self.kline_duration}秒")
        
        try:
            self.klines = self.api.get_kline_serial(self.contract, self.kline_duration)
            self.logger.info(f"成功订阅 {self.contract} 的 K 线数据")
        except Exception as e:
            self.logger.error(f"订阅 K 线数据失败: {str(e)}", exc_info=True)
            raise
    
    @staticmethod
    def _calculate_klines_per_day(kline_duration: int) -> int:
        seconds_per_day = 14400
        if kline_duration <= 0:
            return 240
        return max(1, int(seconds_per_day / kline_duration))
    
    def _init_dataframe(self):
        if not PANDAS_AVAILABLE:
            return
        
        if self._price_list:
            self._price_df = pd.DataFrame({
                'close': self._price_list
            })
            self._calculate_all_indicators()
    
    def _calculate_all_indicators(self):
        if self._price_df is None or len(self._price_df) < self.long_period:
            return
        
        close_prices = self._price_df['close'].values
        
        if TALIB_AVAILABLE:
            np_close = np.array(close_prices, dtype=float)
            if self.use_ema:
                self._price_df['short_ma'] = talib.EMA(np_close, timeperiod=self.short_period)
                self._price_df['long_ma'] = talib.EMA(np_close, timeperiod=self.long_period)
            else:
                self._price_df['short_ma'] = talib.SMA(np_close, timeperiod=self.short_period)
                self._price_df['long_ma'] = talib.SMA(np_close, timeperiod=self.long_period)
            
            if len(self._price_df) >= self.rsi_period + 1:
                self._price_df['rsi'] = talib.RSI(np_close, timeperiod=self.rsi_period)
            else:
                self._price_df['rsi'] = None
        elif PANDAS_AVAILABLE:
            if self.use_ema:
                self._price_df['short_ma'] = self._price_df['close'].ewm(span=self.short_period, adjust=False).mean()
                self._price_df['long_ma'] = self._price_df['close'].ewm(span=self.long_period, adjust=False).mean()
            else:
                self._price_df['short_ma'] = self._price_df['close'].rolling(window=self.short_period).mean()
                self._price_df['long_ma'] = self._price_df['close'].rolling(window=self.long_period).mean()
            
            if len(self._price_df) >= self.rsi_period + 1:
                self._price_df['rsi'] = self._calculate_rsi_pandas(self._price_df['close'], self.rsi_period)
            else:
                self._price_df['rsi'] = None
        
        self.short_ma_series = self._price_df['short_ma'].tolist()
        self.long_ma_series = self._price_df['long_ma'].tolist()
    
    @staticmethod
    def _calculate_rsi_pandas(series: pd.Series, period: int) -> pd.Series:
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _log_ma_trajectory(self, current_price: float):
        if not self.debug_logging:
            return
        
        logger = get_debug_logger()
        
        n = min(5, len(self._price_list))
        if n < 1:
            return
        
        logger.info("")
        logger.info("=" * 80)
        logger.info(f"[均线轨迹追踪] cycle={self._cycle_count} - 最近{n}根K线")
        logger.info("-" * 80)
        logger.info(f"{'周期':<8} {'收盘价':<12} {'MA' + str(self.short_period):<15} {'MA' + str(self.long_period):<15} {'状态':<10}")
        logger.info("-" * 80)
        
        for i in range(-n, 0):
            idx = len(self._price_list) + i
            cycle_num = self._cycle_count + i + 1
            
            close_val = self._price_list[idx] if idx >= 0 else 0.0
            
            short_ma_val = self.short_ma_series[idx] if idx < len(self.short_ma_series) else None
            long_ma_val = self.long_ma_series[idx] if idx < len(self.long_ma_series) else None
            
            short_ma_str = f"{short_ma_val:.2f}" if short_ma_val is not None else "N/A"
            long_ma_str = f"{long_ma_val:.2f}" if long_ma_val is not None else "N/A"
            
            status = "N/A"
            if short_ma_val is not None and long_ma_val is not None:
                if short_ma_val > long_ma_val:
                    status = "多头排列"
                elif short_ma_val < long_ma_val:
                    status = "空头排列"
                else:
                    status = "交叉"
            
            marker = " <-- 当前" if i == -1 else ""
            logger.info(f"{cycle_num:<8} {close_val:<12.2f} {short_ma_str:<15} {long_ma_str:<15} {status:<10}{marker}")
        
        if len(self.short_ma_series) >= 2 and len(self.long_ma_series) >= 2:
            prev_short = self.short_ma_series[-2] if len(self.short_ma_series) >= 2 else None
            curr_short = self.short_ma_series[-1] if len(self.short_ma_series) >= 1 else None
            prev_long = self.long_ma_series[-2] if len(self.long_ma_series) >= 2 else None
            curr_long = self.long_ma_series[-1] if len(self.long_ma_series) >= 1 else None
            
            if prev_short is not None and curr_short is not None and prev_long is not None and curr_long is not None:
                short_change = curr_short - prev_short
                long_change = curr_long - prev_long
                
                logger.info("-" * 80)
                logger.info(f"均线变动: MA{self.short_period} 变化={short_change:+.4f}, MA{self.long_period} 变化={long_change:+.4f}")
                
                prev_above = prev_short > prev_long
                curr_above = curr_short > curr_long
                
                if prev_above != curr_above:
                    if curr_above:
                        logger.info(f"⚠️ 金叉信号: MA{self.short_period} 上穿 MA{self.long_period}")
                    else:
                        logger.info(f"⚠️ 死叉信号: MA{self.short_period} 下穿 MA{self.long_period}")
        
        logger.info("=" * 80)
        logger.info("")
    
    def update_prices(self, close_price: float):
        self._cycle_count += 1
        
        if close_price is None or (isinstance(close_price, float) and close_price != close_price):
            self.logger.debug(f"[cycle={self._cycle_count}] 收到无效的收盘价，跳过")
            if self.debug_logging:
                log_signal_debug(self.contract, self._cycle_count, {
                    'close_price': 0.0 if close_price is None else close_price,
                    'signal': 'INVALID_PRICE',
                })
            return
        
        self.prev_short_ma = self.short_ma
        self.prev_long_ma = self.long_ma
        self.prev_rsi = self.rsi
        
        self._price_list.append(close_price)
        
        if PANDAS_AVAILABLE:
            if self._price_df is None:
                self._init_dataframe()
            else:
                new_row = pd.DataFrame({'close': [close_price]})
                self._price_df = pd.concat([self._price_df, new_row], ignore_index=True)
                self._calculate_all_indicators()
            
            if len(self._price_df) >= self.short_period:
                last_row = self._price_df.iloc[-1]
                self.short_ma = float(last_row['short_ma']) if pd.notna(last_row.get('short_ma')) else None
                self.long_ma = float(last_row['long_ma']) if pd.notna(last_row.get('long_ma')) else None
                self.rsi = float(last_row['rsi']) if pd.notna(last_row.get('rsi')) else None
        else:
            self._fallback_calculate_ma()
        
        if self.debug_logging and len(self._price_list) >= self.long_period:
            self._log_ma_trajectory(close_price)
        
        if self.debug_logging:
            debug_data = {
                'close_price': close_price,
                'short_period': self.short_period,
                'long_period': self.long_period,
                'short_ma': self.short_ma,
                'long_ma': self.long_ma,
                'prev_short_ma': self.prev_short_ma,
                'prev_long_ma': self.prev_long_ma,
                'rsi': self.rsi,
                'data_warmed_up': self._data_warmed_up,
                'is_ready': self.is_ready(),
                'position': self._position,
            }
            
            if self.prev_short_ma is not None and self.prev_long_ma is not None:
                debug_data['prev_above'] = self.prev_short_ma > self.prev_long_ma
                debug_data['prev_below'] = self.prev_short_ma < self.prev_long_ma
            
            if self.short_ma is not None and self.long_ma is not None:
                debug_data['curr_above'] = self.short_ma > self.long_ma
                debug_data['curr_below'] = self.short_ma < self.long_ma
            
            log_signal_debug(self.contract, self._cycle_count, debug_data)
        
        if self.force_trade_test:
            self._force_trade(close_price)
            return
        
        if self._position != 0:
            self._check_take_profit_stop_loss(close_price)
        
        if self._position == 0:
            self._detect_signal(close_price)
        else:
            self.signal = SignalType.HOLD
    
    def _fallback_calculate_ma(self):
        if len(self._price_list) < self.short_period:
            self.short_ma = None
        else:
            if self.use_ema:
                self.short_ma = self._calculate_ema_fallback(self._price_list, self.short_period)
            else:
                self.short_ma = sum(self._price_list[-self.short_period:]) / self.short_period
        
        if len(self._price_list) < self.long_period:
            self.long_ma = None
        else:
            if self.use_ema:
                self.long_ma = self._calculate_ema_fallback(self._price_list, self.long_period)
            else:
                self.long_ma = sum(self._price_list[-self.long_period:]) / self.long_period
        
        self.short_ma_series.append(self.short_ma)
        self.long_ma_series.append(self.long_ma)
    
    @staticmethod
    def _calculate_ema_fallback(prices: List[float], period: int) -> Optional[float]:
        if len(prices) < period:
            return None
        
        multiplier = 2.0 / (period + 1.0)
        ema = sum(prices[:period]) / period
        
        for i in range(period, len(prices)):
            ema = (prices[i] - ema) * multiplier + ema
        
        return ema
    
    def _force_trade(self, close_price: float):
        if self.api is None:
            self.logger.warning(f"[FORCE_TRADE] API 未初始化，无法执行交易")
            if self.debug_logging:
                log_signal_debug(self.contract, self._cycle_count, {
                    'close_price': close_price,
                    'force_trade': 'SKIP_NO_API',
                    'position': self._position,
                })
            return
        
        try:
            quote = self.api.get_quote(self.contract)
            
            if self._position == 0:
                if self._cycle_count % 2 == 1:
                    self._place_order(quote, direction="BUY", offset="OPEN", volume=1, limit_price=close_price)
                    self._position = 1
                    self._entry_price = close_price
                    self.logger.info(f"[FORCE_TRADE] cycle={self._cycle_count} 开多单, 价格={close_price:.2f}")
                    if self.debug_logging:
                        log_signal_debug(self.contract, self._cycle_count, {
                            'close_price': close_price,
                            'force_trade': 'BUY_OPEN',
                            'position': 1,
                            'order_price': close_price,
                        })
                else:
                    self._place_order(quote, direction="SELL", offset="OPEN", volume=1, limit_price=close_price)
                    self._position = -1
                    self._entry_price = close_price
                    self.logger.info(f"[FORCE_TRADE] cycle={self._cycle_count} 开空单, 价格={close_price:.2f}")
                    if self.debug_logging:
                        log_signal_debug(self.contract, self._cycle_count, {
                            'close_price': close_price,
                            'force_trade': 'SELL_OPEN',
                            'position': -1,
                            'order_price': close_price,
                        })
            else:
                if self._position > 0:
                    self._place_order(quote, direction="SELL", offset="CLOSE", volume=self._position, limit_price=close_price)
                    self.logger.info(f"[FORCE_TRADE] cycle={self._cycle_count} 平多单, 价格={close_price:.2f}")
                    if self.debug_logging:
                        log_signal_debug(self.contract, self._cycle_count, {
                            'close_price': close_price,
                            'force_trade': 'SELL_CLOSE',
                            'position': 0,
                            'order_price': close_price,
                        })
                else:
                    self._place_order(quote, direction="BUY", offset="CLOSE", volume=abs(self._position), limit_price=close_price)
                    self.logger.info(f"[FORCE_TRADE] cycle={self._cycle_count} 平空单, 价格={close_price:.2f}")
                    if self.debug_logging:
                        log_signal_debug(self.contract, self._cycle_count, {
                            'close_price': close_price,
                            'force_trade': 'BUY_CLOSE',
                            'position': 0,
                            'order_price': close_price,
                        })
                self._position = 0
                self._entry_price = None
        
        except Exception as e:
            self.logger.error(f"[FORCE_TRADE] 交易失败: {e}")
            if self.debug_logging:
                log_signal_debug(self.contract, self._cycle_count, {
                    'close_price': close_price,
                    'force_trade': f'ERROR_{e}',
                    'position': self._position,
                })
    
    def update_from_kline(self, kline_data: Dict[str, Any]):
        if kline_data is None:
            self.logger.warning("收到空的 K 线数据")
            return
        
        close_price = kline_data.get('close')
        
        if close_price is None:
            return
        
        try:
            close_price = float(close_price)
            self.update_prices(close_price)
        except (ValueError, TypeError) as e:
            self.logger.warning(f"无效的收盘价: {close_price}, 错误: {str(e)}")
    
    def _detect_signal(self, close_price: float = 0):
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
        
        if not self.use_rsi_filter:
            self.signal = ma_signal
            self.logger.info(f"[EXECUTE] cycle={self._cycle_count} 执行交易: 信号={ma_signal.value} (RSI过滤关闭)")
            if self.signal != SignalType.HOLD:
                self._execute_trade(self.signal, close_price)
            return
        
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
                rsi_status = f"RSI={self.rsi:.2f} > 阈值={self.rsi_threshold:.2f}，信号确认"
            else:
                self._rsi_filtered_signals += 1
                rsi_status = f"RSI={self.rsi:.2f} <= 阈值={self.rsi_threshold:.2f}，信号被过滤（非强势区）"
        
        elif ma_signal == SignalType.SELL:
            if self.rsi < self.rsi_threshold:
                final_signal = SignalType.SELL
                rsi_status = f"RSI={self.rsi:.2f} < 阈值={self.rsi_threshold:.2f}，信号确认"
            else:
                self._rsi_filtered_signals += 1
                rsi_status = f"RSI={self.rsi:.2f} >= 阈值={self.rsi_threshold:.2f}，信号被过滤（非弱势区）"
        
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
    
    def _check_take_profit_stop_loss(self, current_price: float):
        if self._position == 0 or self._entry_price is None:
            return
        
        if self.take_profit_ratio is None and self.stop_loss_ratio is None:
            return
        
        tp_triggered = False
        sl_triggered = False
        
        if self._position > 0:
            if self.take_profit_ratio is not None:
                tp_price = self._entry_price * (1 + self.take_profit_ratio)
                if current_price >= tp_price:
                    tp_triggered = True
                    self.logger.info(f"止盈触发: 多单入场价={self._entry_price:.2f}, 当前价={current_price:.2f}, 止盈价={tp_price:.2f}")
            
            if self.stop_loss_ratio is not None and not tp_triggered:
                sl_price = self._entry_price * (1 - self.stop_loss_ratio)
                if current_price <= sl_price:
                    sl_triggered = True
                    self.logger.info(f"止损触发: 多单入场价={self._entry_price:.2f}, 当前价={current_price:.2f}, 止损价={sl_price:.2f}")
        
        elif self._position < 0:
            if self.take_profit_ratio is not None:
                tp_price = self._entry_price * (1 - self.take_profit_ratio)
                if current_price <= tp_price:
                    tp_triggered = True
                    self.logger.info(f"止盈触发: 空单入场价={self._entry_price:.2f}, 当前价={current_price:.2f}, 止盈价={tp_price:.2f}")
            
            if self.stop_loss_ratio is not None and not tp_triggered:
                sl_price = self._entry_price * (1 + self.stop_loss_ratio)
                if current_price >= sl_price:
                    sl_triggered = True
                    self.logger.info(f"止损触发: 空单入场价={self._entry_price:.2f}, 当前价={current_price:.2f}, 止损价={sl_price:.2f}")
        
        if tp_triggered or sl_triggered:
            if tp_triggered:
                self._tp_triggered += 1
            if sl_triggered:
                self._sl_triggered += 1
            
            self._close_position(current_price)
    
    def _close_position(self, current_price: float):
        if self._position == 0:
            return
        
        if self.api is None:
            self.logger.warning("API 未初始化，无法执行平仓")
            self._position = 0
            self._entry_price = None
            return
        
        try:
            quote = self.api.get_quote(self.contract)
            
            if self._position > 0:
                self._place_order(quote, direction="SELL", offset="CLOSE", volume=self._position, limit_price=current_price)
                self.logger.info(f"止盈/止损平仓: 平多单 {self._position} 手, 当前价格={current_price:.2f}")
            elif self._position < 0:
                self._place_order(quote, direction="BUY", offset="CLOSE", volume=abs(self._position), limit_price=current_price)
                self.logger.info(f"止盈/止损平仓: 平空单 {abs(self._position)} 手, 当前价格={current_price:.2f}")
            
            self._position = 0
            self._entry_price = None
        except Exception as e:
            self.logger.error(f"止盈/止损平仓失败: {e}")
    
    def _execute_trade(self, signal: SignalType, close_price: float = 0):
        if self.api is None:
            self.logger.warning("API 未初始化，无法执行交易")
            return
        
        try:
            quote = self.api.get_quote(self.contract)
            
            if hasattr(quote, 'price_tick') and quote.price_tick > 0:
                self._price_tick = float(quote.price_tick)
            
            current_price = close_price if close_price > 0 else None
            if current_price is None:
                if hasattr(quote, 'last_price'):
                    current_price = quote.last_price
                elif hasattr(quote, 'close'):
                    current_price = quote.close
            
            if current_price is None or current_price != current_price:
                self.logger.warning(f"无法获取有效价格，跳过交易")
                return
            
            if signal == SignalType.BUY:
                buy_price = current_price + self.slippage_ticks * self._price_tick
                self.logger.debug(f"[滑点模拟] 买入: 原价={current_price:.2f}, 滑点后={buy_price:.2f} (滑点={self.slippage_ticks}跳, 最小变动={self._price_tick})")
                
                if self._position < 0:
                    self._place_order(quote, direction="BUY", offset="CLOSE", volume=abs(self._position), limit_price=buy_price)
                    self._position = 0
                    self._entry_price = None
                
                if self._position == 0:
                    self._place_order(quote, direction="BUY", offset="OPEN", volume=1, limit_price=buy_price)
                    self._position = 1
                    self._entry_price = buy_price
                    
            elif signal == SignalType.SELL:
                sell_price = current_price - self.slippage_ticks * self._price_tick
                self.logger.debug(f"[滑点模拟] 卖出: 原价={current_price:.2f}, 滑点后={sell_price:.2f} (滑点={self.slippage_ticks}跳, 最小变动={self._price_tick})")
                
                if self._position > 0:
                    self._place_order(quote, direction="SELL", offset="CLOSE", volume=self._position, limit_price=sell_price)
                    self._position = 0
                    self._entry_price = None
                
                if self._position == 0:
                    self._place_order(quote, direction="SELL", offset="OPEN", volume=1, limit_price=sell_price)
                    self._position = -1
                    self._entry_price = sell_price
                    
        except Exception as e:
            self.logger.error(f"执行交易失败: {e}")
    
    def _place_order(self, quote, direction: str, offset: str, volume: int, limit_price: float = 0):
        try:
            order = self.api.insert_order(
                quote,
                direction=direction,
                offset=offset,
                volume=volume,
                limit_price=limit_price,
            )
            self._trade_count += 1
            self.logger.info(f"下单成功: {direction} {offset} {volume}手 {self.contract}, 价格={limit_price if limit_price > 0 else '市价'}, 交易次数={self._trade_count}")
            return order
        except Exception as e:
            self.logger.error(f"下单失败: {direction} {offset} {volume}手 {self.contract}, 错误={e}")
            return None
    
    def get_signal(self) -> SignalType:
        return self.signal
    
    def get_ma_values(self) -> Dict[str, Optional[float]]:
        result = {
            f"ma_{self.short_period}": self.short_ma,
            f"ma_{self.long_period}": self.long_ma,
            f"prev_ma_{self.short_period}": self.prev_short_ma,
            f"prev_ma_{self.long_period}": self.prev_long_ma,
        }
        
        if self.use_rsi_filter:
            result['rsi'] = self.rsi
            result['prev_rsi'] = self.prev_rsi
            result['rsi_threshold'] = self.rsi_threshold
        
        if self.take_profit_ratio is not None:
            result['take_profit_ratio'] = self.take_profit_ratio
        if self.stop_loss_ratio is not None:
            result['stop_loss_ratio'] = self.stop_loss_ratio
        
        return result
    
    def get_rsi_value(self) -> Optional[float]:
        return self.rsi
    
    def get_rsi_filter_stats(self) -> Dict[str, Any]:
        return {
            'rsi_period': self.rsi_period,
            'rsi_threshold': self.rsi_threshold,
            'use_rsi_filter': self.use_rsi_filter,
            'filtered_signals': self._rsi_filtered_signals,
        }
    
    def get_tp_sl_stats(self) -> Dict[str, Any]:
        return {
            'take_profit_ratio': self.take_profit_ratio,
            'stop_loss_ratio': self.stop_loss_ratio,
            'tp_triggered': self._tp_triggered,
            'sl_triggered': self._sl_triggered,
            'entry_price': self._entry_price,
            'position': self._position,
        }
    
    def is_ready(self) -> bool:
        if not self._data_warmed_up:
            return False
        
        if self.use_rsi_filter:
            return (self.short_ma is not None and 
                    self.long_ma is not None and 
                    self.rsi is not None)
        return self.short_ma is not None and self.long_ma is not None
    
    def on_bar(self, bar_data: Dict[str, Any]):
        if bar_data is None:
            self.logger.warning("on_bar 收到空数据")
            return
        
        try:
            self.update_from_kline(bar_data)
            
            if self.is_ready():
                ma_values = self.get_ma_values()
                current_signal = self.get_signal()
                
                rsi_info = ""
                if self.use_rsi_filter and self.rsi is not None:
                    rsi_info = f", RSI={self.rsi:.2f}"
                
                self.logger.debug(
                    f"K线更新 - MA{self.short_period}: {ma_values[f'ma_{self.short_period}']:.2f}, "
                    f"MA{self.long_period}: {ma_values[f'ma_{self.long_period}']:.2f}{rsi_info}, "
                    f"信号: {current_signal.value}"
                )
        except Exception as e:
            self.logger.error(f"处理 K 线数据时出错: {str(e)}", exc_info=True)
    
    def _on_update(self):
        if self.klines is None:
            return
        
        try:
            if len(self.klines) > 0:
                if not self.is_ready():
                    self._warmup_klines()
                
                latest_kline = self.klines.iloc[-1]
                self.on_bar(latest_kline.to_dict())
        except Exception as e:
            self.logger.error(f"处理更新时出错: {str(e)}", exc_info=True)
    
    def _warmup_klines(self):
        if self._data_warmed_up:
            return
        
        required_period = max(
            self.long_period,
            self.rsi_period,
            self._required_warmup_klines,
        )
        
        if len(self.klines) <= required_period:
            self.logger.debug(
                f"K 线数量不足，跳过预热: "
                f"{len(self.klines)} < {required_period + 1}, "
                f"需要 {self.initial_data_days} 天数据"
            )
            return
        
        warmup_count = min(len(self.klines) - 1, self._required_warmup_klines)
        start_idx = max(0, len(self.klines) - warmup_count)
        
        self.logger.info(
            f"开始预热 K 线 (向量化模式): initial_data_days={self.initial_data_days}天, "
            f"需要预热K线数={self._required_warmup_klines}根, "
            f"从索引 {start_idx} 到 {len(self.klines) - 2}, 共 {warmup_count - 1} 根"
        )
        
        for i in range(start_idx, len(self.klines) - 1):
            try:
                kline = self.klines.iloc[i]
                kline_dict = kline.to_dict() if hasattr(kline, 'to_dict') else dict(kline)
                self.update_from_kline(kline_dict)
            except Exception as e:
                self.logger.warning(f"预热 K 线 {i} 时出错: {e}")
        
        self._data_warmed_up = True
        
        rsi_status = f", RSI={self.rsi:.2f}" if self.rsi is not None else ""
        self.logger.info(
            f"冷启动预热完成: 已收集 {len(self._price_list)} 条价格记录, "
            f"短期均线={self.short_ma}, 长期均线={self.long_ma}{rsi_status}"
        )
    
    def run(self):
        if not self._initialized:
            self.logger.info("策略未初始化，正在执行初始化...")
            self.initialize()
        
        self.logger.info("向量化双均线策略开始运行")
        self.logger.info(f"短期均线周期: {self.short_period}")
        self.logger.info(f"长期均线周期: {self.long_period}")
        self.logger.info(f"交易合约: {self.contract}")
        self.logger.info(f"均线类型: {'EMA' if self.use_ema else 'SMA'}")
        self.logger.info(f"RSI 过滤: {'开启' if self.use_rsi_filter else '关闭'}")
        if self.use_rsi_filter:
            self.logger.info(f"RSI 周期: {self.rsi_period}")
            self.logger.info(f"RSI 阈值: {self.rsi_threshold}")
        
        if self.take_profit_ratio is not None:
            self.logger.info(f"止盈比例: {self.take_profit_ratio*100:.1f}%")
        else:
            self.logger.info("止盈比例: 未设置")
        
        if self.stop_loss_ratio is not None:
            self.logger.info(f"止损比例: {self.stop_loss_ratio*100:.1f}%")
        else:
            self.logger.info("止损比例: 未设置")
        
        self.logger.info(f"pandas可用: {PANDAS_AVAILABLE}")
        self.logger.info(f"talib可用: {TALIB_AVAILABLE}")
        
        try:
            self._run_loop()
        except KeyboardInterrupt:
            self.logger.info("用户中断，停止策略")
        except Exception as e:
            self.logger.error(f"策略运行出错: {str(e)}", exc_info=True)
            raise
    
    def save_state(self) -> Dict[str, Any]:
        state = {
            'short_period': self.short_period,
            'long_period': self.long_period,
            'contract': self.contract,
            'kline_duration': self.kline_duration,
            'use_ema': self.use_ema,
            'rsi_period': self.rsi_period,
            'rsi_threshold': self.rsi_threshold,
            'use_rsi_filter': self.use_rsi_filter,
            'take_profit_ratio': self.take_profit_ratio,
            'stop_loss_ratio': self.stop_loss_ratio,
            'short_ma': self.short_ma,
            'long_ma': self.long_ma,
            'prev_short_ma': self.prev_short_ma,
            'prev_long_ma': self.prev_long_ma,
            'rsi': self.rsi,
            'prev_rsi': self.prev_rsi,
            'signal': self.signal.value if hasattr(self.signal, 'value') else str(self.signal),
            'prev_signal': self.prev_signal.value if hasattr(self.prev_signal, 'value') else str(self.prev_signal),
            '_price_list': list(self._price_list) if self._price_list else [],
            'short_ma_series': list(self.short_ma_series) if self.short_ma_series else [],
            'long_ma_series': list(self.long_ma_series) if self.long_ma_series else [],
            '_rsi_filtered_signals': self._rsi_filtered_signals,
            '_tp_triggered': self._tp_triggered,
            '_sl_triggered': self._sl_triggered,
            '_position': self._position,
            '_entry_price': self._entry_price,
        }
        
        self.logger.info(f"保存策略状态: 已收集 {len(self._price_list)} 条价格记录, RSI过滤信号数={self._rsi_filtered_signals}, 止盈触发={self._tp_triggered}, 止损触发={self._sl_triggered}")
        return state
    
    def load_state(self, state: Dict[str, Any]) -> None:
        if not state:
            self.logger.warning("状态为空，跳过加载")
            return
        
        if 'short_period' in state:
            if state['short_period'] != self.short_period:
                self.logger.warning(f"保存的短期均线周期 ({state['short_period']}) 与当前配置 ({self.short_period}) 不一致")
        
        if 'long_period' in state:
            if state['long_period'] != self.long_period:
                self.logger.warning(f"保存的长期均线周期 ({state['long_period']}) 与当前配置 ({self.long_period}) 不一致")
        
        if 'rsi_period' in state:
            if state['rsi_period'] != self.rsi_period:
                self.logger.warning(f"保存的 RSI 周期 ({state['rsi_period']}) 与当前配置 ({self.rsi_period}) 不一致")
        
        if 'rsi_threshold' in state:
            if state['rsi_threshold'] != self.rsi_threshold:
                self.logger.warning(f"保存的 RSI 阈值 ({state['rsi_threshold']}) 与当前配置 ({self.rsi_threshold}) 不一致")
        
        if 'take_profit_ratio' in state:
            saved_tp = state['take_profit_ratio']
            if saved_tp != self.take_profit_ratio:
                self.logger.warning(f"保存的止盈比例 ({saved_tp}) 与当前配置 ({self.take_profit_ratio}) 不一致")
        
        if 'stop_loss_ratio' in state:
            saved_sl = state['stop_loss_ratio']
            if saved_sl != self.stop_loss_ratio:
                self.logger.warning(f"保存的止损比例 ({saved_sl}) 与当前配置 ({self.stop_loss_ratio}) 不一致")
        
        if 'contract' in state:
            if state['contract'] != self.contract:
                self.logger.warning(f"保存的合约 ({state['contract']}) 与当前配置 ({self.contract}) 不一致")
        
        if '_price_list' in state:
            prices = state['_price_list']
            if prices:
                self._price_list = [float(p) for p in prices]
                self.logger.info(f"恢复价格历史: {len(self._price_list)} 条记录")
        
        if 'short_ma_series' in state:
            series = state['short_ma_series']
            if series:
                self.short_ma_series = [float(s) if s is not None else None for s in series]
        
        if 'long_ma_series' in state:
            series = state['long_ma_series']
            if series:
                self.long_ma_series = [float(s) if s is not None else None for s in series]
        
        if 'short_ma' in state and state['short_ma'] is not None:
            self.short_ma = float(state['short_ma'])
        
        if 'long_ma' in state and state['long_ma'] is not None:
            self.long_ma = float(state['long_ma'])
        
        if 'prev_short_ma' in state and state['prev_short_ma'] is not None:
            self.prev_short_ma = float(state['prev_short_ma'])
        
        if 'prev_long_ma' in state and state['prev_long_ma'] is not None:
            self.prev_long_ma = float(state['prev_long_ma'])
        
        if 'rsi' in state and state['rsi'] is not None:
            self.rsi = float(state['rsi'])
        
        if 'prev_rsi' in state and state['prev_rsi'] is not None:
            self.prev_rsi = float(state['prev_rsi'])
        
        if '_rsi_filtered_signals' in state:
            self._rsi_filtered_signals = int(state['_rsi_filtered_signals'])
        
        if '_tp_triggered' in state:
            self._tp_triggered = int(state['_tp_triggered'])
        
        if '_sl_triggered' in state:
            self._sl_triggered = int(state['_sl_triggered'])
        
        if '_position' in state:
            self._position = int(state['_position'])
        
        if '_entry_price' in state and state['_entry_price'] is not None:
            self._entry_price = float(state['_entry_price'])
        
        if 'signal' in state:
            try:
                signal_value = state['signal']
                if isinstance(signal_value, str):
                    for sig_type in SignalType:
                        if sig_type.value == signal_value:
                            self.signal = sig_type
                            break
            except Exception as e:
                self.logger.warning(f"恢复信号状态失败: {e}")
        
        if 'prev_signal' in state:
            try:
                signal_value = state['prev_signal']
                if isinstance(signal_value, str):
                    for sig_type in SignalType:
                        if sig_type.value == signal_value:
                            self.prev_signal = sig_type
                            break
            except Exception as e:
                self.logger.warning(f"恢复历史信号状态失败: {e}")
        
        rsi_status = f", RSI={self.rsi:.2f}" if self.rsi is not None else ""
        self.logger.info(f"策略状态已恢复: 短期均线={self.short_ma}, 长期均线={self.long_ma}{rsi_status}, 信号={self.signal.value}")


DoubleMAStrategy = VectorizedMAStrategy
