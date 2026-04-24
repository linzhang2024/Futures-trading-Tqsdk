import logging
from typing import Optional, List, Dict, Any, Tuple
from collections import deque

from strategies.base_strategy import StrategyBase, SignalType

try:
    import talib
    import numpy as np
    TALIB_AVAILABLE = True
    logging.getLogger(__name__).info("talib 库已加载，将使用 talib 计算技术指标")
except ImportError:
    TALIB_AVAILABLE = False
    logging.getLogger(__name__).warning("talib 库未安装，将使用手动计算技术指标")


class DoubleMAStrategy(StrategyBase):
    def __init__(
        self,
        connector: Any = None,
        short_period: int = 5,
        long_period: int = 10,
        contract: str = "SHFE.rb2410",
        kline_duration: int = 60,
        use_ema: bool = False,
        rsi_period: int = 14,
        rsi_threshold: float = 50.0,
        use_rsi_filter: bool = False,
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
        
        self.short_period = short_period
        self.long_period = long_period
        self.contract = contract
        self.kline_duration = kline_duration
        self.use_ema = use_ema
        self.rsi_period = rsi_period
        self.rsi_threshold = rsi_threshold
        self.use_rsi_filter = use_rsi_filter
        
        self.short_prices: deque = deque(maxlen=long_period)
        self.long_prices: deque = deque(maxlen=long_period)
        
        self.short_ma: Optional[float] = None
        self.long_ma: Optional[float] = None
        
        self.prev_short_ma: Optional[float] = None
        self.prev_long_ma: Optional[float] = None
        
        self.rsi: Optional[float] = None
        self.prev_rsi: Optional[float] = None
        
        self.prev_signal: SignalType = SignalType.HOLD
        self.klines = None
        
        self._all_prices: List[float] = []
        self._price_changes: List[float] = []
        self._gains: deque = deque(maxlen=rsi_period)
        self._losses: deque = deque(maxlen=rsi_period)
        self._avg_gain: Optional[float] = None
        self._avg_loss: Optional[float] = None
        
        self._position = 0
        self._entry_price = None
        self._trade_count = 0
        
        self._rsi_filtered_signals = 0
        
        rsi_info = f", RSI周期={rsi_period}, RSI阈值={rsi_threshold}, RSI过滤={'开启' if use_rsi_filter else '关闭'}"
        self.logger.info(f"双均线策略初始化: 短期周期={short_period}, 长期周期={long_period}, 合约={contract}, K线周期={kline_duration}秒, 均线类型={'EMA' if use_ema else 'SMA'}{rsi_info}")

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
    def calculate_sma(prices: List[float], period: int) -> Optional[float]:
        if not prices or len(prices) < period:
            return None
        
        valid_prices = [p for p in prices[-period:] if p is not None and not (isinstance(p, float) and p != p)]
        if len(valid_prices) < period:
            return None
        
        return sum(valid_prices) / period

    @staticmethod
    def calculate_ema(prices: List[float], period: int, prev_ema: Optional[float] = None) -> Optional[float]:
        if not prices:
            return None
        
        if len(prices) < period and prev_ema is None:
            return None
        
        multiplier = 2.0 / (period + 1.0)
        
        if prev_ema is None:
            return DoubleMAStrategy.calculate_sma(prices, period)
        
        current_price = prices[-1]
        if current_price is None or (isinstance(current_price, float) and current_price != current_price):
            return prev_ema
        
        return (current_price - prev_ema) * multiplier + prev_ema

    @staticmethod
    def calculate_ma_with_talib(prices: List[float], period: int, use_ema: bool = False) -> Optional[float]:
        if not TALIB_AVAILABLE or not prices or len(prices) < period:
            return None
        
        valid_prices = [p for p in prices if p is not None and not (isinstance(p, float) and p != p)]
        if len(valid_prices) < period:
            return None
        
        try:
            np_prices = np.array(valid_prices, dtype=float)
            
            if use_ema:
                result = talib.EMA(np_prices, timeperiod=period)
            else:
                result = talib.SMA(np_prices, timeperiod=period)
            
            last_value = result[-1]
            
            if np.isnan(last_value):
                return None
            
            return float(last_value)
        except Exception as e:
            logging.getLogger(__name__).warning(f"talib 计算失败，将使用手动计算: {str(e)}")
            return None

    def _calculate_ma(self, prices: List[float], period: int, use_ema: bool = False) -> Optional[float]:
        if TALIB_AVAILABLE and len(self._all_prices) >= period:
            result = self.calculate_ma_with_talib(self._all_prices, period, use_ema)
            if result is not None:
                return result
        
        if use_ema:
            return self.calculate_ema(prices, period)
        else:
            return self.calculate_sma(prices, period)

    @staticmethod
    def calculate_rsi_with_talib(prices: List[float], period: int) -> Optional[float]:
        if not TALIB_AVAILABLE or not prices or len(prices) < period + 1:
            return None
        
        valid_prices = [p for p in prices if p is not None and not (isinstance(p, float) and p != p)]
        if len(valid_prices) < period + 1:
            return None
        
        try:
            np_prices = np.array(valid_prices, dtype=float)
            result = talib.RSI(np_prices, timeperiod=period)
            
            last_value = result[-1]
            
            if np.isnan(last_value):
                return None
            
            return float(last_value)
        except Exception as e:
            logging.getLogger(__name__).warning(f"talib 计算 RSI 失败，将使用手动计算: {str(e)}")
            return None

    @staticmethod
    def calculate_rsi(
        prices: List[float],
        period: int,
        prev_avg_gain: Optional[float] = None,
        prev_avg_loss: Optional[float] = None
    ) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        if not prices or len(prices) < 2:
            return None, prev_avg_gain, prev_avg_loss
        
        price_change = prices[-1] - prices[-2]
        
        gain = max(price_change, 0.0)
        loss = abs(min(price_change, 0.0))
        
        if len(prices) < period + 1 and prev_avg_gain is None:
            if gain > 0:
                gains = [gain]
            else:
                gains = []
            if loss > 0:
                losses = [loss]
            else:
                losses = []
            return None, gains, losses
        
        if prev_avg_gain is None:
            if len(prices) < period + 1:
                return None, None, None
            
            initial_changes = []
            for i in range(len(prices) - period, len(prices)):
                if i > 0:
                    change = prices[i] - prices[i-1]
                    initial_changes.append(change)
            
            if len(initial_changes) < period:
                return None, None, None
            
            initial_gains = [max(c, 0.0) for c in initial_changes]
            initial_losses = [abs(min(c, 0.0)) for c in initial_changes]
            
            avg_gain = sum(initial_gains) / period
            avg_loss = sum(initial_losses) / period
            
            if avg_loss == 0:
                rsi = 100.0
            else:
                rs = avg_gain / avg_loss
                rsi = 100.0 - (100.0 / (1.0 + rs))
            
            return rsi, avg_gain, avg_loss
        else:
            new_avg_gain = (prev_avg_gain * (period - 1) + gain) / period
            new_avg_loss = (prev_avg_loss * (period - 1) + loss) / period
            
            if new_avg_loss == 0:
                rsi = 100.0
            else:
                rs = new_avg_gain / new_avg_loss
                rsi = 100.0 - (100.0 / (1.0 + rs))
            
            return rsi, new_avg_gain, new_avg_loss

    def _calculate_rsi(self) -> Optional[float]:
        if TALIB_AVAILABLE and len(self._all_prices) >= self.rsi_period + 1:
            result = self.calculate_rsi_with_talib(self._all_prices, self.rsi_period)
            if result is not None:
                return result
        
        if len(self._all_prices) < self.rsi_period + 1:
            return None
        
        rsi, avg_gain, avg_loss = self.calculate_rsi(
            self._all_prices,
            self.rsi_period,
            self._avg_gain,
            self._avg_loss
        )
        
        if isinstance(avg_gain, list):
            if avg_gain:
                self._gains.extend(avg_gain)
            if isinstance(avg_loss, list):
                self._losses.extend(avg_loss)
            return None
        
        self._avg_gain = avg_gain
        self._avg_loss = avg_loss
        
        return rsi

    def update_prices(self, close_price: float):
        if close_price is None or (isinstance(close_price, float) and close_price != close_price):
            self.logger.debug("收到无效的收盘价，跳过")
            return
        
        self.prev_short_ma = self.short_ma
        self.prev_long_ma = self.long_ma
        self.prev_rsi = self.rsi
        
        self.short_prices.append(close_price)
        self.long_prices.append(close_price)
        self._all_prices.append(close_price)
        
        self.short_ma = self._calculate_ma(list(self.short_prices), self.short_period, self.use_ema)
        self.long_ma = self._calculate_ma(list(self.long_prices), self.long_period, self.use_ema)
        
        self.rsi = self._calculate_rsi()
        
        self._detect_signal()

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

    def _detect_signal(self):
        self.prev_signal = self.signal
        
        if self.short_ma is None or self.long_ma is None:
            self.signal = SignalType.HOLD
            return
        
        if self.prev_short_ma is None or self.prev_long_ma is None:
            self.signal = SignalType.HOLD
            rsi_status = f", RSI={self.rsi:.2f}" if self.rsi is not None else ""
            self.logger.debug(f"均线数据准备中... 短期均线={self.short_ma:.2f}, 长期均线={self.long_ma:.2f}{rsi_status}")
            return
        
        short_above_prev = self.prev_short_ma > self.prev_long_ma
        short_above_current = self.short_ma > self.long_ma
        
        short_below_prev = self.prev_short_ma < self.prev_long_ma
        short_below_current = self.short_ma < self.long_ma
        
        ma_signal: SignalType = SignalType.HOLD
        
        if short_below_prev and short_above_current:
            ma_signal = SignalType.BUY
            self.logger.info(f"金叉信号: 短期均线({self.short_ma:.2f}) 上穿 长期均线({self.long_ma:.2f})")
        elif short_above_prev and short_below_current:
            ma_signal = SignalType.SELL
            self.logger.info(f"死叉信号: 短期均线({self.short_ma:.2f}) 下穿 长期均线({self.long_ma:.2f})")
        else:
            ma_signal = SignalType.HOLD
        
        if ma_signal == SignalType.HOLD:
            self.signal = SignalType.HOLD
            return
        
        if not self.use_rsi_filter:
            self.signal = ma_signal
            if self.signal != SignalType.HOLD:
                self._execute_trade(self.signal)
            return
        
        if self.rsi is None:
            self.signal = SignalType.HOLD
            self.logger.debug(f"RSI 数据未准备好，跳过信号: {ma_signal.value}")
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
            self.logger.info(f"RSI 过滤: 原信号={ma_signal.value}, {rsi_status}")
        elif final_signal != SignalType.HOLD:
            self.logger.info(f"信号确认: {final_signal.value}, {rsi_status}")
        
        if self.signal != SignalType.HOLD:
            self._execute_trade(self.signal)

    def _execute_trade(self, signal: SignalType):
        """执行交易下单"""
        if self.api is None:
            self.logger.warning("API 未初始化，无法执行交易")
            return
        
        try:
            quote = self.api.get_quote(self.contract)
            
            if signal == SignalType.BUY:
                if self._position < 0:
                    self._place_order(quote, direction="BUY", offset="CLOSE", volume=abs(self._position))
                    self._position = 0
                
                if self._position == 0:
                    self._place_order(quote, direction="BUY", offset="OPEN", volume=1)
                    self._position = 1
                    
            elif signal == SignalType.SELL:
                if self._position > 0:
                    self._place_order(quote, direction="SELL", offset="CLOSE", volume=self._position)
                    self._position = 0
                
                if self._position == 0:
                    self._place_order(quote, direction="SELL", offset="OPEN", volume=1)
                    self._position = -1
                    
        except Exception as e:
            self.logger.error(f"执行交易失败: {e}")

    def _place_order(self, quote, direction: str, offset: str, volume: int):
        """下单"""
        try:
            order = self.api.insert_order(
                quote,
                direction=direction,
                offset=offset,
                volume=volume,
            )
            self._trade_count += 1
            self.logger.info(f"下单成功: {direction} {offset} {volume}手 {self.contract}, 交易次数={self._trade_count}")
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

    def is_ready(self) -> bool:
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
        """预加载历史 K 线数据，用于计算初始均线和 RSI"""
        required_period = max(self.long_period, self.rsi_period) if self.use_rsi_filter else self.long_period
        
        if len(self.klines) <= required_period:
            self.logger.debug(f"K 线数量不足，跳过预热: {len(self.klines)} < {required_period + 1}")
            return
        
        warmup_count = min(len(self.klines) - 1, max(200, required_period * 3))
        start_idx = max(0, len(self.klines) - warmup_count)
        
        self.logger.info(f"预热 K 线: 从索引 {start_idx} 到 {len(self.klines) - 2}, 共 {warmup_count - 1} 根")
        
        for i in range(start_idx, len(self.klines) - 1):
            try:
                kline = self.klines.iloc[i]
                kline_dict = kline.to_dict() if hasattr(kline, 'to_dict') else dict(kline)
                self.update_from_kline(kline_dict)
            except Exception as e:
                self.logger.warning(f"预热 K 线 {i} 时出错: {e}")
        
        rsi_status = f", RSI={self.rsi:.2f}" if self.rsi is not None else ""
        self.logger.info(f"预热完成: 已收集 {len(self._all_prices)} 条价格记录, 短期均线={self.short_ma}, 长期均线={self.long_ma}{rsi_status}")

    def run(self):
        if not self._initialized:
            self.logger.info("策略未初始化，正在执行初始化...")
            self.initialize()
        
        self.logger.info("双均线策略开始运行")
        self.logger.info(f"短期均线周期: {self.short_period}")
        self.logger.info(f"长期均线周期: {self.long_period}")
        self.logger.info(f"交易合约: {self.contract}")
        self.logger.info(f"均线类型: {'EMA' if self.use_ema else 'SMA'}")
        self.logger.info(f"RSI 过滤: {'开启' if self.use_rsi_filter else '关闭'}")
        if self.use_rsi_filter:
            self.logger.info(f"RSI 周期: {self.rsi_period}")
            self.logger.info(f"RSI 阈值: {self.rsi_threshold}")
        self.logger.info(f"talib 可用: {TALIB_AVAILABLE}")
        
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
            'short_ma': self.short_ma,
            'long_ma': self.long_ma,
            'prev_short_ma': self.prev_short_ma,
            'prev_long_ma': self.prev_long_ma,
            'rsi': self.rsi,
            'prev_rsi': self.prev_rsi,
            '_avg_gain': self._avg_gain,
            '_avg_loss': self._avg_loss,
            'signal': self.signal.value if hasattr(self.signal, 'value') else str(self.signal),
            'prev_signal': self.prev_signal.value if hasattr(self.prev_signal, 'value') else str(self.prev_signal),
            '_all_prices': list(self._all_prices) if self._all_prices else [],
            'short_prices': list(self.short_prices) if self.short_prices else [],
            'long_prices': list(self.long_prices) if self.long_prices else [],
            '_gains': list(self._gains) if self._gains else [],
            '_losses': list(self._losses) if self._losses else [],
            '_rsi_filtered_signals': self._rsi_filtered_signals,
        }
        
        self.logger.info(f"保存策略状态: 已收集 {len(self._all_prices)} 条价格记录, RSI过滤信号数={self._rsi_filtered_signals}")
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
        
        if 'contract' in state:
            if state['contract'] != self.contract:
                self.logger.warning(f"保存的合约 ({state['contract']}) 与当前配置 ({self.contract}) 不一致")
        
        if '_all_prices' in state:
            from collections import deque
            prices = state['_all_prices']
            if prices:
                self._all_prices = [float(p) for p in prices]
                self.logger.info(f"恢复价格历史: {len(self._all_prices)} 条记录")
        
        if 'short_prices' in state:
            from collections import deque
            prices = state['short_prices']
            if prices:
                maxlen = self.short_prices.maxlen if hasattr(self.short_prices, 'maxlen') else None
                self.short_prices = deque([float(p) for p in prices], maxlen=maxlen)
        
        if 'long_prices' in state:
            from collections import deque
            prices = state['long_prices']
            if prices:
                maxlen = self.long_prices.maxlen if hasattr(self.long_prices, 'maxlen') else None
                self.long_prices = deque([float(p) for p in prices], maxlen=maxlen)
        
        if '_gains' in state:
            from collections import deque
            gains = state['_gains']
            if gains:
                maxlen = self._gains.maxlen if hasattr(self._gains, 'maxlen') else None
                self._gains = deque([float(g) for g in gains], maxlen=maxlen)
        
        if '_losses' in state:
            from collections import deque
            losses = state['_losses']
            if losses:
                maxlen = self._losses.maxlen if hasattr(self._losses, 'maxlen') else None
                self._losses = deque([float(l) for l in losses], maxlen=maxlen)
        
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
        
        if '_avg_gain' in state and state['_avg_gain'] is not None:
            self._avg_gain = float(state['_avg_gain'])
        
        if '_avg_loss' in state and state['_avg_loss'] is not None:
            self._avg_loss = float(state['_avg_loss'])
        
        if '_rsi_filtered_signals' in state:
            self._rsi_filtered_signals = int(state['_rsi_filtered_signals'])
        
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
