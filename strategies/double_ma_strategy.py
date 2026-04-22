import logging
from typing import Optional, List, Dict, Any
from collections import deque
from enum import Enum


class SignalType(Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class DoubleMAStrategy:
    def __init__(
        self,
        short_period: int = 5,
        long_period: int = 10,
        contract: str = "SHFE.rb2410",
        kline_duration: str = 60,
    ):
        self.short_period = short_period
        self.long_period = long_period
        self.contract = contract
        self.kline_duration = kline_duration
        
        self.logger = logging.getLogger(__name__)
        
        self.short_prices: deque = deque(maxlen=short_period)
        self.long_prices: deque = deque(maxlen=long_period)
        
        self.short_ma: Optional[float] = None
        self.long_ma: Optional[float] = None
        
        self.prev_short_ma: Optional[float] = None
        self.prev_long_ma: Optional[float] = None
        
        self.current_signal: SignalType = SignalType.HOLD
        self.prev_signal: SignalType = SignalType.HOLD
        
        self.api = None
        self.klines = None
        self.position = 0

    def set_api(self, api):
        self.api = api

    def subscribe(self):
        if not self.api:
            raise RuntimeError("API 未设置，请先调用 set_api()")
        
        self.logger.info(f"订阅合约: {self.contract}, K线周期: {self.kline_duration}秒")
        self.klines = self.api.get_kline_serial(self.contract, self.kline_duration)

    @staticmethod
    def calculate_sma(prices: List[float], period: int) -> Optional[float]:
        if not prices or len(prices) < period:
            return None
        
        valid_prices = [p for p in prices[-period:] if p is not None]
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
        if current_price is None:
            return prev_ema
        
        return (current_price - prev_ema) * multiplier + prev_ema

    def update_prices(self, close_price: float):
        if close_price is None:
            return
        
        self.prev_short_ma = self.short_ma
        self.prev_long_ma = self.long_ma
        
        self.short_prices.append(close_price)
        self.long_prices.append(close_price)
        
        self.short_ma = self.calculate_sma(list(self.short_prices), self.short_period)
        self.long_ma = self.calculate_sma(list(self.long_prices), self.long_period)
        
        self._detect_signal()

    def update_from_kline(self, kline_data: Dict[str, Any]):
        close_price = kline_data.get('close')
        if close_price is not None:
            self.update_prices(float(close_price))

    def _detect_signal(self):
        self.prev_signal = self.current_signal
        
        if self.short_ma is None or self.long_ma is None:
            self.current_signal = SignalType.HOLD
            return
        
        if self.prev_short_ma is None or self.prev_long_ma is None:
            self.current_signal = SignalType.HOLD
            return
        
        short_above_prev = self.prev_short_ma > self.prev_long_ma
        short_above_current = self.short_ma > self.long_ma
        
        short_below_prev = self.prev_short_ma < self.prev_long_ma
        short_below_current = self.short_ma < self.long_ma
        
        if short_below_prev and short_above_current:
            self.current_signal = SignalType.BUY
            self.logger.info(f"金叉信号: 短期均线({self.short_ma:.2f}) 上穿 长期均线({self.long_ma:.2f})")
        elif short_above_prev and short_below_current:
            self.current_signal = SignalType.SELL
            self.logger.info(f"死叉信号: 短期均线({self.short_ma:.2f}) 下穿 长期均线({self.long_ma:.2f})")
        else:
            self.current_signal = SignalType.HOLD

    def get_signal(self) -> SignalType:
        return self.current_signal

    def get_ma_values(self) -> Dict[str, Optional[float]]:
        return {
            f"ma_{self.short_period}": self.short_ma,
            f"ma_{self.long_period}": self.long_ma,
            f"prev_ma_{self.short_period}": self.prev_short_ma,
            f"prev_ma_{self.long_period}": self.prev_long_ma,
        }

    def is_ready(self) -> bool:
        return self.short_ma is not None and self.long_ma is not None

    def on_tick(self):
        if self.klines is None:
            return
        
        if len(self.klines) > 0:
            latest_kline = self.klines.iloc[-1]
            self.update_from_kline(latest_kline.to_dict())

    def run(self):
        if not self.api:
            raise RuntimeError("API 未设置，请先调用 set_api()")
        
        self.subscribe()
        
        self.logger.info("双均线策略启动")
        self.logger.info(f"短期均线周期: {self.short_period}")
        self.logger.info(f"长期均线周期: {self.long_period}")
        self.logger.info(f"交易合约: {self.contract}")
        
        while True:
            self.api.wait_update()
            self.on_tick()
            
            if self.is_ready():
                signal = self.get_signal()
                if signal != SignalType.HOLD:
                    self.logger.info(f"策略信号: {signal.value}")
