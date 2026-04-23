import pytest
import sys
import os
from unittest.mock import Mock, patch, MagicMock
from abc import ABC

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategies.base_strategy import StrategyBase, SignalType
from strategies.double_ma_strategy import DoubleMAStrategy, TALIB_AVAILABLE


class TestStrategyBase:
    def test_strategy_base_is_abstract(self):
        assert ABC in StrategyBase.__bases__
    
    def test_signal_type_enum(self):
        assert SignalType.BUY.value == "BUY"
        assert SignalType.SELL.value == "SELL"
        assert SignalType.HOLD.value == "HOLD"
    
    def test_init_without_connector(self):
        strategy = DoubleMAStrategy()
        assert strategy.connector is None
        assert strategy.api is None
        assert strategy.signal == SignalType.HOLD
    
    def test_init_with_connector(self):
        mock_connector = Mock()
        mock_api = Mock()
        mock_connector.get_api.return_value = mock_api
        
        strategy = DoubleMAStrategy(connector=mock_connector)
        
        assert strategy.connector == mock_connector
        assert strategy.api == mock_api
    
    def test_set_connector(self):
        mock_connector = Mock()
        mock_api = Mock()
        mock_connector.get_api.return_value = mock_api
        
        strategy = DoubleMAStrategy()
        strategy.set_connector(mock_connector)
        
        assert strategy.connector == mock_connector
        assert strategy.api == mock_api
    
    def test_set_connector_none_raises_error(self):
        strategy = DoubleMAStrategy()
        
        with pytest.raises(ValueError, match="Connector 不能为 None"):
            strategy.set_connector(None)
    
    def test_get_api(self):
        mock_connector = Mock()
        mock_api = Mock()
        mock_connector.get_api.return_value = mock_api
        
        strategy = DoubleMAStrategy(connector=mock_connector)
        
        assert strategy.get_api() == mock_api


class TestDoubleMAStrategy:
    def test_init(self):
        strategy = DoubleMAStrategy(
            short_period=5,
            long_period=10,
            contract="SHFE.rb2410",
            kline_duration=60,
        )
        
        assert strategy.short_period == 5
        assert strategy.long_period == 10
        assert strategy.contract == "SHFE.rb2410"
        assert strategy.kline_duration == 60
        assert strategy.short_ma is None
        assert strategy.long_ma is None
        assert strategy.signal == SignalType.HOLD
        assert strategy.is_ready() is False
    
    def test_init_with_invalid_period_raises_error(self):
        with pytest.raises(ValueError, match="均线周期必须大于 0"):
            DoubleMAStrategy(short_period=0, long_period=10)
        
        with pytest.raises(ValueError, match="均线周期必须大于 0"):
            DoubleMAStrategy(short_period=5, long_period=-1)
        
        with pytest.raises(ValueError, match="短期均线周期必须小于长期均线周期"):
            DoubleMAStrategy(short_period=10, long_period=5)
    
    def test_calculate_sma_basic(self):
        prices = [1.0, 2.0, 3.0, 4.0, 5.0]
        period = 5
        
        result = DoubleMAStrategy.calculate_sma(prices, period)
        
        assert result == 3.0
    
    def test_calculate_sma_with_different_period(self):
        prices = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        period = 5
        
        result = DoubleMAStrategy.calculate_sma(prices, period)
        
        assert result == (6.0 + 7.0 + 8.0 + 9.0 + 10.0) / 5
    
    def test_calculate_sma_insufficient_data(self):
        prices = [1.0, 2.0, 3.0]
        period = 5
        
        result = DoubleMAStrategy.calculate_sma(prices, period)
        
        assert result is None
    
    def test_calculate_sma_empty_prices(self):
        prices = []
        period = 5
        
        result = DoubleMAStrategy.calculate_sma(prices, period)
        
        assert result is None
    
    def test_calculate_sma_with_none_values(self):
        prices = [1.0, None, 3.0, 4.0, 5.0]
        period = 5
        
        result = DoubleMAStrategy.calculate_sma(prices, period)
        
        assert result is None
    
    def test_calculate_ema_basic(self):
        prices = [1.0, 2.0, 3.0, 4.0, 5.0]
        period = 5
        
        result = DoubleMAStrategy.calculate_ema(prices, period)
        
        assert result == 3.0
    
    def test_calculate_ema_with_prev_ema(self):
        prices = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        period = 5
        prev_ema = 3.0
        
        multiplier = 2.0 / (5 + 1.0)
        expected = (6.0 - 3.0) * multiplier + 3.0
        
        result = DoubleMAStrategy.calculate_ema(prices, period, prev_ema)
        
        assert result == expected
    
    def test_update_prices_single(self):
        strategy = DoubleMAStrategy(short_period=3, long_period=5)
        
        strategy.update_prices(10.0)
        
        assert len(strategy.short_prices) == 1
        assert len(strategy.long_prices) == 1
        assert strategy.short_ma is None
        assert strategy.long_ma is None
        assert strategy.is_ready() is False
    
    def test_update_prices_enough_for_short_ma(self):
        strategy = DoubleMAStrategy(short_period=3, long_period=5)
        
        strategy.update_prices(10.0)
        strategy.update_prices(11.0)
        strategy.update_prices(12.0)
        
        assert strategy.short_ma == (10.0 + 11.0 + 12.0) / 3
        assert strategy.long_ma is None
        assert strategy.is_ready() is False
    
    def test_update_prices_enough_for_both_ma(self):
        strategy = DoubleMAStrategy(short_period=3, long_period=5)
        
        prices = [10.0, 11.0, 12.0, 13.0, 14.0]
        for p in prices:
            strategy.update_prices(p)
        
        assert strategy.short_ma == (12.0 + 13.0 + 14.0) / 3
        assert strategy.long_ma == sum(prices) / 5
        assert strategy.is_ready() is True
    
    def test_update_prices_sliding_window(self):
        strategy = DoubleMAStrategy(short_period=3, long_period=5)
        
        prices = [10.0, 11.0, 12.0, 13.0, 14.0, 15.0]
        for p in prices:
            strategy.update_prices(p)
        
        assert strategy.short_ma == (13.0 + 14.0 + 15.0) / 3
        assert strategy.long_ma == (11.0 + 12.0 + 13.0 + 14.0 + 15.0) / 5
    
    def test_golden_cross_signal(self):
        strategy = DoubleMAStrategy(short_period=2, long_period=3)
        
        strategy.update_prices(10.0)
        strategy.update_prices(11.0)
        strategy.update_prices(9.0)
        
        assert strategy.short_ma == (11.0 + 9.0) / 2
        assert strategy.long_ma == (10.0 + 11.0 + 9.0) / 3
        
        strategy.update_prices(15.0)
        
        assert strategy.prev_short_ma == (11.0 + 9.0) / 2
        assert strategy.prev_long_ma == (10.0 + 11.0 + 9.0) / 3
        assert strategy.short_ma == (9.0 + 15.0) / 2
        assert strategy.long_ma == (11.0 + 9.0 + 15.0) / 3
        
        prev_short_below = strategy.prev_short_ma < strategy.prev_long_ma
        current_short_above = strategy.short_ma > strategy.long_ma
        
        if prev_short_below and current_short_above:
            assert strategy.signal == SignalType.BUY
    
    def test_death_cross_signal(self):
        strategy = DoubleMAStrategy(short_period=2, long_period=3)
        
        strategy.update_prices(10.0)
        strategy.update_prices(15.0)
        strategy.update_prices(12.0)
        
        strategy.update_prices(5.0)
        
        prev_short_above = strategy.prev_short_ma > strategy.prev_long_ma
        current_short_below = strategy.short_ma < strategy.long_ma
        
        if prev_short_above and current_short_below:
            assert strategy.signal == SignalType.SELL
    
    def test_hold_signal_when_no_cross(self):
        strategy = DoubleMAStrategy(short_period=2, long_period=3)
        
        strategy.update_prices(10.0)
        strategy.update_prices(11.0)
        strategy.update_prices(12.0)
        strategy.update_prices(13.0)
        
        assert strategy.signal == SignalType.HOLD
    
    def test_get_ma_values(self):
        strategy = DoubleMAStrategy(short_period=2, long_period=3)
        
        strategy.update_prices(10.0)
        strategy.update_prices(11.0)
        strategy.update_prices(12.0)
        
        ma_values = strategy.get_ma_values()
        
        assert "ma_2" in ma_values
        assert "ma_3" in ma_values
        assert "prev_ma_2" in ma_values
        assert "prev_ma_3" in ma_values
        assert ma_values["ma_2"] == (11.0 + 12.0) / 2
        assert ma_values["ma_3"] == (10.0 + 11.0 + 12.0) / 3
    
    def test_update_from_kline(self):
        strategy = DoubleMAStrategy(short_period=2, long_period=3)
        
        strategy.update_from_kline({'close': 10.0})
        strategy.update_from_kline({'close': 11.0})
        strategy.update_from_kline({'close': 12.0})
        
        assert strategy.short_ma == (11.0 + 12.0) / 2
        assert strategy.long_ma == (10.0 + 11.0 + 12.0) / 3
    
    def test_update_from_kline_none_close(self):
        strategy = DoubleMAStrategy(short_period=2, long_period=3)
        
        strategy.update_from_kline({'close': None})
        
        assert len(strategy.short_prices) == 0
        assert len(strategy.long_prices) == 0
    
    def test_update_from_kline_none_data(self):
        strategy = DoubleMAStrategy(short_period=2, long_period=3)
        
        strategy.update_from_kline(None)
        
        assert len(strategy.short_prices) == 0
        assert len(strategy.long_prices) == 0
    
    def test_subscribe_with_api(self):
        strategy = DoubleMAStrategy()
        mock_api = Mock()
        mock_kline = Mock()
        mock_api.get_kline_serial.return_value = mock_kline
        
        strategy.api = mock_api
        strategy.subscribe()
        
        mock_api.get_kline_serial.assert_called_once_with("SHFE.rb2410", 60)
        assert strategy.klines == mock_kline
    
    def test_subscribe_without_api_raises_error(self):
        strategy = DoubleMAStrategy()
        
        with pytest.raises(RuntimeError, match="API 未设置"):
            strategy.subscribe()
    
    def test_on_bar(self):
        strategy = DoubleMAStrategy(short_period=2, long_period=3)
        
        strategy.on_bar({'close': 10.0})
        strategy.on_bar({'close': 11.0})
        strategy.on_bar({'close': 12.0})
        
        assert strategy.short_ma == (11.0 + 12.0) / 2
        assert strategy.long_ma == (10.0 + 11.0 + 12.0) / 3
    
    def test_on_bar_none_data(self):
        strategy = DoubleMAStrategy(short_period=2, long_period=3)
        
        strategy.on_bar(None)
        
        assert len(strategy.short_prices) == 0
        assert len(strategy.long_prices) == 0
    
    @pytest.mark.parametrize("prices,period,expected", [
        ([1, 2, 3, 4, 5], 5, 3.0),
        ([2, 4, 6, 8, 10], 5, 6.0),
        ([1, 1, 1, 1, 1], 5, 1.0),
        ([10, 20, 30], 3, 20.0),
    ])
    def test_calculate_sma_parametrized(self, prices, period, expected):
        result = DoubleMAStrategy.calculate_sma(prices, period)
        assert result == expected
    
    def test_signal_transition_hold_to_buy(self):
        strategy = DoubleMAStrategy(short_period=2, long_period=3)
        
        strategy.update_prices(20.0)
        strategy.update_prices(15.0)
        strategy.update_prices(18.0)
        
        assert strategy.signal == SignalType.HOLD
        
        strategy.update_prices(25.0)
        
        if strategy.prev_short_ma < strategy.prev_long_ma and strategy.short_ma > strategy.long_ma:
            assert strategy.signal == SignalType.BUY
            assert strategy.prev_signal == SignalType.HOLD
    
    def test_initialize_without_connector_raises_error(self):
        strategy = DoubleMAStrategy()
        
        with pytest.raises(RuntimeError, match="Connector 未设置"):
            strategy.initialize()
    
    def test_initialize_with_connector(self):
        mock_connector = Mock()
        mock_api = Mock()
        mock_kline = Mock()
        mock_connector.get_api.return_value = mock_api
        mock_api.get_kline_serial.return_value = mock_kline
        
        strategy = DoubleMAStrategy(connector=mock_connector)
        strategy.initialize()
        
        assert strategy._initialized is True
        mock_api.get_kline_serial.assert_called_once()
    
    def test_double_initialization_skips(self):
        mock_connector = Mock()
        mock_api = Mock()
        mock_kline = Mock()
        mock_connector.get_api.return_value = mock_api
        mock_api.get_kline_serial.return_value = mock_kline
        
        strategy = DoubleMAStrategy(connector=mock_connector)
        strategy.initialize()
        strategy.initialize()
        
        mock_api.get_kline_serial.assert_called_once()


class TestTalibIntegration:
    def test_talib_available_flag(self):
        assert isinstance(TALIB_AVAILABLE, bool)
    
    def test_calculate_ma_with_talib_insufficient_data(self):
        prices = [1.0, 2.0, 3.0]
        period = 5
        
        result = DoubleMAStrategy.calculate_ma_with_talib(prices, period)
        
        assert result is None
    
    def test_calculate_ma_with_talib_empty_prices(self):
        prices = []
        period = 5
        
        result = DoubleMAStrategy.calculate_ma_with_talib(prices, period)
        
        assert result is None
