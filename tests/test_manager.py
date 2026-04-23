import pytest
import sys
import os
from unittest.mock import Mock, patch, MagicMock, call
from abc import ABC

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategies.base_strategy import StrategyBase, SignalType
from strategies.double_ma_strategy import DoubleMAStrategy
from core.manager import StrategyManager


class TestStrategyManagerRegistration:
    def test_register_strategy(self):
        manager = StrategyManager()
        strategy = DoubleMAStrategy(short_period=5, long_period=10)
        
        manager.register_strategy("TestStrategy", strategy)
        
        assert manager.get_strategy("TestStrategy") == strategy
        assert "TestStrategy" in manager.get_all_strategies()
    
    def test_register_duplicate_strategy_overwrites(self):
        manager = StrategyManager()
        strategy1 = DoubleMAStrategy(short_period=5, long_period=10)
        strategy2 = DoubleMAStrategy(short_period=10, long_period=20)
        
        manager.register_strategy("TestStrategy", strategy1)
        manager.register_strategy("TestStrategy", strategy2)
        
        assert manager.get_strategy("TestStrategy") == strategy2
    
    def test_unregister_strategy(self):
        manager = StrategyManager()
        strategy = DoubleMAStrategy(short_period=5, long_period=10)
        
        manager.register_strategy("TestStrategy", strategy)
        result = manager.unregister_strategy("TestStrategy")
        
        assert result is True
        assert manager.get_strategy("TestStrategy") is None
    
    def test_unregister_nonexistent_strategy_returns_false(self):
        manager = StrategyManager()
        
        result = manager.unregister_strategy("Nonexistent")
        
        assert result is False
    
    def test_get_all_strategies(self):
        manager = StrategyManager()
        strategy1 = DoubleMAStrategy(short_period=5, long_period=10)
        strategy2 = DoubleMAStrategy(short_period=10, long_period=20)
        
        manager.register_strategy("Strategy1", strategy1)
        manager.register_strategy("Strategy2", strategy2)
        
        all_strategies = manager.get_all_strategies()
        
        assert len(all_strategies) == 2
        assert "Strategy1" in all_strategies
        assert "Strategy2" in all_strategies


class TestStrategyManagerConfiguration:
    def test_load_strategies_from_config(self):
        config = {
            'strategies': [
                {
                    'name': 'ShortTerm_MA',
                    'class': 'DoubleMAStrategy',
                    'params': {
                        'fast': 5,
                        'slow': 10,
                        'contract': 'SHFE.rb2410',
                        'kline_duration': 60
                    }
                },
                {
                    'name': 'MidTerm_MA',
                    'class': 'DoubleMAStrategy',
                    'params': {
                        'fast': 10,
                        'slow': 20,
                        'contract': 'SHFE.rb2410',
                        'kline_duration': 60
                    }
                }
            ]
        }
        
        manager = StrategyManager()
        manager.load_strategies_from_config(config)
        
        strategies = manager.get_all_strategies()
        
        assert len(strategies) == 2
        assert 'ShortTerm_MA' in strategies
        assert 'MidTerm_MA' in strategies
        
        short_strategy = strategies['ShortTerm_MA']
        assert short_strategy.short_period == 5
        assert short_strategy.long_period == 10
        assert short_strategy.contract == 'SHFE.rb2410'
        
        mid_strategy = strategies['MidTerm_MA']
        assert mid_strategy.short_period == 10
        assert mid_strategy.long_period == 20
        assert mid_strategy.contract == 'SHFE.rb2410'
    
    def test_load_strategies_from_empty_config(self):
        config = {
            'strategies': []
        }
        
        manager = StrategyManager()
        manager.load_strategies_from_config(config)
        
        assert len(manager.get_all_strategies()) == 0
    
    def test_load_strategies_without_strategies_key(self):
        config = {}
        
        manager = StrategyManager()
        manager.load_strategies_from_config(config)
        
        assert len(manager.get_all_strategies()) == 0


class TestStrategyManagerDataDistribution:
    def test_distribute_bar_to_all_strategies(self):
        manager = StrategyManager()
        
        strategy1 = Mock(spec=DoubleMAStrategy)
        strategy1.contract = 'SHFE.rb2410'
        strategy1.short_period = 5
        strategy1.long_period = 10
        
        strategy2 = Mock(spec=DoubleMAStrategy)
        strategy2.contract = 'SHFE.rb2410'
        strategy2.short_period = 10
        strategy2.long_period = 20
        
        manager.register_strategy('Strategy1', strategy1)
        manager.register_strategy('Strategy2', strategy2)
        
        bar_data = {'close': 100.0, 'open': 99.0, 'high': 101.0, 'low': 98.0}
        
        manager._distribute_bar_to_all(bar_data)
        
        strategy1.on_bar.assert_called_once_with(bar_data)
        strategy2.on_bar.assert_called_once_with(bar_data)
    
    def test_distribute_bar_by_contract(self):
        manager = StrategyManager()
        
        strategy1 = Mock(spec=DoubleMAStrategy)
        strategy1.contract = 'SHFE.rb2410'
        
        strategy2 = Mock(spec=DoubleMAStrategy)
        strategy2.contract = 'SHFE.hc2410'
        
        manager.register_strategy('RB_Strategy', strategy1)
        manager.register_strategy('HC_Strategy', strategy2)
        
        rb_bar = {'close': 100.0}
        hc_bar = {'close': 200.0}
        
        manager._distribute_bar('SHFE.rb2410', rb_bar)
        manager._distribute_bar('SHFE.hc2410', hc_bar)
        
        strategy1.on_bar.assert_called_once_with(rb_bar)
        strategy2.on_bar.assert_called_once_with(hc_bar)
        
        assert strategy1.on_bar.call_args == call(rb_bar)
        assert strategy2.on_bar.call_args == call(hc_bar)
    
    def test_distribute_same_kline_to_different_param_strategies(self):
        manager = StrategyManager()
        
        strategy1 = Mock(spec=DoubleMAStrategy)
        strategy1.contract = 'SHFE.rb2410'
        strategy1.short_period = 5
        strategy1.long_period = 10
        
        strategy2 = Mock(spec=DoubleMAStrategy)
        strategy2.contract = 'SHFE.rb2410'
        strategy2.short_period = 10
        strategy2.long_period = 20
        
        manager.register_strategy('ShortTerm', strategy1)
        manager.register_strategy('MidTerm', strategy2)
        
        bar_data = {'close': 100.0, 'open': 99.0, 'high': 101.0, 'low': 98.0}
        
        manager._distribute_bar_to_all(bar_data)
        
        strategy1.on_bar.assert_called_once_with(bar_data)
        strategy2.on_bar.assert_called_once_with(bar_data)
        
        call_args1 = strategy1.on_bar.call_args
        call_args2 = strategy2.on_bar.call_args
        
        assert call_args1 == call_args2
        assert call_args1[0][0] is bar_data
        assert call_args2[0][0] is bar_data
    
    def test_multiple_klines_distributed_correctly(self):
        manager = StrategyManager()
        
        strategy1 = Mock(spec=DoubleMAStrategy)
        strategy1.contract = 'SHFE.rb2410'
        
        strategy2 = Mock(spec=DoubleMAStrategy)
        strategy2.contract = 'SHFE.rb2410'
        
        manager.register_strategy('Strategy1', strategy1)
        manager.register_strategy('Strategy2', strategy2)
        
        bars = [
            {'close': 100.0},
            {'close': 101.0},
            {'close': 102.0},
        ]
        
        for bar in bars:
            manager._distribute_bar_to_all(bar)
        
        assert strategy1.on_bar.call_count == 3
        assert strategy2.on_bar.call_count == 3
        
        expected_calls = [call(bar) for bar in bars]
        strategy1.on_bar.assert_has_calls(expected_calls)
        strategy2.on_bar.assert_has_calls(expected_calls)


class TestStrategyManagerWithRealStrategies:
    def test_same_kline_to_different_param_strategies_produces_different_results(self):
        manager = StrategyManager()
        
        strategy_short = DoubleMAStrategy(short_period=2, long_period=3, contract='SHFE.rb2410')
        strategy_mid = DoubleMAStrategy(short_period=3, long_period=5, contract='SHFE.rb2410')
        
        manager.register_strategy('ShortMA', strategy_short)
        manager.register_strategy('MidMA', strategy_mid)
        
        prices = [10.0, 11.0, 12.0, 13.0, 14.0, 15.0]
        
        for price in prices:
            bar_data = {'close': price}
            manager._distribute_bar_to_all(bar_data)
        
        assert strategy_short.is_ready() is True
        assert strategy_mid.is_ready() is True
        
        assert strategy_short.short_period == 2
        assert strategy_short.long_period == 3
        assert strategy_mid.short_period == 3
        assert strategy_mid.long_period == 5
        
        expected_short_ma = (14.0 + 15.0) / 2
        expected_short_long_ma = (13.0 + 14.0 + 15.0) / 3
        
        expected_mid_ma = (13.0 + 14.0 + 15.0) / 3
        expected_mid_long_ma = (11.0 + 12.0 + 13.0 + 14.0 + 15.0) / 5
        
        assert strategy_short.short_ma == expected_short_ma
        assert strategy_short.long_ma == expected_short_long_ma
        assert strategy_mid.short_ma == expected_mid_ma
        assert strategy_mid.long_ma == expected_mid_long_ma
        
        assert strategy_short.short_ma != strategy_mid.short_ma
        assert strategy_short.long_ma != strategy_mid.long_ma
    
    def test_get_all_signals(self):
        manager = StrategyManager()
        
        strategy1 = DoubleMAStrategy(short_period=2, long_period=3)
        strategy2 = DoubleMAStrategy(short_period=2, long_period=3)
        
        manager.register_strategy('Strategy1', strategy1)
        manager.register_strategy('Strategy2', strategy2)
        
        signals = manager.get_all_signals()
        
        assert len(signals) == 2
        assert 'Strategy1' in signals
        assert 'Strategy2' in signals
        assert signals['Strategy1'] == SignalType.HOLD
        assert signals['Strategy2'] == SignalType.HOLD
    
    def test_get_strategy_state(self):
        manager = StrategyManager()
        
        strategy = DoubleMAStrategy(
            short_period=5, 
            long_period=10, 
            contract='SHFE.rb2410',
            kline_duration=60
        )
        
        manager.register_strategy('TestStrategy', strategy)
        
        state = manager.get_strategy_state('TestStrategy')
        
        assert state is not None
        assert state['name'] == 'TestStrategy'
        assert state['class'] == 'DoubleMAStrategy'
        assert state['short_period'] == 5
        assert state['long_period'] == 10
        assert state['contract'] == 'SHFE.rb2410'
        assert state['kline_duration'] == 60
        assert state['is_ready'] == False
        assert state['signal'] == SignalType.HOLD
    
    def test_get_all_states(self):
        manager = StrategyManager()
        
        strategy1 = DoubleMAStrategy(short_period=5, long_period=10, contract='SHFE.rb2410')
        strategy2 = DoubleMAStrategy(short_period=10, long_period=20, contract='SHFE.hc2410')
        
        manager.register_strategy('Strategy1', strategy1)
        manager.register_strategy('Strategy2', strategy2)
        
        all_states = manager.get_all_states()
        
        assert len(all_states) == 2
        assert 'Strategy1' in all_states
        assert 'Strategy2' in all_states
        
        state1 = all_states['Strategy1']
        state2 = all_states['Strategy2']
        
        assert state1['contract'] == 'SHFE.rb2410'
        assert state1['short_period'] == 5
        assert state1['long_period'] == 10
        
        assert state2['contract'] == 'SHFE.hc2410'
        assert state2['short_period'] == 10
        assert state2['long_period'] == 20


class TestStrategyManagerLifecycle:
    def test_initialize_without_connector_raises_error(self):
        manager = StrategyManager()
        
        with pytest.raises(RuntimeError, match="Connector 未设置"):
            manager.initialize()
    
    def test_initialize_with_connector(self):
        mock_connector = Mock()
        mock_api = Mock()
        mock_kline = Mock()
        mock_connector.get_api.return_value = mock_api
        mock_api.get_kline_serial.return_value = mock_kline
        
        manager = StrategyManager(connector=mock_connector)
        
        strategy = DoubleMAStrategy(connector=mock_connector)
        manager.register_strategy('TestStrategy', strategy)
        
        manager.initialize()
        
        assert manager._initialized is True
        assert strategy._initialized is True
    
    def test_stop_all(self):
        manager = StrategyManager()
        
        strategy1 = Mock(spec=DoubleMAStrategy)
        strategy2 = Mock(spec=DoubleMAStrategy)
        
        manager.register_strategy('Strategy1', strategy1)
        manager.register_strategy('Strategy2', strategy2)
        
        manager.stop_all()
        
        strategy1.stop.assert_called_once()
        strategy2.stop.assert_called_once()
    
    def test_set_connector(self):
        manager = StrategyManager()
        
        mock_connector = Mock()
        mock_api = Mock()
        mock_connector.get_api.return_value = mock_api
        
        manager.set_connector(mock_connector)
        
        assert manager.connector == mock_connector
        assert manager.api == mock_api
    
    def test_set_connector_none_raises_error(self):
        manager = StrategyManager()
        
        with pytest.raises(ValueError, match="Connector 不能为 None"):
            manager.set_connector(None)


class TestStrategyManagerIntegration:
    def test_complete_workflow(self):
        mock_connector = Mock()
        mock_api = Mock()
        mock_kline = Mock()
        mock_connector.get_api.return_value = mock_api
        mock_api.get_kline_serial.return_value = mock_kline
        
        config = {
            'strategies': [
                {
                    'name': 'ShortMA',
                    'class': 'DoubleMAStrategy',
                    'params': {
                        'fast': 2,
                        'slow': 3,
                        'contract': 'SHFE.rb2410',
                        'kline_duration': 60
                    }
                },
                {
                    'name': 'MidMA',
                    'class': 'DoubleMAStrategy',
                    'params': {
                        'fast': 3,
                        'slow': 5,
                        'contract': 'SHFE.rb2410',
                        'kline_duration': 60
                    }
                }
            ]
        }
        
        manager = StrategyManager(connector=mock_connector)
        manager.load_strategies_from_config(config)
        
        assert len(manager.get_all_strategies()) == 2
        
        manager.initialize()
        
        assert manager._initialized is True
        
        prices = [10.0, 11.0, 12.0, 13.0, 14.0, 15.0]
        for price in prices:
            bar_data = {'close': price}
            manager._distribute_bar_to_all(bar_data)
        
        short_strategy = manager.get_strategy('ShortMA')
        mid_strategy = manager.get_strategy('MidMA')
        
        assert short_strategy.is_ready() is True
        assert mid_strategy.is_ready() is True
        
        signals = manager.get_all_signals()
        assert 'ShortMA' in signals
        assert 'MidMA' in signals
        
        all_states = manager.get_all_states()
        assert len(all_states) == 2
        
        manager.stop_all()
        
        short_strategy = manager.get_strategy('ShortMA')
        mid_strategy = manager.get_strategy('MidMA')
        
        assert short_strategy.is_ready() is True
        assert mid_strategy.is_ready() is True
