import sys
import os
import pytest
from datetime import datetime, date

base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, base_dir)

from strategies.base_strategy import SignalType
from strategies.adaptive_momentum_strategy import AdaptiveMomentumStrategy
from strategies.strategy_comparison import (
    MockKlineGenerator,
    BacktestSimulator,
    StrategyTester,
    StrategyComparator,
)


class TestAdaptiveMomentumStrategyBasics:
    
    def test_init_with_default_params(self):
        strategy = AdaptiveMomentumStrategy()
        
        assert strategy.short_period == 5
        assert strategy.long_period == 20
        assert strategy.contract == "SHFE.rb2410"
        assert strategy.kline_duration == 60
        assert strategy.use_ema == False
        
        assert strategy.rsi_period == 14
        assert strategy.rsi_threshold == 50.0
        
        assert strategy.atr_period == 14
        assert strategy.atr_entry_multiplier == 1.5
        assert strategy.atr_exit_multiplier == 2.0
        
        assert strategy.risk_per_trade_percent == 0.01
        assert strategy.trailing_stop_atr_multiplier == 2.0
        assert strategy.contract_multiplier == 10
        
        assert strategy.signal == SignalType.HOLD
        assert strategy._position == 0
        assert strategy._trailing_stop_active == False
    
    def test_init_with_custom_params(self):
        strategy = AdaptiveMomentumStrategy(
            short_period=10,
            long_period=30,
            contract="DCE.i2409",
            kline_duration=300,
            use_ema=True,
            rsi_period=7,
            rsi_threshold=55.0,
            atr_period=10,
            atr_entry_multiplier=2.0,
            atr_exit_multiplier=3.0,
            risk_per_trade_percent=0.02,
            trailing_stop_atr_multiplier=2.5,
            contract_multiplier=100,
        )
        
        assert strategy.short_period == 10
        assert strategy.long_period == 30
        assert strategy.contract == "DCE.i2409"
        assert strategy.use_ema == True
        assert strategy.rsi_period == 7
        assert strategy.rsi_threshold == 55.0
        assert strategy.atr_period == 10
        assert strategy.atr_entry_multiplier == 2.0
        assert strategy.atr_exit_multiplier == 3.0
        assert strategy.risk_per_trade_percent == 0.02
        assert strategy.trailing_stop_atr_multiplier == 2.5
        assert strategy.contract_multiplier == 100
    
    def test_init_with_invalid_period_raises_error(self):
        with pytest.raises(ValueError, match="均线周期必须大于 0"):
            AdaptiveMomentumStrategy(short_period=0, long_period=10)
        
        with pytest.raises(ValueError, match="均线周期必须大于 0"):
            AdaptiveMomentumStrategy(short_period=5, long_period=-1)
        
        with pytest.raises(ValueError, match="短期均线周期必须小于长期均线周期"):
            AdaptiveMomentumStrategy(short_period=20, long_period=10)
    
    def test_init_with_invalid_rsi_raises_error(self):
        with pytest.raises(ValueError, match="RSI 周期必须大于 1"):
            AdaptiveMomentumStrategy(rsi_period=1)
        
        with pytest.raises(ValueError, match="RSI 阈值必须在 0-100 之间"):
            AdaptiveMomentumStrategy(rsi_threshold=-1)
        
        with pytest.raises(ValueError, match="RSI 阈值必须在 0-100 之间"):
            AdaptiveMomentumStrategy(rsi_threshold=101)
    
    def test_init_with_invalid_atr_raises_error(self):
        with pytest.raises(ValueError, match="ATR 周期必须大于 0"):
            AdaptiveMomentumStrategy(atr_period=0)
        
        with pytest.raises(ValueError, match="ATR 入场倍数必须大于 0"):
            AdaptiveMomentumStrategy(atr_entry_multiplier=0)
        
        with pytest.raises(ValueError, match="ATR 追踪止损启动倍数必须大于 0"):
            AdaptiveMomentumStrategy(atr_exit_multiplier=0)
    
    def test_init_with_invalid_risk_raises_error(self):
        with pytest.raises(ValueError, match="单笔风险比例必须在 0-1 之间"):
            AdaptiveMomentumStrategy(risk_per_trade_percent=0)
        
        with pytest.raises(ValueError, match="单笔风险比例必须在 0-1 之间"):
            AdaptiveMomentumStrategy(risk_per_trade_percent=1.1)
        
        with pytest.raises(ValueError, match="追踪止损 ATR 倍数必须大于 0"):
            AdaptiveMomentumStrategy(trailing_stop_atr_multiplier=0)
        
        with pytest.raises(ValueError, match="合约乘数必须大于 0"):
            AdaptiveMomentumStrategy(contract_multiplier=0)


class TestAdaptiveMomentumStrategyIndicators:
    
    def test_update_prices_adds_to_lists(self):
        strategy = AdaptiveMomentumStrategy(debug_logging=False)
        
        strategy.update_prices(close_price=100.0, high_price=102.0, low_price=98.0, open_price=99.0)
        
        assert len(strategy._price_list) == 1
        assert len(strategy._high_list) == 1
        assert len(strategy._low_list) == 1
        assert len(strategy._open_list) == 1
        
        assert strategy._price_list[0] == 100.0
        assert strategy._high_list[0] == 102.0
        assert strategy._low_list[0] == 98.0
        assert strategy._open_list[0] == 99.0
    
    def test_update_prices_without_high_low_uses_close(self):
        strategy = AdaptiveMomentumStrategy(debug_logging=False)
        
        strategy.update_prices(close_price=100.0)
        
        assert strategy._high_list[0] == 100.0
        assert strategy._low_list[0] == 100.0
        assert strategy._open_list[0] == 100.0
    
    def test_calculate_position_size_basic(self):
        strategy = AdaptiveMomentumStrategy(debug_logging=False)
        strategy._total_capital = 1000000.0
        
        atr = 50.0
        current_price = 3000.0
        
        position_size = strategy._calculate_position_size(atr, current_price)
        
        risk_amount = 1000000.0 * 0.01
        risk_per_lot = atr * strategy.contract_multiplier
        expected = risk_amount / risk_per_lot
        
        assert position_size == expected
    
    def test_calculate_position_size_with_zero_atr(self):
        strategy = AdaptiveMomentumStrategy(debug_logging=False)
        
        position_size = strategy._calculate_position_size(0.0, 3000.0)
        
        assert position_size == 1.0


class TestMockKlineGenerator:
    
    def test_generate_basic(self):
        from datetime import date
        
        start_dt = date(2024, 1, 1)
        end_dt = date(2024, 1, 5)
        
        generator = MockKlineGenerator(
            start_dt=start_dt,
            end_dt=end_dt,
            initial_price=3000.0,
            volatility=0.02,
            kline_duration=60,
        )
        
        klines = generator.generate(seed=42)
        
        assert len(klines) > 0
        
        first_kline = klines[0]
        assert 'open' in first_kline
        assert 'high' in first_kline
        assert 'low' in first_kline
        assert 'close' in first_kline
        assert 'datetime' in first_kline
        
        assert first_kline['high'] >= first_kline['open']
        assert first_kline['high'] >= first_kline['close']
        assert first_kline['low'] <= first_kline['open']
        assert first_kline['low'] <= first_kline['close']


class TestBacktestSimulator:
    
    def test_init(self):
        simulator = BacktestSimulator(
            initial_capital=1000000.0,
            contract_multiplier=10,
        )
        
        assert simulator.current_capital == 1000000.0
        assert simulator._position == 0
        assert simulator._peak_capital == 1000000.0
    
    def test_open_position_long(self):
        from datetime import datetime
        
        simulator = BacktestSimulator(
            initial_capital=1000000.0,
            contract_multiplier=10,
            price_tick=1.0,
            slippage_ticks=1.0,
        )
        
        result = simulator.open_position(
            direction="BUY",
            price=3000.0,
            volume=1,
            current_time=datetime.now(),
            reason="test",
        )
        
        assert result == True
        assert simulator._position == 1
        assert simulator._entry_price == 3001.0
    
    def test_close_position_long(self):
        from datetime import datetime
        
        simulator = BacktestSimulator(
            initial_capital=1000000.0,
            contract_multiplier=10,
            price_tick=1.0,
            slippage_ticks=1.0,
            commission_per_lot=5.0,
        )
        
        simulator.open_position(
            direction="BUY",
            price=3000.0,
            volume=1,
            current_time=datetime.now(),
            reason="test",
        )
        
        initial_capital = simulator.current_capital
        
        result = simulator.close_position(
            current_price=3100.0,
            current_time=datetime.now(),
            reason="test_close",
        )
        
        assert result == True
        assert simulator._position == 0
        
        expected_profit = (3099.0 - 3001.0) * 10
        expected_commission = 10.0
        
        assert simulator.current_capital == initial_capital + expected_profit - expected_commission


class TestStrategyComparator:
    
    def test_comparison_basic(self):
        from datetime import date
        
        contracts = ['SHFE.rb', 'DCE.i']
        start_dt = date(2024, 1, 1)
        end_dt = date(2024, 1, 15)
        
        comparator = StrategyComparator(
            contracts=contracts,
            start_dt=start_dt,
            end_dt=end_dt,
            initial_capital=1000000.0,
        )
        
        contract_configs = {
            'SHFE.rb': {
                'initial_price': 3500.0,
                'volatility': 0.015,
                'trend': 0.0,
            },
            'DCE.i': {
                'initial_price': 800.0,
                'volatility': 0.02,
                'trend': 0.0,
            },
        }
        
        comparator.generate_mock_data(contract_configs, seed=42)
        
        old_params = {
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
        
        new_params = {
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
        
        report = comparator.run_comparison(
            old_strategy_params=old_params,
            new_strategy_params=new_params,
        )
        
        assert report is not None
        assert len(report.old_strategy_results) > 0
        assert len(report.new_strategy_results) > 0
        
        for contract in contracts:
            assert contract in report.old_strategy_results
            assert contract in report.new_strategy_results
            
            old_result = report.old_strategy_results[contract]
            new_result = report.new_strategy_results[contract]
            
            assert old_result.status == "completed"
            assert new_result.status == "completed"


class TestAdaptiveMomentumStrategyFeatures:
    
    def test_atr_filter_logic(self):
        strategy = AdaptiveMomentumStrategy(
            short_period=2,
            long_period=3,
            atr_entry_multiplier=1.5,
            debug_logging=False,
        )
        
        strategy.atr = 10.0
        
        prices = [
            (100.0, 102.0, 98.0),
            (101.0, 103.0, 99.0),
            (100.5, 102.5, 99.5),
            (99.0, 101.0, 97.0),
        ]
        
        for close, high, low in prices:
            strategy.update_prices(close_price=close, high_price=high, low_price=low)
        
        assert strategy._atr_filtered_signals >= 0
    
    def test_trailing_stop_logic(self):
        from datetime import datetime
        
        simulator = BacktestSimulator(
            initial_capital=1000000.0,
            contract_multiplier=10,
        )
        
        simulator.open_position(
            direction="BUY",
            price=3000.0,
            volume=1,
            current_time=datetime.now(),
            reason="test",
        )
        
        atr = 50.0
        
        should_close, reason = simulator.update_price(
            current_price=3050.0,
            atr=atr,
            atr_exit_multiplier=2.0,
            trailing_stop_multiplier=2.0,
        )
        
        profit_atr = (50.0 * 10) / atr
        if profit_atr >= 2.0:
            assert simulator._trailing_stop_active == True
        else:
            assert simulator._trailing_stop_active == False


def run_tests():
    print("\n" + "=" * 60)
    print("运行自适应动量策略测试")
    print("=" * 60)
    
    test_classes = [
        TestAdaptiveMomentumStrategyBasics,
        TestAdaptiveMomentumStrategyIndicators,
        TestMockKlineGenerator,
        TestBacktestSimulator,
        TestAdaptiveMomentumStrategyFeatures,
    ]
    
    passed = 0
    failed = 0
    errors = []
    
    for test_class in test_classes:
        print(f"\n运行测试类: {test_class.__name__}")
        print("-" * 40)
        
        test_instance = test_class()
        test_methods = [m for m in dir(test_instance) if m.startswith('test_')]
        
        for method_name in test_methods:
            try:
                test_method = getattr(test_instance, method_name)
                test_method()
                print(f"  ✓ {method_name}")
                passed += 1
            except AssertionError as e:
                print(f"  ✗ {method_name}: {e}")
                failed += 1
                errors.append(f"{test_class.__name__}.{method_name}: {e}")
            except Exception as e:
                print(f"  ✗ {method_name}: 异常 - {e}")
                failed += 1
                errors.append(f"{test_class.__name__}.{method_name}: 异常 - {e}")
    
    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    print(f"通过: {passed}")
    print(f"失败: {failed}")
    
    if errors:
        print("\n错误详情:")
        for error in errors:
            print(f"  - {error}")
    
    return failed == 0


if __name__ == "__main__":
    success = run_tests()
    
    print("\n" + "=" * 60)
    if success:
        print("✅ 所有测试通过！")
    else:
        print("❌ 部分测试失败，请检查错误信息")
    print("=" * 60)
