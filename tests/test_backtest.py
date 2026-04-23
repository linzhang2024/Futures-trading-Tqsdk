import pytest
import sys
import os
from datetime import date
from unittest.mock import Mock, patch, MagicMock, call
from typing import Dict, Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.backtest import (
    BacktestEngine,
    BacktestResult,
    BacktestMode,
    ParameterRange,
    PerformanceMetrics,
    CostConfig,
    PerformanceConfig,
)
from strategies.base_strategy import StrategyBase, SignalType
from strategies.double_ma_strategy import DoubleMAStrategy
from core.risk_manager import RiskManager, RiskLevel


class TestParameterRange:
    def test_parameter_range_int(self):
        pr = ParameterRange(
            name='short_period',
            min_val=5,
            max_val=10,
            step=1,
            param_type=int
        )
        
        assert pr.name == 'short_period'
        assert pr.min_val == 5
        assert pr.max_val == 10
        assert pr.step == 1
        assert pr.param_type == int

    def test_parameter_range_float(self):
        pr = ParameterRange(
            name='some_float',
            min_val=0.5,
            max_val=2.0,
            step=0.5,
            param_type=float
        )
        
        assert pr.name == 'some_float'
        assert pr.min_val == 0.5
        assert pr.max_val == 2.0
        assert pr.step == 0.5
        assert pr.param_type == float


class TestPerformanceMetrics:
    def test_performance_metrics_default_values(self):
        metrics = PerformanceMetrics()
        
        assert metrics.total_return == 0.0
        assert metrics.total_return_percent == 0.0
        assert metrics.annualized_return == 0.0
        assert metrics.annualized_return_percent == 0.0
        assert metrics.max_drawdown == 0.0
        assert metrics.max_drawdown_percent == 0.0
        assert metrics.sharpe_ratio == 0.0
        assert metrics.sortino_ratio == 0.0
        assert metrics.calmar_ratio == 0.0
        assert metrics.total_trades == 0
        assert metrics.win_rate == 0.0
        assert metrics.profit_factor == 0.0
        assert metrics.total_commission_cost == 0.0
        assert metrics.total_slippage_cost == 0.0
        assert metrics.total_cost == 0.0

    def test_performance_metrics_with_values(self):
        metrics = PerformanceMetrics(
            total_return=100000.0,
            total_return_percent=10.0,
            annualized_return=0.40,
            annualized_return_percent=40.0,
            max_drawdown=50000.0,
            max_drawdown_percent=5.0,
            sharpe_ratio=1.5,
            sortino_ratio=2.0,
            calmar_ratio=8.0,
            total_trades=20,
            winning_trades=12,
            losing_trades=8,
            win_rate=60.0,
            profit_factor=1.5,
            total_commission_cost=100.0,
            total_slippage_cost=200.0,
            total_cost=300.0,
        )
        
        assert metrics.total_return == 100000.0
        assert metrics.total_return_percent == 10.0
        assert metrics.annualized_return_percent == 40.0
        assert metrics.max_drawdown_percent == 5.0
        assert metrics.sharpe_ratio == 1.5
        assert metrics.sortino_ratio == 2.0
        assert metrics.calmar_ratio == 8.0
        assert metrics.total_trades == 20
        assert metrics.win_rate == 60.0
        assert metrics.profit_factor == 1.5


class TestCostConfig:
    def test_cost_config_default_values(self):
        config = CostConfig()
        
        assert config.default_commission_per_lot == 0.0
        assert config.default_slippage_points == 0.0
        assert config.contract_configs == {}

    def test_cost_config_get_commission_default(self):
        config = CostConfig(
            default_commission_per_lot=5.0,
            default_slippage_points=1.0,
        )
        
        assert config.get_commission('unknown_contract') == 5.0
        assert config.get_slippage('unknown_contract') == 1.0

    def test_cost_config_get_contract_specific(self):
        config = CostConfig(
            default_commission_per_lot=5.0,
            default_slippage_points=1.0,
            contract_configs={
                'SHFE.rb2410': {
                    'commission_per_lot': 3.0,
                    'slippage_points': 0.5,
                }
            }
        )
        
        assert config.get_commission('SHFE.rb2410') == 3.0
        assert config.get_slippage('SHFE.rb2410') == 0.5
        
        assert config.get_commission('SHFE.hc2410') == 5.0
        assert config.get_slippage('SHFE.hc2410') == 1.0


class TestBacktestResult:
    def test_backtest_result_default_values(self):
        result = BacktestResult(
            strategy_name='TestStrategy',
            params={'param1': 1, 'param2': 2},
            start_dt=date(2024, 1, 1),
            end_dt=date(2024, 3, 31),
        )
        
        assert result.strategy_name == 'TestStrategy'
        assert result.params == {'param1': 1, 'param2': 2}
        assert result.start_dt == date(2024, 1, 1)
        assert result.end_dt == date(2024, 3, 31)
        
        assert result.initial_equity == 0.0
        assert result.final_equity == 0.0
        assert result.performance.total_return == 0.0
        assert result.performance.max_drawdown == 0.0
        assert result.performance.total_trades == 0
        assert result.risk_triggered == False
        assert result.frozen_during_backtest == False
        assert result.status == 'completed'
        assert result.equity_curve == []
        assert result.trade_records == []
        assert result.risk_events == []

    def test_backtest_result_with_performance(self):
        performance = PerformanceMetrics(
            total_return=100000.0,
            total_return_percent=10.0,
            max_drawdown_percent=5.0,
            total_trades=20,
            win_rate=60.0,
            profit_factor=1.5,
        )
        
        result = BacktestResult(
            strategy_name='DoubleMAStrategy',
            params={'short_period': 5, 'long_period': 10},
            start_dt=date(2024, 1, 1),
            end_dt=date(2024, 3, 31),
            initial_equity=1000000.0,
            final_equity=1100000.0,
            performance=performance,
            risk_triggered=False,
            frozen_during_backtest=False,
            status='completed',
        )
        
        assert result.performance.total_return_percent == 10.0
        assert result.performance.max_drawdown_percent == 5.0
        assert result.performance.total_trades == 20
        assert result.performance.win_rate == 60.0
        assert result.performance.profit_factor == 1.5


class TestBacktestEngineInitialization:
    def test_engine_initialization_without_config(self):
        engine = BacktestEngine()
        
        assert engine.config == {}
        assert engine._results == []
        assert engine._best_result is None

    def test_engine_initialization_with_config(self):
        config = {
            'backtest': {
                'init_balance': 2000000.0,
                'start_dt': '2024-01-01',
                'end_dt': '2024-03-31',
            }
        }
        
        engine = BacktestEngine(config=config)
        
        assert engine.config == config
        assert engine._init_balance == 2000000.0

    def test_engine_initialization_with_tq_credentials(self):
        config = {
            'backtest': {
                'init_balance': 1000000.0,
                'tq_account': 'test_account',
                'tq_password': 'test_password',
            }
        }
        
        engine = BacktestEngine(config=config)
        
        assert engine._tq_account == 'test_account'
        assert engine._tq_password == 'test_password'

    def test_engine_initialization_with_cost_config(self):
        config = {
            'backtest': {
                'init_balance': 1000000.0,
                'costs': {
                    'default_commission_per_lot': 5.0,
                    'default_slippage_points': 1.0,
                    'contracts': {
                        'SHFE.rb2410': {
                            'commission_per_lot': 3.0,
                            'slippage_points': 0.5,
                        }
                    }
                }
            }
        }
        
        engine = BacktestEngine(config=config)
        
        assert engine._cost_config.default_commission_per_lot == 5.0
        assert engine._cost_config.default_slippage_points == 1.0
        assert engine._cost_config.get_commission('SHFE.rb2410') == 3.0

    def test_engine_initialization_with_performance_config(self):
        config = {
            'backtest': {
                'init_balance': 1000000.0,
                'performance': {
                    'risk_free_rate': 0.04,
                    'trading_days_per_year': 250,
                }
            }
        }
        
        engine = BacktestEngine(config=config)
        
        assert engine._performance_config.risk_free_rate == 0.04
        assert engine._performance_config.trading_days_per_year == 250

    def test_engine_initialization_with_max_workers(self):
        config = {
            'backtest': {
                'init_balance': 1000000.0,
                'optimization': {
                    'max_workers': 2,
                }
            }
        }
        
        engine = BacktestEngine(config=config)
        
        assert engine._max_workers == 2


class TestBacktestEngineDateExtraction:
    def test_extract_date_range_from_config(self):
        config = {
            'backtest': {
                'start_dt': '2024-01-01',
                'end_dt': '2024-03-31',
            }
        }
        
        engine = BacktestEngine(config=config)
        start_dt, end_dt = engine._extract_date_range()
        
        assert start_dt == date(2024, 1, 1)
        assert end_dt == date(2024, 3, 31)

    def test_extract_date_range_different_formats(self):
        test_cases = [
            (('2024-01-01', '2024-03-31', date(2024, 1, 1), date(2024, 3, 31))),
            (('20240101', '20240331', date(2024, 1, 1), date(2024, 3, 31))),
            (('2024/01/01', '2024/03/31', date(2024, 1, 1), date(2024, 3, 31))),
        ]
        
        for start_str, end_str, expected_start, expected_end in test_cases:
            config = {
                'backtest': {
                    'start_dt': start_str,
                    'end_dt': end_str,
                }
            }
            
            engine = BacktestEngine(config=config)
            start_dt, end_dt = engine._extract_date_range()
            
            assert start_dt == expected_start
            assert end_dt == expected_end

    def test_extract_date_range_missing_start_raises_error(self):
        config = {
            'backtest': {
                'end_dt': '2024-03-31',
            }
        }
        
        engine = BacktestEngine(config=config)
        
        with pytest.raises(ValueError, match="缺少回测时间段"):
            engine._extract_date_range()

    def test_extract_date_range_start_after_end_raises_error(self):
        config = {
            'backtest': {
                'start_dt': '2024-03-31',
                'end_dt': '2024-01-01',
            }
        }
        
        engine = BacktestEngine(config=config)
        
        with pytest.raises(ValueError, match="必须早于结束日期"):
            engine._extract_date_range()


class TestBacktestEngineParamGrid:
    def test_generate_param_grid_single_param(self):
        engine = BacktestEngine()
        
        param_ranges = {
            'short_period': ParameterRange(
                name='short_period',
                min_val=5,
                max_val=7,
                step=1,
                param_type=int
            ),
        }
        
        grids = engine._generate_param_grid(param_ranges)
        
        assert len(grids) == 3
        assert {'short_period': 5} in grids
        assert {'short_period': 6} in grids
        assert {'short_period': 7} in grids

    def test_generate_param_grid_multiple_params(self):
        engine = BacktestEngine()
        
        param_ranges = {
            'short_period': ParameterRange(
                name='short_period',
                min_val=5,
                max_val=6,
                step=1,
                param_type=int
            ),
            'long_period': ParameterRange(
                name='long_period',
                min_val=10,
                max_val=12,
                step=2,
                param_type=int
            ),
        }
        
        grids = engine._generate_param_grid(param_ranges)
        
        assert len(grids) == 4
        
        expected_combinations = [
            {'short_period': 5, 'long_period': 10},
            {'short_period': 5, 'long_period': 12},
            {'short_period': 6, 'long_period': 10},
            {'short_period': 6, 'long_period': 12},
        ]
        
        for combo in expected_combinations:
            assert combo in grids

    def test_generate_param_grid_float_step(self):
        engine = BacktestEngine()
        
        param_ranges = {
            'some_param': ParameterRange(
                name='some_param',
                min_val=0.0,
                max_val=1.0,
                step=0.5,
                param_type=float
            ),
        }
        
        grids = engine._generate_param_grid(param_ranges)
        
        assert len(grids) == 3
        assert {'some_param': 0.0} in grids
        assert {'some_param': 0.5} in grids
        assert {'some_param': 1.0} in grids


class TestBacktestEngineResults:
    def test_get_results_empty(self):
        engine = BacktestEngine()
        
        results = engine.get_results()
        
        assert results == []
        assert engine._best_result is None

    def test_get_results_with_data(self):
        engine = BacktestEngine()
        
        result1 = BacktestResult(
            strategy_name='Strategy1',
            params={'a': 1},
            start_dt=date(2024, 1, 1),
            end_dt=date(2024, 3, 31),
        )
        result2 = BacktestResult(
            strategy_name='Strategy2',
            params={'a': 2},
            start_dt=date(2024, 1, 1),
            end_dt=date(2024, 3, 31),
        )
        
        engine._results = [result1, result2]
        engine._best_result = result1
        
        results = engine.get_results()
        
        assert len(results) == 2
        assert results[0].strategy_name == 'Strategy1'
        assert results[1].strategy_name == 'Strategy2'
        
        assert engine.get_best_result() == result1

    def test_clear_results(self):
        engine = BacktestEngine()
        
        result = BacktestResult(
            strategy_name='Test',
            params={},
            start_dt=date(2024, 1, 1),
            end_dt=date(2024, 3, 31),
        )
        
        engine._results = [result]
        engine._best_result = result
        
        engine.clear_results()
        
        assert engine._results == []
        assert engine._best_result is None


class TestBacktestEngineRiskIntegration:
    def test_create_risk_manager_with_config(self):
        config = {
            'risk': {
                'max_drawdown_percent': 10.0,
                'max_strategy_margin_percent': 40.0,
                'max_total_margin_percent': 90.0,
            }
        }
        
        engine = BacktestEngine(config=config)
        
        mock_api = Mock()
        mock_connector = type('MockConnector', (), {
            'get_api': lambda self: mock_api,
        })()
        
        with patch('core.backtest.RiskManager') as mock_risk_manager_class:
            mock_risk_manager = Mock()
            mock_risk_manager_class.return_value = mock_risk_manager
            
            risk_manager = engine._create_risk_manager(mock_api)
            
            mock_risk_manager_class.assert_called_once()
            
            call_kwargs = mock_risk_manager_class.call_args.kwargs
            assert 'connector' in call_kwargs

    def test_risk_config_applied_correctly(self):
        config = {
            'risk': {
                'max_drawdown_percent': 15.0,
            }
        }
        
        engine = BacktestEngine(config=config)
        
        mock_api = Mock()
        
        risk_manager = engine._create_risk_manager(mock_api)
        
        assert risk_manager.max_drawdown_percent == 15.0


class TestBacktestEngineReportGeneration:
    def test_generate_report_empty_results(self):
        engine = BacktestEngine()
        
        report = engine.generate_report()
        
        assert report == {}

    def test_generate_report_with_results(self, tmp_path):
        engine = BacktestEngine()
        
        performance = PerformanceMetrics(
            total_return=100000.0,
            total_return_percent=10.0,
            max_drawdown_percent=5.0,
            total_trades=20,
        )
        
        result = BacktestResult(
            strategy_name='DoubleMAStrategy',
            params={'short_period': 5, 'long_period': 10},
            start_dt=date(2024, 1, 1),
            end_dt=date(2024, 3, 31),
            initial_equity=1000000.0,
            final_equity=1100000.0,
            performance=performance,
            status='completed',
        )
        
        engine._results = [result]
        engine._best_result = result
        
        report = engine.generate_report(output_dir=str(tmp_path))
        
        assert 'generated_at' in report
        assert report['total_backtests'] == 1
        assert report['best_result'] is not None
        assert report['best_result']['strategy_name'] == 'DoubleMAStrategy'
        
        csv_files = list(tmp_path.glob('*.csv'))
        json_files = list(tmp_path.glob('*.json'))
        
        assert len(csv_files) >= 1
        assert len(json_files) >= 1


class TestBacktestEngineIntegration:
    def test_create_strategy_manager(self):
        engine = BacktestEngine()
        
        mock_api = Mock()
        
        strategy_params = {
            'short_period': 5,
            'long_period': 10,
            'contract': 'SHFE.rb2410',
            'kline_duration': 60,
        }
        
        manager = engine._create_strategy_manager(
            api=mock_api,
            strategy_class=DoubleMAStrategy,
            strategy_params=strategy_params,
        )
        
        assert manager is not None
        assert 'BacktestStrategy' in manager.get_all_strategies()
        
        strategy = manager.get_strategy('BacktestStrategy')
        assert strategy.short_period == 5
        assert strategy.long_period == 10
        assert strategy.contract == 'SHFE.rb2410'

    def test_extract_date_range_with_date_objects(self):
        config = {
            'backtest': {
                'start_dt': date(2024, 1, 15),
                'end_dt': date(2024, 2, 15),
            }
        }
        
        engine = BacktestEngine(config=config)
        start_dt, end_dt = engine._extract_date_range()
        
        assert start_dt == date(2024, 1, 15)
        assert end_dt == date(2024, 2, 15)


class TestParseCostConfig:
    def test_parse_cost_config_from_dict(self):
        backtest_config = {
            'costs': {
                'default_commission_per_lot': 5.0,
                'default_slippage_points': 1.0,
                'contracts': {
                    'SHFE.rb2410': {
                        'commission_per_lot': 3.0,
                        'slippage_points': 0.5,
                    },
                    'SHFE.hc2410': {
                        'commission_per_lot': 4.0,
                        'slippage_points': 0.8,
                    }
                }
            }
        }
        
        engine = BacktestEngine(config={'backtest': backtest_config})
        
        assert engine._cost_config.default_commission_per_lot == 5.0
        assert engine._cost_config.default_slippage_points == 1.0
        assert engine._cost_config.get_commission('SHFE.rb2410') == 3.0
        assert engine._cost_config.get_slippage('SHFE.rb2410') == 0.5
        assert engine._cost_config.get_commission('SHFE.hc2410') == 4.0
        assert engine._cost_config.get_slippage('SHFE.hc2410') == 0.8

    def test_parse_cost_config_empty(self):
        backtest_config = {}
        
        engine = BacktestEngine(config={'backtest': backtest_config})
        
        assert engine._cost_config.default_commission_per_lot == 0.0
        assert engine._cost_config.default_slippage_points == 0.0
        assert engine._cost_config.contract_configs == {}


class TestPerformanceConfig:
    def test_performance_config_default(self):
        config = PerformanceConfig()
        
        assert config.risk_free_rate == 0.03
        assert config.trading_days_per_year == 252

    def test_performance_config_custom(self):
        config = PerformanceConfig(
            risk_free_rate=0.04,
            trading_days_per_year=250,
        )
        
        assert config.risk_free_rate == 0.04
        assert config.trading_days_per_year == 250
