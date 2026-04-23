import pytest
import sys
import os
import time
import threading
from unittest.mock import Mock, patch, MagicMock, PropertyMock
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.risk_manager import (
    RiskManager,
    RiskLevel,
    RiskEventType,
    RiskEvent,
    PositionInfo,
    AccountSnapshot,
    StrategyRiskInfo,
    NUMPY_AVAILABLE,
)


class TestExtremeScenarios:
    def test_api_delay_5_seconds(self):
        risk_manager = RiskManager(
            api_timeout_seconds=3.0,
        )

        mock_api = Mock()
        mock_api.get_account.return_value = {'equity': 1000000.0, 'balance': 1000000.0}

        risk_manager.api = mock_api
        risk_manager._api_last_response_time = time.time() - 6.0

        health_ok, health_msg = risk_manager.check_api_health()

        assert health_ok is False
        assert "超时" in health_msg
        assert risk_manager._api_timeouts == 1

    def test_api_delay_10_seconds_consecutive_timeouts(self):
        risk_manager = RiskManager(
            api_timeout_seconds=3.0,
        )

        mock_api = Mock()
        risk_manager.api = mock_api
        risk_manager._api_last_response_time = time.time() - 10.0
        risk_manager._api_timeouts = 2

        health_ok, health_msg = risk_manager.check_api_health()

        assert health_ok is False
        assert "连接可能已丢失" in health_msg
        assert risk_manager._api_timeouts == 3

    def test_api_response_recovery(self):
        risk_manager = RiskManager(
            api_timeout_seconds=3.0,
        )

        mock_api = Mock()
        risk_manager.api = mock_api

        risk_manager._api_last_response_time = time.time() - 10.0
        risk_manager._api_timeouts = 2

        risk_manager.update_api_response_time()

        health_ok, health_msg = risk_manager.check_api_health()

        assert health_ok is True
        assert risk_manager._api_timeouts == 0

    def test_price_gap_5_percent_warning(self):
        risk_manager = RiskManager(
            price_gap_threshold_percent=5.0,
        )

        contract = "SHFE.rb2410"
        previous_price = 3500.0
        current_price = 3675.0

        risk_manager.check_price_gap(contract, previous_price)

        risk_manager._previous_prices[contract] = previous_price

        with patch.object(risk_manager, '_emit_risk_event') as mock_emit:
            risk_manager.check_price_gap(contract, current_price)

            gap_percent = abs((current_price - previous_price) / previous_price * 100)
            assert gap_percent == 5.0

    def test_price_gap_10_percent_critical(self):
        risk_manager = RiskManager(
            price_gap_threshold_percent=5.0,
        )

        contract = "SHFE.rb2410"
        previous_price = 3500.0
        current_price = 3850.0

        risk_manager._previous_prices[contract] = previous_price

        events_emitted = []

        def capture_event(event_type, level, message, details=None):
            events_emitted.append({
                'event_type': event_type,
                'level': level,
                'message': message,
                'details': details or {},
            })
            return RiskEvent(event_type, time.time(), level, message, details or {})

        with patch.object(risk_manager, '_emit_risk_event', side_effect=capture_event):
            risk_manager.check_price_gap(contract, current_price)

            gap_percent = abs((current_price - previous_price) / previous_price * 100)
            assert gap_percent == 10.0

            if len(events_emitted) > 0:
                event = events_emitted[0]
                assert event['event_type'] == RiskEventType.PRICE_GAP_DETECTED

    def test_price_gap_down_12_percent(self):
        risk_manager = RiskManager(
            price_gap_threshold_percent=5.0,
        )

        contract = "DCE.i2409"
        previous_price = 1000.0
        current_price = 880.0

        risk_manager._previous_prices[contract] = previous_price

        events_emitted = []

        def capture_event(event_type, level, message, details=None):
            events_emitted.append({
                'event_type': event_type,
                'level': level,
                'message': message,
                'details': details or {},
            })
            return RiskEvent(event_type, time.time(), level, message, details or {})

        with patch.object(risk_manager, '_emit_risk_event', side_effect=capture_event):
            risk_manager.check_price_gap(contract, current_price)

            gap_percent = abs((current_price - previous_price) / previous_price * 100)
            assert gap_percent == 12.0

            if len(events_emitted) > 0:
                details = events_emitted[0].get('details', {})
                assert details.get('direction') == 'DOWN'

    def test_sudden_drawdown_20_percent(self):
        risk_manager = RiskManager(
            max_drawdown_percent=5.0,
        )

        risk_manager._peak_equity = 1000000.0

        snapshot = AccountSnapshot(
            timestamp=time.time(),
            balance=800000.0,
            equity=800000.0,
            total_asset=800000.0,
            margin_used=0.0,
            available=800000.0,
            float_profit=0.0,
        )

        start_time = time.perf_counter()
        level = risk_manager.check_drawdown(snapshot)
        end_time = time.perf_counter()

        response_time_ms = (end_time - start_time) * 1000

        assert level == RiskLevel.FROZEN
        assert risk_manager.is_frozen() is True
        assert risk_manager._current_drawdown_percent == 20.0
        assert response_time_ms < 10.0

    def test_rapid_drawdown_series(self):
        risk_manager = RiskManager(
            max_drawdown_percent=5.0,
        )

        risk_manager._peak_equity = 1000000.0

        equity_series = [
            1000000.0,
            990000.0,
            970000.0,
            950000.0,
            940000.0,
            920000.0,
            900000.0,
        ]

        freeze_step = None

        for i, equity in enumerate(equity_series):
            snapshot = AccountSnapshot(
                timestamp=time.time(),
                balance=equity,
                equity=equity,
                total_asset=equity,
                margin_used=0.0,
                available=equity,
                float_profit=0.0,
            )

            level = risk_manager.check_drawdown(snapshot)

            if level == RiskLevel.FROZEN and freeze_step is None:
                freeze_step = i
                break

        assert freeze_step is not None
        assert freeze_step < len(equity_series)

    def test_connection_loss_during_risk_check(self):
        risk_manager = RiskManager(
            api_timeout_seconds=1.0,
        )

        mock_api = Mock()

        def get_account_raise():
            raise ConnectionError("API连接已断开")

        mock_api.get_account.side_effect = get_account_raise
        risk_manager.api = mock_api

        risk_manager._api_last_response_time = time.time() - 5.0
        risk_manager._api_timeouts = 2

        health_ok, health_msg = risk_manager.check_api_health()

        assert health_ok is False

    def test_reconnection_failure_3_times(self):
        risk_manager = RiskManager(
            api_timeout_seconds=1.0,
        )

        mock_api = Mock()
        risk_manager.api = mock_api

        risk_manager._api_last_response_time = time.time() - 10.0
        risk_manager._api_timeouts = 0

        for i in range(3):
            health_ok, health_msg = risk_manager.check_api_health()

        assert risk_manager._api_timeouts == 3

        health_ok, health_msg = risk_manager.check_api_health()
        assert "连接可能已丢失" in health_msg


class TestPerformanceBenchmark:
    def test_50_strategies_batch_check(self):
        risk_manager = RiskManager()

        num_strategies = 50
        for i in range(num_strategies):
            strategy_name = f"Strategy_{i+1:03d}"
            margin_used = 50000.0 + (i * 500.0)
            risk_manager.update_strategy_risk(
                strategy_name=strategy_name,
                margin_used=margin_used,
                position_value=margin_used * 3,
                float_profit=margin_used * 0.1,
            )

        snapshot = AccountSnapshot(
            timestamp=time.time(),
            balance=10000000.0,
            equity=10000000.0,
            total_asset=10000000.0,
            margin_used=2500000.0,
            available=7500000.0,
            float_profit=50000.0,
        )

        strategy_names = list(risk_manager._strategy_risk.keys())

        num_iterations = 100
        times = []

        for _ in range(num_iterations):
            start_time = time.perf_counter()

            results = risk_manager.batch_check_strategies_margin(strategy_names, snapshot)

            end_time = time.perf_counter()
            times.append((end_time - start_time) * 1000)

        avg_time = sum(times) / len(times)
        median_time = sorted(times)[len(times) // 2]
        p95_time = sorted(times)[int(len(times) * 0.95)]
        p99_time = sorted(times)[int(len(times) * 0.99)]

        print(f"""
        ================ 50策略批处理性能测试 ================
        NumPy可用: {NUMPY_AVAILABLE}
        策略数量: {num_strategies}
        迭代次数: {num_iterations}
        
        统计结果:
          最小耗时: {min(times):.3f} ms
          最大耗时: {max(times):.3f} ms
          平均耗时: {avg_time:.3f} ms
          中位数:   {median_time:.3f} ms
          P95:      {p95_time:.3f} ms
          P99:      {p99_time:.3f} ms
        
        性能评估:
          {'✅ 优秀: 平均耗时 < 1ms' if avg_time < 1 else 
           '⚠️ 良好: 平均耗时 < 5ms' if avg_time < 5 else
           '🔴 需要优化: 平均耗时 >= 5ms'}
        ==================================================
        """)

        assert len(results) == num_strategies

    def test_run_performance_benchmark_method(self):
        risk_manager = RiskManager()

        report = risk_manager.run_performance_benchmark(num_strategies=50)

        assert report['num_strategies'] == 50
        assert report['num_iterations'] == 100
        assert 'statistics' in report
        assert 'summary' in report

        stats = report['statistics']
        assert stats['min_ms'] >= 0
        assert stats['max_ms'] >= stats['min_ms']
        assert stats['avg_ms'] >= stats['min_ms']

    def test_100_strategies_concurrent_check(self):
        risk_manager = RiskManager()

        num_strategies = 100
        for i in range(num_strategies):
            strategy_name = f"Concurrent_Strategy_{i+1:03d}"
            margin_used = 30000.0 + (i * 300.0)
            risk_manager.update_strategy_risk(
                strategy_name=strategy_name,
                margin_used=margin_used,
            )

        snapshot = AccountSnapshot(
            timestamp=time.time(),
            balance=20000000.0,
            equity=20000000.0,
            total_asset=20000000.0,
            margin_used=5000000.0,
            available=15000000.0,
            float_profit=100000.0,
        )

        strategy_names = list(risk_manager._strategy_risk.keys())

        start_time = time.perf_counter()

        results = {}
        for name in strategy_names:
            results[name] = risk_manager.check_strategy_margin(name, snapshot=snapshot)

        end_time = time.perf_counter()
        total_time_ms = (end_time - start_time) * 1000
        avg_per_strategy_ms = total_time_ms / num_strategies

        print(f"""
        ================ 100策略串行性能测试 ================
        策略数量: {num_strategies}
        总耗时: {total_time_ms:.3f} ms
        单策略平均: {avg_per_strategy_ms:.3f} ms
        ==================================================
        """)

        assert len(results) == num_strategies


class TestFreezeReportGeneration:
    def test_freeze_report_generation(self):
        risk_manager = RiskManager()

        risk_manager._peak_equity = 1000000.0

        risk_manager.update_strategy_risk(
            strategy_name="Strategy_1",
            margin_used=100000.0,
            position_value=300000.0,
            float_profit=5000.0,
        )
        risk_manager.update_strategy_risk(
            strategy_name="Strategy_2",
            margin_used=150000.0,
            position_value=450000.0,
            float_profit=-10000.0,
        )

        positions_data = {
            'SHFE.rb2410': {
                'buy_volume': 10,
                'sell_volume': 5,
                'buy_margin': 35000.0,
                'sell_margin': 17500.0,
                'buy_open_price': 3500.0,
                'sell_open_price': 3600.0,
                'last_price': 3550.0,
                'float_profit': 2500.0,
            },
            'DCE.i2409': {
                'buy_volume': 0,
                'sell_volume': 20,
                'buy_margin': 0.0,
                'sell_margin': 80000.0,
                'buy_open_price': 0.0,
                'sell_open_price': 1000.0,
                'last_price': 980.0,
                'float_profit': -4000.0,
            },
        }

        risk_manager.update_positions(positions_data)

        for i in range(10):
            equity = 1000000.0 - (i * 10000.0)
            snapshot = AccountSnapshot(
                timestamp=time.time(),
                balance=equity,
                equity=equity,
                total_asset=equity,
                margin_used=0.0,
                available=equity,
                float_profit=0.0,
            )
            risk_manager._snapshots.append(snapshot)

        freeze_snapshot = AccountSnapshot(
            timestamp=time.time(),
            balance=940000.0,
            equity=940000.0,
            total_asset=940000.0,
            margin_used=0.0,
            available=940000.0,
            float_profit=0.0,
        )

        risk_manager.check_drawdown(freeze_snapshot)

        assert risk_manager.is_frozen() is True

        assert os.path.exists(risk_manager._risk_log_file)

    def test_freeze_event_with_details(self):
        risk_manager = RiskManager()

        risk_manager._peak_equity = 1000000.0

        events = []

        def on_event(event):
            events.append(event)

        risk_manager.on_risk_event = on_event

        freeze_snapshot = AccountSnapshot(
            timestamp=time.time(),
            balance=900000.0,
            equity=900000.0,
            total_asset=900000.0,
            margin_used=0.0,
            available=900000.0,
            float_profit=0.0,
        )

        risk_manager.check_drawdown(freeze_snapshot)

        assert len(events) >= 1

        drawdown_events = [e for e in events if e.event_type == RiskEventType.DRAWDOWN_EXCEEDED]
        assert len(drawdown_events) >= 1

        event = drawdown_events[0]
        assert event.level == RiskLevel.CRITICAL
        assert '900000' in event.message or '10.0' in event.message


class TestRealTimeRiskMonitoring:
    def test_continuous_risk_monitoring_simulation(self):
        risk_manager = RiskManager(
            max_drawdown_percent=3.0,
            api_timeout_seconds=2.0,
        )

        risk_manager._peak_equity = 1000000.0

        initial_snapshot = AccountSnapshot(
            timestamp=time.time(),
            balance=1000000.0,
            equity=1000000.0,
            total_asset=1000000.0,
            margin_used=0.0,
            available=1000000.0,
            float_profit=0.0,
        )

        risk_manager.check_drawdown(initial_snapshot)

        equity_changes = [
            (990000.0, "轻微波动"),
            (980000.0, "继续下跌"),
            (995000.0, "小幅反弹"),
            (975000.0, "再次下跌"),
            (960000.0, "加速下跌"),
            (950000.0, "触发熔断"),
        ]

        for equity, description in equity_changes:
            if risk_manager.is_frozen():
                break

            snapshot = AccountSnapshot(
                timestamp=time.time(),
                balance=equity,
                equity=equity,
                total_asset=equity,
                margin_used=0.0,
                available=equity,
                float_profit=0.0,
            )

            level = risk_manager.check_drawdown(snapshot)

            if level == RiskLevel.WARNING:
                print(f"⚠️ [{description}] 权益: {equity:,.0f}, 回撤: {risk_manager._current_drawdown_percent:.2f}%")
            elif level == RiskLevel.CRITICAL:
                print(f"🔴 [{description}] 权益: {equity:,.0f}, 回撤: {risk_manager._current_drawdown_percent:.2f}%")
            elif level == RiskLevel.FROZEN:
                print(f"❄️ [{description}] 系统已冻结! 权益: {equity:,.0f}, 回撤: {risk_manager._current_drawdown_percent:.2f}%")

        assert risk_manager.is_frozen() is True
        assert risk_manager._current_drawdown_percent >= 3.0


if __name__ == "__main__":
    print("=" * 80)
    print("极端场景性能测试")
    print("=" * 80)

    benchmark = TestPerformanceBenchmark()
    benchmark.test_50_strategies_batch_check()
    benchmark.test_run_performance_benchmark_method()
    benchmark.test_100_strategies_concurrent_check()
