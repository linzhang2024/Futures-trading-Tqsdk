import pytest
import sys
import os
import time
from unittest.mock import Mock, patch, MagicMock, PropertyMock
from dataclasses import dataclass, field
from typing import Dict, Any, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.risk_manager import (
    RiskManager,
    RiskLevel,
    RiskEventType,
    RiskEvent,
    PositionInfo,
    AccountSnapshot,
    StrategyRiskInfo,
)
from core.manager import StrategyManager
from strategies.base_strategy import StrategyBase, SignalType
from strategies.double_ma_strategy import DoubleMAStrategy


class TestRiskManagerBasics:
    def test_initialization(self):
        risk_manager = RiskManager()
        
        assert risk_manager.is_frozen() is False
        assert risk_manager.get_frozen_reason() is None
        assert risk_manager._peak_equity == 0.0
        assert risk_manager._current_drawdown_percent == 0.0
        assert risk_manager.max_drawdown_percent == 5.0
        assert risk_manager.max_strategy_margin_percent == 30.0
        assert risk_manager.max_total_margin_percent == 80.0
    
    def test_custom_thresholds(self):
        risk_manager = RiskManager(
            max_drawdown_percent=10.0,
            max_strategy_margin_percent=40.0,
            max_total_margin_percent=90.0,
        )
        
        assert risk_manager.max_drawdown_percent == 10.0
        assert risk_manager.max_strategy_margin_percent == 40.0
        assert risk_manager.max_total_margin_percent == 90.0
    
    def test_freeze_and_unfreeze(self):
        risk_manager = RiskManager()
        
        assert risk_manager.is_frozen() is False
        
        risk_manager.freeze("测试冻结原因")
        
        assert risk_manager.is_frozen() is True
        assert risk_manager.get_frozen_reason() == "测试冻结原因"
        
        risk_manager.unfreeze()
        
        assert risk_manager.is_frozen() is False
        assert risk_manager.get_frozen_reason() is None


class TestRiskManagerDrawdown:
    def test_peak_equity_tracking(self):
        risk_manager = RiskManager(max_drawdown_percent=5.0)
        
        snapshot1 = AccountSnapshot(
            timestamp=time.time(),
            balance=1000000.0,
            equity=1000000.0,
            total_asset=1000000.0,
            margin_used=0.0,
            available=1000000.0,
            float_profit=0.0,
        )
        
        risk_manager.check_drawdown(snapshot1)
        assert risk_manager._peak_equity == 1000000.0
        
        snapshot2 = AccountSnapshot(
            timestamp=time.time(),
            balance=1100000.0,
            equity=1100000.0,
            total_asset=1100000.0,
            margin_used=0.0,
            available=1100000.0,
            float_profit=0.0,
        )
        
        risk_manager.check_drawdown(snapshot2)
        assert risk_manager._peak_equity == 1100000.0
    
    def test_drawdown_calculation(self):
        risk_manager = RiskManager(max_drawdown_percent=5.0)
        
        peak_snapshot = AccountSnapshot(
            timestamp=time.time(),
            balance=1000000.0,
            equity=1000000.0,
            total_asset=1000000.0,
            margin_used=0.0,
            available=1000000.0,
            float_profit=0.0,
        )
        risk_manager.check_drawdown(peak_snapshot)
        
        snapshot_2percent = AccountSnapshot(
            timestamp=time.time(),
            balance=980000.0,
            equity=980000.0,
            total_asset=980000.0,
            margin_used=0.0,
            available=980000.0,
            float_profit=0.0,
        )
        
        level = risk_manager.check_drawdown(snapshot_2percent)
        assert risk_manager._current_drawdown_percent == 2.0
        assert level == RiskLevel.SAFE
        assert risk_manager.is_frozen() is False
    
    def test_drawdown_warning(self):
        risk_manager = RiskManager(max_drawdown_percent=5.0)
        
        peak_snapshot = AccountSnapshot(
            timestamp=time.time(),
            balance=1000000.0,
            equity=1000000.0,
            total_asset=1000000.0,
            margin_used=0.0,
            available=1000000.0,
            float_profit=0.0,
        )
        risk_manager.check_drawdown(peak_snapshot)
        
        snapshot_warning = AccountSnapshot(
            timestamp=time.time(),
            balance=965000.0,
            equity=965000.0,
            total_asset=965000.0,
            margin_used=0.0,
            available=965000.0,
            float_profit=0.0,
        )
        
        level = risk_manager.check_drawdown(snapshot_warning)
        assert abs(risk_manager._current_drawdown_percent - 3.5) < 0.001
        assert level == RiskLevel.WARNING
        assert risk_manager.is_frozen() is False
    
    def test_drawdown_freeze(self):
        risk_manager = RiskManager(max_drawdown_percent=5.0)
        
        peak_snapshot = AccountSnapshot(
            timestamp=time.time(),
            balance=1000000.0,
            equity=1000000.0,
            total_asset=1000000.0,
            margin_used=0.0,
            available=1000000.0,
            float_profit=0.0,
        )
        risk_manager.check_drawdown(peak_snapshot)
        
        snapshot_freeze = AccountSnapshot(
            timestamp=time.time(),
            balance=940000.0,
            equity=940000.0,
            total_asset=940000.0,
            margin_used=0.0,
            available=940000.0,
            float_profit=0.0,
        )
        
        level = risk_manager.check_drawdown(snapshot_freeze)
        assert risk_manager._current_drawdown_percent == 6.0
        assert level == RiskLevel.FROZEN
        assert risk_manager.is_frozen() is True
        assert "回撤" in risk_manager.get_frozen_reason()
    
    def test_drawdown_rapid_decline(self):
        risk_manager = RiskManager(max_drawdown_percent=5.0)
        
        peak_snapshot = AccountSnapshot(
            timestamp=time.time(),
            balance=1000000.0,
            equity=1000000.0,
            total_asset=1000000.0,
            margin_used=0.0,
            available=1000000.0,
            float_profit=0.0,
        )
        risk_manager.check_drawdown(peak_snapshot)
        
        rapid_decline = AccountSnapshot(
            timestamp=time.time(),
            balance=500000.0,
            equity=500000.0,
            total_asset=500000.0,
            margin_used=0.0,
            available=500000.0,
            float_profit=0.0,
        )
        
        level = risk_manager.check_drawdown(rapid_decline)
        assert risk_manager._current_drawdown_percent == 50.0
        assert level == RiskLevel.FROZEN
        assert risk_manager.is_frozen() is True


class TestRiskManagerMargin:
    def test_strategy_margin_check(self):
        risk_manager = RiskManager(max_strategy_margin_percent=30.0)
        
        snapshot = AccountSnapshot(
            timestamp=time.time(),
            balance=1000000.0,
            equity=1000000.0,
            total_asset=1000000.0,
            margin_used=0.0,
            available=1000000.0,
            float_profit=0.0,
        )
        
        risk_manager.update_strategy_risk(
            strategy_name="TestStrategy",
            margin_used=200000.0,
        )
        
        level = risk_manager.check_strategy_margin("TestStrategy", snapshot=snapshot)
        assert level == RiskLevel.SAFE
    
    def test_strategy_margin_warning(self):
        risk_manager = RiskManager(max_strategy_margin_percent=30.0)
        
        snapshot = AccountSnapshot(
            timestamp=time.time(),
            balance=1000000.0,
            equity=1000000.0,
            total_asset=1000000.0,
            margin_used=0.0,
            available=1000000.0,
            float_profit=0.0,
        )
        
        risk_manager.update_strategy_risk(
            strategy_name="TestStrategy",
            margin_used=250000.0,
        )
        
        level = risk_manager.check_strategy_margin("TestStrategy", snapshot=snapshot)
        assert level == RiskLevel.WARNING
    
    def test_strategy_margin_critical(self):
        risk_manager = RiskManager(max_strategy_margin_percent=30.0)
        
        snapshot = AccountSnapshot(
            timestamp=time.time(),
            balance=1000000.0,
            equity=1000000.0,
            total_asset=1000000.0,
            margin_used=0.0,
            available=1000000.0,
            float_profit=0.0,
        )
        
        risk_manager.update_strategy_risk(
            strategy_name="TestStrategy",
            margin_used=350000.0,
        )
        
        level = risk_manager.check_strategy_margin("TestStrategy", snapshot=snapshot)
        assert level == RiskLevel.CRITICAL
    
    def test_total_margin_check(self):
        risk_manager = RiskManager(max_total_margin_percent=80.0)
        
        snapshot = AccountSnapshot(
            timestamp=time.time(),
            balance=1000000.0,
            equity=1000000.0,
            total_asset=1000000.0,
            margin_used=500000.0,
            available=500000.0,
            float_profit=0.0,
        )
        
        level = risk_manager.check_total_margin(snapshot)
        assert level == RiskLevel.SAFE
    
    def test_total_margin_warning(self):
        risk_manager = RiskManager(max_total_margin_percent=80.0)
        
        snapshot = AccountSnapshot(
            timestamp=time.time(),
            balance=1000000.0,
            equity=1000000.0,
            total_asset=1000000.0,
            margin_used=700000.0,
            available=300000.0,
            float_profit=0.0,
        )
        
        level = risk_manager.check_total_margin(snapshot)
        assert level == RiskLevel.WARNING
    
    def test_total_margin_critical(self):
        risk_manager = RiskManager(max_total_margin_percent=80.0)
        
        snapshot = AccountSnapshot(
            timestamp=time.time(),
            balance=1000000.0,
            equity=1000000.0,
            total_asset=1000000.0,
            margin_used=900000.0,
            available=100000.0,
            float_profit=0.0,
        )
        
        level = risk_manager.check_total_margin(snapshot)
        assert level == RiskLevel.CRITICAL


class TestRiskManagerOrderValidation:
    def test_can_place_order_allowed(self):
        risk_manager = RiskManager(max_strategy_margin_percent=30.0)
        
        with patch.object(risk_manager, 'get_account_snapshot') as mock_snapshot:
            mock_snapshot.return_value = AccountSnapshot(
                timestamp=time.time(),
                balance=1000000.0,
                equity=1000000.0,
                total_asset=1000000.0,
                margin_used=100000.0,
                available=900000.0,
                float_profit=0.0,
            )
            
            can_place, message, level = risk_manager.can_place_order(
                strategy_name="TestStrategy",
                contract="SHFE.rb2410",
                direction="BUY",
                volume=10,
                price=3500.0,
                margin_per_contract=3500.0,
            )
            
            assert can_place is True
            assert level == RiskLevel.SAFE
    
    def test_can_place_order_insufficient_funds(self):
        risk_manager = RiskManager(max_strategy_margin_percent=30.0)
        
        with patch.object(risk_manager, 'get_account_snapshot') as mock_snapshot:
            mock_snapshot.return_value = AccountSnapshot(
                timestamp=time.time(),
                balance=100000.0,
                equity=100000.0,
                total_asset=100000.0,
                margin_used=90000.0,
                available=10000.0,
                float_profit=0.0,
            )
            
            can_place, message, level = risk_manager.can_place_order(
                strategy_name="TestStrategy",
                contract="SHFE.rb2410",
                direction="BUY",
                volume=10,
                price=3500.0,
                margin_per_contract=3500.0,
            )
            
            assert can_place is False
            assert level == RiskLevel.CRITICAL
    
    def test_can_place_order_frozen(self):
        risk_manager = RiskManager()
        risk_manager.freeze("测试冻结")
        
        can_place, message, level = risk_manager.can_place_order(
            strategy_name="TestStrategy",
            contract="SHFE.rb2410",
            direction="BUY",
            volume=10,
            price=3500.0,
        )
        
        assert can_place is False
        assert level == RiskLevel.FROZEN
    
    def test_can_place_order_invalid_volume(self):
        risk_manager = RiskManager()
        
        can_place, message, level = risk_manager.can_place_order(
            strategy_name="TestStrategy",
            contract="SHFE.rb2410",
            direction="BUY",
            volume=0,
            price=3500.0,
        )
        
        assert can_place is False


class TestRiskManagerPerformance:
    def test_drawdown_check_performance_ms_level(self):
        risk_manager = RiskManager(max_drawdown_percent=5.0)
        
        peak_snapshot = AccountSnapshot(
            timestamp=time.time(),
            balance=1000000.0,
            equity=1000000.0,
            total_asset=1000000.0,
            margin_used=0.0,
            available=1000000.0,
            float_profit=0.0,
        )
        risk_manager.check_drawdown(peak_snapshot)
        
        num_iterations = 1000
        total_time = 0.0
        
        for i in range(num_iterations):
            equity = 1000000.0 - (i * 100)
            snapshot = AccountSnapshot(
                timestamp=time.time(),
                balance=equity,
                equity=equity,
                total_asset=equity,
                margin_used=0.0,
                available=equity,
                float_profit=0.0,
            )
            
            start_time = time.perf_counter()
            risk_manager.check_drawdown(snapshot)
            end_time = time.perf_counter()
            
            total_time += (end_time - start_time)
        
        avg_time_ms = (total_time / num_iterations) * 1000
        
        assert avg_time_ms < 1.0, f"平均检查时间 {avg_time_ms:.3f}ms 超过 1ms"
    
    def test_rapid_drawdown_freeze_response_time(self):
        risk_manager = RiskManager(max_drawdown_percent=5.0)
        
        peak_snapshot = AccountSnapshot(
            timestamp=time.time(),
            balance=1000000.0,
            equity=1000000.0,
            total_asset=1000000.0,
            margin_used=0.0,
            available=1000000.0,
            float_profit=0.0,
        )
        risk_manager.check_drawdown(peak_snapshot)
        
        crash_snapshot = AccountSnapshot(
            timestamp=time.time(),
            balance=900000.0,
            equity=900000.0,
            total_asset=900000.0,
            margin_used=0.0,
            available=900000.0,
            float_profit=0.0,
        )
        
        start_time = time.perf_counter()
        
        for _ in range(100):
            level = risk_manager.check_drawdown(crash_snapshot)
            if level == RiskLevel.FROZEN:
                break
        
        end_time = time.perf_counter()
        response_time_ms = (end_time - start_time) * 1000
        
        assert risk_manager.is_frozen() is True
        assert response_time_ms < 10.0, f"熔断响应时间 {response_time_ms:.3f}ms 超过 10ms"
    
    def test_full_risk_check_performance(self):
        risk_manager = RiskManager(
            max_drawdown_percent=5.0,
            max_strategy_margin_percent=30.0,
            max_total_margin_percent=80.0,
        )
        
        risk_manager.update_strategy_risk("Strategy1", margin_used=100000.0)
        risk_manager.update_strategy_risk("Strategy2", margin_used=150000.0)
        risk_manager.update_strategy_risk("Strategy3", margin_used=50000.0)
        
        with patch.object(risk_manager, 'get_account_snapshot') as mock_snapshot:
            mock_snapshot.return_value = AccountSnapshot(
                timestamp=time.time(),
                balance=1000000.0,
                equity=1000000.0,
                total_asset=1000000.0,
                margin_used=300000.0,
                available=700000.0,
                float_profit=0.0,
            )
            
            with patch.object(risk_manager, 'update_positions') as mock_positions:
                mock_positions.return_value = {}
                
                num_iterations = 100
                total_time = 0.0
                
                for _ in range(num_iterations):
                    start_time = time.perf_counter()
                    risk_manager.run_risk_checks()
                    end_time = time.perf_counter()
                    total_time += (end_time - start_time)
                
                avg_time_ms = (total_time / num_iterations) * 1000
                
                assert avg_time_ms < 5.0, f"平均完整风控检查时间 {avg_time_ms:.3f}ms 超过 5ms"


class TestRiskManagerEventCallback:
    def test_risk_event_callback(self):
        events_received = []
        
        def on_event(event: RiskEvent):
            events_received.append(event)
        
        risk_manager = RiskManager(
            max_drawdown_percent=5.0,
            on_risk_event=on_event,
        )
        
        peak_snapshot = AccountSnapshot(
            timestamp=time.time(),
            balance=1000000.0,
            equity=1000000.0,
            total_asset=1000000.0,
            margin_used=0.0,
            available=1000000.0,
            float_profit=0.0,
        )
        risk_manager.check_drawdown(peak_snapshot)
        
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
        
        assert len(events_received) >= 1
        drawdown_events = [e for e in events_received if e.event_type == RiskEventType.DRAWDOWN_EXCEEDED]
        assert len(drawdown_events) >= 1
        assert drawdown_events[0].level == RiskLevel.CRITICAL
    
    def test_frozen_callback(self):
        frozen_called = [False]
        
        def on_frozen():
            frozen_called[0] = True
        
        risk_manager = RiskManager(
            max_drawdown_percent=5.0,
            on_frozen=on_frozen,
        )
        
        peak_snapshot = AccountSnapshot(
            timestamp=time.time(),
            balance=1000000.0,
            equity=1000000.0,
            total_asset=1000000.0,
            margin_used=0.0,
            available=1000000.0,
            float_profit=0.0,
        )
        risk_manager.check_drawdown(peak_snapshot)
        
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
        
        assert frozen_called[0] is True


class TestRiskManagerInfo:
    def test_get_drawdown_info(self):
        risk_manager = RiskManager(max_drawdown_percent=5.0)
        
        peak_snapshot = AccountSnapshot(
            timestamp=time.time(),
            balance=1000000.0,
            equity=1000000.0,
            total_asset=1000000.0,
            margin_used=0.0,
            available=1000000.0,
            float_profit=0.0,
        )
        risk_manager.check_drawdown(peak_snapshot)
        
        info = risk_manager.get_drawdown_info()
        
        assert info['peak_equity'] == 1000000.0
        assert info['current_drawdown_percent'] == 0.0
        assert info['max_drawdown_percent'] == 5.0
        assert info['is_frozen'] is False
    
    def test_get_total_risk_info(self):
        risk_manager = RiskManager(max_drawdown_percent=5.0)
        
        with patch.object(risk_manager, 'get_account_snapshot') as mock_snapshot:
            mock_snapshot.return_value = AccountSnapshot(
                timestamp=time.time(),
                balance=1000000.0,
                equity=1000000.0,
                total_asset=1000000.0,
                margin_used=200000.0,
                available=800000.0,
                float_profit=5000.0,
            )
            
            info = risk_manager.get_total_risk_info()
            
            assert info['balance'] == 1000000.0
            assert info['equity'] == 1000000.0
            assert info['margin_used'] == 200000.0
            assert info['available'] == 800000.0
            assert info['float_profit'] == 5000.0
    
    def test_get_risk_events(self):
        risk_manager = RiskManager(max_drawdown_percent=5.0)
        
        peak_snapshot = AccountSnapshot(
            timestamp=time.time(),
            balance=1000000.0,
            equity=1000000.0,
            total_asset=1000000.0,
            margin_used=0.0,
            available=1000000.0,
            float_profit=0.0,
        )
        risk_manager.check_drawdown(peak_snapshot)
        
        warning_snapshot = AccountSnapshot(
            timestamp=time.time(),
            balance=965000.0,
            equity=965000.0,
            total_asset=965000.0,
            margin_used=0.0,
            available=965000.0,
            float_profit=0.0,
        )
        risk_manager.check_drawdown(warning_snapshot)
        
        events = risk_manager.get_risk_events(limit=10)
        
        assert len(events) >= 1
        drawdown_events = [e for e in events if e['event_type'] == RiskEventType.DRAWDOWN_EXCEEDED.value]
        assert len(drawdown_events) >= 1


class TestRiskManagerPositionTracking:
    def test_update_positions(self):
        risk_manager = RiskManager()
        
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
            }
        }
        
        positions = risk_manager.update_positions(positions_data)
        
        assert 'SHFE.rb2410' in positions
        
        pos = positions['SHFE.rb2410']
        assert pos.long_volume == 10
        assert pos.short_volume == 5
        assert pos.long_margin == 35000.0
        assert pos.short_margin == 17500.0
        assert pos.total_volume == 15
        assert pos.total_margin == 52500.0
        assert pos.net_position == 5


class TestStrategyManagerRiskIntegration:
    def test_manager_with_risk_manager(self):
        risk_manager = RiskManager(max_drawdown_percent=5.0)
        manager = StrategyManager(risk_manager=risk_manager)
        
        assert manager.get_risk_manager() is risk_manager
    
    def test_manager_configure_risk(self):
        manager = StrategyManager()
        
        config = {
            'risk': {
                'max_drawdown_percent': 10.0,
                'max_strategy_margin_percent': 40.0,
                'max_total_margin_percent': 90.0,
                'risk_check_interval': 0.5,
            }
        }
        
        manager.configure_from_dict(config)
        
        risk_manager = manager.get_risk_manager()
        assert risk_manager is not None
        assert risk_manager.max_drawdown_percent == 10.0
        assert risk_manager.max_strategy_margin_percent == 40.0
        assert risk_manager.max_total_margin_percent == 90.0
        assert manager._risk_check_interval == 0.5
    
    def test_manager_emergency_stop(self):
        mock_api = Mock()
        mock_connector = Mock()
        mock_connector.get_api.return_value = mock_api
        
        risk_manager = RiskManager(connector=mock_connector)
        manager = StrategyManager(connector=mock_connector, risk_manager=risk_manager)
        
        result = manager.emergency_stop("测试紧急停止")
        
        assert manager.is_risk_frozen() is True
        assert result['status'] == 'success'
        assert result['reason'] == "测试紧急停止"
    
    def test_manager_get_risk_info(self):
        manager = StrategyManager()
        
        info = manager.get_risk_info()
        assert info['risk_enabled'] is False
        
        risk_manager = RiskManager(max_drawdown_percent=5.0)
        manager.set_risk_manager(risk_manager)
        
        with patch.object(risk_manager, 'get_account_snapshot') as mock_snapshot:
            mock_snapshot.return_value = AccountSnapshot(
                timestamp=time.time(),
                balance=1000000.0,
                equity=1000000.0,
                total_asset=1000000.0,
                margin_used=0.0,
                available=1000000.0,
                float_profit=0.0,
            )
            
            info = manager.get_risk_info()
            assert info['risk_enabled'] is True
            assert info['equity'] == 1000000.0


class TestExtremeMarketScenarios:
    def test_flash_crash_scenario(self):
        risk_manager = RiskManager(max_drawdown_percent=5.0)
        
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
        
        num_steps = 10
        equity_values = [
            1000000.0,
            980000.0,
            950000.0,
            920000.0,
            900000.0,
            880000.0,
            850000.0,
            820000.0,
            800000.0,
            750000.0,
        ]
        
        freeze_step = None
        for i, equity in enumerate(equity_values):
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
        assert freeze_step < 10
        
        drawdown_at_freeze = risk_manager._current_drawdown_percent
        assert drawdown_at_freeze >= 5.0
    
    def test_margin_call_scenario(self):
        risk_manager = RiskManager(
            max_strategy_margin_percent=30.0,
            max_total_margin_percent=80.0,
        )
        
        initial_snapshot = AccountSnapshot(
            timestamp=time.time(),
            balance=1000000.0,
            equity=1000000.0,
            total_asset=1000000.0,
            margin_used=200000.0,
            available=800000.0,
            float_profit=0.0,
        )
        
        risk_manager.update_strategy_risk(
            strategy_name="StrategyA",
            margin_used=200000.0,
        )
        
        level = risk_manager.check_strategy_margin("StrategyA", snapshot=initial_snapshot)
        assert level == RiskLevel.SAFE
        
        moderately_stressed_snapshot = AccountSnapshot(
            timestamp=time.time(),
            balance=800000.0,
            equity=800000.0,
            total_asset=800000.0,
            margin_used=200000.0,
            available=600000.0,
            float_profit=-200000.0,
        )
        
        level = risk_manager.check_strategy_margin("StrategyA", snapshot=moderately_stressed_snapshot)
        assert level == RiskLevel.WARNING
        
        severely_stressed_snapshot = AccountSnapshot(
            timestamp=time.time(),
            balance=500000.0,
            equity=500000.0,
            total_asset=500000.0,
            margin_used=200000.0,
            available=300000.0,
            float_profit=-500000.0,
        )
        
        level = risk_manager.check_strategy_margin("StrategyA", snapshot=severely_stressed_snapshot)
        assert level == RiskLevel.CRITICAL
