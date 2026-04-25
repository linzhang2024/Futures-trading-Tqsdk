import pytest
import sys
import os
import time
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, base_dir)

from core.realtime_runner import (
    OrderManager,
    OrderRecord,
    OrderStatus,
    PositionSynchronizer,
    TargetPosition,
    WebhookNotifier,
    WebhookConfig,
    RealtimeRunner,
    load_webhook_config_from_settings,
)


class TestOrderStatus:
    def test_order_status_enum(self):
        assert OrderStatus.PENDING.value == "PENDING"
        assert OrderStatus.PLACED.value == "PLACED"
        assert OrderStatus.FILLED.value == "FILLED"
        assert OrderStatus.CANCELED.value == "CANCELED"
        assert OrderStatus.REJECTED.value == "REJECTED"
        assert OrderStatus.TIMEOUT.value == "TIMEOUT"


class TestOrderRecord:
    def test_order_record_creation(self):
        record = OrderRecord(
            order_id="test_123",
            contract="SHFE.rb2410",
            direction="BUY",
            offset="OPEN",
            volume=1,
            limit_price=3500.0,
            status=OrderStatus.PLACED,
            placed_time=time.time(),
        )
        
        assert record.order_id == "test_123"
        assert record.contract == "SHFE.rb2410"
        assert record.direction == "BUY"
        assert record.offset == "OPEN"
        assert record.volume == 1
        assert record.limit_price == 3500.0
        assert record.status == OrderStatus.PLACED
        assert record.filled_volume == 0
        assert record.retry_count == 0
        assert record.max_retries == 3


class TestTargetPosition:
    def test_target_position_creation(self):
        tp = TargetPosition(
            contract="SHFE.rb2410",
            target_long=5,
            target_short=0,
            sync_interval_seconds=60.0,
        )
        
        assert tp.contract == "SHFE.rb2410"
        assert tp.target_long == 5
        assert tp.target_short == 0
        assert tp.current_long == 0
        assert tp.current_short == 0
        assert tp.sync_interval_seconds == 60.0


class TestWebhookConfig:
    def test_webhook_config_defaults(self):
        config = WebhookConfig()
        
        assert config.enabled is False
        assert config.url == ""
        assert config.secret == ""
        assert config.timeout_seconds == 10.0
        assert config.retry_count == 3
        assert config.retry_delay_seconds == 2.0
        assert config.notify_on_trade is True
        assert config.notify_on_risk is True
        assert config.notify_on_order is False
    
    def test_webhook_config_custom(self):
        config = WebhookConfig(
            enabled=True,
            url="https://example.com/webhook",
            secret="test_secret",
            timeout_seconds=15.0,
            retry_count=5,
            retry_delay_seconds=3.0,
            notify_on_trade=True,
            notify_on_risk=True,
            notify_on_order=True,
        )
        
        assert config.enabled is True
        assert config.url == "https://example.com/webhook"
        assert config.secret == "test_secret"
        assert config.timeout_seconds == 15.0
        assert config.retry_count == 5
        assert config.retry_delay_seconds == 3.0
        assert config.notify_on_order is True


class TestWebhookNotifier:
    @patch('core.realtime_runner.requests.Session')
    def test_webhook_notifier_send_disabled(self, mock_session_class):
        config = WebhookConfig(enabled=False)
        logger = Mock()
        notifier = WebhookNotifier(config=config, logger=logger)
        
        result = notifier._send_request({'test': 'data'})
        
        assert result is False
        mock_session_class.return_value.post.assert_not_called()
    
    @patch('core.realtime_runner.requests.Session')
    def test_webhook_notifier_send_no_url(self, mock_session_class):
        config = WebhookConfig(enabled=True, url="")
        logger = Mock()
        notifier = WebhookNotifier(config=config, logger=logger)
        
        result = notifier._send_request({'test': 'data'})
        
        assert result is False
        mock_session_class.return_value.post.assert_not_called()


class TestOrderManager:
    def test_order_manager_init(self):
        mock_api = Mock()
        mock_logger = Mock()
        
        manager = OrderManager(
            api=mock_api,
            logger=mock_logger,
            timeout_seconds=30.0,
            max_retries=3,
            price_protection_percent=0.001,
        )
        
        assert manager.timeout_seconds == 30.0
        assert manager.max_retries == 3
        assert manager.price_protection_percent == 0.001
    
    def test_generate_order_id(self):
        mock_api = Mock()
        mock_logger = Mock()
        manager = OrderManager(api=mock_api, logger=mock_logger)
        
        order_id1 = manager._generate_order_id("SHFE.rb2410", "BUY")
        time.sleep(0.002)
        order_id2 = manager._generate_order_id("SHFE.rb2410", "BUY")
        
        assert "SHFE.rb2410" in order_id1
        assert "BUY" in order_id1
        assert order_id1 != order_id2
    
    def test_calculate_protected_price_buy(self):
        mock_api = Mock()
        mock_logger = Mock()
        manager = OrderManager(
            api=mock_api,
            logger=mock_logger,
            price_protection_percent=0.01,
        )
        
        with patch.object(manager, '_get_current_price', return_value=3500.0):
            protected_price = manager._calculate_protected_price(
                contract="SHFE.rb2410",
                direction="BUY",
                base_price=3550.0,
            )
            
            max_allowed = 3500.0 * (1 + 0.01)
            assert protected_price == min(3550.0, max_allowed)
    
    def test_calculate_protected_price_sell(self):
        mock_api = Mock()
        mock_logger = Mock()
        manager = OrderManager(
            api=mock_api,
            logger=mock_logger,
            price_protection_percent=0.01,
        )
        
        with patch.object(manager, '_get_current_price', return_value=3500.0):
            protected_price = manager._calculate_protected_price(
                contract="SHFE.rb2410",
                direction="SELL",
                base_price=3450.0,
            )
            
            min_allowed = 3500.0 * (1 - 0.01)
            assert protected_price == max(3450.0, min_allowed)
    
    def test_get_active_orders_empty(self):
        mock_api = Mock()
        mock_logger = Mock()
        manager = OrderManager(api=mock_api, logger=mock_logger)
        
        active_orders = manager.get_active_orders()
        
        assert active_orders == []


class TestPositionSynchronizer:
    def test_position_synchronizer_init(self):
        mock_api = Mock()
        mock_logger = Mock()
        
        sync = PositionSynchronizer(
            api=mock_api,
            logger=mock_logger,
            sync_interval_seconds=60.0,
        )
        
        assert sync.sync_interval_seconds == 60.0
    
    def test_set_target_position(self):
        mock_api = Mock()
        mock_logger = Mock()
        sync = PositionSynchronizer(api=mock_api, logger=mock_logger)
        
        sync.set_target_position("SHFE.rb2410", target_long=5, target_short=0)
        
        tp = sync.get_target_position("SHFE.rb2410")
        assert tp is not None
        assert tp.target_long == 5
        assert tp.target_short == 0
    
    def test_check_need_sync_true(self):
        mock_api = Mock()
        mock_logger = Mock()
        sync = PositionSynchronizer(
            api=mock_api,
            logger=mock_logger,
            sync_interval_seconds=1.0,
        )
        
        sync._last_sync_time = time.time() - 2.0
        
        assert sync.check_need_sync() is True
    
    def test_check_need_sync_false(self):
        mock_api = Mock()
        mock_logger = Mock()
        sync = PositionSynchronizer(
            api=mock_api,
            logger=mock_logger,
            sync_interval_seconds=60.0,
        )
        
        sync._last_sync_time = time.time()
        
        assert sync.check_need_sync() is False


class TestRealtimeRunner:
    def test_realtime_runner_init(self):
        mock_connector = Mock()
        mock_api = Mock()
        mock_connector.get_api.return_value = mock_api
        
        config = {
            'realtime': {
                'heartbeat_interval_seconds': 60.0,
                'status_report_interval_seconds': 30.0,
                'order_timeout_seconds': 30.0,
                'max_order_retries': 3,
                'price_protection_percent': 0.1,
                'position_sync_interval_seconds': 60.0,
            },
            'notification': {
                'webhook_enabled': False,
                'webhook_url': '',
            },
        }
        
        runner = RealtimeRunner(connector=mock_connector, config=config)
        
        assert runner.connector == mock_connector
        assert runner.api == mock_api
        assert runner._heartbeat_interval_seconds == 60.0
        assert runner._status_report_interval_seconds == 30.0
    
    def test_register_strategy(self):
        mock_connector = Mock()
        mock_api = Mock()
        mock_connector.get_api.return_value = mock_api
        mock_strategy = Mock()
        
        runner = RealtimeRunner(connector=mock_connector, config={})
        runner.register_strategy("TestStrategy", mock_strategy)
        
        assert runner.get_strategy("TestStrategy") == mock_strategy
    
    def test_get_strategy_nonexistent(self):
        mock_connector = Mock()
        mock_api = Mock()
        mock_connector.get_api.return_value = mock_api
        
        runner = RealtimeRunner(connector=mock_connector, config={})
        
        assert runner.get_strategy("Nonexistent") is None


class TestWebhookConfigLoading:
    def test_load_webhook_config_from_settings(self):
        config = load_webhook_config_from_settings()
        
        assert isinstance(config, WebhookConfig)


class TestOrderManagerEdgeCases:
    def test_place_order_zero_volume(self):
        mock_api = Mock()
        mock_logger = Mock()
        manager = OrderManager(api=mock_api, logger=mock_logger)
        
        result = manager.place_order(
            contract="SHFE.rb2410",
            direction="BUY",
            offset="OPEN",
            volume=0,
        )
        
        assert result is None
    
    def test_calculate_protected_price_no_current_price(self):
        mock_api = Mock()
        mock_logger = Mock()
        manager = OrderManager(api=mock_api, logger=mock_logger)
        
        with patch.object(manager, '_get_current_price', return_value=None):
            protected_price = manager._calculate_protected_price(
                contract="SHFE.rb2410",
                direction="BUY",
                base_price=3500.0,
            )
            
            assert protected_price == 3500.0


class TestOrderTimeout:
    def test_check_order_timeout_no_timeout(self):
        mock_api = Mock()
        mock_logger = Mock()
        manager = OrderManager(
            api=mock_api,
            logger=mock_logger,
            timeout_seconds=30.0,
        )
        
        record = OrderRecord(
            order_id="test_1",
            contract="SHFE.rb2410",
            direction="BUY",
            offset="OPEN",
            volume=1,
            limit_price=3500.0,
            status=OrderStatus.PLACED,
            placed_time=time.time() - 10.0,
            timeout_seconds=30.0,
        )
        
        with manager._lock:
            manager._orders["test_1"] = record
            manager._orders_by_contract["SHFE.rb2410"] = ["test_1"]
        
        timeout_orders = manager.check_order_timeout()
        
        assert len(timeout_orders) == 0


class TestPositionSynchronizerEdgeCases:
    def test_get_target_position_nonexistent(self):
        mock_api = Mock()
        mock_logger = Mock()
        sync = PositionSynchronizer(api=mock_api, logger=mock_logger)
        
        assert sync.get_target_position("Nonexistent") is None
    
    def test_get_position_info_nonexistent(self):
        mock_api = Mock()
        mock_logger = Mock()
        sync = PositionSynchronizer(api=mock_api, logger=mock_logger)
        
        assert sync.get_position_info("Nonexistent") is None


class TestRealtimeRunnerEdgeCases:
    def test_init_without_connector(self):
        runner = RealtimeRunner(connector=None, config={})
        
        assert runner.connector is None
        assert runner.api is None
    
    def test_set_connector(self):
        mock_connector = Mock()
        mock_api = Mock()
        mock_connector.get_api.return_value = mock_api
        
        runner = RealtimeRunner(connector=None, config={})
        runner.set_connector(mock_connector)
        
        assert runner.connector == mock_connector
        assert runner.api == mock_api
    
    def test_set_connector_none_raises_error(self):
        runner = RealtimeRunner(connector=None, config={})
        
        with pytest.raises(ValueError, match="Connector 不能为 None"):
            runner.set_connector(None)
