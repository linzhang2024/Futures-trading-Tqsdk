import logging
import time
import json
import os
import threading
import requests
from typing import Dict, Any, List, Optional, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict
from threading import Lock

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

from tqsdk import TqApi, TqSim, TqAuth, TqKq, TqAccount

from strategies.base_strategy import StrategyBase, SignalType
from core.manager import StrategyManager
from core.risk_manager import (
    RiskManager,
    RiskLevel,
    RiskEvent,
    RiskEventType,
    PositionInfo,
    AccountSnapshot,
)


class OrderStatus(Enum):
    PENDING = "PENDING"
    PLACED = "PLACED"
    FILLED = "FILLED"
    CANCELED = "CANCELED"
    REJECTED = "REJECTED"
    TIMEOUT = "TIMEOUT"


class PositionDirection(Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    FLAT = "FLAT"


@dataclass
class OrderRecord:
    order_id: str
    contract: str
    direction: str
    offset: str
    volume: int
    limit_price: float
    status: OrderStatus
    placed_time: float
    filled_volume: int = 0
    filled_price: float = 0.0
    canceled_time: Optional[float] = None
    timeout_seconds: float = 30.0
    retry_count: int = 0
    max_retries: int = 3
    parent_order_id: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TargetPosition:
    contract: str
    target_long: int = 0
    target_short: int = 0
    current_long: int = 0
    current_short: int = 0
    last_sync_time: float = 0.0
    sync_interval_seconds: float = 60.0
    pending_orders: List[str] = field(default_factory=list)


@dataclass
class WebhookConfig:
    enabled: bool = False
    url: str = ""
    secret: str = ""
    timeout_seconds: float = 10.0
    retry_count: int = 3
    retry_delay_seconds: float = 2.0
    notify_on_trade: bool = True
    notify_on_risk: bool = True
    notify_on_order: bool = False


class OrderManager:
    
    def __init__(
        self,
        api: TqApi,
        logger: logging.Logger,
        timeout_seconds: float = 30.0,
        max_retries: int = 3,
        price_protection_percent: float = 0.1,
        on_order_filled: Callable = None,
        on_order_canceled: Callable = None,
        on_order_timeout: Callable = None,
    ):
        self.api = api
        self.logger = logger
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.price_protection_percent = price_protection_percent
        self.on_order_filled = on_order_filled
        self.on_order_canceled = on_order_canceled
        self.on_order_timeout = on_order_timeout
        
        self._orders: Dict[str, OrderRecord] = {}
        self._orders_by_contract: Dict[str, List[str]] = defaultdict(list)
        self._lock = Lock()
        
        self.logger.info(
            f"OrderManager 初始化: 超时={timeout_seconds}秒, "
            f"最大重试={max_retries}次, "
            f"价格保护偏离={price_protection_percent*100:.1f}%"
        )
    
    def _generate_order_id(self, contract: str, direction: str) -> str:
        timestamp = int(time.time() * 1000)
        return f"{contract}_{direction}_{timestamp}"
    
    def _get_current_price(self, contract: str) -> Optional[float]:
        try:
            quote = self.api.get_quote(contract)
            if hasattr(quote, 'last_price') and quote.last_price > 0:
                return float(quote.last_price)
            if hasattr(quote, 'close') and quote.close > 0:
                return float(quote.close)
            return None
        except Exception as e:
            self.logger.warning(f"获取 {contract} 当前价格失败: {e}")
            return None
    
    def _calculate_protected_price(
        self,
        contract: str,
        direction: str,
        base_price: float,
    ) -> float:
        current_price = self._get_current_price(contract)
        
        if current_price is None or current_price <= 0:
            return base_price
        
        protection_amount = current_price * self.price_protection_percent
        
        if direction == "BUY":
            max_allowed = current_price + protection_amount
            protected_price = min(base_price, max_allowed)
            if protected_price < base_price:
                self.logger.debug(
                    f"价格保护: 买入价 {base_price:.2f} 超过保护阈值, "
                    f"调整为 {protected_price:.2f} (当前价={current_price:.2f}, "
                    f"保护偏离={self.price_protection_percent*100:.1f}%)"
                )
            return protected_price
        else:
            min_allowed = current_price - protection_amount
            protected_price = max(base_price, min_allowed)
            if protected_price > base_price:
                self.logger.debug(
                    f"价格保护: 卖出价 {base_price:.2f} 超过保护阈值, "
                    f"调整为 {protected_price:.2f} (当前价={current_price:.2f}, "
                    f"保护偏离={self.price_protection_percent*100:.1f}%)"
                )
            return protected_price
    
    def place_order(
        self,
        contract: str,
        direction: str,
        offset: str,
        volume: int,
        limit_price: float = 0.0,
        timeout_seconds: float = None,
        max_retries: int = None,
    ) -> Optional[OrderRecord]:
        if volume <= 0:
            self.logger.warning(f"订单数量无效: {volume}, 跳过下单")
            return None
        
        timeout = timeout_seconds or self.timeout_seconds
        retries = max_retries if max_retries is not None else self.max_retries
        
        try:
            quote = self.api.get_quote(contract)
            
            if limit_price <= 0:
                current_price = self._get_current_price(contract)
                if current_price is None:
                    self.logger.error(f"无法获取 {contract} 当前价格，且未指定限价，无法下单")
                    return None
                limit_price = current_price
                self.logger.debug(f"未指定限价，使用当前价 {limit_price:.2f} 作为限价")
            
            protected_price = self._calculate_protected_price(
                contract=contract,
                direction=direction,
                base_price=limit_price,
            )
            
            order_id = self._generate_order_id(contract, direction)
            
            self.logger.info(
                f"[下单] {contract} | 方向={direction} | 开平={offset} | "
                f"数量={volume}手 | 限价={limit_price:.2f} | 保护价={protected_price:.2f} | "
                f"超时={timeout}秒 | 最大重试={retries}次"
            )
            
            tq_order = self.api.insert_order(
                quote,
                direction=direction,
                offset=offset,
                volume=volume,
                limit_price=protected_price,
            )
            
            if tq_order is None:
                self.logger.error(f"下单失败: TqApi 返回 None")
                return None
            
            record = OrderRecord(
                order_id=order_id,
                contract=contract,
                direction=direction,
                offset=offset,
                volume=volume,
                limit_price=protected_price,
                status=OrderStatus.PLACED,
                placed_time=time.time(),
                timeout_seconds=timeout,
                max_retries=retries,
                details={
                    'original_price': limit_price,
                    'tq_order_id': getattr(tq_order, 'order_id', None),
                },
            )
            
            with self._lock:
                self._orders[order_id] = record
                self._orders_by_contract[contract].append(order_id)
            
            self.logger.info(f"订单已提交: order_id={order_id}")
            
            return record
            
        except Exception as e:
            self.logger.error(f"下单异常: {e}", exc_info=True)
            return None
    
    def check_order_timeout(self) -> List[OrderRecord]:
        now = time.time()
        timeout_orders = []
        
        with self._lock:
            for order_id, record in list(self._orders.items()):
                if record.status != OrderStatus.PLACED:
                    continue
                
                elapsed = now - record.placed_time
                if elapsed >= record.timeout_seconds:
                    timeout_orders.append(record)
        
        for record in timeout_orders:
            self._handle_timeout(record)
        
        return timeout_orders
    
    def _handle_timeout(self, record: OrderRecord):
        self.logger.warning(
            f"[订单超时] order_id={record.order_id}, "
            f"合约={record.contract}, "
            f"已等待={time.time() - record.placed_time:.1f}秒, "
            f"超时={record.timeout_seconds}秒"
        )
        
        try:
            self.api.cancel_order(record.order_id)
            self.logger.info(f"已发送撤单请求: order_id={record.order_id}")
        except Exception as e:
            self.logger.warning(f"撤单请求失败: {e}")
        
        with self._lock:
            record.status = OrderStatus.TIMEOUT
            record.canceled_time = time.time()
        
        if record.retry_count < record.max_retries:
            self._retry_order(record)
        else:
            self.logger.error(
                f"订单已达最大重试次数 {record.max_retries}, 放弃: "
                f"order_id={record.order_id}"
            )
            if self.on_order_timeout:
                self.on_order_timeout(record)
    
    def _retry_order(self, record: OrderRecord):
        record.retry_count += 1
        
        current_price = self._get_current_price(record.contract)
        if current_price is None:
            self.logger.warning(
                f"无法获取当前价格，跳过重试: order_id={record.order_id}"
            )
            return
        
        new_price = current_price
        
        if record.direction == "BUY":
            new_price = current_price * 1.001
        else:
            new_price = current_price * 0.999
        
        self.logger.info(
            f"[追单重试] 第{record.retry_count}次, "
            f"合约={record.contract}, "
            f"原价格={record.limit_price:.2f}, "
            f"新价格={new_price:.2f}"
        )
        
        new_record = self.place_order(
            contract=record.contract,
            direction=record.direction,
            offset=record.offset,
            volume=record.volume - record.filled_volume,
            limit_price=new_price,
            timeout_seconds=record.timeout_seconds,
            max_retries=record.max_retries,
        )
        
        if new_record:
            new_record.parent_order_id = record.order_id
            with self._lock:
                if record.contract not in self._orders_by_contract:
                    self._orders_by_contract[record.contract] = []
                self._orders_by_contract[record.contract].append(new_record.order_id)
    
    def update_order_status(self):
        with self._lock:
            for order_id, record in list(self._orders.items()):
                if record.status not in [OrderStatus.PLACED, OrderStatus.PENDING]:
                    continue
                
                try:
                    tq_orders = self.api.get_order()
                    
                    tq_order = None
                    for oid, o in tq_orders.items():
                        if (o.get('contract') == record.contract and
                            o.get('volume_orign') == record.volume):
                            tq_order = o
                            break
                    
                    if tq_order is None:
                        continue
                    
                    status = tq_order.get('status', '')
                    volume_left = tq_order.get('volume_left', record.volume)
                    volume_trade = record.volume - volume_left
                    
                    if status == 'FINISHED' or volume_left <= 0:
                        record.status = OrderStatus.FILLED
                        record.filled_volume = record.volume
                        trade_price = tq_order.get('trade_price', record.limit_price)
                        record.filled_price = float(trade_price) if trade_price else record.limit_price
                        
                        self.logger.info(
                            f"[订单成交] order_id={order_id}, "
                            f"合约={record.contract}, "
                            f"数量={record.filled_volume}手, "
                            f"价格={record.filled_price:.2f}"
                        )
                        
                        if self.on_order_filled:
                            self.on_order_filled(record)
                    
                    elif status in ['CANCELED', 'CANCELLED']:
                        record.status = OrderStatus.CANCELED
                        record.canceled_time = time.time()
                        record.filled_volume = volume_trade
                        
                        self.logger.info(
                            f"[订单已撤单] order_id={order_id}, "
                            f"合约={record.contract}, "
                            f"已成交={volume_trade}手"
                        )
                        
                        if self.on_order_canceled:
                            self.on_order_canceled(record)
                    
                    elif status == 'REJECTED':
                        record.status = OrderStatus.REJECTED
                        self.logger.error(
                            f"[订单被拒] order_id={order_id}, "
                            f"合约={record.contract}"
                        )
                        
                except Exception as e:
                    self.logger.debug(f"更新订单状态异常 (可忽略): {e}")
    
    def get_active_orders(self, contract: str = None) -> List[OrderRecord]:
        with self._lock:
            if contract:
                order_ids = self._orders_by_contract.get(contract, [])
                return [
                    self._orders[oid] for oid in order_ids
                    if self._orders[oid].status in [OrderStatus.PLACED, OrderStatus.PENDING]
                ]
            else:
                return [
                    record for record in self._orders.values()
                    if record.status in [OrderStatus.PLACED, OrderStatus.PENDING]
                ]
    
    def get_order(self, order_id: str) -> Optional[OrderRecord]:
        with self._lock:
            return self._orders.get(order_id)
    
    def cancel_all_orders(self, contract: str = None) -> int:
        canceled_count = 0
        active_orders = self.get_active_orders(contract)
        
        for record in active_orders:
            try:
                self.api.cancel_order(record.order_id)
                with self._lock:
                    record.status = OrderStatus.CANCELED
                    record.canceled_time = time.time()
                canceled_count += 1
                self.logger.info(f"已撤单: order_id={record.order_id}")
            except Exception as e:
                self.logger.warning(f"撤单失败 order_id={record.order_id}: {e}")
        
        return canceled_count


class PositionSynchronizer:
    
    def __init__(
        self,
        api: TqApi,
        logger: logging.Logger,
        sync_interval_seconds: float = 60.0,
        on_sync_complete: Callable = None,
        on_position_mismatch: Callable = None,
    ):
        self.api = api
        self.logger = logger
        self.sync_interval_seconds = sync_interval_seconds
        self.on_sync_complete = on_sync_complete
        self.on_position_mismatch = on_position_mismatch
        
        self._target_positions: Dict[str, TargetPosition] = {}
        self._exchange_positions: Dict[str, PositionInfo] = {}
        self._last_sync_time: float = 0.0
        self._lock = Lock()
        
        self.logger.info(
            f"PositionSynchronizer 初始化: 同步间隔={sync_interval_seconds}秒"
        )
    
    def set_target_position(
        self,
        contract: str,
        target_long: int = 0,
        target_short: int = 0,
    ):
        with self._lock:
            if contract not in self._target_positions:
                self._target_positions[contract] = TargetPosition(
                    contract=contract,
                    sync_interval_seconds=self.sync_interval_seconds,
                )
            
            tp = self._target_positions[contract]
            tp.target_long = target_long
            tp.target_short = target_short
            
            self.logger.info(
                f"[目标持仓设置] 合约={contract}, "
                f"目标多单={target_long}手, "
                f"目标空单={target_short}手"
            )
    
    def get_target_position(self, contract: str) -> Optional[TargetPosition]:
        with self._lock:
            return self._target_positions.get(contract)
    
    def _fetch_exchange_positions(self) -> Dict[str, PositionInfo]:
        positions = {}
        
        try:
            tq_positions = self.api.get_position()
            
            for contract, pos_data in tq_positions.items():
                if not pos_data:
                    continue
                
                try:
                    position = PositionInfo(
                        contract=contract,
                        long_volume=int(pos_data.get('buy_volume', 0)),
                        short_volume=int(pos_data.get('sell_volume', 0)),
                        long_margin=float(pos_data.get('buy_margin', 0)),
                        short_margin=float(pos_data.get('sell_margin', 0)),
                        long_open_price=float(pos_data.get('buy_open_price', 0)),
                        short_open_price=float(pos_data.get('sell_open_price', 0)),
                        current_price=float(pos_data.get('last_price', 0)),
                        float_profit=float(pos_data.get('float_profit', 0)),
                    )
                    positions[contract] = position
                except (ValueError, TypeError) as e:
                    self.logger.warning(f"解析持仓数据失败 [{contract}]: {e}")
            
        except Exception as e:
            self.logger.error(f"获取交易所持仓失败: {e}")
        
        return positions
    
    def check_need_sync(self) -> bool:
        now = time.time()
        if now - self._last_sync_time >= self.sync_interval_seconds:
            return True
        return False
    
    def sync_positions(self) -> Dict[str, Any]:
        now = time.time()
        
        self.logger.info("[持仓同步] 开始同步本地与交易所持仓...")
        
        exchange_positions = self._fetch_exchange_positions()
        
        sync_report = {
            'sync_time': now,
            'sync_datetime': datetime.fromtimestamp(now).isoformat(),
            'contracts': {},
            'mismatches': [],
            'total_mismatches': 0,
        }
        
        with self._lock:
            for contract, tp in self._target_positions.items():
                exchange_pos = exchange_positions.get(contract)
                
                if exchange_pos:
                    tp.current_long = exchange_pos.long_volume
                    tp.current_short = exchange_pos.short_volume
                    self._exchange_positions[contract] = exchange_pos
                else:
                    tp.current_long = 0
                    tp.current_short = 0
                
                long_mismatch = tp.target_long != tp.current_long
                short_mismatch = tp.target_short != tp.current_short
                
                contract_report = {
                    'contract': contract,
                    'target_long': tp.target_long,
                    'target_short': tp.target_short,
                    'current_long': tp.current_long,
                    'current_short': tp.current_short,
                    'long_mismatch': long_mismatch,
                    'short_mismatch': short_mismatch,
                    'in_sync': not (long_mismatch or short_mismatch),
                }
                
                sync_report['contracts'][contract] = contract_report
                
                if long_mismatch or short_mismatch:
                    sync_report['mismatches'].append({
                        'contract': contract,
                        'type': 'LONG_MISMATCH' if long_mismatch else 'SHORT_MISMATCH',
                        'target': tp.target_long if long_mismatch else tp.target_short,
                        'current': tp.current_long if long_mismatch else tp.current_short,
                    })
                    sync_report['total_mismatches'] += 1
                    
                    self.logger.warning(
                        f"[持仓不匹配] 合约={contract}, "
                        f"目标多单={tp.target_long}, 当前多单={tp.current_long}, "
                        f"目标空单={tp.target_short}, 当前空单={tp.current_short}"
                    )
                    
                    if self.on_position_mismatch:
                        self.on_position_mismatch(contract, tp, exchange_pos)
                
                tp.last_sync_time = now
            
            self._last_sync_time = now
        
        if sync_report['total_mismatches'] > 0:
            self.logger.warning(
                f"[持仓同步完成] 发现 {sync_report['total_mismatches']} 个持仓不匹配"
            )
        else:
            self.logger.info("[持仓同步完成] 所有持仓匹配正常")
        
        if self.on_sync_complete:
            self.on_sync_complete(sync_report)
        
        return sync_report
    
    def get_position_info(self, contract: str) -> Optional[PositionInfo]:
        with self._lock:
            return self._exchange_positions.get(contract)
    
    def get_all_positions(self) -> Dict[str, PositionInfo]:
        with self._lock:
            return self._exchange_positions.copy()


class WebhookNotifier:
    
    def __init__(
        self,
        config: WebhookConfig,
        logger: logging.Logger,
    ):
        self.config = config
        self.logger = logger
        self._session = requests.Session()
        self._lock = Lock()
        
        self.logger.info(
            f"WebhookNotifier 初始化: "
            f"enabled={config.enabled}, "
            f"url={config.url[:50] + '...' if config.url and len(config.url) > 50 else config.url}"
        )
    
    def _build_signature(self, payload: Dict[str, Any], timestamp: str) -> str:
        if not self.config.secret:
            return ""
        
        import hmac
        import hashlib
        
        message = f"{timestamp}\n{self.config.secret}"
        signature = hmac.new(
            self.config.secret.encode('utf-8'),
            message.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        return signature
    
    def _send_request(self, payload: Dict[str, Any]) -> bool:
        if not self.config.enabled:
            return False
        
        if not self.config.url:
            self.logger.warning("Webhook URL 未配置")
            return False
        
        timestamp = str(int(time.time()))
        signature = self._build_signature(payload, timestamp)
        
        headers = {
            'Content-Type': 'application/json',
            'X-Timestamp': timestamp,
        }
        
        if signature:
            headers['X-Signature'] = signature
        
        for attempt in range(self.config.retry_count):
            try:
                response = self._session.post(
                    self.config.url,
                    json=payload,
                    headers=headers,
                    timeout=self.config.timeout_seconds,
                )
                
                if response.status_code in [200, 201, 202, 204]:
                    self.logger.debug(f"Webhook 发送成功: status_code={response.status_code}")
                    return True
                else:
                    self.logger.warning(
                        f"Webhook 响应异常: status_code={response.status_code}, "
                        f"response={response.text[:200]}"
                    )
                    
            except requests.exceptions.Timeout:
                self.logger.warning(f"Webhook 超时 (尝试 {attempt + 1}/{self.config.retry_count})")
            except requests.exceptions.RequestException as e:
                self.logger.warning(f"Webhook 请求异常 (尝试 {attempt + 1}/{self.config.retry_count}): {e}")
            
            if attempt < self.config.retry_count - 1:
                time.sleep(self.config.retry_delay_seconds)
        
        self.logger.error(f"Webhook 发送失败，已重试 {self.config.retry_count} 次")
        return False
    
    def notify_trade(
        self,
        event_type: str,
        strategy_name: str,
        contract: str,
        direction: str,
        offset: str,
        volume: int,
        price: float,
        equity: float = 0.0,
        profit_loss: float = None,
        order_id: str = None,
    ):
        if not self.config.notify_on_trade:
            return
        
        payload = {
            'msgtype': 'markdown',
            'markdown': {
                'title': '交易通知',
                'text': self._format_trade_message(
                    event_type=event_type,
                    strategy_name=strategy_name,
                    contract=contract,
                    direction=direction,
                    offset=offset,
                    volume=volume,
                    price=price,
                    equity=equity,
                    profit_loss=profit_loss,
                    order_id=order_id,
                ),
            },
            'timestamp': datetime.now().isoformat(),
            'event_type': 'TRADE',
            'data': {
                'event_type': event_type,
                'strategy_name': strategy_name,
                'contract': contract,
                'direction': direction,
                'offset': offset,
                'volume': volume,
                'price': price,
                'equity': equity,
                'profit_loss': profit_loss,
                'order_id': order_id,
            },
        }
        
        self._send_request(payload)
        self.logger.info(f"[交易通知] 已发送: {event_type} {contract} {direction} {volume}手 @ {price:.2f}")
    
    def notify_risk(
        self,
        risk_event: RiskEvent,
        equity: float = 0.0,
        frozen_reason: str = None,
    ):
        if not self.config.notify_on_risk:
            return
        
        payload = {
            'msgtype': 'markdown',
            'markdown': {
                'title': '风控预警',
                'text': self._format_risk_message(
                    risk_event=risk_event,
                    equity=equity,
                    frozen_reason=frozen_reason,
                ),
            },
            'timestamp': datetime.now().isoformat(),
            'event_type': 'RISK',
            'data': {
                'event_type': risk_event.event_type.value,
                'level': risk_event.level.value,
                'message': risk_event.message,
                'details': risk_event.details,
                'equity': equity,
                'frozen_reason': frozen_reason,
            },
        }
        
        self._send_request(payload)
        self.logger.warning(f"[风控通知] 已发送: {risk_event.event_type.value} - {risk_event.message}")
    
    def notify_order(
        self,
        order_record: OrderRecord,
        event_type: str,
        equity: float = 0.0,
    ):
        if not self.config.notify_on_order:
            return
        
        payload = {
            'msgtype': 'markdown',
            'markdown': {
                'title': '订单状态更新',
                'text': self._format_order_message(
                    order_record=order_record,
                    event_type=event_type,
                    equity=equity,
                ),
            },
            'timestamp': datetime.now().isoformat(),
            'event_type': 'ORDER',
            'data': {
                'order_id': order_record.order_id,
                'contract': order_record.contract,
                'direction': order_record.direction,
                'offset': order_record.offset,
                'volume': order_record.volume,
                'price': order_record.limit_price,
                'status': order_record.status.value,
                'event_type': event_type,
                'equity': equity,
            },
        }
        
        self._send_request(payload)
    
    def _format_trade_message(
        self,
        event_type: str,
        strategy_name: str,
        contract: str,
        direction: str,
        offset: str,
        volume: int,
        price: float,
        equity: float = 0.0,
        profit_loss: float = None,
        order_id: str = None,
    ) -> str:
        direction_emoji = "🔴" if direction == "SELL" else "🟢"
        offset_emoji = "📉" if offset == "CLOSE" else "📈"
        
        lines = [
            f"## 📊 交易通知",
            f"",
            f"**事件类型**: {event_type}",
            f"**策略名称**: {strategy_name}",
            f"**交易合约**: {contract}",
            f"",
            f"**交易方向**: {direction_emoji} {direction}",
            f"**开平标志**: {offset_emoji} {offset}",
            f"**交易数量**: {volume} 手",
            f"**成交价格**: {price:.2f}",
            f"",
        ]
        
        if profit_loss is not None:
            pl_emoji = "🟢" if profit_loss >= 0 else "🔴"
            lines.append(f"**盈亏情况**: {pl_emoji} {profit_loss:,.2f} 元")
            lines.append(f"")
        
        if equity > 0:
            lines.append(f"**当前权益**: 💰 {equity:,.2f} 元")
            lines.append(f"")
        
        if order_id:
            lines.append(f"**订单ID**: `{order_id}`")
            lines.append(f"")
        
        lines.append(f"---")
        lines.append(f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        return "\n".join(lines)
    
    def _format_risk_message(
        self,
        risk_event: RiskEvent,
        equity: float = 0.0,
        frozen_reason: str = None,
    ) -> str:
        level_emoji = {
            RiskLevel.SAFE: "🟢",
            RiskLevel.WARNING: "🟡",
            RiskLevel.CRITICAL: "🔴",
            RiskLevel.FROZEN: "🔥",
        }.get(risk_event.level, "⚪")
        
        lines = [
            f"## ⚠️ 风控预警",
            f"",
            f"**风险级别**: {level_emoji} {risk_event.level.value}",
            f"**事件类型**: {risk_event.event_type.value}",
            f"",
            f"**详细信息**:",
            f"> {risk_event.message}",
            f"",
        ]
        
        if risk_event.details:
            lines.append(f"**附加信息**:")
            for key, value in risk_event.details.items():
                lines.append(f"- {key}: {value}")
            lines.append(f"")
        
        if frozen_reason:
            lines.append(f"### 🔥 系统已冻结")
            lines.append(f"**冻结原因**: {frozen_reason}")
            lines.append(f"")
        
        if equity > 0:
            lines.append(f"**当前权益**: 💰 {equity:,.2f} 元")
            lines.append(f"")
        
        lines.append(f"---")
        lines.append(f"⏰ {datetime.fromtimestamp(risk_event.timestamp).strftime('%Y-%m-%d %H:%M:%S')}")
        
        return "\n".join(lines)
    
    def _format_order_message(
        self,
        order_record: OrderRecord,
        event_type: str,
        equity: float = 0.0,
    ) -> str:
        status_emoji = {
            OrderStatus.PLACED: "📤",
            OrderStatus.FILLED: "✅",
            OrderStatus.CANCELED: "❌",
            OrderStatus.TIMEOUT: "⏰",
            OrderStatus.REJECTED: "🚫",
        }.get(order_record.status, "📋")
        
        lines = [
            f"## 📋 订单状态更新",
            f"",
            f"**订单ID**: `{order_record.order_id}`",
            f"**交易合约**: {order_record.contract}",
            f"",
            f"**当前状态**: {status_emoji} {order_record.status.value}",
            f"**更新事件**: {event_type}",
            f"",
            f"**交易方向**: {order_record.direction}",
            f"**开平标志**: {order_record.offset}",
            f"**订单数量**: {order_record.volume} 手",
            f"**限价**: {order_record.limit_price:.2f}",
            f"",
        ]
        
        if order_record.filled_volume > 0:
            lines.append(f"**已成交**: {order_record.filled_volume} 手")
            if order_record.filled_price > 0:
                lines.append(f"**成交均价**: {order_record.filled_price:.2f}")
            lines.append(f"")
        
        if order_record.retry_count > 0:
            lines.append(f"**重试次数**: {order_record.retry_count}/{order_record.max_retries}")
            lines.append(f"")
        
        if equity > 0:
            lines.append(f"**当前权益**: 💰 {equity:,.2f} 元")
            lines.append(f"")
        
        lines.append(f"---")
        lines.append(f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        return "\n".join(lines)


class RealtimeRunner:
    
    def __init__(
        self,
        connector=None,
        config: Dict[str, Any] = None,
        config_path: str = None,
    ):
        self.connector = connector
        self.api = None
        self.logger = logging.getLogger(self.__class__.__name__)
        
        if connector:
            self.api = connector.get_api()
        
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        if config is None and config_path is None:
            config_path = os.path.join(base_dir, 'config', 'settings.yaml')
        
        if config_path and os.path.exists(config_path) and YAML_AVAILABLE:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            self.logger.info(f"已加载配置文件: {config_path}")
        
        self.config = config or {}
        
        self._running = False
        self._initialized = False
        self._cycle_count = 0
        self._start_time = 0.0
        self._last_heartbeat_time = 0.0
        
        self._heartbeat_interval_seconds = 60.0
        self._status_report_interval_seconds = 30.0
        self._last_status_report_time = 0.0
        
        self._strategy_manager: Optional[StrategyManager] = None
        self._risk_manager: Optional[RiskManager] = None
        self._order_manager: Optional[OrderManager] = None
        self._position_synchronizer: Optional[PositionSynchronizer] = None
        self._webhook_notifier: Optional[WebhookNotifier] = None
        
        self._strategies: Dict[str, StrategyBase] = {}
        
        self._init_components()
        
        self.logger.info("RealtimeRunner 初始化完成")
    
    def _init_components(self):
        realtime_config = self.config.get('realtime', {})
        trading_config = self.config.get('trading', {})
        notification_config = self.config.get('notification', {})
        
        self._heartbeat_interval_seconds = realtime_config.get(
            'heartbeat_interval_seconds', 60.0
        )
        self._status_report_interval_seconds = realtime_config.get(
            'status_report_interval_seconds', 30.0
        )
        
        order_timeout = realtime_config.get('order_timeout_seconds', 30.0)
        max_retries = realtime_config.get('max_order_retries', 3)
        price_protection = realtime_config.get('price_protection_percent', 0.1) / 100.0
        
        sync_interval = realtime_config.get('position_sync_interval_seconds', 60.0)
        
        webhook_config = WebhookConfig(
            enabled=notification_config.get('webhook_enabled', False),
            url=notification_config.get('webhook_url', ''),
            secret=notification_config.get('webhook_secret', ''),
            timeout_seconds=notification_config.get('webhook_timeout_seconds', 10.0),
            retry_count=notification_config.get('webhook_retry_count', 3),
            retry_delay_seconds=notification_config.get('webhook_retry_delay_seconds', 2.0),
            notify_on_trade=notification_config.get('notify_on_trade', True),
            notify_on_risk=notification_config.get('notify_on_risk', True),
            notify_on_order=notification_config.get('notify_on_order', False),
        )
        
        if self.api:
            self._order_manager = OrderManager(
                api=self.api,
                logger=self.logger,
                timeout_seconds=order_timeout,
                max_retries=max_retries,
                price_protection_percent=price_protection,
                on_order_filled=self._on_order_filled,
                on_order_canceled=self._on_order_canceled,
                on_order_timeout=self._on_order_timeout,
            )
            
            self._position_synchronizer = PositionSynchronizer(
                api=self.api,
                logger=self.logger,
                sync_interval_seconds=sync_interval,
                on_sync_complete=self._on_position_sync_complete,
                on_position_mismatch=self._on_position_mismatch,
            )
        
        self._webhook_notifier = WebhookNotifier(
            config=webhook_config,
            logger=self.logger,
        )
    
    def set_connector(self, connector):
        if connector is None:
            self.logger.error("Connector 不能为 None")
            raise ValueError("Connector 不能为 None")
        
        self.connector = connector
        self.api = connector.get_api()
        self.logger.info("Connector 已设置")
        
        realtime_config = self.config.get('realtime', {})
        order_timeout = realtime_config.get('order_timeout_seconds', 30.0)
        max_retries = realtime_config.get('max_order_retries', 3)
        price_protection = realtime_config.get('price_protection_percent', 0.1) / 100.0
        sync_interval = realtime_config.get('position_sync_interval_seconds', 60.0)
        
        self._order_manager = OrderManager(
            api=self.api,
            logger=self.logger,
            timeout_seconds=order_timeout,
            max_retries=max_retries,
            price_protection_percent=price_protection,
            on_order_filled=self._on_order_filled,
            on_order_canceled=self._on_order_canceled,
            on_order_timeout=self._on_order_timeout,
        )
        
        self._position_synchronizer = PositionSynchronizer(
            api=self.api,
            logger=self.logger,
            sync_interval_seconds=sync_interval,
            on_sync_complete=self._on_position_sync_complete,
            on_position_mismatch=self._on_position_mismatch,
        )
    
    def register_strategy(self, name: str, strategy: StrategyBase) -> None:
        self._strategies[name] = strategy
        self.logger.info(f"已注册策略: {name}")
    
    def get_strategy(self, name: str) -> Optional[StrategyBase]:
        return self._strategies.get(name)
    
    def initialize(self) -> None:
        if self._initialized:
            self.logger.info("RealtimeRunner 已初始化，跳过重复初始化")
            return
        
        if self.api is None:
            raise RuntimeError("API 未初始化，请确保 Connector 已连接")
        
        self.logger.info("正在初始化 RealtimeRunner 组件...")
        
        self._strategy_manager = StrategyManager(connector=self.connector)
        self._strategy_manager.configure_from_dict(self.config)
        
        for name, strategy in self._strategies.items():
            self._strategy_manager.register_strategy(name, strategy)
        
        self._risk_manager = RiskManager(connector=self.connector)
        self._strategy_manager.set_risk_manager(self._risk_manager)
        
        risk_config = self.config.get('risk', {})
        if risk_config.get('max_drawdown_percent'):
            self._risk_manager.max_drawdown_percent = float(risk_config['max_drawdown_percent'])
        if risk_config.get('max_strategy_margin_percent'):
            self._risk_manager.max_strategy_margin_percent = float(risk_config['max_strategy_margin_percent'])
        if risk_config.get('max_total_margin_percent'):
            self._risk_manager.max_total_margin_percent = float(risk_config['max_total_margin_percent'])
        
        self._strategy_manager.initialize(load_saved_states=False)
        
        if self._risk_manager:
            self._risk_manager.initialize()
        
        self._initialized = True
        self.logger.info("RealtimeRunner 初始化完成")
    
    def _on_order_filled(self, order_record: OrderRecord):
        self.logger.info(
            f"[订单成交回调] order_id={order_record.order_id}, "
            f"合约={order_record.contract}, "
            f"数量={order_record.filled_volume}手, "
            f"价格={order_record.filled_price:.2f}"
        )
        
        equity = self._get_current_equity()
        
        if self._webhook_notifier:
            self._webhook_notifier.notify_trade(
                event_type="ORDER_FILLED",
                strategy_name="RealtimeRunner",
                contract=order_record.contract,
                direction=order_record.direction,
                offset=order_record.offset,
                volume=order_record.filled_volume,
                price=order_record.filled_price,
                equity=equity,
                order_id=order_record.order_id,
            )
    
    def _on_order_canceled(self, order_record: OrderRecord):
        self.logger.info(
            f"[订单撤单回调] order_id={order_record.order_id}, "
            f"合约={order_record.contract}"
        )
        
        if self._webhook_notifier and self._webhook_notifier.config.notify_on_order:
            equity = self._get_current_equity()
            self._webhook_notifier.notify_order(
                order_record=order_record,
                event_type="CANCELED",
                equity=equity,
            )
    
    def _on_order_timeout(self, order_record: OrderRecord):
        self.logger.warning(
            f"[订单超时回调] order_id={order_record.order_id}, "
            f"合约={order_record.contract}, "
            f"重试次数={order_record.retry_count}/{order_record.max_retries}"
        )
        
        if self._webhook_notifier and self._webhook_notifier.config.notify_on_order:
            equity = self._get_current_equity()
            self._webhook_notifier.notify_order(
                order_record=order_record,
                event_type="TIMEOUT",
                equity=equity,
            )
    
    def _on_position_sync_complete(self, sync_report: Dict[str, Any]):
        self.logger.debug(f"[持仓同步完成回调] 同步报告: {json.dumps(sync_report, ensure_ascii=False, default=str)[:200]}")
    
    def _on_position_mismatch(
        self,
        contract: str,
        target_position: TargetPosition,
        exchange_position: Optional[PositionInfo],
    ):
        self.logger.warning(
            f"[持仓不匹配回调] 合约={contract}, "
            f"目标=多{target_position.target_long}/空{target_position.target_short}, "
            f"当前=多{target_position.current_long}/空{target_position.current_short}"
        )
    
    def _get_current_equity(self) -> float:
        if self._risk_manager:
            snapshot = self._risk_manager.get_account_snapshot()
            return snapshot.equity
        elif self.api:
            try:
                account = self.api.get_account()
                return float(account.get('equity', 0))
            except:
                pass
        return 0.0
    
    def _should_send_heartbeat(self) -> bool:
        now = time.time()
        if now - self._last_heartbeat_time >= self._heartbeat_interval_seconds:
            self._last_heartbeat_time = now
            return True
        return False
    
    def _should_report_status(self) -> bool:
        now = time.time()
        if now - self._last_status_report_time >= self._status_report_interval_seconds:
            self._last_status_report_time = now
            return True
        return False
    
    def _send_heartbeat(self):
        self.logger.info("")
        self.logger.info("=" * 80)
        self.logger.info(f"❤️  心跳检测 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info(f"   运行周期: {self._cycle_count} 次")
        self.logger.info(f"   运行时间: {timedelta(seconds=int(time.time() - self._start_time))}")
        self.logger.info("")
        self.logger.info("📊 策略状态:")
        
        for name, strategy in self._strategies.items():
            is_ready = strategy.is_ready() if hasattr(strategy, 'is_ready') else False
            signal = strategy.signal if hasattr(strategy, 'signal') else SignalType.HOLD
            
            status_icon = "✅" if is_ready else "⏳"
            signal_icon = "🟢" if signal == SignalType.BUY else "🔴" if signal == SignalType.SELL else "⚪"
            
            self.logger.info(f"   {status_icon} {name}:")
            self.logger.info(f"      就绪状态: {'已就绪' if is_ready else '数据收集中'}")
            self.logger.info(f"      当前信号: {signal_icon} {signal.value}")
            
            if hasattr(strategy, 'get_ma_values'):
                ma_values = strategy.get_ma_values()
                if ma_values:
                    ma_str = ", ".join([f"{k}={v:.2f}" if v else f"{k}=N/A" for k, v in ma_values.items()])
                    self.logger.info(f"      技术指标: {ma_str}")
        
        self.logger.info("")
        self.logger.info("⏳ 等待信号中...")
        self.logger.info("=" * 80)
        self.logger.info("")
    
    def _report_status(self):
        if not self._strategy_manager:
            return
        
        try:
            status_report = self._strategy_manager._format_status_report()
            self.logger.info(status_report)
            
            if self._risk_manager:
                risk_info = self._risk_manager.get_total_risk_info()
                drawdown_info = risk_info.get('drawdown_info', {})
                self.logger.info(
                    f"[风控状态] 权益: {risk_info.get('equity', 0):.2f}, "
                    f"回撤: {drawdown_info.get('current_drawdown_percent', 0):.2f}%, "
                    f"冻结: {risk_info.get('is_frozen', False)}"
                )
            
            if self._position_synchronizer:
                positions = self._position_synchronizer.get_all_positions()
                if positions:
                    self.logger.info("[持仓状态]")
                    for contract, pos in positions.items():
                        self.logger.info(
                            f"   {contract}: 多单={pos.long_volume}手, 空单={pos.short_volume}手, "
                            f"浮动盈亏={pos.float_profit:,.2f}元"
                        )
            
            active_orders = self._order_manager.get_active_orders() if self._order_manager else []
            if active_orders:
                self.logger.info(f"[活跃订单] 共 {len(active_orders)} 笔待成交订单")
                for order in active_orders:
                    elapsed = time.time() - order.placed_time
                    self.logger.info(
                        f"   {order.contract}: {order.direction} {order.offset} {order.volume}手 "
                        f"@ {order.limit_price:.2f}, 已等待 {elapsed:.1f}秒"
                    )
            
            self.logger.info("⏳ 等待信号中...")
            
        except Exception as e:
            self.logger.error(f"生成状态报告失败: {e}")
    
    def _run_main_loop(self):
        if self.api is None:
            raise RuntimeError("API 未初始化")
        
        if self._risk_manager and not self._risk_manager._initialized:
            self._risk_manager.initialize()
        
        self.logger.info("开始运行 RealtimeRunner 主循环...")
        self._running = True
        self._start_time = time.time()
        self._cycle_count = 0
        self._last_heartbeat_time = time.time()
        self._last_status_report_time = time.time()
        
        self.logger.info("=" * 80)
        self.logger.info("                🚀 实时交易系统启动")
        self.logger.info("=" * 80)
        self.logger.info(f"启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info(f"策略数量: {len(self._strategies)}")
        self.logger.info(f"心跳间隔: {self._heartbeat_interval_seconds} 秒")
        self.logger.info(f"状态报告间隔: {self._status_report_interval_seconds} 秒")
        self.logger.info(f"订单超时: {self._order_manager.timeout_seconds if self._order_manager else 'N/A'} 秒")
        self.logger.info(f"持仓同步间隔: {self._position_synchronizer.sync_interval_seconds if self._position_synchronizer else 'N/A'} 秒")
        self.logger.info("=" * 80)
        self.logger.info("")
        
        self._send_heartbeat()
        
        while self._running:
            try:
                if self._strategy_manager and self._strategy_manager.is_risk_frozen():
                    self.logger.critical("系统已被风控冻结，停止运行")
                    self._running = False
                    break
                
                self.api.wait_update()
                self._cycle_count += 1
                
                if self._order_manager:
                    self._order_manager.update_order_status()
                    self._order_manager.check_order_timeout()
                
                if self._position_synchronizer and self._position_synchronizer.check_need_sync():
                    self._position_synchronizer.sync_positions()
                
                if self._strategy_manager:
                    if self._strategy_manager._should_run_risk_check():
                        risk_results = self._strategy_manager.run_risk_checks()
                        
                        if self._strategy_manager.is_risk_frozen():
                            frozen_reason = self._risk_manager.get_frozen_reason() if self._risk_manager else "未知原因"
                            self.logger.critical(f"风控检查触发冻结: {frozen_reason}")
                            
                            if self._webhook_notifier:
                                equity = self._get_current_equity()
                                self._webhook_notifier.notify_risk(
                                    risk_event=RiskEvent(
                                        event_type=RiskEventType.DRAWDOWN_EXCEEDED,
                                        timestamp=time.time(),
                                        level=RiskLevel.FROZEN,
                                        message=f"系统冻结: {frozen_reason}",
                                        details={'reason': frozen_reason},
                                    ),
                                    equity=equity,
                                    frozen_reason=frozen_reason,
                                )
                            
                            self._running = False
                            break
                
                for name, strategy in self._strategies.items():
                    try:
                        if hasattr(strategy, '_on_update'):
                            strategy._on_update()
                    except Exception as e:
                        self.logger.error(f"策略 {name} 更新时出错: {e}", exc_info=True)
                
                if self._should_send_heartbeat():
                    self._send_heartbeat()
                
                if self._should_report_status():
                    self._report_status()
                    
                    if self._strategy_manager:
                        self.logger.info("自动保存所有策略状态...")
                        saved_count = self._strategy_manager.save_all_states()
                        self.logger.info(f"已保存 {saved_count} 个策略的状态")
                    
            except KeyboardInterrupt:
                self.logger.info("用户中断，停止 RealtimeRunner")
                self._running = False
                break
            except Exception as e:
                self.logger.error(f"主循环出错: {e}", exc_info=True)
                self._running = False
                raise
    
    def run(self, load_saved_states: bool = True) -> None:
        if not self._initialized:
            self.logger.info("RealtimeRunner 未初始化，正在执行初始化...")
            self.initialize()
        
        self.logger.info("=" * 80)
        self.logger.info("                🚀 启动实时交易系统")
        self.logger.info("=" * 80)
        self.logger.info(f"已注册策略数量: {len(self._strategies)}")
        for name in self._strategies.keys():
            strategy = self._strategies[name]
            status_line = f"  - {name}: {strategy.__class__.__name__}"
            if hasattr(strategy, 'short_period') and hasattr(strategy, 'long_period'):
                status_line += f" (MA{strategy.short_period}/MA{strategy.long_period})"
            self.logger.info(status_line)
        self.logger.info("=" * 80)
        self.logger.info(f"心跳间隔: {self._heartbeat_interval_seconds} 秒")
        self.logger.info(f"状态报告间隔: {self._status_report_interval_seconds} 秒")
        self.logger.info("提示: 按 Ctrl+C 停止实时交易系统")
        self.logger.info("=" * 80)
        
        try:
            self._run_main_loop()
        except KeyboardInterrupt:
            self.logger.info("用户中断，停止实时交易系统")
        except Exception as e:
            self.logger.error(f"实时交易系统运行出错: {e}", exc_info=True)
            raise
        finally:
            self.logger.info("正在保存所有策略状态...")
            if self._strategy_manager:
                saved_count = self._strategy_manager.save_all_states()
                self.logger.info(f"已保存 {saved_count} 个策略的状态")
            
            self.stop()
    
    def stop(self) -> None:
        self._running = False
        
        self.logger.info("正在停止 RealtimeRunner...")
        
        if self._order_manager:
            canceled = self._order_manager.cancel_all_orders()
            if canceled > 0:
                self.logger.info(f"已取消 {canceled} 笔活跃订单")
        
        for name, strategy in self._strategies.items():
            try:
                self.logger.info(f"停止策略: {name}")
                strategy.stop()
                self.logger.info(f"策略 {name} 已停止")
            except Exception as e:
                self.logger.error(f"停止策略 {name} 时出错: {e}")
        
        self.logger.info("RealtimeRunner 已停止")
    
    def place_order(
        self,
        contract: str,
        direction: str,
        offset: str,
        volume: int,
        limit_price: float = 0.0,
    ) -> Optional[OrderRecord]:
        if not self._order_manager:
            self.logger.error("OrderManager 未初始化")
            return None
        
        return self._order_manager.place_order(
            contract=contract,
            direction=direction,
            offset=offset,
            volume=volume,
            limit_price=limit_price,
        )
    
    def set_target_position(
        self,
        contract: str,
        target_long: int = 0,
        target_short: int = 0,
    ):
        if not self._position_synchronizer:
            self.logger.error("PositionSynchronizer 未初始化")
            return
        
        self._position_synchronizer.set_target_position(
            contract=contract,
            target_long=target_long,
            target_short=target_short,
        )
    
    def get_risk_info(self) -> Dict[str, Any]:
        if self._risk_manager:
            return self._risk_manager.get_total_risk_info()
        return {'risk_enabled': False}
    
    def get_account_info(self) -> Dict[str, Any]:
        if self._risk_manager:
            snapshot = self._risk_manager.get_account_snapshot()
            return snapshot.to_dict()
        elif self.api:
            try:
                account = self.api.get_account()
                return {
                    'balance': float(account.get('balance', 0)),
                    'equity': float(account.get('equity', 0)),
                    'margin_used': float(account.get('margin', 0)),
                    'available': float(account.get('available', 0)),
                    'float_profit': float(account.get('float_profit', 0)),
                }
            except:
                pass
        return {}
    
    def get_positions(self) -> Dict[str, PositionInfo]:
        if self._position_synchronizer:
            return self._position_synchronizer.get_all_positions()
        return {}
    
    def get_active_orders(self, contract: str = None) -> List[OrderRecord]:
        if self._order_manager:
            return self._order_manager.get_active_orders(contract)
        return []
    
    def emergency_stop(self, reason: str) -> Dict[str, Any]:
        self.logger.critical(f"[紧急停止] 原因: {reason}")
        
        if self._order_manager:
            canceled = self._order_manager.cancel_all_orders()
            self.logger.warning(f"已取消所有订单: {canceled} 笔")
        
        self._running = False
        
        if self._risk_manager:
            self._risk_manager.freeze(reason)
        
        return {
            'status': 'emergency_stopped',
            'reason': reason,
            'canceled_orders': canceled if self._order_manager else 0,
            'timestamp': time.time(),
        }


def load_webhook_config_from_settings(config_path: str = None) -> WebhookConfig:
    if not config_path:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_path = os.path.join(base_dir, 'config', 'settings.yaml')
    
    if not os.path.exists(config_path) or not YAML_AVAILABLE:
        return WebhookConfig()
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    notification_config = config.get('notification', {})
    
    return WebhookConfig(
        enabled=notification_config.get('webhook_enabled', False),
        url=notification_config.get('webhook_url', ''),
        secret=notification_config.get('webhook_secret', ''),
        timeout_seconds=notification_config.get('webhook_timeout_seconds', 10.0),
        retry_count=notification_config.get('webhook_retry_count', 3),
        retry_delay_seconds=notification_config.get('webhook_retry_delay_seconds', 2.0),
        notify_on_trade=notification_config.get('notify_on_trade', True),
        notify_on_risk=notification_config.get('notify_on_risk', True),
        notify_on_order=notification_config.get('notify_on_order', False),
    )
