import logging
import time
from enum import Enum
from typing import Dict, Any, Optional, List, Callable
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime


class RiskLevel(Enum):
    SAFE = "SAFE"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"
    FROZEN = "FROZEN"


class RiskEventType(Enum):
    DRAWDOWN_EXCEEDED = "DRAWDOWN_EXCEEDED"
    STRATEGY_MARGIN_EXCEEDED = "STRATEGY_MARGIN_EXCEEDED"
    TOTAL_MARGIN_EXCEEDED = "TOTAL_MARGIN_EXCEEDED"
    POSITION_LIMIT_EXCEEDED = "POSITION_LIMIT_EXCEEDED"


@dataclass
class PositionInfo:
    contract: str
    long_volume: int = 0
    short_volume: int = 0
    long_margin: float = 0.0
    short_margin: float = 0.0
    long_open_price: float = 0.0
    short_open_price: float = 0.0
    current_price: float = 0.0
    float_profit: float = 0.0

    @property
    def total_volume(self) -> int:
        return self.long_volume + self.short_volume

    @property
    def total_margin(self) -> float:
        return self.long_margin + self.short_margin

    @property
    def net_position(self) -> int:
        return self.long_volume - self.short_volume


@dataclass
class StrategyRiskInfo:
    strategy_name: str
    margin_used: float = 0.0
    position_value: float = 0.0
    float_profit: float = 0.0
    positions: Dict[str, PositionInfo] = field(default_factory=dict)


@dataclass
class AccountSnapshot:
    timestamp: float
    balance: float
    equity: float
    total_asset: float
    margin_used: float
    available: float
    float_profit: float


@dataclass
class RiskEvent:
    event_type: RiskEventType
    timestamp: float
    level: RiskLevel
    message: str
    details: Dict[str, Any] = field(default_factory=dict)


class RiskManager:
    DEFAULT_MAX_DRAWDOWN_PERCENT = 5.0
    DEFAULT_MAX_STRATEGY_MARGIN_PERCENT = 30.0
    DEFAULT_MAX_TOTAL_MARGIN_PERCENT = 80.0

    def __init__(
        self,
        connector: Any = None,
        max_drawdown_percent: float = None,
        max_strategy_margin_percent: float = None,
        max_total_margin_percent: float = None,
        on_risk_event: Optional[Callable[[RiskEvent], None]] = None,
        on_frozen: Optional[Callable[[], None]] = None,
    ):
        self.connector = connector
        self.api = None
        self.logger = logging.getLogger(self.__class__.__name__)

        if connector:
            self.api = connector.get_api()

        self.max_drawdown_percent = max_drawdown_percent or self.DEFAULT_MAX_DRAWDOWN_PERCENT
        self.max_strategy_margin_percent = max_strategy_margin_percent or self.DEFAULT_MAX_STRATEGY_MARGIN_PERCENT
        self.max_total_margin_percent = max_total_margin_percent or self.DEFAULT_MAX_TOTAL_MARGIN_PERCENT

        self.on_risk_event = on_risk_event
        self.on_frozen = on_frozen

        self._initialized = False
        self._frozen = False
        self._frozen_reason: Optional[str] = None
        self._frozen_time: Optional[float] = None

        self._peak_equity: float = 0.0
        self._initial_equity: float = 0.0
        self._current_drawdown_percent: float = 0.0

        self._strategy_risk: Dict[str, StrategyRiskInfo] = defaultdict(
            lambda: StrategyRiskInfo(strategy_name="")
        )
        self._positions: Dict[str, PositionInfo] = defaultdict(PositionInfo)

        self._snapshots: List[AccountSnapshot] = []
        self._max_snapshots = 1000

        self._risk_events: List[RiskEvent] = []
        self._max_events = 100

        self.logger.info(
            f"RiskManager 初始化: "
            f"最大回撤={self.max_drawdown_percent}%, "
            f"单策略最大保证金={self.max_strategy_margin_percent}%, "
            f"总最大保证金={self.max_total_margin_percent}%"
        )

    def set_connector(self, connector: Any):
        if connector is None:
            self.logger.error("Connector 不能为 None")
            raise ValueError("Connector 不能为 None")

        self.connector = connector
        self.api = connector.get_api()
        self.logger.info("Connector 已设置")

    def initialize(self) -> None:
        if self._initialized:
            self.logger.info("RiskManager 已初始化，跳过重复初始化")
            return

        if self.api is None:
            self.logger.warning("API 未初始化，将使用模拟模式")

        self._load_account_state()
        self._initialized = True
        self.logger.info("RiskManager 初始化完成")

    def _load_account_state(self) -> None:
        if self.api is None:
            self.logger.info("无 API 连接，使用默认账户状态")
            return

        try:
            account = self.api.get_account()
            if account:
                self._initial_equity = float(account.get('equity', 0))
                self._peak_equity = self._initial_equity
                self.logger.info(f"初始账户权益: {self._initial_equity:.2f}")
        except Exception as e:
            self.logger.warning(f"加载账户状态失败: {e}，将使用默认值")

    def is_frozen(self) -> bool:
        return self._frozen

    def get_frozen_reason(self) -> Optional[str]:
        return self._frozen_reason

    def freeze(self, reason: str) -> None:
        if self._frozen:
            return

        self._frozen = True
        self._frozen_reason = reason
        self._frozen_time = time.time()

        self.logger.critical(f"系统已冻结: {reason}")

        if self.on_frozen:
            try:
                self.on_frozen()
            except Exception as e:
                self.logger.error(f"执行冻结回调失败: {e}")

    def unfreeze(self) -> None:
        if not self._frozen:
            return

        self._frozen = False
        self._frozen_reason = None
        self._frozen_time = None
        self.logger.info("系统已解冻")

    def _emit_risk_event(
        self,
        event_type: RiskEventType,
        level: RiskLevel,
        message: str,
        details: Dict[str, Any] = None,
    ) -> RiskEvent:
        event = RiskEvent(
            event_type=event_type,
            timestamp=time.time(),
            level=level,
            message=message,
            details=details or {},
        )

        self._risk_events.append(event)
        if len(self._risk_events) > self._max_events:
            self._risk_events = self._risk_events[-self._max_events:]

        if level == RiskLevel.CRITICAL or level == RiskLevel.FROZEN:
            self.logger.critical(f"风险事件 [{event_type.value}]: {message}")
        elif level == RiskLevel.WARNING:
            self.logger.warning(f"风险事件 [{event_type.value}]: {message}")
        else:
            self.logger.info(f"风险事件 [{event_type.value}]: {message}")

        if self.on_risk_event:
            try:
                self.on_risk_event(event)
            except Exception as e:
                self.logger.error(f"执行风险事件回调失败: {e}")

        return event

    def get_account_snapshot(self) -> AccountSnapshot:
        if self.api is None:
            return AccountSnapshot(
                timestamp=time.time(),
                balance=0.0,
                equity=0.0,
                total_asset=0.0,
                margin_used=0.0,
                available=0.0,
                float_profit=0.0,
            )

        try:
            account = self.api.get_account()
            if not account:
                return self._create_default_snapshot()

            balance = float(account.get('balance', 0))
            equity = float(account.get('equity', balance))
            margin_used = float(account.get('margin', 0))
            available = float(account.get('available', balance - margin_used))
            float_profit = float(account.get('float_profit', 0))

            total_asset = equity

            snapshot = AccountSnapshot(
                timestamp=time.time(),
                balance=balance,
                equity=equity,
                total_asset=total_asset,
                margin_used=margin_used,
                available=available,
                float_profit=float_profit,
            )

            self._snapshots.append(snapshot)
            if len(self._snapshots) > self._max_snapshots:
                self._snapshots = self._snapshots[-self._max_snapshots:]

            return snapshot

        except Exception as e:
            self.logger.error(f"获取账户快照失败: {e}")
            return self._create_default_snapshot()

    def _create_default_snapshot(self) -> AccountSnapshot:
        return AccountSnapshot(
            timestamp=time.time(),
            balance=0.0,
            equity=0.0,
            total_asset=0.0,
            margin_used=0.0,
            available=0.0,
            float_profit=0.0,
        )

    def update_positions(self, positions_data: Dict[str, Any] = None) -> Dict[str, PositionInfo]:
        if positions_data is None and self.api is None:
            return self._positions.copy()

        if positions_data is None:
            try:
                positions_data = self.api.get_position()
            except Exception as e:
                self.logger.warning(f"获取持仓数据失败: {e}")
                return self._positions.copy()

        if not positions_data:
            return self._positions.copy()

        new_positions: Dict[str, PositionInfo] = defaultdict(PositionInfo)

        for contract, pos_data in positions_data.items():
            if not pos_data:
                continue

            position = PositionInfo(contract=contract)

            try:
                position.long_volume = int(pos_data.get('buy_volume', 0))
                position.short_volume = int(pos_data.get('sell_volume', 0))
                position.long_margin = float(pos_data.get('buy_margin', 0))
                position.short_margin = float(pos_data.get('sell_margin', 0))
                position.long_open_price = float(pos_data.get('buy_open_price', 0))
                position.short_open_price = float(pos_data.get('sell_open_price', 0))
                position.current_price = float(pos_data.get('last_price', 0))
                position.float_profit = float(pos_data.get('float_profit', 0))
            except (ValueError, TypeError) as e:
                self.logger.warning(f"解析持仓数据失败 [{contract}]: {e}")

            new_positions[contract] = position

        self._positions = new_positions
        return self._positions.copy()

    def update_strategy_risk(
        self,
        strategy_name: str,
        margin_used: float = 0.0,
        position_value: float = 0.0,
        float_profit: float = 0.0,
        positions: Dict[str, PositionInfo] = None,
    ) -> None:
        if strategy_name not in self._strategy_risk:
            self._strategy_risk[strategy_name] = StrategyRiskInfo(strategy_name=strategy_name)

        risk_info = self._strategy_risk[strategy_name]
        risk_info.margin_used = margin_used
        risk_info.position_value = position_value
        risk_info.float_profit = float_profit

        if positions:
            risk_info.positions = positions.copy()

    def check_drawdown(self, snapshot: AccountSnapshot = None) -> RiskLevel:
        if self._frozen:
            return RiskLevel.FROZEN

        if snapshot is None:
            snapshot = self.get_account_snapshot()

        current_equity = snapshot.equity

        if current_equity <= 0:
            return RiskLevel.SAFE

        if current_equity > self._peak_equity:
            self._peak_equity = current_equity

        if self._peak_equity > 0:
            drawdown = self._peak_equity - current_equity
            self._current_drawdown_percent = (drawdown / self._peak_equity) * 100
        else:
            self._current_drawdown_percent = 0.0

        if self._current_drawdown_percent >= self.max_drawdown_percent:
            self._emit_risk_event(
                event_type=RiskEventType.DRAWDOWN_EXCEEDED,
                level=RiskLevel.CRITICAL,
                message=f"净值回撤超过阈值: 当前回撤 {self._current_drawdown_percent:.2f}% > 阈值 {self.max_drawdown_percent}%",
                details={
                    'peak_equity': self._peak_equity,
                    'current_equity': current_equity,
                    'drawdown_percent': self._current_drawdown_percent,
                    'threshold_percent': self.max_drawdown_percent,
                },
            )

            self.freeze(f"净值回撤超过 {self.max_drawdown_percent}%")
            return RiskLevel.FROZEN

        warning_threshold = self.max_drawdown_percent * 0.6
        if self._current_drawdown_percent >= warning_threshold:
            self._emit_risk_event(
                event_type=RiskEventType.DRAWDOWN_EXCEEDED,
                level=RiskLevel.WARNING,
                message=f"净值回撤接近阈值: 当前回撤 {self._current_drawdown_percent:.2f}%",
                details={
                    'peak_equity': self._peak_equity,
                    'current_equity': current_equity,
                    'drawdown_percent': self._current_drawdown_percent,
                    'warning_threshold': warning_threshold,
                },
            )
            return RiskLevel.WARNING

        return RiskLevel.SAFE

    def check_strategy_margin(
        self,
        strategy_name: str,
        proposed_margin: float = None,
        snapshot: AccountSnapshot = None,
    ) -> RiskLevel:
        if self._frozen:
            return RiskLevel.FROZEN

        if snapshot is None:
            snapshot = self.get_account_snapshot()

        total_asset = snapshot.total_asset
        if total_asset <= 0:
            return RiskLevel.SAFE

        risk_info = self._strategy_risk.get(strategy_name)
        if risk_info is None:
            current_margin = 0.0
        else:
            current_margin = risk_info.margin_used

        if proposed_margin is not None:
            check_margin = proposed_margin
        else:
            check_margin = current_margin

        margin_percent = (check_margin / total_asset) * 100

        if margin_percent >= self.max_strategy_margin_percent:
            self._emit_risk_event(
                event_type=RiskEventType.STRATEGY_MARGIN_EXCEEDED,
                level=RiskLevel.CRITICAL,
                message=f"策略 [{strategy_name}] 保证金超过阈值: {margin_percent:.2f}% > {self.max_strategy_margin_percent}%",
                details={
                    'strategy_name': strategy_name,
                    'margin_used': check_margin,
                    'total_asset': total_asset,
                    'margin_percent': margin_percent,
                    'threshold_percent': self.max_strategy_margin_percent,
                },
            )
            return RiskLevel.CRITICAL

        warning_threshold = self.max_strategy_margin_percent * 0.7
        if margin_percent >= warning_threshold:
            self._emit_risk_event(
                event_type=RiskEventType.STRATEGY_MARGIN_EXCEEDED,
                level=RiskLevel.WARNING,
                message=f"策略 [{strategy_name}] 保证金接近阈值: {margin_percent:.2f}%",
                details={
                    'strategy_name': strategy_name,
                    'margin_used': check_margin,
                    'margin_percent': margin_percent,
                    'warning_threshold': warning_threshold,
                },
            )
            return RiskLevel.WARNING

        return RiskLevel.SAFE

    def check_total_margin(self, snapshot: AccountSnapshot = None) -> RiskLevel:
        if self._frozen:
            return RiskLevel.FROZEN

        if snapshot is None:
            snapshot = self.get_account_snapshot()

        total_asset = snapshot.total_asset
        if total_asset <= 0:
            return RiskLevel.SAFE

        total_margin = snapshot.margin_used
        margin_percent = (total_margin / total_asset) * 100

        if margin_percent >= self.max_total_margin_percent:
            self._emit_risk_event(
                event_type=RiskEventType.TOTAL_MARGIN_EXCEEDED,
                level=RiskLevel.CRITICAL,
                message=f"总保证金超过阈值: {margin_percent:.2f}% > {self.max_total_margin_percent}%",
                details={
                    'total_margin': total_margin,
                    'total_asset': total_asset,
                    'margin_percent': margin_percent,
                    'threshold_percent': self.max_total_margin_percent,
                },
            )
            return RiskLevel.CRITICAL

        warning_threshold = self.max_total_margin_percent * 0.8
        if margin_percent >= warning_threshold:
            self._emit_risk_event(
                event_type=RiskEventType.TOTAL_MARGIN_EXCEEDED,
                level=RiskLevel.WARNING,
                message=f"总保证金接近阈值: {margin_percent:.2f}%",
                details={
                    'total_margin': total_margin,
                    'margin_percent': margin_percent,
                    'warning_threshold': warning_threshold,
                },
            )
            return RiskLevel.WARNING

        return RiskLevel.SAFE

    def can_place_order(
        self,
        strategy_name: str,
        contract: str,
        direction: str,
        volume: int,
        price: float,
        margin_per_contract: float = None,
    ) -> tuple[bool, str, RiskLevel]:
        if self._frozen:
            return False, f"系统已冻结: {self._frozen_reason}", RiskLevel.FROZEN

        if volume <= 0:
            return False, "下单数量必须大于 0", RiskLevel.SAFE

        snapshot = self.get_account_snapshot()

        if margin_per_contract is None:
            estimated_margin = price * volume * 0.1
        else:
            estimated_margin = margin_per_contract * volume

        current_risk = self._strategy_risk.get(strategy_name)
        current_margin = current_risk.margin_used if current_risk else 0.0
        proposed_margin = current_margin + estimated_margin

        strategy_level = self.check_strategy_margin(
            strategy_name=strategy_name,
            proposed_margin=proposed_margin,
            snapshot=snapshot,
        )

        if strategy_level == RiskLevel.CRITICAL:
            return (
                False,
                f"策略 [{strategy_name}] 保证金将超过限制",
                strategy_level,
            )

        total_margin_check = snapshot.margin_used + estimated_margin
        total_asset = snapshot.total_asset
        if total_asset > 0:
            total_percent = (total_margin_check / total_asset) * 100
            if total_percent >= self.max_total_margin_percent:
                return (
                    False,
                    f"总保证金将超过限制: {total_percent:.2f}%",
                    RiskLevel.CRITICAL,
                )

        available = snapshot.available
        if estimated_margin > available:
            return (
                False,
                f"可用资金不足: 需要 {estimated_margin:.2f}，可用 {available:.2f}",
                RiskLevel.CRITICAL,
            )

        return True, "风控检查通过", RiskLevel.SAFE

    def run_risk_checks(self) -> Dict[str, RiskLevel]:
        if self._frozen:
            return {'drawdown': RiskLevel.FROZEN, 'total_margin': RiskLevel.FROZEN}

        snapshot = self.get_account_snapshot()
        self.update_positions()

        results = {}

        results['drawdown'] = self.check_drawdown(snapshot)

        if not self._frozen:
            results['total_margin'] = self.check_total_margin(snapshot)

            for strategy_name in self._strategy_risk.keys():
                strategy_key = f'strategy_{strategy_name}'
                results[strategy_key] = self.check_strategy_margin(
                    strategy_name=strategy_name,
                    snapshot=snapshot,
                )

        return results

    def get_drawdown_info(self) -> Dict[str, Any]:
        return {
            'peak_equity': self._peak_equity,
            'current_drawdown_percent': self._current_drawdown_percent,
            'max_drawdown_percent': self.max_drawdown_percent,
            'is_frozen': self._frozen,
            'frozen_reason': self._frozen_reason,
        }

    def get_total_risk_info(self) -> Dict[str, Any]:
        snapshot = self.get_account_snapshot()

        return {
            'timestamp': datetime.fromtimestamp(snapshot.timestamp).isoformat(),
            'balance': snapshot.balance,
            'equity': snapshot.equity,
            'total_asset': snapshot.total_asset,
            'margin_used': snapshot.margin_used,
            'available': snapshot.available,
            'float_profit': snapshot.float_profit,
            'margin_percent': (snapshot.margin_used / snapshot.total_asset * 100)
            if snapshot.total_asset > 0
            else 0.0,
            'drawdown_info': self.get_drawdown_info(),
            'is_frozen': self._frozen,
            'frozen_reason': self._frozen_reason,
            'position_count': len(self._positions),
            'strategy_count': len(self._strategy_risk),
        }

    def get_risk_events(self, limit: int = 10) -> List[Dict[str, Any]]:
        events = self._risk_events[-limit:] if limit > 0 else self._risk_events
        return [
            {
                'event_type': e.event_type.value,
                'timestamp': datetime.fromtimestamp(e.timestamp).isoformat(),
                'level': e.level.value,
                'message': e.message,
                'details': e.details,
            }
            for e in events
        ]

    def close_all_positions(self) -> Dict[str, Any]:
        if self.api is None:
            self.logger.warning("无 API 连接，无法执行平仓操作")
            return {'status': 'error', 'message': '无 API 连接'}

        results = {'status': 'success', 'closed': [], 'failed': []}

        try:
            positions = self.update_positions()

            for contract, position in positions.items():
                if position.long_volume > 0:
                    try:
                        self.api.insert_order(
                            self.api.get_quote(contract),
                            direction="SELL",
                            offset="CLOSE",
                            volume=position.long_volume,
                        )
                        results['closed'].append(
                            {
                                'contract': contract,
                                'direction': 'SELL',
                                'volume': position.long_volume,
                            }
                        )
                        self.logger.info(f"已发送平多单指令: {contract}, 数量: {position.long_volume}")
                    except Exception as e:
                        results['failed'].append(
                            {
                                'contract': contract,
                                'direction': 'SELL',
                                'volume': position.long_volume,
                                'error': str(e),
                            }
                        )
                        self.logger.error(f"平多单失败 [{contract}]: {e}")

                if position.short_volume > 0:
                    try:
                        self.api.insert_order(
                            self.api.get_quote(contract),
                            direction="BUY",
                            offset="CLOSE",
                            volume=position.short_volume,
                        )
                        results['closed'].append(
                            {
                                'contract': contract,
                                'direction': 'BUY',
                                'volume': position.short_volume,
                            }
                        )
                        self.logger.info(f"已发送平空单指令: {contract}, 数量: {position.short_volume}")
                    except Exception as e:
                        results['failed'].append(
                            {
                                'contract': contract,
                                'direction': 'BUY',
                                'volume': position.short_volume,
                                'error': str(e),
                            }
                        )
                        self.logger.error(f"平空单失败 [{contract}]: {e}")

        except Exception as e:
            self.logger.error(f"执行平仓操作失败: {e}")
            results['status'] = 'error'
            results['message'] = str(e)

        return results

    def emergency_stop(self, reason: str) -> Dict[str, Any]:
        self.logger.critical(f"执行紧急停止: {reason}")

        close_result = self.close_all_positions()

        self.freeze(f"紧急停止: {reason}")

        return {
            'status': 'success',
            'reason': reason,
            'close_result': close_result,
            'is_frozen': self._frozen,
        }

    def reset_peak_equity(self, new_peak: float = None) -> None:
        if new_peak is not None:
            self._peak_equity = float(new_peak)
        else:
            snapshot = self.get_account_snapshot()
            self._peak_equity = snapshot.equity

        self._current_drawdown_percent = 0.0
        self.logger.info(f"峰值权益已重置为: {self._peak_equity:.2f}")
