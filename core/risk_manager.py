import logging
import logging.handlers
import os
import time
import json
from enum import Enum
from typing import Dict, Any, Optional, List, Callable, Tuple
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False


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
    PRICE_GAP_DETECTED = "PRICE_GAP_DETECTED"
    API_TIMEOUT = "API_TIMEOUT"
    CONNECTION_LOST = "CONNECTION_LOST"


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

    def to_dict(self) -> Dict[str, Any]:
        return {
            'contract': self.contract,
            'long_volume': self.long_volume,
            'short_volume': self.short_volume,
            'long_margin': self.long_margin,
            'short_margin': self.short_margin,
            'long_open_price': self.long_open_price,
            'short_open_price': self.short_open_price,
            'current_price': self.current_price,
            'float_profit': self.float_profit,
            'total_volume': self.total_volume,
            'total_margin': self.total_margin,
            'net_position': self.net_position,
        }


@dataclass
class StrategyRiskInfo:
    strategy_name: str
    margin_used: float = 0.0
    position_value: float = 0.0
    float_profit: float = 0.0
    positions: Dict[str, PositionInfo] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'strategy_name': self.strategy_name,
            'margin_used': self.margin_used,
            'position_value': self.position_value,
            'float_profit': self.float_profit,
            'positions': {k: v.to_dict() for k, v in self.positions.items()},
        }


@dataclass
class AccountSnapshot:
    timestamp: float
    balance: float
    equity: float
    total_asset: float
    margin_used: float
    available: float
    float_profit: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp,
            'datetime': datetime.fromtimestamp(self.timestamp).isoformat(),
            'balance': self.balance,
            'equity': self.equity,
            'total_asset': self.total_asset,
            'margin_used': self.margin_used,
            'available': self.available,
            'float_profit': self.float_profit,
        }


@dataclass
class RiskEvent:
    event_type: RiskEventType
    timestamp: float
    level: RiskLevel
    message: str
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'event_type': self.event_type.value,
            'timestamp': self.timestamp,
            'datetime': datetime.fromtimestamp(self.timestamp).isoformat(),
            'level': self.level.value,
            'message': self.message,
            'details': self.details,
        }


@dataclass
class PriceGapInfo:
    contract: str
    previous_price: float
    current_price: float
    gap_percent: float
    timestamp: float
    direction: str


class RiskManager:
    DEFAULT_MAX_DRAWDOWN_PERCENT = 5.0
    DEFAULT_MAX_STRATEGY_MARGIN_PERCENT = 30.0
    DEFAULT_MAX_TOTAL_MARGIN_PERCENT = 80.0
    DEFAULT_PRICE_GAP_THRESHOLD_PERCENT = 5.0
    DEFAULT_API_TIMEOUT_SECONDS = 3.0

    _risk_logger = None
    _risk_log_file = None

    def __init__(
        self,
        connector: Any = None,
        max_drawdown_percent: float = None,
        max_strategy_margin_percent: float = None,
        max_total_margin_percent: float = None,
        price_gap_threshold_percent: float = None,
        api_timeout_seconds: float = None,
        on_risk_event: Optional[Callable[[RiskEvent], None]] = None,
        on_frozen: Optional[Callable[[], None]] = None,
        risk_log_file: str = None,
    ):
        self.connector = connector
        self.api = None
        self.logger = logging.getLogger(self.__class__.__name__)

        if connector:
            self.api = connector.get_api()

        self.max_drawdown_percent = max_drawdown_percent or self.DEFAULT_MAX_DRAWDOWN_PERCENT
        self.max_strategy_margin_percent = max_strategy_margin_percent or self.DEFAULT_MAX_STRATEGY_MARGIN_PERCENT
        self.max_total_margin_percent = max_total_margin_percent or self.DEFAULT_MAX_TOTAL_MARGIN_PERCENT
        self.price_gap_threshold_percent = price_gap_threshold_percent or self.DEFAULT_PRICE_GAP_THRESHOLD_PERCENT
        self.api_timeout_seconds = api_timeout_seconds or self.DEFAULT_API_TIMEOUT_SECONDS

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
        self._previous_prices: Dict[str, float] = {}

        self._snapshots: List[AccountSnapshot] = []
        self._max_snapshots = 10000

        self._risk_events: List[RiskEvent] = []
        self._max_events = 1000

        self._api_last_response_time: float = time.time()
        self._api_timeouts: int = 0
        self._max_api_timeouts = 3

        self._setup_risk_logger(risk_log_file)

        self.logger.info(
            f"RiskManager 初始化: "
            f"最大回撤={self.max_drawdown_percent}%, "
            f"单策略最大保证金={self.max_strategy_margin_percent}%, "
            f"总最大保证金={self.max_total_margin_percent}%, "
            f"价格跳空阈值={self.price_gap_threshold_percent}%, "
            f"API超时={self.api_timeout_seconds}秒"
        )

    def _setup_risk_logger(self, log_file: str = None):
        if log_file is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            log_file = os.path.join(base_dir, 'logs', 'risk_event.log')

        self._risk_log_file = log_file

        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)

        risk_logger = logging.getLogger('RiskEventLogger')
        risk_logger.setLevel(logging.INFO)
        risk_logger.propagate = False

        if not risk_logger.handlers:
            file_handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=10 * 1024 * 1024,
                backupCount=5,
                encoding='utf-8',
            )
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            risk_logger.addHandler(file_handler)

        self._risk_logger = risk_logger
        self.logger.info(f"风险日志已配置: {log_file}")

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

        fire_ascii = """
        🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥
        🔥                                          🔥
        🔥     ⚠️  系 统 风 控 熔 断 触 发  ⚠️     🔥
        🔥                                          🔥
        🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥
        """

        self.logger.critical(fire_ascii)
        self.logger.critical(f"熔断原因: {reason}")
        self.logger.critical(f"熔断时间: {datetime.fromtimestamp(self._frozen_time).strftime('%Y-%m-%d %H:%M:%S')}")

        self._generate_freeze_report(reason)

        if self.on_frozen:
            try:
                self.on_frozen()
            except Exception as e:
                self.logger.error(f"执行冻结回调失败: {e}")

    def _generate_freeze_report(self, reason: str):
        freeze_time = datetime.fromtimestamp(self._frozen_time if self._frozen_time else time.time())
        timestamp_str = freeze_time.strftime('%Y%m%d_%H%M%S')

        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        report_dir = os.path.join(base_dir, 'logs', 'freeze_reports')
        os.makedirs(report_dir, exist_ok=True)

        report_file = os.path.join(report_dir, f'freeze_report_{timestamp_str}.json')

        report = {
            'report_type': 'FROZEN_REPORT',
            'version': '1.0',
            'generated_at': freeze_time.isoformat(),
            'freeze_reason': reason,
            'freeze_timestamp': self._frozen_time,
            'freeze_datetime': datetime.fromtimestamp(self._frozen_time).isoformat() if self._frozen_time else None,
            'thresholds': {
                'max_drawdown_percent': self.max_drawdown_percent,
                'max_strategy_margin_percent': self.max_strategy_margin_percent,
                'max_total_margin_percent': self.max_total_margin_percent,
                'price_gap_threshold_percent': self.price_gap_threshold_percent,
            },
            'current_state': {
                'peak_equity': self._peak_equity,
                'current_drawdown_percent': self._current_drawdown_percent,
                'total_strategies': len(self._strategy_risk),
                'total_positions': len(self._positions),
                'api_timeouts': self._api_timeouts,
            },
            'account_snapshots': [],
            'positions_snapshot': {},
            'strategies_snapshot': {},
            'equity_curve': [],
            'risk_events': [],
        }

        if self._snapshots:
            report['account_snapshots'] = [s.to_dict() for s in self._snapshots[-100:]]
            report['equity_curve'] = [
                {'timestamp': s.timestamp, 'equity': s.equity, 'drawdown': (self._peak_equity - s.equity) / self._peak_equity * 100 if self._peak_equity > 0 else 0}
                for s in self._snapshots[-500:]
            ]

        for contract, pos in self._positions.items():
            report['positions_snapshot'][contract] = pos.to_dict()

        for name, strategy in self._strategy_risk.items():
            report['strategies_snapshot'][name] = strategy.to_dict()

        report['risk_events'] = [e.to_dict() for e in self._risk_events[-50:]]

        try:
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            self.logger.critical(f"熔断复盘报告已保存: {report_file}")
        except Exception as e:
            self.logger.error(f"保存熔断报告失败: {e}")

        log_message = f"""
        ================ 风 险 熔 断 复 盘 报 告 ================
        时间: {freeze_time.strftime('%Y-%m-%d %H:%M:%S')}
        原因: {reason}
        
        ---------- 阈值配置 ----------
        最大回撤阈值: {self.max_drawdown_percent}%
        单策略最大保证金: {self.max_strategy_margin_percent}%
        总最大保证金: {self.max_total_margin_percent}%
        价格跳空阈值: {self.price_gap_threshold_percent}%
        
        ---------- 当前状态 ----------
        峰值权益: {self._peak_equity:,.2f}
        当前回撤: {self._current_drawdown_percent:.2f}%
        策略数量: {len(self._strategy_risk)}
        持仓数量: {len(self._positions)}
        
        ---------- 持仓快照 ----------
        """

        for contract, pos in self._positions.items():
            log_message += f"""
        合约: {contract}
          多单: {pos.long_volume} 手 @ {pos.long_open_price:.2f}
          空单: {pos.short_volume} 手 @ {pos.short_open_price:.2f}
          现价: {pos.current_price:.2f}
          浮动盈亏: {pos.float_profit:,.2f}
          占用保证金: {pos.total_margin:,.2f}
            """

        log_message += f"""
        ---------- 策略风险快照 ----------
        """

        for name, strategy in self._strategy_risk.items():
            log_message += f"""
        策略: {name}
          占用保证金: {strategy.margin_used:,.2f}
          持仓价值: {strategy.position_value:,.2f}
          浮动盈亏: {strategy.float_profit:,.2f}
            """

        log_message += f"""
        ---------- 最近净值曲线 (最近20条) ----------
        """

        recent_snapshots = self._snapshots[-20:]
        for i, s in enumerate(recent_snapshots):
            drawdown = (self._peak_equity - s.equity) / self._peak_equity * 100 if self._peak_equity > 0 else 0
            log_message += f"  [{i+1}] {datetime.fromtimestamp(s.timestamp).strftime('%H:%M:%S')} - 权益: {s.equity:,.2f}, 回撤: {drawdown:.2f}%\n"

        log_message += """
        ========================================================
        """

        if self._risk_logger:
            self._risk_logger.critical(log_message)
            self._risk_logger.critical(f"完整报告已保存至: {report_file}")

    def unfreeze(self) -> None:
        if not self._frozen:
            return

        self._frozen = False
        self._frozen_reason = None
        self._frozen_time = None
        self.logger.info("系统已解冻")

        if self._risk_logger:
            self._risk_logger.info(f"系统已解冻，时间: {datetime.now().isoformat()}")

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

        log_msg = f"[{event_type.value}] {message}"

        if level == RiskLevel.CRITICAL or level == RiskLevel.FROZEN:
            self.logger.critical(log_msg)
            if self._risk_logger:
                self._risk_logger.critical(json.dumps(event.to_dict(), ensure_ascii=False))
        elif level == RiskLevel.WARNING:
            self.logger.warning(log_msg)
            if self._risk_logger:
                self._risk_logger.warning(json.dumps(event.to_dict(), ensure_ascii=False))
        else:
            self.logger.info(log_msg)

        if self.on_risk_event:
            try:
                self.on_risk_event(event)
            except Exception as e:
                self.logger.error(f"执行风险事件回调失败: {e}")

        return event

    def check_api_health(self) -> Tuple[bool, str]:
        if self.api is None:
            return True, "无API连接，跳过健康检查"

        try:
            current_time = time.time()
            elapsed = current_time - self._api_last_response_time

            if elapsed > self.api_timeout_seconds:
                self._api_timeouts += 1

                self._emit_risk_event(
                    event_type=RiskEventType.API_TIMEOUT,
                    level=RiskLevel.WARNING,
                    message=f"API响应超时: {elapsed:.2f}秒 > 阈值 {self.api_timeout_seconds}秒",
                    details={
                        'elapsed_seconds': elapsed,
                        'timeout_threshold': self.api_timeout_seconds,
                        'consecutive_timeouts': self._api_timeouts,
                    },
                )

                if self._api_timeouts >= self._max_api_timeouts:
                    self._emit_risk_event(
                        event_type=RiskEventType.CONNECTION_LOST,
                        level=RiskLevel.CRITICAL,
                        message=f"连接可能已丢失: 连续 {self._api_timeouts} 次超时",
                        details={
                            'consecutive_timeouts': self._api_timeouts,
                            'max_timeouts': self._max_api_timeouts,
                        },
                    )
                    return False, f"连接可能已丢失: 连续 {self._api_timeouts} 次超时"

                return False, f"API响应超时: {elapsed:.2f}秒"

            self._api_timeouts = 0
            return True, "API健康"

        except Exception as e:
            return False, f"API健康检查异常: {e}"

    def update_api_response_time(self):
        self._api_last_response_time = time.time()
        self._api_timeouts = 0

    def check_price_gap(self, contract: str, current_price: float) -> Optional[PriceGapInfo]:
        if contract not in self._previous_prices:
            self._previous_prices[contract] = current_price
            return None

        previous_price = self._previous_prices[contract]

        if previous_price <= 0 or current_price <= 0:
            self._previous_prices[contract] = current_price
            return None

        gap_percent = abs((current_price - previous_price) / previous_price * 100)
        direction = "UP" if current_price > previous_price else "DOWN"

        if gap_percent >= self.price_gap_threshold_percent:
            gap_info = PriceGapInfo(
                contract=contract,
                previous_price=previous_price,
                current_price=current_price,
                gap_percent=gap_percent,
                timestamp=time.time(),
                direction=direction,
            )

            self._emit_risk_event(
                event_type=RiskEventType.PRICE_GAP_DETECTED,
                level=RiskLevel.WARNING if gap_percent < self.price_gap_threshold_percent * 2 else RiskLevel.CRITICAL,
                message=f"检测到价格跳空 [{direction}]: {contract} 从 {previous_price:.2f} 到 {current_price:.2f}, 跳空 {gap_percent:.2f}%",
                details={
                    'contract': contract,
                    'previous_price': previous_price,
                    'current_price': current_price,
                    'gap_percent': gap_percent,
                    'direction': direction,
                    'threshold_percent': self.price_gap_threshold_percent,
                },
            )

        self._previous_prices[contract] = current_price
        return None

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

            self.update_api_response_time()

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
                self.update_api_response_time()
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

                if position.current_price > 0:
                    self.check_price_gap(contract, position.current_price)

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

        api_healthy, api_msg = self.check_api_health()
        if not api_healthy:
            return False, f"API健康检查失败: {api_msg}", RiskLevel.WARNING

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

    def batch_check_strategies_margin(
        self,
        strategy_names: List[str],
        snapshot: AccountSnapshot = None,
    ) -> Dict[str, RiskLevel]:
        if self._frozen:
            return {name: RiskLevel.FROZEN for name in strategy_names}

        if snapshot is None:
            snapshot = self.get_account_snapshot()

        if NUMPY_AVAILABLE and len(strategy_names) > 10:
            return self._batch_check_strategies_margin_numpy(strategy_names, snapshot)

        results = {}
        for name in strategy_names:
            results[name] = self.check_strategy_margin(name, snapshot=snapshot)

        return results

    def _batch_check_strategies_margin_numpy(
        self,
        strategy_names: List[str],
        snapshot: AccountSnapshot,
    ) -> Dict[str, RiskLevel]:
        if snapshot.total_asset <= 0:
            return {name: RiskLevel.SAFE for name in strategy_names}

        margins = []
        for name in strategy_names:
            risk_info = self._strategy_risk.get(name)
            margins.append(risk_info.margin_used if risk_info else 0.0)

        margins_array = np.array(margins, dtype=np.float64)
        total_asset = snapshot.total_asset

        margin_percents = (margins_array / total_asset) * 100

        warning_threshold = self.max_strategy_margin_percent * 0.7
        critical_threshold = self.max_strategy_margin_percent

        results = {}
        for i, name in enumerate(strategy_names):
            percent = margin_percents[i]
            if percent >= critical_threshold:
                results[name] = RiskLevel.CRITICAL
            elif percent >= warning_threshold:
                results[name] = RiskLevel.WARNING
            else:
                results[name] = RiskLevel.SAFE

        return results

    def run_risk_checks(self) -> Dict[str, RiskLevel]:
        if self._frozen:
            return {'drawdown': RiskLevel.FROZEN, 'total_margin': RiskLevel.FROZEN}

        self.check_api_health()

        snapshot = self.get_account_snapshot()
        self.update_positions()

        results = {}

        results['drawdown'] = self.check_drawdown(snapshot)

        if not self._frozen:
            results['total_margin'] = self.check_total_margin(snapshot)

            strategy_names = list(self._strategy_risk.keys())
            if strategy_names:
                strategy_results = self.batch_check_strategies_margin(strategy_names, snapshot)
                for name, level in strategy_results.items():
                    results[f'strategy_{name}'] = level

        return results

    def run_performance_benchmark(self, num_strategies: int = 50) -> Dict[str, Any]:
        for i in range(num_strategies):
            strategy_name = f"Benchmark_Strategy_{i+1}"
            margin_used = 50000.0 + (i * 1000.0)
            self.update_strategy_risk(
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

        self._peak_equity = 10000000.0

        strategy_names = list(self._strategy_risk.keys())

        num_iterations = 100
        times = []

        for _ in range(num_iterations):
            start_time = time.perf_counter()

            self.check_api_health()
            self.check_drawdown(snapshot)
            self.check_total_margin(snapshot)
            self.batch_check_strategies_margin(strategy_names, snapshot)

            end_time = time.perf_counter()
            times.append((end_time - start_time) * 1000)

        times_array = np.array(times) if NUMPY_AVAILABLE else None

        report = {
            'benchmark_time': datetime.now().isoformat(),
            'num_strategies': num_strategies,
            'num_iterations': num_iterations,
            'numpy_available': NUMPY_AVAILABLE,
            'statistics': {
                'min_ms': min(times),
                'max_ms': max(times),
                'avg_ms': sum(times) / len(times),
                'median_ms': sorted(times)[len(times) // 2],
                'p95_ms': sorted(times)[int(len(times) * 0.95)],
                'p99_ms': sorted(times)[int(len(times) * 0.99)],
            },
            'summary': f"""
        ================ 性 能 测 试 报 告 ================
        测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        策略数量: {num_strategies}
        迭代次数: {num_iterations}
        NumPy可用: {NUMPY_AVAILABLE}
        
        ---------- 单次风控检查耗时统计 ----------
        最小: {min(times):.3f} ms
        最大: {max(times):.3f} ms
        平均: {sum(times)/len(times):.3f} ms
        中位数: {sorted(times)[len(times)//2]:.3f} ms
        P95: {sorted(times)[int(len(times)*0.95)]:.3f} ms
        P99: {sorted(times)[int(len(times)*0.99)]:.3f} ms
        
        ---------- 性能评估 ----------
        {'✅ 优秀: 平均耗时 < 1ms' if sum(times)/len(times) < 1 else 
         '⚠️ 良好: 平均耗时 < 5ms' if sum(times)/len(times) < 5 else
         '🔴 需要优化: 平均耗时 >= 5ms'}
        
        ==================================================
        """
        }

        self.logger.info(report['summary'])

        if self._risk_logger:
            self._risk_logger.info(json.dumps(report, ensure_ascii=False, indent=2))

        return report

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
            'api_health': {
                'last_response_time': datetime.fromtimestamp(self._api_last_response_time).isoformat(),
                'consecutive_timeouts': self._api_timeouts,
                'max_timeouts': self._max_api_timeouts,
            },
        }

    def get_risk_events(self, limit: int = 10) -> List[Dict[str, Any]]:
        events = self._risk_events[-limit:] if limit > 0 else self._risk_events
        return [e.to_dict() for e in events]

    def close_all_positions(self) -> Dict[str, Any]:
        if self.api is None:
            self.logger.warning("无 API 连接，无法执行平仓操作")
            return {'status': 'error', 'message': '无 API 连接'}

        results = {'status': 'success', 'closed': [], 'failed': []}

        self.logger.critical("🔥🔥🔥 正在执行紧急平仓操作...")

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
                        self.logger.critical(f"✅ 已发送平多单指令: {contract}, 数量: {position.long_volume}")
                    except Exception as e:
                        results['failed'].append(
                            {
                                'contract': contract,
                                'direction': 'SELL',
                                'volume': position.long_volume,
                                'error': str(e),
                            }
                        )
                        self.logger.error(f"❌ 平多单失败 [{contract}]: {e}")

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
                        self.logger.critical(f"✅ 已发送平空单指令: {contract}, 数量: {position.short_volume}")
                    except Exception as e:
                        results['failed'].append(
                            {
                                'contract': contract,
                                'direction': 'BUY',
                                'volume': position.short_volume,
                                'error': str(e),
                            }
                        )
                        self.logger.error(f"❌ 平空单失败 [{contract}]: {e}")

        except Exception as e:
            self.logger.error(f"执行平仓操作失败: {e}")
            results['status'] = 'error'
            results['message'] = str(e)

        return results

    def emergency_stop(self, reason: str) -> Dict[str, Any]:
        fire_ascii = """
        ⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️
        ⚠️                                          ⚠️
        ⚠️     🔴 执 行 紧 急 停 止 🔴     ⚠️
        ⚠️                                          ⚠️
        ⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️
        """

        self.logger.critical(fire_ascii)
        self.logger.critical(f"紧急停止原因: {reason}")

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
