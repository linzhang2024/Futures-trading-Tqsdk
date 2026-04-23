import logging
import os
import sys
import csv
import json
import copy
import itertools
import math
import statistics
from datetime import date, datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple, Callable, Type, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Manager

from tqsdk import TqApi, TqSim, TqAuth, TqBacktest
from tqsdk.exceptions import BacktestFinished

from strategies.base_strategy import StrategyBase, SignalType
from core.manager import StrategyManager
from core.risk_manager import RiskManager, RiskLevel, AccountSnapshot


class BacktestMode(Enum):
    SINGLE = "single"
    OPTIMIZATION = "optimization"


@dataclass
class PerformanceMetrics:
    total_return: float = 0.0
    total_return_percent: float = 0.0
    annualized_return: float = 0.0
    annualized_return_percent: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_percent: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_trade_return: float = 0.0
    total_commission_cost: float = 0.0
    total_slippage_cost: float = 0.0
    total_cost: float = 0.0


@dataclass
class BacktestResult:
    strategy_name: str
    params: Dict[str, Any]
    start_dt: date
    end_dt: date
    initial_equity: float = 0.0
    final_equity: float = 0.0
    performance: PerformanceMetrics = field(default_factory=PerformanceMetrics)
    risk_triggered: bool = False
    frozen_during_backtest: bool = False
    frozen_reason: Optional[str] = None
    status: str = "completed"
    error_message: Optional[str] = None
    equity_curve: List[Dict[str, Any]] = field(default_factory=list)
    trade_records: List[Dict[str, Any]] = field(default_factory=list)
    risk_events: List[Dict[str, Any]] = field(default_factory=list)
    config_snapshot: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ParameterRange:
    name: str
    min_val: Union[int, float]
    max_val: Union[int, float]
    step: Union[int, float] = 1
    param_type: Type = int


@dataclass
class CostConfig:
    default_commission_per_lot: float = 0.0
    default_slippage_points: float = 0.0
    contract_configs: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    def get_commission(self, symbol: str) -> float:
        if symbol in self.contract_configs:
            return self.contract_configs[symbol].get('commission_per_lot', self.default_commission_per_lot)
        return self.default_commission_per_lot
    
    def get_slippage(self, symbol: str) -> float:
        if symbol in self.contract_configs:
            return self.contract_configs[symbol].get('slippage_points', self.default_slippage_points)
        return self.default_slippage_points


@dataclass
class PerformanceConfig:
    risk_free_rate: float = 0.03
    trading_days_per_year: int = 252


class BacktestEngine:
    def __init__(
        self,
        config: Dict[str, Any] = None,
        on_progress: Callable[[int, int, str], None] = None,
    ):
        self.config = config or {}
        self.on_progress = on_progress
        self.logger = logging.getLogger(self.__class__.__name__)
        
        self._results: List[BacktestResult] = []
        self._best_result: Optional[BacktestResult] = None
        
        backtest_config = self.config.get('backtest', {})
        self._init_balance = backtest_config.get('init_balance', 1000000.0)
        
        self._tq_account = None
        if backtest_config.get('tq_account') and backtest_config.get('tq_password'):
            self._tq_account = backtest_config['tq_account']
            self._tq_password = backtest_config['tq_password']
        
        self._cost_config = self._parse_cost_config(backtest_config)
        
        self._performance_config = PerformanceConfig(
            risk_free_rate=backtest_config.get('performance', {}).get('risk_free_rate', 0.03),
            trading_days_per_year=backtest_config.get('performance', {}).get('trading_days_per_year', 252),
        )
        
        optimization_config = backtest_config.get('optimization', {})
        self._max_workers = optimization_config.get('max_workers', os.cpu_count() or 4)
        
        self.logger.info(f"BacktestEngine 初始化完成，初始资金: {self._init_balance}")
        self.logger.info(f"成本配置: 手续费={self._cost_config.default_commission_per_lot}元/手, 滑点={self._cost_config.default_slippage_points}点")
        self.logger.info(f"并行配置: 最大工作进程={self._max_workers}")

    def _parse_cost_config(self, backtest_config: Dict[str, Any]) -> CostConfig:
        costs_config = backtest_config.get('costs', {})
        
        default_commission = costs_config.get('default_commission_per_lot', 0.0)
        default_slippage = costs_config.get('default_slippage_points', 0.0)
        
        contract_configs = {}
        contracts = costs_config.get('contracts', {})
        for symbol, cfg in contracts.items():
            contract_configs[symbol] = {
                'commission_per_lot': cfg.get('commission_per_lot', default_commission),
                'slippage_points': cfg.get('slippage_points', default_slippage),
            }
        
        return CostConfig(
            default_commission_per_lot=default_commission,
            default_slippage_points=default_slippage,
            contract_configs=contract_configs,
        )

    def _create_backtest_api(
        self, 
        start_dt: date, 
        end_dt: date,
        cost_config: CostConfig = None,
    ) -> Tuple[TqApi, TqSim]:
        self.logger.info(f"创建回测 API，时间段: {start_dt} 至 {end_dt}")
        
        account = TqSim(init_balance=self._init_balance)
        backtest = TqBacktest(start_dt=start_dt, end_dt=end_dt)
        
        if cost_config:
            for symbol, cfg in cost_config.contract_configs.items():
                commission = cfg.get('commission_per_lot', 0.0)
                if commission > 0:
                    account.set_commission(symbol, commission)
                    self.logger.debug(f"设置合约 {symbol} 手续费: {commission}元/手")
        
        if self._tq_account and self._tq_password:
            auth = TqAuth(self._tq_account, self._tq_password)
            api = TqApi(account=account, backtest=backtest, auth=auth)
        else:
            api = TqApi(account=account, backtest=backtest)
        
        return api, account

    def _extract_date_range(self) -> Tuple[date, date]:
        backtest_config = self.config.get('backtest', {})
        
        start_dt_str = backtest_config.get('start_dt')
        end_dt_str = backtest_config.get('end_dt')
        
        if not start_dt_str or not end_dt_str:
            raise ValueError("配置中缺少回测时间段 (start_dt 或 end_dt)")
        
        def parse_date(date_str: str) -> date:
            if isinstance(date_str, date):
                return date_str
            if isinstance(date_str, datetime):
                return date_str.date()
            for fmt in ['%Y-%m-%d', '%Y%m%d', '%Y/%m/%d']:
                try:
                    return datetime.strptime(str(date_str), fmt).date()
                except ValueError:
                    continue
            raise ValueError(f"无法解析日期格式: {date_str}")
        
        start_dt = parse_date(start_dt_str)
        end_dt = parse_date(end_dt_str)
        
        if start_dt >= end_dt:
            raise ValueError(f"开始日期 ({start_dt}) 必须早于结束日期 ({end_dt})")
        
        return start_dt, end_dt

    def _create_strategy_manager(
        self,
        api: TqApi,
        strategy_class: Type[StrategyBase],
        strategy_params: Dict[str, Any],
    ) -> StrategyManager:
        mock_connector = type('MockConnector', (), {
            'get_api': lambda self: api,
            'get_config': lambda self: self.config,
        })()
        
        manager = StrategyManager(connector=mock_connector)
        manager.configure_from_dict(self.config)
        
        strategy = strategy_class(**strategy_params)
        strategy.set_connector(mock_connector)
        
        manager.register_strategy('BacktestStrategy', strategy)
        
        return manager

    def _create_risk_manager(
        self,
        api: TqApi,
    ) -> RiskManager:
        mock_connector = type('MockConnector', (), {
            'get_api': lambda self: api,
        })()
        
        risk_manager = RiskManager(connector=mock_connector)
        
        risk_config = self.config.get('risk', {})
        if risk_config.get('max_drawdown_percent'):
            risk_manager.max_drawdown_percent = float(risk_config['max_drawdown_percent'])
        if risk_config.get('max_strategy_margin_percent'):
            risk_manager.max_strategy_margin_percent = float(risk_config['max_strategy_margin_percent'])
        if risk_config.get('max_total_margin_percent'):
            risk_manager.max_total_margin_percent = float(risk_config['max_total_margin_percent'])
        
        return risk_manager

    def _calculate_returns_from_equity_curve(
        self, 
        equity_curve: List[Dict[str, Any]],
        initial_equity: float,
    ) -> Tuple[List[float], List[float]]:
        if not equity_curve:
            return [], []
        
        returns = []
        log_returns = []
        prev_equity = initial_equity
        
        for point in equity_curve:
            equity = point.get('equity', prev_equity)
            if prev_equity > 0:
                simple_return = (equity - prev_equity) / prev_equity
                log_return = math.log(equity / prev_equity) if equity > 0 else 0
                returns.append(simple_return)
                log_returns.append(log_return)
            prev_equity = equity
        
        return returns, log_returns

    def _calculate_performance_metrics(
        self,
        result: BacktestResult,
        start_dt: date,
        end_dt: date,
        equity_curve: List[Dict[str, Any]],
        initial_equity: float,
        final_equity: float,
        total_trades: int = 0,
    ) -> PerformanceMetrics:
        metrics = PerformanceMetrics()
        
        metrics.total_return = final_equity - initial_equity
        metrics.total_return_percent = (metrics.total_return / initial_equity * 100) if initial_equity > 0 else 0.0
        
        days = (end_dt - start_dt).days
        if days > 0 and initial_equity > 0:
            daily_return = (final_equity / initial_equity) ** (1.0 / days) - 1
            metrics.annualized_return = (1 + daily_return) ** 365 - 1
            metrics.annualized_return_percent = metrics.annualized_return * 100
        
        if equity_curve:
            equities = [initial_equity] + [p.get('equity', initial_equity) for p in equity_curve]
            
            peak = initial_equity
            max_dd = 0.0
            max_dd_percent = 0.0
            
            for eq in equities:
                if eq > peak:
                    peak = eq
                dd = peak - eq
                dd_percent = (dd / peak * 100) if peak > 0 else 0.0
                
                if dd > max_dd:
                    max_dd = dd
                if dd_percent > max_dd_percent:
                    max_dd_percent = dd_percent
            
            metrics.max_drawdown = max_dd
            metrics.max_drawdown_percent = max_dd_percent
            
            returns, log_returns = self._calculate_returns_from_equity_curve(equity_curve, initial_equity)
            
            if returns and len(returns) > 1:
                try:
                    avg_return = statistics.mean(returns)
                    std_return = statistics.stdev(returns)
                    
                    if std_return > 0:
                        daily_rf = (1 + self._performance_config.risk_free_rate) ** (1/365) - 1
                        daily_sharpe = (avg_return - daily_rf) / std_return
                        metrics.sharpe_ratio = daily_sharpe * math.sqrt(252)
                    
                    negative_returns = [r for r in returns if r < 0]
                    if negative_returns and len(negative_returns) > 1:
                        std_negative = statistics.stdev(negative_returns)
                        if std_negative > 0:
                            daily_sortino = (avg_return - daily_rf) / std_negative
                            metrics.sortino_ratio = daily_sortino * math.sqrt(252)
                    
                    if max_dd_percent > 0:
                        metrics.calmar_ratio = metrics.annualized_return / (max_dd_percent / 100)
                        
                except Exception as e:
                    self.logger.debug(f"计算风险调整收益指标时出错: {e}")
        
        metrics.total_trades = total_trades
        
        metrics.total_commission_cost = self._cost_config.default_commission_per_lot * total_trades
        metrics.total_slippage_cost = 0.0
        metrics.total_cost = metrics.total_commission_cost + metrics.total_slippage_cost
        
        if total_trades > 0:
            metrics.avg_trade_return = metrics.total_return / total_trades
        
        return metrics

    def _run_single_backtest_internal(
        self,
        start_dt: date,
        end_dt: date,
        strategy_class: Type[StrategyBase],
        strategy_params: Dict[str, Any],
        strategy_name: str = "BacktestStrategy",
        config: Dict[str, Any] = None,
        cost_config: CostConfig = None,
        performance_config: PerformanceConfig = None,
    ) -> Dict[str, Any]:
        result_dict = {
            'strategy_name': strategy_name,
            'params': strategy_params.copy(),
            'start_dt': start_dt.isoformat(),
            'end_dt': end_dt.isoformat(),
            'initial_equity': 0.0,
            'final_equity': 0.0,
            'performance': {},
            'risk_triggered': False,
            'frozen_during_backtest': False,
            'frozen_reason': None,
            'status': 'completed',
            'error_message': None,
            'equity_curve': [],
            'trade_records': [],
            'risk_events': [],
        }
        
        api = None
        account = None
        manager = None
        risk_manager = None
        
        try:
            from tqsdk import TqApi, TqSim, TqAuth, TqBacktest
            from tqsdk.exceptions import BacktestFinished
            
            init_balance = config.get('backtest', {}).get('init_balance', 1000000.0) if config else 1000000.0
            
            account = TqSim(init_balance=init_balance)
            backtest = TqBacktest(start_dt=start_dt, end_dt=end_dt)
            
            tq_account = config.get('backtest', {}).get('tq_account') if config else None
            tq_password = config.get('backtest', {}).get('tq_password') if config else None
            
            if cost_config:
                for symbol, cfg in cost_config.contract_configs.items():
                    commission = cfg.get('commission_per_lot', 0.0)
                    if commission > 0:
                        account.set_commission(symbol, commission)
            
            if tq_account and tq_password:
                auth = TqAuth(tq_account, tq_password)
                api = TqApi(account=account, backtest=backtest, auth=auth)
            else:
                api = TqApi(account=account, backtest=backtest)
            
            from strategies.base_strategy import StrategyBase
            from core.manager import StrategyManager
            from core.risk_manager import RiskManager
            
            mock_connector = type('MockConnector', (), {
                'get_api': lambda self: api,
                'get_config': lambda self: config or {},
            })()
            
            manager = StrategyManager(connector=mock_connector)
            manager.configure_from_dict(config or {})
            
            strategy = strategy_class(**strategy_params)
            strategy.set_connector(mock_connector)
            manager.register_strategy('BacktestStrategy', strategy)
            
            risk_manager = RiskManager(connector=mock_connector)
            risk_config = config.get('risk', {}) if config else {}
            if risk_config.get('max_drawdown_percent'):
                risk_manager.max_drawdown_percent = float(risk_config['max_drawdown_percent'])
            manager.set_risk_manager(risk_manager)
            
            manager.initialize(load_saved_states=False)
            
            initial_snapshot = risk_manager.get_account_snapshot()
            result_dict['initial_equity'] = initial_snapshot.equity
            
            equity_curve = []
            risk_events = []
            cycle_count = 0
            
            while True:
                try:
                    api.wait_update()
                    cycle_count += 1
                    
                    if manager._should_run_risk_check():
                        manager.run_risk_checks()
                        
                        if manager.is_risk_frozen():
                            result_dict['risk_triggered'] = True
                            result_dict['frozen_during_backtest'] = True
                            result_dict['frozen_reason'] = risk_manager.get_frozen_reason()
                            break
                    
                    for name, strat in manager._strategies.items():
                        health = manager._strategy_health.get(name)
                        if health and health.can_run() and hasattr(strat, '_on_update'):
                            try:
                                strat._on_update()
                                health.record_success()
                            except Exception as e:
                                health.record_error(str(e))
                    
                    if cycle_count % 50 == 0:
                        snapshot = risk_manager.get_account_snapshot()
                        equity_curve.append({
                            'cycle': cycle_count,
                            'timestamp': snapshot.timestamp,
                            'equity': snapshot.equity,
                            'margin_used': snapshot.margin_used,
                        })
                        
                except BacktestFinished:
                    break
            
            final_snapshot = risk_manager.get_account_snapshot()
            result_dict['final_equity'] = final_snapshot.equity
            result_dict['equity_curve'] = equity_curve
            
            events = risk_manager.get_risk_events(limit=100)
            result_dict['risk_events'] = events
            
            drawdown_info = risk_manager.get_drawdown_info()
            current_dd_percent = drawdown_info.get('current_drawdown_percent', 0.0)
            
            total_trades = 0
            try:
                if hasattr(account, 'trades') and account.trades:
                    total_trades = len(account.trades)
            except Exception:
                pass
            
            performance_config = performance_config or PerformanceConfig()
            initial_eq = result_dict['initial_equity']
            final_eq = result_dict['final_equity']
            eq_curve = result_dict['equity_curve']
            
            total_return = final_eq - initial_eq
            total_return_percent = (total_return / initial_eq * 100) if initial_eq > 0 else 0.0
            
            days = (end_dt - start_dt).days
            annualized_return = 0.0
            annualized_return_percent = 0.0
            if days > 0 and initial_eq > 0:
                daily_return = (final_eq / initial_eq) ** (1.0 / days) - 1
                annualized_return = (1 + daily_return) ** 365 - 1
                annualized_return_percent = annualized_return * 100
            
            max_dd = 0.0
            max_dd_percent = 0.0
            if eq_curve:
                equities = [initial_eq] + [p.get('equity', initial_eq) for p in eq_curve]
                peak = initial_eq
                for eq in equities:
                    if eq > peak:
                        peak = eq
                    dd = peak - eq
                    dd_percent = (dd / peak * 100) if peak > 0 else 0.0
                    if dd > max_dd:
                        max_dd = dd
                    if dd_percent > max_dd_percent:
                        max_dd_percent = dd_percent
            
            result_dict['performance'] = {
                'total_return': total_return,
                'total_return_percent': total_return_percent,
                'annualized_return': annualized_return,
                'annualized_return_percent': annualized_return_percent,
                'max_drawdown': max_dd,
                'max_drawdown_percent': max_dd_percent,
                'sharpe_ratio': 0.0,
                'sortino_ratio': 0.0,
                'calmar_ratio': 0.0,
                'total_trades': total_trades,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'avg_trade_return': total_return / total_trades if total_trades > 0 else 0.0,
                'total_commission_cost': cost_config.default_commission_per_lot * total_trades if cost_config else 0.0,
                'total_slippage_cost': 0.0,
                'total_cost': cost_config.default_commission_per_lot * total_trades if cost_config else 0.0,
            }
            
            result_dict['config_snapshot'] = {
                'cost_config': {
                    'default_commission': cost_config.default_commission_per_lot if cost_config else 0.0,
                    'default_slippage': cost_config.default_slippage_points if cost_config else 0.0,
                },
            }
            
        except Exception as e:
            result_dict['status'] = 'error'
            result_dict['error_message'] = str(e)
        finally:
            if api and not api.is_closed():
                api.close()
        
        return result_dict

    def _run_single_backtest(
        self,
        start_dt: date,
        end_dt: date,
        strategy_class: Type[StrategyBase],
        strategy_params: Dict[str, Any],
        strategy_name: str = "BacktestStrategy",
    ) -> BacktestResult:
        result_dict = self._run_single_backtest_internal(
            start_dt=start_dt,
            end_dt=end_dt,
            strategy_class=strategy_class,
            strategy_params=strategy_params,
            strategy_name=strategy_name,
            config=self.config,
            cost_config=self._cost_config,
            performance_config=self._performance_config,
        )
        
        perf_dict = result_dict.get('performance', {})
        performance = PerformanceMetrics(
            total_return=perf_dict.get('total_return', 0.0),
            total_return_percent=perf_dict.get('total_return_percent', 0.0),
            annualized_return=perf_dict.get('annualized_return', 0.0),
            annualized_return_percent=perf_dict.get('annualized_return_percent', 0.0),
            max_drawdown=perf_dict.get('max_drawdown', 0.0),
            max_drawdown_percent=perf_dict.get('max_drawdown_percent', 0.0),
            sharpe_ratio=perf_dict.get('sharpe_ratio', 0.0),
            sortino_ratio=perf_dict.get('sortino_ratio', 0.0),
            calmar_ratio=perf_dict.get('calmar_ratio', 0.0),
            total_trades=perf_dict.get('total_trades', 0),
            winning_trades=perf_dict.get('winning_trades', 0),
            losing_trades=perf_dict.get('losing_trades', 0),
            win_rate=perf_dict.get('win_rate', 0.0),
            profit_factor=perf_dict.get('profit_factor', 0.0),
            avg_trade_return=perf_dict.get('avg_trade_return', 0.0),
            total_commission_cost=perf_dict.get('total_commission_cost', 0.0),
            total_slippage_cost=perf_dict.get('total_slippage_cost', 0.0),
            total_cost=perf_dict.get('total_cost', 0.0),
        )
        
        def parse_date_str(date_str: str) -> date:
            try:
                return datetime.fromisoformat(date_str).date()
            except Exception:
                return date.today()
        
        result = BacktestResult(
            strategy_name=result_dict['strategy_name'],
            params=result_dict['params'],
            start_dt=parse_date_str(result_dict['start_dt']),
            end_dt=parse_date_str(result_dict['end_dt']),
            initial_equity=result_dict['initial_equity'],
            final_equity=result_dict['final_equity'],
            performance=performance,
            risk_triggered=result_dict['risk_triggered'],
            frozen_during_backtest=result_dict['frozen_during_backtest'],
            frozen_reason=result_dict['frozen_reason'],
            status=result_dict['status'],
            error_message=result_dict['error_message'],
            equity_curve=result_dict.get('equity_curve', []),
            trade_records=result_dict.get('trade_records', []),
            risk_events=result_dict.get('risk_events', []),
            config_snapshot=result_dict.get('config_snapshot', {}),
        )
        
        if result.status == 'error':
            self.logger.error(f"回测失败: {result.error_message}")
        else:
            self.logger.info(
                f"回测完成: 初始权益={result.initial_equity:.2f}, "
                f"最终权益={result.final_equity:.2f}, "
                f"收益率={result.performance.total_return_percent:.2f}%"
            )
        
        return result

    def run_backtest(
        self,
        strategy_class: Type[StrategyBase],
        strategy_params: Dict[str, Any] = None,
        start_dt: date = None,
        end_dt: date = None,
    ) -> BacktestResult:
        if start_dt is None or end_dt is None:
            start_dt, end_dt = self._extract_date_range()
        
        if strategy_params is None:
            strategy_params = {}
        
        self.logger.info(f"开始单参数回测: {strategy_class.__name__}")
        self.logger.info(f"回测时间段: {start_dt} 至 {end_dt}")
        self.logger.info(f"策略参数: {strategy_params}")
        self.logger.info(f"成本配置: 手续费={self._cost_config.default_commission_per_lot}元/手, 滑点={self._cost_config.default_slippage_points}点")
        
        result = self._run_single_backtest(
            start_dt=start_dt,
            end_dt=end_dt,
            strategy_class=strategy_class,
            strategy_params=strategy_params,
            strategy_name=strategy_class.__name__,
        )
        
        self._results = [result]
        self._best_result = result
        
        return result

    def _generate_param_grid(
        self,
        param_ranges: Dict[str, ParameterRange],
    ) -> List[Dict[str, Any]]:
        param_names = list(param_ranges.keys())
        param_values = []
        
        for name, pr in param_ranges.items():
            values = []
            current = pr.min_val
            while current <= pr.max_val:
                values.append(pr.param_type(current))
                current += pr.step
            param_values.append(values)
        
        grids = []
        for combo in itertools.product(*param_values):
            grid = dict(zip(param_names, combo))
            grids.append(grid)
        
        return grids

    def run_optimization(
        self,
        strategy_class: Type[StrategyBase],
        param_ranges: Dict[str, ParameterRange],
        base_params: Dict[str, Any] = None,
        start_dt: date = None,
        end_dt: date = None,
        optimize_by: str = 'total_return_percent',
    ) -> List[BacktestResult]:
        if start_dt is None or end_dt is None:
            start_dt, end_dt = self._extract_date_range()
        
        base_params = base_params or {}
        
        param_grid = self._generate_param_grid(param_ranges)
        total_combinations = len(param_grid)
        
        self.logger.info(f"开始参数寻优: {strategy_class.__name__}")
        self.logger.info(f"回测时间段: {start_dt} 至 {end_dt}")
        self.logger.info(f"参数组合数量: {total_combinations}")
        self.logger.info(f"优化目标: {optimize_by}")
        self.logger.info(f"并行工作进程数: {self._max_workers}")
        
        results = []
        
        if self._max_workers > 1 and total_combinations > 1:
            self.logger.info("使用多进程并行执行回测...")
            
            tasks = []
            for i, params in enumerate(param_grid, 1):
                full_params = base_params.copy()
                full_params.update(params)
                tasks.append((i, full_params))
            
            with ProcessPoolExecutor(max_workers=self._max_workers) as executor:
                future_to_task = {
                    executor.submit(
                        _worker_run_backtest,
                        start_dt,
                        end_dt,
                        strategy_class,
                        full_params,
                        f"{strategy_class.__name__}_Opt_{i}",
                        self.config,
                        self._cost_config,
                        self._performance_config,
                    ): (i, full_params)
                    for i, full_params in tasks
                }
                
                completed = 0
                for future in as_completed(future_to_task):
                    i, params = future_to_task[future]
                    completed += 1
                    
                    progress_msg = f"完成进度: {completed}/{total_combinations}, 参数: {params}"
                    self.logger.info(progress_msg)
                    
                    if self.on_progress:
                        self.on_progress(completed, total_combinations, progress_msg)
                    
                    try:
                        result_dict = future.result()
                        
                        perf_dict = result_dict.get('performance', {})
                        performance = PerformanceMetrics(
                            total_return=perf_dict.get('total_return', 0.0),
                            total_return_percent=perf_dict.get('total_return_percent', 0.0),
                            annualized_return=perf_dict.get('annualized_return', 0.0),
                            annualized_return_percent=perf_dict.get('annualized_return_percent', 0.0),
                            max_drawdown=perf_dict.get('max_drawdown', 0.0),
                            max_drawdown_percent=perf_dict.get('max_drawdown_percent', 0.0),
                            sharpe_ratio=perf_dict.get('sharpe_ratio', 0.0),
                            sortino_ratio=perf_dict.get('sortino_ratio', 0.0),
                            calmar_ratio=perf_dict.get('calmar_ratio', 0.0),
                            total_trades=perf_dict.get('total_trades', 0),
                            winning_trades=perf_dict.get('winning_trades', 0),
                            losing_trades=perf_dict.get('losing_trades', 0),
                            win_rate=perf_dict.get('win_rate', 0.0),
                            profit_factor=perf_dict.get('profit_factor', 0.0),
                            avg_trade_return=perf_dict.get('avg_trade_return', 0.0),
                            total_commission_cost=perf_dict.get('total_commission_cost', 0.0),
                            total_slippage_cost=perf_dict.get('total_slippage_cost', 0.0),
                            total_cost=perf_dict.get('total_cost', 0.0),
                        )
                        
                        def parse_date_str(date_str: str) -> date:
                            try:
                                return datetime.fromisoformat(date_str).date()
                            except Exception:
                                return date.today()
                        
                        result = BacktestResult(
                            strategy_name=result_dict['strategy_name'],
                            params=result_dict['params'],
                            start_dt=parse_date_str(result_dict['start_dt']),
                            end_dt=parse_date_str(result_dict['end_dt']),
                            initial_equity=result_dict['initial_equity'],
                            final_equity=result_dict['final_equity'],
                            performance=performance,
                            risk_triggered=result_dict['risk_triggered'],
                            frozen_during_backtest=result_dict['frozen_during_backtest'],
                            frozen_reason=result_dict['frozen_reason'],
                            status=result_dict['status'],
                            error_message=result_dict['error_message'],
                            equity_curve=result_dict.get('equity_curve', []),
                            trade_records=result_dict.get('trade_records', []),
                            risk_events=result_dict.get('risk_events', []),
                            config_snapshot=result_dict.get('config_snapshot', {}),
                        )
                        
                        results.append(result)
                        
                    except Exception as e:
                        self.logger.error(f"回测任务 {i} 执行失败: {e}")
        else:
            self.logger.info("使用单进程串行执行回测...")
            
            for i, params in enumerate(param_grid, 1):
                full_params = base_params.copy()
                full_params.update(params)
                
                progress_msg = f"正在测试组合 {i}/{total_combinations}: {params}"
                self.logger.info(progress_msg)
                
                if self.on_progress:
                    self.on_progress(i, total_combinations, progress_msg)
                
                result = self._run_single_backtest(
                    start_dt=start_dt,
                    end_dt=end_dt,
                    strategy_class=strategy_class,
                    strategy_params=full_params,
                    strategy_name=f"{strategy_class.__name__}_Opt_{i}",
                )
                
                results.append(result)
                
                self.logger.info(
                    f"组合 {i} 结果: 收益率={result.performance.total_return_percent:.2f}%, "
                    f"最大回撤={result.performance.max_drawdown_percent:.2f}%, "
                    f"状态={result.status}"
                )
        
        def get_optimize_value(r: BacktestResult) -> float:
            if optimize_by == 'total_return_percent':
                return r.performance.total_return_percent
            elif optimize_by == 'annualized_return_percent':
                return r.performance.annualized_return_percent
            elif optimize_by == 'sharpe_ratio':
                return r.performance.sharpe_ratio
            elif optimize_by == 'sortino_ratio':
                return r.performance.sortino_ratio
            elif optimize_by == 'calmar_ratio':
                return r.performance.calmar_ratio
            elif optimize_by == 'max_drawdown_percent':
                return -r.performance.max_drawdown_percent
            elif optimize_by == 'win_rate':
                return r.performance.win_rate
            elif optimize_by == 'profit_factor':
                return r.performance.profit_factor
            return r.performance.total_return_percent
        
        results.sort(key=get_optimize_value, reverse=True)
        
        self._results = results
        if results:
            self._best_result = results[0]
            self.logger.info(f"最优参数组合: {results[0].params}")
            self.logger.info(
                f"最优结果: 收益率={results[0].performance.total_return_percent:.2f}%, "
                f"年化收益率={results[0].performance.annualized_return_percent:.2f}%, "
                f"最大回撤={results[0].performance.max_drawdown_percent:.2f}%, "
                f"夏普比率={results[0].performance.sharpe_ratio:.2f}"
            )
        
        return results

    def _print_performance_table(self, result: BacktestResult) -> None:
        p = result.performance
        
        print("\n" + "=" * 80)
        print("                          回测绩效报告")
        print("=" * 80)
        
        print(f"\n【基本信息】")
        print(f"  策略名称: {result.strategy_name}")
        print(f"  策略参数: {result.params}")
        print(f"  回测区间: {result.start_dt} 至 {result.end_dt}")
        print(f"  运行状态: {result.status}")
        
        print(f"\n【收益指标】")
        row1 = f"{'初始权益':<15} {result.initial_equity:>15,.2f}  {'最终权益':<15} {result.final_equity:>15,.2f}"
        print(f"  {row1}")
        
        row2 = f"{'总收益':<15} {p.total_return:>15,.2f}  {'总收益率':<15} {p.total_return_percent:>13.2f}%"
        print(f"  {row2}")
        
        row3 = f"{'年化收益':<15} {p.annualized_return_percent:>13.2f}%"
        print(f"  {row3}")
        
        print(f"\n【风险指标】")
        row4 = f"{'最大回撤':<15} {p.max_drawdown:>15,.2f}  {'最大回撤率':<15} {p.max_drawdown_percent:>13.2f}%"
        print(f"  {row4}")
        
        row5 = f"{'夏普比率':<15} {p.sharpe_ratio:>15.2f}  {'索提诺比率':<15} {p.sortino_ratio:>15.2f}"
        print(f"  {row5}")
        
        row6 = f"{'卡尔玛比率':<15} {p.calmar_ratio:>15.2f}"
        print(f"  {row6}")
        
        print(f"\n【交易统计】")
        row7 = f"{'总交易次数':<15} {p.total_trades:>15,d}  {'胜率':<15} {p.win_rate:>13.2f}%"
        print(f"  {row7}")
        
        row8 = f"{'盈亏比':<15} {p.profit_factor:>15.2f}  {'平均每笔收益':<15} {p.avg_trade_return:>15,.2f}"
        print(f"  {row8}")
        
        print(f"\n【成本统计】")
        row9 = f"{'总手续费':<15} {p.total_commission_cost:>15,.2f}  {'总滑点成本':<15} {p.total_slippage_cost:>15,.2f}"
        print(f"  {row9}")
        
        row10 = f"{'总成本':<15} {p.total_cost:>15,.2f}"
        print(f"  {row10}")
        
        print(f"\n【风控状态】")
        row11 = f"{'风控触发':<15} {'是' if result.risk_triggered else '否':<15}  {'期间冻结':<15} {'是' if result.frozen_during_backtest else '否':<15}"
        print(f"  {row11}")
        
        if result.frozen_reason:
            print(f"  冻结原因: {result.frozen_reason}")
        
        if result.error_message:
            print(f"\n【错误信息】")
            print(f"  {result.error_message}")
        
        print("\n" + "=" * 80)

    def _print_optimization_table(self, results: List[BacktestResult], top_n: int = 20) -> None:
        if not results:
            return
        
        print("\n" + "=" * 120)
        print("                                      参数寻优结果排名")
        print("=" * 120)
        
        header = (
            f"{'排名':<4} "
            f"{'参数组合':<25} "
            f"{'总收益率(%)':<12} "
            f"{'年化(%)':<10} "
            f"{'最大回撤(%)':<12} "
            f"{'夏普比率':<10} "
            f"{'交易次数':<10} "
            f"{'风控冻结':<8} "
            f"{'状态':<10}"
        )
        print(header)
        print("-" * 120)
        
        display_count = min(top_n, len(results))
        for i, r in enumerate(results[:display_count], 1):
            params_str = str(r.params)[:23]
            if len(str(r.params)) > 23:
                params_str += "..."
            
            row = (
                f"{i:<4} "
                f"{params_str:<25} "
                f"{r.performance.total_return_percent:<12.2f} "
                f"{r.performance.annualized_return_percent:<10.2f} "
                f"{r.performance.max_drawdown_percent:<12.2f} "
                f"{r.performance.sharpe_ratio:<10.2f} "
                f"{r.performance.total_trades:<10} "
                f"{'是' if r.frozen_during_backtest else '否':<8} "
                f"{r.status:<10}"
            )
            print(row)
        
        if len(results) > display_count:
            print(f"... 等共 {len(results)} 个组合")
        
        print("\n" + "=" * 120)

    def generate_report(self, output_dir: str = None) -> Dict[str, Any]:
        if not self._results:
            self.logger.warning("没有回测结果，无法生成报告")
            return {}
        
        if output_dir is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            output_dir = os.path.join(base_dir, 'logs', 'backtest_reports')
        
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        def result_to_dict(r: BacktestResult) -> Dict[str, Any]:
            return {
                'strategy_name': r.strategy_name,
                'params': r.params,
                'start_dt': r.start_dt.isoformat(),
                'end_dt': r.end_dt.isoformat(),
                'initial_equity': r.initial_equity,
                'final_equity': r.final_equity,
                'performance': asdict(r.performance),
                'risk_triggered': r.risk_triggered,
                'frozen_during_backtest': r.frozen_during_backtest,
                'frozen_reason': r.frozen_reason,
                'status': r.status,
                'error_message': r.error_message,
                'equity_curve': r.equity_curve,
                'trade_records': r.trade_records,
                'risk_events': r.risk_events,
                'config_snapshot': r.config_snapshot,
            }
        
        report = {
            'generated_at': datetime.now().isoformat(),
            'total_backtests': len(self._results),
            'best_result': result_to_dict(self._best_result) if self._best_result else None,
            'all_results': [result_to_dict(r) for r in self._results],
        }
        
        if len(self._results) == 1 and self._best_result:
            self._print_performance_table(self._best_result)
        else:
            if self._best_result:
                self._print_performance_table(self._best_result)
            self._print_optimization_table(self._results)
        
        json_file = os.path.join(output_dir, f'backtest_report_{timestamp}.json')
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2, default=str)
        self.logger.info(f"JSON 报告已保存: {json_file}")
        
        csv_file = os.path.join(output_dir, f'backtest_results_{timestamp}.csv')
        self._write_csv_report(csv_file)
        self.logger.info(f"CSV 报告已保存: {csv_file}")
        
        if self._best_result and self._best_result.equity_curve:
            equity_file = os.path.join(output_dir, f'equity_curve_{timestamp}.csv')
            self._write_equity_curve_csv(equity_file, self._best_result.equity_curve)
            self.logger.info(f"权益曲线 CSV 已保存: {equity_file}")
        
        return report

    def _write_csv_report(self, csv_file: str) -> None:
        if not self._results:
            return
        
        fieldnames = [
            'rank', 'strategy_name', 'params', 'start_dt', 'end_dt',
            'initial_equity', 'final_equity',
            'total_return', 'total_return_percent',
            'annualized_return_percent',
            'max_drawdown', 'max_drawdown_percent',
            'sharpe_ratio', 'sortino_ratio', 'calmar_ratio',
            'total_trades', 'win_rate', 'profit_factor',
            'total_commission_cost', 'total_slippage_cost', 'total_cost',
            'risk_triggered', 'frozen_during_backtest', 'frozen_reason',
            'status', 'error_message'
        ]
        
        with open(csv_file, 'w', encoding='utf-8-sig', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for i, r in enumerate(self._results, 1):
                p = r.performance
                row = {
                    'rank': i,
                    'strategy_name': r.strategy_name,
                    'params': json.dumps(r.params, ensure_ascii=False),
                    'start_dt': r.start_dt,
                    'end_dt': r.end_dt,
                    'initial_equity': r.initial_equity,
                    'final_equity': r.final_equity,
                    'total_return': p.total_return,
                    'total_return_percent': p.total_return_percent,
                    'annualized_return_percent': p.annualized_return_percent,
                    'max_drawdown': p.max_drawdown,
                    'max_drawdown_percent': p.max_drawdown_percent,
                    'sharpe_ratio': p.sharpe_ratio,
                    'sortino_ratio': p.sortino_ratio,
                    'calmar_ratio': p.calmar_ratio,
                    'total_trades': p.total_trades,
                    'win_rate': p.win_rate,
                    'profit_factor': p.profit_factor,
                    'total_commission_cost': p.total_commission_cost,
                    'total_slippage_cost': p.total_slippage_cost,
                    'total_cost': p.total_cost,
                    'risk_triggered': r.risk_triggered,
                    'frozen_during_backtest': r.frozen_during_backtest,
                    'frozen_reason': r.frozen_reason or '',
                    'status': r.status,
                    'error_message': r.error_message or '',
                }
                writer.writerow(row)

    def _write_equity_curve_csv(self, csv_file: str, equity_curve: List[Dict[str, Any]]) -> None:
        if not equity_curve:
            return
        
        fieldnames = ['cycle', 'timestamp', 'equity', 'margin_used']
        
        with open(csv_file, 'w', encoding='utf-8-sig', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for point in equity_curve:
                row = {
                    'cycle': point.get('cycle', 0),
                    'timestamp': point.get('timestamp', 0),
                    'equity': point.get('equity', 0),
                    'margin_used': point.get('margin_used', 0),
                }
                writer.writerow(row)

    def get_results(self) -> List[BacktestResult]:
        return self._results.copy()

    def get_best_result(self) -> Optional[BacktestResult]:
        return self._best_result

    def clear_results(self) -> None:
        self._results = []
        self._best_result = None
        self.logger.info("回测结果已清空")


def _worker_run_backtest(
    start_dt: date,
    end_dt: date,
    strategy_class: Type[StrategyBase],
    strategy_params: Dict[str, Any],
    strategy_name: str,
    config: Dict[str, Any],
    cost_config: CostConfig,
    performance_config: PerformanceConfig,
) -> Dict[str, Any]:
    engine = BacktestEngine(config=config)
    engine._cost_config = cost_config
    engine._performance_config = performance_config
    
    return engine._run_single_backtest_internal(
        start_dt=start_dt,
        end_dt=end_dt,
        strategy_class=strategy_class,
        strategy_params=strategy_params,
        strategy_name=strategy_name,
        config=config,
        cost_config=cost_config,
        performance_config=performance_config,
    )
