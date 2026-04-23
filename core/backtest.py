import logging
import os
import csv
import json
import itertools
from datetime import date, datetime
from typing import Dict, Any, List, Optional, Tuple, Callable, Type, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import defaultdict

from tqsdk import TqApi, TqSim, TqAuth, TqBacktest
from tqsdk.exceptions import BacktestFinished

from strategies.base_strategy import StrategyBase, SignalType
from core.manager import StrategyManager
from core.risk_manager import RiskManager, RiskLevel, AccountSnapshot


class BacktestMode(Enum):
    SINGLE = "single"
    OPTIMIZATION = "optimization"


@dataclass
class BacktestResult:
    strategy_name: str
    params: Dict[str, Any]
    start_dt: date
    end_dt: date
    initial_equity: float = 0.0
    final_equity: float = 0.0
    total_return: float = 0.0
    total_return_percent: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_percent: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    total_profit: float = 0.0
    total_loss: float = 0.0
    profit_factor: float = 0.0
    avg_trade: float = 0.0
    risk_triggered: bool = False
    frozen_during_backtest: bool = False
    frozen_reason: Optional[str] = None
    status: str = "completed"
    error_message: Optional[str] = None
    equity_curve: List[Dict[str, Any]] = field(default_factory=list)
    trade_records: List[Dict[str, Any]] = field(default_factory=list)
    risk_events: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class ParameterRange:
    name: str
    min_val: Union[int, float]
    max_val: Union[int, float]
    step: Union[int, float] = 1
    param_type: Type = int


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
        
        self.logger.info(f"BacktestEngine 初始化完成，初始资金: {self._init_balance}")

    def _create_backtest_api(self, start_dt: date, end_dt: date) -> TqApi:
        self.logger.info(f"创建回测 API，时间段: {start_dt} 至 {end_dt}")
        
        account = TqSim(init_balance=self._init_balance)
        backtest = TqBacktest(start_dt=start_dt, end_dt=end_dt)
        
        if self._tq_account and self._tq_password:
            auth = TqAuth(self._tq_account, self._tq_password)
            api = TqApi(account=account, backtest=backtest, auth=auth)
        else:
            api = TqApi(account=account, backtest=backtest)
        
        return api

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
        from core.connection import TqConnector
        
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

    def _run_single_backtest(
        self,
        start_dt: date,
        end_dt: date,
        strategy_class: Type[StrategyBase],
        strategy_params: Dict[str, Any],
        strategy_name: str = "BacktestStrategy",
    ) -> BacktestResult:
        result = BacktestResult(
            strategy_name=strategy_name,
            params=strategy_params.copy(),
            start_dt=start_dt,
            end_dt=end_dt,
        )
        
        api = None
        manager = None
        risk_manager = None
        
        try:
            api = self._create_backtest_api(start_dt, end_dt)
            
            strategy_params_with_connector = strategy_params.copy()
            mock_connector = type('MockConnector', (), {
                'get_api': lambda self: api,
                'get_config': lambda self: self.config,
            })()
            strategy_params_with_connector['connector'] = mock_connector
            
            manager = self._create_strategy_manager(api, strategy_class, strategy_params)
            
            risk_manager = self._create_risk_manager(api)
            manager.set_risk_manager(risk_manager)
            
            manager.initialize(load_saved_states=False)
            
            initial_snapshot = risk_manager.get_account_snapshot()
            result.initial_equity = initial_snapshot.equity
            self.logger.info(f"初始账户权益: {result.initial_equity}")
            
            equity_curve = []
            trade_records = []
            risk_events = []
            
            cycle_count = 0
            last_report_time = 0.0
            
            while True:
                try:
                    api.wait_update()
                    cycle_count += 1
                    
                    if manager._should_run_risk_check():
                        risk_results = manager.run_risk_checks()
                        
                        if manager.is_risk_frozen():
                            result.risk_triggered = True
                            result.frozen_during_backtest = True
                            result.frozen_reason = risk_manager.get_frozen_reason()
                            self.logger.warning(f"风控触发冻结: {result.frozen_reason}")
                            break
                    
                    for name, strategy in manager._strategies.items():
                        health = manager._strategy_health.get(name)
                        
                        if not health or not health.can_run():
                            continue
                        
                        try:
                            if hasattr(strategy, '_on_update'):
                                strategy._on_update()
                            health.record_success()
                        except Exception as e:
                            error_msg = f"策略 {name} 更新时出错: {e}"
                            self.logger.error(error_msg)
                            health.record_error(str(e))
                    
                    if cycle_count % 100 == 0:
                        snapshot = risk_manager.get_account_snapshot()
                        equity_curve.append({
                            'cycle': cycle_count,
                            'timestamp': snapshot.timestamp,
                            'equity': snapshot.equity,
                            'margin_used': snapshot.margin_used,
                        })
                        
                        events = risk_manager.get_risk_events(limit=10)
                        for event in events:
                            if event not in risk_events:
                                risk_events.append(event)
                    
                except BacktestFinished:
                    self.logger.info("回测已完成")
                    break
            
            final_snapshot = risk_manager.get_account_snapshot()
            result.final_equity = final_snapshot.equity
            result.total_return = result.final_equity - result.initial_equity
            result.total_return_percent = (result.total_return / result.initial_equity * 100) if result.initial_equity > 0 else 0.0
            
            result.equity_curve = equity_curve
            result.risk_events = risk_events
            
            drawdown_info = risk_manager.get_drawdown_info()
            result.max_drawdown_percent = drawdown_info.get('current_drawdown_percent', 0.0)
            
            peak_equity = drawdown_info.get('peak_equity', result.initial_equity)
            result.max_drawdown = peak_equity - result.final_equity
            
            try:
                account = api._account if hasattr(api, '_account') else None
                if account and hasattr(account, 'trades'):
                    result.total_trades = len(account.trades)
            except Exception as e:
                self.logger.debug(f"获取交易记录失败: {e}")
            
            self._calculate_statistics(result)
            
            result.status = "completed"
            self.logger.info(f"回测完成: 初始权益={result.initial_equity:.2f}, 最终权益={result.final_equity:.2f}, 收益率={result.total_return_percent:.2f}%")
            
        except Exception as e:
            result.status = "error"
            result.error_message = str(e)
            self.logger.error(f"回测过程中出错: {e}", exc_info=True)
        finally:
            if api and not api.is_closed():
                api.close()
        
        return result

    def _calculate_statistics(self, result: BacktestResult) -> None:
        if result.total_trades > 0:
            result.avg_trade = result.total_return / result.total_trades if result.total_trades > 0 else 0.0
        
        if result.equity_curve:
            equities = [s['equity'] for s in result.equity_curve]
            if equities:
                peak = max(equities)
                current = equities[-1]
                result.max_drawdown = peak - current
                result.max_drawdown_percent = (result.max_drawdown / peak * 100) if peak > 0 else 0.0

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
        
        results = []
        
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
                f"组合 {i} 结果: 收益率={result.total_return_percent:.2f}%, "
                f"最大回撤={result.max_drawdown_percent:.2f}%, "
                f"状态={result.status}"
            )
        
        def get_optimize_value(r: BacktestResult) -> float:
            if optimize_by == 'total_return_percent':
                return r.total_return_percent
            elif optimize_by == 'profit_factor':
                return r.profit_factor
            elif optimize_by == 'win_rate':
                return r.win_rate
            elif optimize_by == 'sharpe_ratio':
                return 0.0
            elif optimize_by == 'max_drawdown_percent':
                return -r.max_drawdown_percent
            return r.total_return_percent
        
        results.sort(key=get_optimize_value, reverse=True)
        
        self._results = results
        if results:
            self._best_result = results[0]
            self.logger.info(f"最优参数组合: {results[0].params}")
            self.logger.info(f"最优结果: 收益率={results[0].total_return_percent:.2f}%, 最大回撤={results[0].max_drawdown_percent:.2f}%")
        
        return results

    def generate_report(self, output_dir: str = None) -> Dict[str, Any]:
        if not self._results:
            self.logger.warning("没有回测结果，无法生成报告")
            return {}
        
        if output_dir is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            output_dir = os.path.join(base_dir, 'logs', 'backtest_reports')
        
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        report = {
            'generated_at': datetime.now().isoformat(),
            'total_backtests': len(self._results),
            'best_result': asdict(self._best_result) if self._best_result else None,
            'all_results': [asdict(r) for r in self._results],
        }
        
        self._print_summary_report()
        
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

    def _print_summary_report(self) -> None:
        if not self._results:
            return
        
        print("\n" + "=" * 80)
        print("                    回测绩效报告")
        print("=" * 80)
        
        if self._best_result:
            r = self._best_result
            print("\n【最优策略参数】")
            print(f"  策略名称: {r.strategy_name}")
            print(f"  策略参数: {r.params}")
            print(f"  回测区间: {r.start_dt} 至 {r.end_dt}")
            
            print("\n【收益指标】")
            print(f"  初始权益: {r.initial_equity:,.2f}")
            print(f"  最终权益: {r.final_equity:,.2f}")
            print(f"  总收益: {r.total_return:,.2f}")
            print(f"  总收益率: {r.total_return_percent:.2f}%")
            
            print("\n【风险指标】")
            print(f"  最大回撤: {r.max_drawdown:,.2f}")
            print(f"  最大回撤率: {r.max_drawdown_percent:.2f}%")
            
            print("\n【交易统计】")
            print(f"  总交易次数: {r.total_trades}")
            print(f"  胜率: {r.win_rate:.2f}%")
            print(f"  盈亏比: {r.profit_factor:.2f}")
            
            print("\n【风控状态】")
            print(f"  风控触发: {'是' if r.risk_triggered else '否'}")
            print(f"  期间冻结: {'是' if r.frozen_during_backtest else '否'}")
            if r.frozen_reason:
                print(f"  冻结原因: {r.frozen_reason}")
            
            print(f"  运行状态: {r.status}")
            if r.error_message:
                print(f"  错误信息: {r.error_message}")
        
        if len(self._results) > 1:
            print("\n" + "-" * 80)
            print("                    所有参数组合结果排名")
            print("-" * 80)
            print(f"{'排名':<4} {'参数组合':<30} {'收益率(%)':<12} {'最大回撤(%)':<12} {'交易次数':<10} {'状态':<10}")
            print("-" * 80)
            
            for i, r in enumerate(self._results[:20], 1):
                params_str = str(r.params)[:28]
                print(
                    f"{i:<4} "
                    f"{params_str:<30} "
                    f"{r.total_return_percent:<12.2f} "
                    f"{r.max_drawdown_percent:<12.2f} "
                    f"{r.total_trades:<10} "
                    f"{r.status:<10}"
                )
            
            if len(self._results) > 20:
                print(f"... 等共 {len(self._results)} 个组合")
        
        print("\n" + "=" * 80)

    def _write_csv_report(self, csv_file: str) -> None:
        if not self._results:
            return
        
        fieldnames = [
            'rank', 'strategy_name', 'params', 'start_dt', 'end_dt',
            'initial_equity', 'final_equity', 'total_return', 'total_return_percent',
            'max_drawdown', 'max_drawdown_percent',
            'total_trades', 'winning_trades', 'losing_trades',
            'win_rate', 'total_profit', 'total_loss', 'profit_factor', 'avg_trade',
            'risk_triggered', 'frozen_during_backtest', 'frozen_reason',
            'status', 'error_message'
        ]
        
        with open(csv_file, 'w', encoding='utf-8-sig', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for i, r in enumerate(self._results, 1):
                row = {
                    'rank': i,
                    'strategy_name': r.strategy_name,
                    'params': json.dumps(r.params, ensure_ascii=False),
                    'start_dt': r.start_dt,
                    'end_dt': r.end_dt,
                    'initial_equity': r.initial_equity,
                    'final_equity': r.final_equity,
                    'total_return': r.total_return,
                    'total_return_percent': r.total_return_percent,
                    'max_drawdown': r.max_drawdown,
                    'max_drawdown_percent': r.max_drawdown_percent,
                    'total_trades': r.total_trades,
                    'winning_trades': r.winning_trades,
                    'losing_trades': r.losing_trades,
                    'win_rate': r.win_rate,
                    'total_profit': r.total_profit,
                    'total_loss': r.total_loss,
                    'profit_factor': r.profit_factor,
                    'avg_trade': r.avg_trade,
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
