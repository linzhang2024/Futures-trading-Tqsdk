import os
import sys
import logging
import copy
from datetime import date, datetime
from typing import Dict, Any, List, Optional, Type
from dataclasses import dataclass, field, asdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Manager

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, base_dir)

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    yaml = None

from strategies.base_strategy import StrategyBase
from strategies.double_ma_strategy import DoubleMAStrategy
from core.backtest import BacktestEngine, BacktestResult, CostConfig, PerformanceConfig
from core.equity_plotter import (
    EquityPlotter, 
    ContractResult, 
    create_contract_result_from_backtest
)


@dataclass
class ContractBacktestConfig:
    contract: str
    strategy_class: Type[StrategyBase]
    strategy_params: Dict[str, Any]
    start_dt: date
    end_dt: date
    initial_balance: float = 1000000.0


@dataclass
class MultiContractResult:
    generated_at: datetime
    contract_results: Dict[str, ContractResult] = field(default_factory=dict)
    backtest_results: Dict[str, BacktestResult] = field(default_factory=dict)
    summary_report: Dict[str, Any] = field(default_factory=dict)
    comparison_chart_path: Optional[str] = None
    summary_report_path: Optional[str] = None


class MultiContractRunner:
    
    def __init__(
        self,
        config: Dict[str, Any] = None,
        config_path: str = None,
        max_workers: int = None,
    ):
        self.logger = logging.getLogger(self.__class__.__name__)
        
        if config is None and config_path is None:
            config_path = os.path.join(base_dir, 'config', 'settings.yaml')
        
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            self.logger.info(f"已加载配置文件: {config_path}")
        
        self.config = config or {}
        self._contract_configs: List[ContractBacktestConfig] = []
        self._results: Dict[str, ContractResult] = {}
        self._backtest_results: Dict[str, BacktestResult] = {}
        
        if max_workers is None:
            import multiprocessing
            max_workers = min(4, multiprocessing.cpu_count() or 4)
        self._max_workers = max_workers
        
        self._plotter = EquityPlotter(
            output_dir=os.path.join(base_dir, 'results')
        )
        
        self._load_contracts_from_config()
    
    def _load_contracts_from_config(self):
        trading_config = self.config.get('trading', {})
        contracts = trading_config.get('contracts', [])
        
        if not contracts:
            default_contract = trading_config.get('default_contract', 'SHFE.rb2410')
            contracts = [default_contract]
            self.logger.warning(f"配置中未找到合约列表，使用默认合约: {default_contract}")
        
        backtest_config = self.config.get('backtest', {})
        start_dt_str = backtest_config.get('start_dt', '2024-01-01')
        end_dt_str = backtest_config.get('end_dt', '2024-03-31')
        init_balance = backtest_config.get('init_balance', 1000000.0)
        
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
            return date(2024, 1, 1)
        
        start_dt = parse_date(start_dt_str)
        end_dt = parse_date(end_dt_str)
        
        strategies_config = self.config.get('strategies', [])
        
        for contract in contracts:
            strategy_params = {}
            strategy_class = DoubleMAStrategy
            
            for strat_cfg in strategies_config:
                params = strat_cfg.get('params', {})
                if params.get('contract') == contract:
                    strategy_params = self._translate_params(params)
                    if strategy_params:
                        break
            
            if not strategy_params:
                strategy_params = {
                    'short_period': 5,
                    'long_period': 20,
                    'kline_duration': 60,
                    'use_ema': False,
                    'rsi_period': 14,
                    'rsi_threshold': 50.0,
                    'use_rsi_filter': False,
                    'initial_data_days': 5,
                }
            
            contract_config = ContractBacktestConfig(
                contract=contract,
                strategy_class=strategy_class,
                strategy_params=strategy_params,
                start_dt=start_dt,
                end_dt=end_dt,
                initial_balance=init_balance,
            )
            self._contract_configs.append(contract_config)
            
            self.logger.info(
                f"已配置合约: {contract}, "
                f"策略参数: {strategy_params}, "
                f"回测区间: {start_dt} 至 {end_dt}"
            )
    
    def _translate_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        param_mapping = {
            'fast': 'short_period',
            'slow': 'long_period',
            'period': 'kline_duration',
        }
        
        translated = {}
        for key, value in params.items():
            mapped_key = param_mapping.get(key, key)
            translated[mapped_key] = value
        
        if 'short_period' not in translated and 'fast' in params:
            translated['short_period'] = params['fast']
        if 'long_period' not in translated and 'slow' in params:
            translated['long_period'] = params['slow']
        if 'initial_data_days' not in translated:
            translated['initial_data_days'] = 5
        
        if 'short_period' not in translated:
            return {}
        
        return translated
    
    def add_contract(self, contract_config: ContractBacktestConfig):
        self._contract_configs.append(contract_config)
        self.logger.info(f"已添加合约配置: {contract_config.contract}")
    
    def get_contracts(self) -> List[str]:
        return [cfg.contract for cfg in self._contract_configs]
    
    def _run_single_contract_backtest(
        self,
        contract: str,
        strategy_class: Type[StrategyBase],
        strategy_params: Dict[str, Any],
        start_dt: date,
        end_dt: date,
        initial_balance: float,
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        from core.backtest import BacktestEngine
        
        engine_config = copy.deepcopy(config)
        backtest_cfg = engine_config.get('backtest', {})
        backtest_cfg['init_balance'] = initial_balance
        engine_config['backtest'] = backtest_cfg
        
        engine = BacktestEngine(config=engine_config)
        
        self._setup_cost_config(engine, contract)
        
        result = engine.run_backtest(
            strategy_class=strategy_class,
            strategy_params=strategy_params,
            start_dt=start_dt,
            end_dt=end_dt,
        )
        
        contract_result = self._convert_to_contract_result(
            contract=contract,
            backtest_result=result,
            initial_balance=initial_balance,
        )
        
        return {
            'contract': contract,
            'backtest_result': result,
            'contract_result': contract_result,
            'status': 'completed' if result.status == 'completed' else 'error',
            'error_message': result.error_message,
        }
    
    def _setup_cost_config(self, engine: BacktestEngine, contract: str):
        cost_config = engine._cost_config
        if cost_config is None:
            cost_config = CostConfig(
                default_commission_per_lot=5.0,
                default_slippage_points=1.0,
            )
        
        if contract not in cost_config.contract_configs:
            backtest_cfg = self.config.get('backtest', {})
            costs_cfg = backtest_cfg.get('costs', {})
            contract_costs = costs_cfg.get('contracts', {}).get(contract, {})
            
            cost_config.contract_configs[contract] = {
                'commission_per_lot': contract_costs.get('commission_per_lot', 5.0),
                'slippage_points': contract_costs.get('slippage_points', 1.0),
            }
        
        engine._cost_config = cost_config
    
    def _convert_to_contract_result(
        self,
        contract: str,
        backtest_result: BacktestResult,
        initial_balance: float,
    ) -> ContractResult:
        return create_contract_result_from_backtest(
            contract=contract,
            initial_equity=initial_balance,
            final_equity=backtest_result.final_equity,
            equity_curve_data=backtest_result.equity_curve,
            trade_records_data=backtest_result.trade_records,
        )
    
    def run_all(self, parallel: bool = True) -> MultiContractResult:
        if not self._contract_configs:
            self.logger.error("没有配置任何合约")
            return MultiContractResult(generated_at=datetime.now())
        
        print("\n" + "=" * 80)
        print("                    多合约回测任务")
        print("=" * 80)
        print(f"启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"合约数量: {len(self._contract_configs)}")
        print(f"运行模式: {'并行' if parallel else '串行'}")
        print(f"最大并行数: {self._max_workers}")
        print("-" * 80)
        
        for cfg in self._contract_configs:
            print(f"  - {cfg.contract}: {cfg.strategy_class.__name__}")
            print(f"    回测区间: {cfg.start_dt} 至 {cfg.end_dt}")
            print(f"    初始资金: {cfg.initial_balance:,.0f}")
        print("=" * 80 + "\n")
        
        self._results = {}
        self._backtest_results = {}
        
        if parallel and len(self._contract_configs) > 1:
            self._run_parallel()
        else:
            self._run_serial()
        
        return self._generate_final_report()
    
    def _run_serial(self):
        for idx, cfg in enumerate(self._contract_configs, 1):
            print(f"\n[{idx}/{len(self._contract_configs)}] 开始回测合约: {cfg.contract}")
            
            try:
                result = self._run_single_contract_backtest(
                    contract=cfg.contract,
                    strategy_class=cfg.strategy_class,
                    strategy_params=cfg.strategy_params,
                    start_dt=cfg.start_dt,
                    end_dt=cfg.end_dt,
                    initial_balance=cfg.initial_balance,
                    config=self.config,
                )
                
                self._results[cfg.contract] = result['contract_result']
                self._backtest_results[cfg.contract] = result['backtest_result']
                
                perf = result['backtest_result'].performance
                print(f"    合约 {cfg.contract} 回测完成:")
                print(f"      收益率: {perf.total_return_percent:.2f}%")
                print(f"      最大回撤: {perf.max_drawdown_percent:.2f}%")
                print(f"      交易次数: {perf.total_trades}")
                
            except Exception as e:
                self.logger.error(f"合约 {cfg.contract} 回测失败: {e}", exc_info=True)
                print(f"    合约 {cfg.contract} 回测失败: {e}")
    
    def _run_parallel(self):
        with ProcessPoolExecutor(max_workers=self._max_workers) as executor:
            future_to_contract = {}
            
            for idx, cfg in enumerate(self._contract_configs, 1):
                future = executor.submit(
                    self._run_single_contract_backtest,
                    contract=cfg.contract,
                    strategy_class=cfg.strategy_class,
                    strategy_params=cfg.strategy_params,
                    start_dt=cfg.start_dt,
                    end_dt=cfg.end_dt,
                    initial_balance=cfg.initial_balance,
                    config=self.config,
                )
                future_to_contract[future] = (idx, cfg.contract)
            
            completed = 0
            for future in as_completed(future_to_contract):
                idx, contract = future_to_contract[future]
                completed += 1
                
                try:
                    result = future.result()
                    
                    self._results[contract] = result['contract_result']
                    self._backtest_results[contract] = result['backtest_result']
                    
                    perf = result['backtest_result'].performance
                    print(f"\n[{completed}/{len(self._contract_configs)}] 合约 {contract} 回测完成:")
                    print(f"    收益率: {perf.total_return_percent:.2f}%")
                    print(f"    最大回撤: {perf.max_drawdown_percent:.2f}%")
                    print(f"    交易次数: {perf.total_trades}")
                    
                except Exception as e:
                    self.logger.error(f"合约 {contract} 回测失败: {e}", exc_info=True)
                    print(f"\n[{completed}/{len(self._contract_configs)}] 合约 {contract} 回测失败: {e}")
    
    def _generate_final_report(self) -> MultiContractResult:
        print("\n" + "=" * 80)
        print("                    生成回测报告...")
        print("=" * 80)
        
        contract_results = list(self._results.values())
        
        comparison_chart_path = None
        if contract_results:
            comparison_chart_path = self._plotter.plot_multi_contract_comparison(
                results=contract_results,
                title=_get_plot_label('多合约回测结果对比', 'Multi-Contract Backtest Comparison'),
            )
        
        summary_report = self._plotter.generate_summary_report(
            results=contract_results,
        )
        
        trade_csv_path = self._plotter.generate_trade_details_csv(
            results=contract_results,
        )
        
        for contract, result in self._results.items():
            if result.equity_curve:
                self._plotter.plot_single_equity_curve(
                    result=result,
                    title=_get_plot_label(
                        f'{contract} 权益曲线',
                        f'{contract} Equity Curve'
                    ),
                )
        
        total_initial = sum(r.initial_equity for r in self._results.values())
        total_final = sum(r.final_equity for r in self._results.values())
        total_return = total_final - total_initial
        total_return_pct = (total_return / total_initial * 100) if total_initial > 0 else 0.0
        
        max_drawdown = max(r.max_drawdown_percent for r in self._results.values()) if self._results else 0.0
        total_trades = sum(r.total_trades for r in self._results.values())
        
        print(f"\n汇总统计:")
        print(f"  合约数量: {len(self._results)}")
        print(f"  总初始资金: {total_initial:,.0f}")
        print(f"  总最终资金: {total_final:,.0f}")
        print(f"  总收益率: {total_return_pct:.2f}%")
        print(f"  最大回撤(跨合约): {max_drawdown:.2f}%")
        print(f"  总交易次数: {total_trades}")
        
        if comparison_chart_path:
            print(f"\n对比图表: {comparison_chart_path}")
        if trade_csv_path:
            print(f"交易明细: {trade_csv_path}")
        
        print("\n" + "=" * 80)
        print("                    多合约回测完成")
        print("=" * 80)
        
        return MultiContractResult(
            generated_at=datetime.now(),
            contract_results=self._results.copy(),
            backtest_results=self._backtest_results.copy(),
            summary_report=summary_report,
            comparison_chart_path=comparison_chart_path,
        )


def _get_plot_label(zh_label: str, en_label: str) -> str:
    try:
        from core.equity_plotter import _check_chinese_font_support
        if _check_chinese_font_support():
            return zh_label
    except:
        pass
    return en_label


def run_all_backtests(
    config_path: str = None,
    contracts: List[str] = None,
    start_dt: str = None,
    end_dt: str = None,
    initial_balance: float = None,
    parallel: bool = True,
    max_workers: int = None,
) -> MultiContractResult:
    runner = MultiContractRunner(
        config_path=config_path,
        max_workers=max_workers,
    )
    
    if contracts:
        runner._contract_configs = []
        
        for contract in contracts:
            contract_config = ContractBacktestConfig(
                contract=contract,
                strategy_class=DoubleMAStrategy,
                strategy_params={
                    'short_period': 5,
                    'long_period': 20,
                    'kline_duration': 60,
                    'use_ema': False,
                    'rsi_period': 14,
                    'rsi_threshold': 50.0,
                    'use_rsi_filter': False,
                    'initial_data_days': 5,
                },
                start_dt=datetime.strptime(start_dt, '%Y-%m-%d').date() if start_dt else date(2024, 1, 1),
                end_dt=datetime.strptime(end_dt, '%Y-%m-%d').date() if end_dt else date(2024, 3, 31),
                initial_balance=initial_balance or 1000000.0,
            )
            runner.add_contract(contract_config)
    
    return runner.run_all(parallel=parallel)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='多合约回测工具')
    parser.add_argument('--config', '-c', type=str, help='配置文件路径')
    parser.add_argument('--contracts', '-s', type=str, nargs='+', help='合约列表')
    parser.add_argument('--start', type=str, help='开始日期 (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, help='结束日期 (YYYY-MM-DD)')
    parser.add_argument('--balance', '-b', type=float, help='初始资金')
    parser.add_argument('--serial', action='store_true', help='串行运行')
    parser.add_argument('--workers', '-w', type=int, help='最大并行数')
    
    args = parser.parse_args()
    
    result = run_all_backtests(
        config_path=args.config,
        contracts=args.contracts,
        start_dt=args.start,
        end_dt=args.end,
        initial_balance=args.balance,
        parallel=not args.serial,
        max_workers=args.workers,
    )
