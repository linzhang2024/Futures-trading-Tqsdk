#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
单合约调试脚本 - 降维调试
只针对 SHFE.rb2410 一个合约，使用最稳的参数
输出详细日志到 logs/debug_signal.log
"""

import os
import sys
import logging
from datetime import date, datetime

base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, base_dir)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)

DEBUG_MODE = True
FORCE_TRADE_TEST = True

if DEBUG_MODE:
    logs_dir = os.path.join(base_dir, 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    
    debug_logger = logging.getLogger('debug_signal')
    debug_logger.setLevel(logging.DEBUG)
    debug_logger.propagate = False
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    debug_file = os.path.join(logs_dir, f'debug_signal_{timestamp}.log')
    
    file_handler = logging.FileHandler(debug_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
    file_handler.setFormatter(file_formatter)
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(asctime)s [DEBUG_SIGNAL] %(message)s')
    console_handler.setFormatter(console_formatter)
    
    debug_logger.addHandler(file_handler)
    debug_logger.addHandler(console_handler)
    
    debug_logger.info("=" * 80)
    debug_logger.info("单合约调试模式启动")
    debug_logger.info(f"调试日志文件: {debug_file}")
    debug_logger.info(f"FORCE_TRADE_TEST: {FORCE_TRADE_TEST}")
    debug_logger.info("=" * 80)


def run_single_contract_debug():
    print("\n" + "=" * 80)
    print("                    单合约调试模式")
    print("=" * 80)
    print(f"合约: SHFE.rb2410")
    print(f"策略参数: MA5/MA20, 60秒K线")
    print(f"force_trade_test: {FORCE_TRADE_TEST}")
    print(f"debug_logging: {DEBUG_MODE}")
    print("=" * 80 + "\n")
    
    from core.backtest import BacktestEngine, BacktestResult
    from strategies.double_ma_strategy import DoubleMAStrategy
    
    config_path = os.path.join(base_dir, 'config', 'settings.yaml')
    config = None
    
    if os.path.exists(config_path):
        try:
            import yaml
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            print(f"已从配置文件加载: {config_path}")
            
            if 'backtest' in config:
                config['backtest']['use_mock_data'] = True
                print(f"已启用模拟数据模式 (use_mock_data: True)")
        except Exception as e:
            print(f"加载配置文件失败: {e}")
    
    if config is None:
        config = {
            'backtest': {
                'start_dt': '2024-01-01',
                'end_dt': '2024-01-31',
                'init_balance': 1000000.0,
                'use_mock_data': True,
                'costs': {
                    'default_commission_per_lot': 5.0,
                    'default_slippage_points': 1.0,
                }
            }
        }
        print(f"使用默认配置")
    
    strategy_params = {
        'short_period': 5,
        'long_period': 20,
        'contract': 'SHFE.rb2410',
        'kline_duration': 60,
        'use_ema': False,
        'rsi_period': 14,
        'rsi_threshold': 50.0,
        'use_rsi_filter': False,
        'initial_data_days': 5,
        'force_trade_test': FORCE_TRADE_TEST,
        'debug_logging': DEBUG_MODE,
    }
    
    print(f"策略参数: {strategy_params}")
    print("")
    
    engine = BacktestEngine(config=config)
    
    start_dt = date(2024, 1, 1)
    end_dt = date(2024, 1, 31)
    
    init_balance = config.get('backtest', {}).get('init_balance', 1000000.0)
    print(f"开始回测: {start_dt} 至 {end_dt}")
    print(f"初始资金: {init_balance:,.0f}")
    print("-" * 80 + "\n")
    
    result = engine.run_backtest(
        strategy_class=DoubleMAStrategy,
        strategy_params=strategy_params,
        start_dt=start_dt,
        end_dt=end_dt,
    )
    
    print("\n" + "=" * 80)
    print("                    回测结果")
    print("=" * 80)
    
    print(f"\n状态: {result.status}")
    if result.error_message:
        print(f"错误: {result.error_message}")
    
    perf = result.performance
    print(f"\n性能统计:")
    print(f"  初始资金: {result.initial_equity:,.0f}")
    print(f"  最终资金: {result.final_equity:,.0f}")
    print(f"  总收益: {perf.total_return:,.0f}")
    print(f"  总收益率: {perf.total_return_percent:.2f}%")
    print(f"  最大回撤: {perf.max_drawdown_percent:.2f}%")
    print(f"  总交易次数: {perf.total_trades}")
    print(f"  盈利交易: {perf.winning_trades}")
    print(f"  亏损交易: {perf.losing_trades}")
    
    print(f"\n权益曲线点数: {len(result.equity_curve)}")
    print(f"交易记录数: {len(result.trade_records)}")
    
    if result.trade_records:
        print(f"\n交易记录示例 (前5笔):")
        for i, trade in enumerate(result.trade_records[:5]):
            print(f"  [{i+1}] {trade}")
    
    results_dir = os.path.join(base_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    from core.equity_plotter import EquityPlotter, create_contract_result_from_backtest
    
    plotter = EquityPlotter(output_dir=results_dir)
    
    contract_result = create_contract_result_from_backtest(
        contract='SHFE.rb2410',
        initial_equity=result.initial_equity,
        final_equity=result.final_equity,
        equity_curve_data=result.equity_curve,
        trade_records_data=result.trade_records,
    )
    
    equity_path = plotter.plot_single_equity_curve(
        result=contract_result,
        title='SHFE.rb2410 Debug Equity Curve',
    )
    
    summary = plotter.generate_summary_report(results=[contract_result])
    
    csv_path = plotter.generate_trade_details_csv(results=[contract_result])
    
    print(f"\n输出文件:")
    if equity_path:
        print(f"  权益曲线图: {equity_path}")
    if csv_path:
        print(f"  交易明细CSV: {csv_path}")
    
    print("\n" + "=" * 80)
    
    if result.trade_records and perf.total_trades > 0:
        print("                    调试成功 - 有交易记录！")
        print("=" * 80)
        return True
    else:
        print("                    调试失败 - 无交易记录")
        print("=" * 80)
        print("\n请检查 logs/debug_signal_*.log 查看详细日志")
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='单合约调试脚本')
    parser.add_argument('--no-force', action='store_true', help='不使用强制交易模式')
    parser.add_argument('--no-debug', action='store_true', help='不输出详细调试日志')
    
    args = parser.parse_args()
    
    if args.no_force:
        FORCE_TRADE_TEST = False
    if args.no_debug:
        DEBUG_MODE = False
    
    success = run_single_contract_debug()
    sys.exit(0 if success else 1)
