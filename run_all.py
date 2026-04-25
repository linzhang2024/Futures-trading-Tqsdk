#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
多合约回测入口脚本
支持同时运行多个品种的回测并输出对比图表

配置说明：
- 修改 CONTRACTS 列表来指定要回测的品种
- 修改 BACKTEST_CONFIG 来调整回测参数

使用方法：
  python run_all.py                          # 使用默认配置
  python run_all.py --contracts rb2410 hc2410  # 指定合约
  python run_all.py --start 2024-01-01 --end 2024-03-31  # 指定日期
"""

import os
import sys
import logging
from datetime import datetime, date

base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, base_dir)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    yaml = None

from core.multi_contract_runner import (
    MultiContractRunner,
    ContractBacktestConfig,
    run_all_backtests,
)
from strategies.double_ma_strategy import DoubleMAStrategy


CONTRACTS = [
    'SHFE.rb2410',
    'SHFE.hc2410',
    'DCE.i2409',
]

BACKTEST_CONFIG = {
    'start_dt': '2024-01-01',
    'end_dt': '2024-03-31',
    'initial_balance': 1000000.0,
    'strategy_params': {
        'short_period': 5,
        'long_period': 13,
        'kline_duration': 60,
        'use_ema': False,
        'rsi_period': 7,
        'rsi_threshold': 50.0,
        'use_rsi_filter': False,
        'initial_data_days': 5,
        'take_profit_ratio': 0.01,
        'stop_loss_ratio': 0.01,
    },
}


def load_config():
    config_path = os.path.join(base_dir, 'config', 'settings.yaml')
    if os.path.exists(config_path) and YAML_AVAILABLE:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        trading_config = config.get('trading', {})
        contracts = trading_config.get('contracts', [])
        if contracts:
            global CONTRACTS
            CONTRACTS = contracts
        
        backtest_config = config.get('backtest', {})
        if backtest_config:
            if backtest_config.get('start_dt'):
                BACKTEST_CONFIG['start_dt'] = backtest_config['start_dt']
            if backtest_config.get('end_dt'):
                BACKTEST_CONFIG['end_dt'] = backtest_config['end_dt']
            if backtest_config.get('init_balance'):
                BACKTEST_CONFIG['initial_balance'] = backtest_config['init_balance']
        
        strategies = config.get('strategies', [])
        if strategies:
            first_strategy = strategies[0]
            params = first_strategy.get('params', {})
            
            param_mapping = {
                'fast': 'short_period',
                'slow': 'long_period',
                'period': 'kline_duration',
            }
            
            for key, value in params.items():
                mapped_key = param_mapping.get(key, key)
                if mapped_key in BACKTEST_CONFIG['strategy_params']:
                    BACKTEST_CONFIG['strategy_params'][mapped_key] = value
        
        return config
    
    return None


def print_header():
    print("\n" + "=" * 80)
    print("                    多合约回测系统启动")
    print("=" * 80)
    print(f"启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"工作目录: {base_dir}")
    print("-" * 80)


def print_config():
    print(f"\n【回测配置】")
    print(f"  回测区间: {BACKTEST_CONFIG['start_dt']} 至 {BACKTEST_CONFIG['end_dt']}")
    print(f"  初始资金: {BACKTEST_CONFIG['initial_balance']:,.0f}")
    print(f"\n  策略参数:")
    for key, value in BACKTEST_CONFIG['strategy_params'].items():
        print(f"    {key}: {value}")
    print(f"\n  回测合约 ({len(CONTRACTS)} 个):")
    for contract in CONTRACTS:
        print(f"    - {contract}")
    print("-" * 80)


def print_guide():
    print("\n【使用说明】")
    print("  1. 修改 CONTRACTS 列表来指定要回测的品种")
    print("  2. 修改 BACKTEST_CONFIG 来调整回测参数")
    print("  3. 或在 config/settings.yaml 中配置 trading.contracts")
    print("  4. 结果将保存到 results/ 目录")
    print("\n【命令行参数】")
    print("  python run_all.py --contracts rb2410 hc2410 i2409")
    print("  python run_all.py --start 2024-01-01 --end 2024-03-31")
    print("  python run_all.py --balance 2000000")
    print("=" * 80 + "\n")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='多合约回测工具')
    parser.add_argument('--contracts', '-s', type=str, nargs='+', help='合约列表')
    parser.add_argument('--start', type=str, help='开始日期 (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, help='结束日期 (YYYY-MM-DD)')
    parser.add_argument('--balance', '-b', type=float, help='初始资金')
    parser.add_argument('--serial', action='store_true', help='串行运行')
    parser.add_argument('--workers', '-w', type=int, default=3, help='最大并行数')
    parser.add_argument('--config', '-c', type=str, help='配置文件路径')
    
    args = parser.parse_args()
    
    print_header()
    
    config = load_config()
    
    if args.contracts:
        global CONTRACTS
        CONTRACTS = args.contracts
    
    if args.start:
        BACKTEST_CONFIG['start_dt'] = args.start
    if args.end:
        BACKTEST_CONFIG['end_dt'] = args.end
    if args.balance:
        BACKTEST_CONFIG['initial_balance'] = args.balance
    
    print_config()
    print_guide()
    
    runner = MultiContractRunner(config=config)
    
    runner._contract_configs = []
    
    def parse_date(date_str: str) -> date:
        for fmt in ['%Y-%m-%d', '%Y%m%d', '%Y/%m/%d']:
            try:
                return datetime.strptime(str(date_str), fmt).date()
            except ValueError:
                continue
        return date(2024, 1, 1)
    
    start_dt = parse_date(BACKTEST_CONFIG['start_dt'])
    end_dt = parse_date(BACKTEST_CONFIG['end_dt'])
    
    for contract in CONTRACTS:
        strategy_params = BACKTEST_CONFIG['strategy_params'].copy()
        strategy_params['contract'] = contract
        
        contract_config = ContractBacktestConfig(
            contract=contract,
            strategy_class=DoubleMAStrategy,
            strategy_params=strategy_params,
            start_dt=start_dt,
            end_dt=end_dt,
            initial_balance=BACKTEST_CONFIG['initial_balance'],
        )
        runner.add_contract(contract_config)
    
    result = runner.run_all(parallel=not args.serial)
    
    print("\n" + "=" * 80)
    print("                    回测完成")
    print("=" * 80)
    
    if result.comparison_chart_path:
        print(f"\n对比图表已生成: {result.comparison_chart_path}")
    
    if result.summary_report:
        print(f"\n回测汇总:")
        summary = result.summary_report.get('summary', {})
        print(f"  合约数量: {summary.get('total_contracts', 0)}")
        print(f"  组合收益率: {summary.get('combined_return_percent', 0):.2f}%")
        print(f"  总交易次数: {summary.get('total_trades', 0)}")
        print(f"  最大回撤: {summary.get('max_drawdown_across_contracts', 0):.2f}%")
    
    print("\n合约详情:")
    for contract, contract_result in result.contract_results.items():
        print(f"\n  {contract}:")
        print(f"    收益率: {contract_result.total_return_percent:.2f}%")
        print(f"    最大回撤: {contract_result.max_drawdown_percent:.2f}%")
        print(f"    交易次数: {contract_result.total_trades}")
    
    print("\n" + "=" * 80)
    print("回测完成，请查看 results/ 目录获取详细图表和报告")
    print("=" * 80 + "\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
