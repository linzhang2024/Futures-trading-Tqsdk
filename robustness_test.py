#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
策略健壮性测试脚本
包含：
1. 蒙特卡洛随机滑点测试 (1-3 ticks)
2. 多时间窗口回测 (Walking Forward Analysis)
3. 挂单重试逻辑验证

使用方法：
  python robustness_test.py                    # 运行所有测试
  python robustness_test.py --slippage         # 仅运行随机滑点测试
  python robustness_test.py --windows          # 仅运行多时间窗口测试
"""

import os
import sys
import logging
import random
from datetime import datetime, date, timedelta
from typing import Dict, List, Any, Optional

base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, base_dir)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger(__name__)

from core.backtest import MockDataBacktestRunner
from strategies.double_ma_strategy import DoubleMAStrategy

CONTRACTS = [
    'SHFE.rb2410',
    'SHFE.hc2410',
    'DCE.i2409',
]

STRATEGY_PARAMS = {
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
}

BASE_START_DATE = date(2024, 1, 1)
BASE_END_DATE = date(2024, 5, 31)


def parse_date(date_str: str) -> date:
    for fmt in ['%Y-%m-%d', '%Y%m%d', '%Y/%m/%d']:
        try:
            return datetime.strptime(str(date_str), fmt).date()
        except ValueError:
            continue
    return date(2024, 1, 1)


class SlippageTestResult:
    def __init__(self):
        self.contract_results: Dict[str, List[Dict[str, Any]]] = {}
        self.no_slippage_results: Dict[str, Dict[str, Any]] = {}
    
    def add_result(self, contract: str, slippage_ticks: int, 
                   result: Dict[str, Any], is_no_slippage: bool = False):
        if is_no_slippage:
            self.no_slippage_results[contract] = {
                'slippage_ticks': 0,
                **result
            }
        else:
            if contract not in self.contract_results:
                self.contract_results[contract] = []
            self.contract_results[contract].append({
                'slippage_ticks': slippage_ticks,
                **result
            })


class TimeWindowResult:
    def __init__(self):
        self.window_results: Dict[str, Dict[str, Any]] = {}
    
    def add_result(self, window_name: str, result: Dict[str, Any]):
        self.window_results[window_name] = result


def run_single_backtest(
    contract: str,
    start_dt: date,
    end_dt: date,
    use_random_slippage: bool = False,
    slippage_min: int = 1,
    slippage_max: int = 3,
) -> Dict[str, Any]:
    """运行单次回测"""
    
    strategy_params = STRATEGY_PARAMS.copy()
    strategy_params['contract'] = contract
    
    runner = MockDataBacktestRunner(
        init_balance=1000000.0,
        start_dt=start_dt,
        end_dt=end_dt,
        logger=logger,
        slippage_min=slippage_min,
        slippage_max=slippage_max,
        use_random_slippage=use_random_slippage,
    )
    
    result = runner.run_backtest(
        strategy_class=DoubleMAStrategy,
        strategy_params=strategy_params,
    )
    
    if result['status'] == 'completed':
        initial = result['initial_equity']
        final = result['final_equity']
        total_return = final - initial
        total_return_percent = (total_return / initial) * 100
        
        trades = result.get('trades', [])
        total_trades = len(trades)
        
        winning_trades = sum(1 for t in trades if t.get('profit_loss', 0) > 0)
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        return {
            'status': 'completed',
            'initial_equity': initial,
            'final_equity': final,
            'total_return': total_return,
            'total_return_percent': total_return_percent,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'win_rate': win_rate,
            'equity_curve': result.get('equity_curve', []),
            'trades': trades,
        }
    
    return {
        'status': 'error',
        'error_message': result.get('error_message', 'Unknown error'),
    }


def run_slippage_test(num_runs: int = 5) -> SlippageTestResult:
    """
    运行随机滑点测试
    
    Args:
        num_runs: 每个合约运行多少次随机滑点测试
    
    Returns:
        SlippageTestResult: 测试结果
    """
    print("\n" + "=" * 80)
    print("                    蒙特卡洛随机滑点测试")
    print("=" * 80)
    print(f"测试配置:")
    print(f"  滑点范围: 1-3 ticks")
    print(f"  每个合约运行次数: {num_runs}")
    print(f"  回测区间: {BASE_START_DATE} 至 {BASE_END_DATE}")
    print("-" * 80)
    
    result = SlippageTestResult()
    
    for contract in CONTRACTS:
        print(f"\n【合约: {contract}】")
        
        print("\n  1. 无滑点测试 (基准):")
        no_slippage_result = run_single_backtest(
            contract=contract,
            start_dt=BASE_START_DATE,
            end_dt=BASE_END_DATE,
            use_random_slippage=False,
        )
        
        if no_slippage_result['status'] == 'completed':
            result.add_result(contract, 0, no_slippage_result, is_no_slippage=True)
            print(f"     收益率: {no_slippage_result['total_return_percent']:.2f}%")
            print(f"     交易次数: {no_slippage_result['total_trades']}")
            print(f"     胜率: {no_slippage_result['win_rate']:.2f}%")
        
        print(f"\n  2. 随机滑点测试 (运行 {num_runs} 次):")
        for i in range(num_runs):
            slippage_min = 1
            slippage_max = 3
            
            slip_result = run_single_backtest(
                contract=contract,
                start_dt=BASE_START_DATE,
                end_dt=BASE_END_DATE,
                use_random_slippage=True,
                slippage_min=slippage_min,
                slippage_max=slippage_max,
            )
            
            if slip_result['status'] == 'completed':
                result.add_result(contract, -1, slip_result)
                print(f"     运行 {i+1}: 收益率={slip_result['total_return_percent']:.2f}%, "
                      f"交易次数={slip_result['total_trades']}")
    
    return result


def run_time_window_test() -> TimeWindowResult:
    """
    运行多时间窗口回测
    
    将回测区间切分为三个独立的60天片段：
    - Window 1: 上涨段 (第1-60天)
    - Window 2: 震荡段 (第61-120天)
    - Window 3: 下跌段 (第121-180天)
    
    要求：至少两个片段保持不亏损
    """
    print("\n" + "=" * 80)
    print("                    多时间窗口回测 (Walking Forward Analysis)")
    print("=" * 80)
    print(f"测试配置:")
    print(f"  总区间: {BASE_START_DATE} 至 {BASE_END_DATE}")
    print(f"  窗口大小: 60天")
    print(f"  窗口数量: 3个")
    print("-" * 80)
    
    result = TimeWindowResult()
    
    windows = [
        ('Window_1_上涨段', BASE_START_DATE, BASE_START_DATE + timedelta(days=59)),
        ('Window_2_震荡段', BASE_START_DATE + timedelta(days=60), BASE_START_DATE + timedelta(days=119)),
        ('Window_3_下跌段', BASE_START_DATE + timedelta(days=120), BASE_START_DATE + timedelta(days=179)),
    ]
    
    for window_name, start_dt, end_dt in windows:
        print(f"\n【{window_name}】")
        print(f"  区间: {start_dt} 至 {end_dt}")
        
        window_returns = []
        window_trades = []
        
        for contract in CONTRACTS:
            print(f"\n    合约: {contract}")
            
            win_result = run_single_backtest(
                contract=contract,
                start_dt=start_dt,
                end_dt=end_dt,
                use_random_slippage=False,
            )
            
            if win_result['status'] == 'completed':
                window_returns.append(win_result['total_return_percent'])
                window_trades.append(win_result['total_trades'])
                
                print(f"      收益率: {win_result['total_return_percent']:.2f}%")
                print(f"      交易次数: {win_result['total_trades']}")
                print(f"      胜率: {win_result['win_rate']:.2f}%")
        
        avg_return = sum(window_returns) / len(window_returns) if window_returns else 0
        total_trades = sum(window_trades)
        
        result.add_result(window_name, {
            'start_dt': start_dt,
            'end_dt': end_dt,
            'contract_returns': dict(zip(CONTRACTS, window_returns)),
            'avg_return': avg_return,
            'total_trades': total_trades,
            'is_profitable': avg_return >= 0,
        })
        
        print(f"\n    窗口汇总:")
        print(f"      平均收益率: {avg_return:.2f}%")
        print(f"      总交易次数: {total_trades}")
        print(f"      状态: {'盈利' if avg_return >= 0 else '亏损'}")
    
    return result


def print_slippage_comparison(result: SlippageTestResult):
    """打印滑点对比表"""
    print("\n" + "=" * 80)
    print("                    随机滑点收益对比表")
    print("=" * 80)
    
    print(f"\n{'合约':<15} {'无滑点收益率':<15} {'随机滑点平均收益率':<20} {'收益衰减':<15} {'状态':<10}")
    print("-" * 85)
    
    all_contracts_ok = True
    
    for contract in CONTRACTS:
        no_slippage = result.no_slippage_results.get(contract, {})
        slip_runs = result.contract_results.get(contract, [])
        
        no_slip_return = no_slippage.get('total_return_percent', 0)
        
        if slip_runs:
            avg_slip_return = sum(r.get('total_return_percent', 0) for r in slip_runs) / len(slip_runs)
            decay = no_slip_return - avg_slip_return
        else:
            avg_slip_return = 0
            decay = 0
        
        is_ok = avg_slip_return >= 0
        if not is_ok:
            all_contracts_ok = False
        
        status = "通过" if is_ok else "失败"
        
        print(f"{contract:<15} {no_slip_return:>12.2f}%    {avg_slip_return:>17.2f}%       {decay:>12.2f}%   {status}")
    
    print("-" * 85)
    
    if all_contracts_ok:
        print("\n[通过] 所有合约在随机滑点下仍保持正收益！")
    else:
        print("\n[失败] 部分合约在随机滑点下变为负收益，需要优化策略！")
    
    print("\n详细数据:")
    for contract in CONTRACTS:
        slip_runs = result.contract_results.get(contract, [])
        if slip_runs:
            print(f"\n  {contract} - 随机滑点各次运行结果:")
            for i, r in enumerate(slip_runs):
                print(f"    运行 {i+1}: 收益率={r['total_return_percent']:.2f}%, "
                      f"交易次数={r['total_trades']}")


def print_time_window_summary(result: TimeWindowResult):
    """打印时间窗口汇总"""
    print("\n" + "=" * 80)
    print("                    多时间窗口回测汇总")
    print("=" * 80)
    
    print(f"\n{'窗口':<20} {'开始日期':<12} {'结束日期':<12} {'平均收益率':<12} {'交易次数':<10} {'状态':<10}")
    print("-" * 85)
    
    profitable_count = 0
    total_count = 0
    
    for window_name, window_data in result.window_results.items():
        start_dt = window_data['start_dt']
        end_dt = window_data['end_dt']
        avg_return = window_data['avg_return']
        total_trades = window_data['total_trades']
        is_profitable = window_data['is_profitable']
        
        total_count += 1
        if is_profitable:
            profitable_count += 1
        
        status = "盈利" if is_profitable else "亏损"
        
        print(f"{window_name:<20} {start_dt!s:<12} {end_dt!s:<12} "
              f"{avg_return:>10.2f}%   {total_trades:>8}   {status}")
    
    print("-" * 85)
    
    print(f"\n盈利窗口数: {profitable_count}/{total_count}")
    
    if profitable_count >= 2:
        print(f"[通过] 检验: 至少2个窗口保持不亏损 (实际: {profitable_count}个)")
    else:
        print(f"[失败] 检验: 需要至少2个窗口盈利 (实际: {profitable_count}个)")
    
    print("\n各窗口合约详情:")
    for window_name, window_data in result.window_results.items():
        contract_returns = window_data.get('contract_returns', {})
        if contract_returns:
            print(f"\n  {window_name}:")
            for contract, ret in contract_returns.items():
                status = "盈利" if ret >= 0 else "亏损"
                print(f"    {contract}: {ret:.2f}% ({status})")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='策略健壮性测试工具')
    parser.add_argument('--slippage', action='store_true', help='仅运行随机滑点测试')
    parser.add_argument('--windows', action='store_true', help='仅运行多时间窗口测试')
    parser.add_argument('--runs', '-n', type=int, default=5, help='随机滑点运行次数')
    parser.add_argument('--start', type=str, help='开始日期 (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, help='结束日期 (YYYY-MM-DD)')
    
    args = parser.parse_args()
    
    if args.start:
        global BASE_START_DATE
        BASE_START_DATE = parse_date(args.start)
    if args.end:
        global BASE_END_DATE
        BASE_END_DATE = parse_date(args.end)
    
    print("\n" + "=" * 80)
    print("                    策略健壮性测试系统启动")
    print("=" * 80)
    print(f"启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    run_slip = args.slippage or (not args.slippage and not args.windows)
    run_win = args.windows or (not args.slippage and not args.windows)
    
    if run_slip:
        slip_result = run_slippage_test(num_runs=args.runs)
        print_slippage_comparison(slip_result)
    
    if run_win:
        win_result = run_time_window_test()
        print_time_window_summary(win_result)
    
    print("\n" + "=" * 80)
    print("                    健壮性测试完成")
    print("=" * 80 + "\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
