#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
简化版健壮性测试脚本
"""

import os
import sys
import logging
from datetime import datetime, date

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
BASE_END_DATE = date(2024, 3, 31)


def run_single_backtest(
    contract: str,
    start_dt: date,
    end_dt: date,
    use_random_slippage: bool = False,
    slippage_min: int = 1,
    slippage_max: int = 3,
) -> dict:
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
        }
    
    return {
        'status': 'error',
        'error_message': result.get('error_message', 'Unknown error'),
    }


def main():
    print("\n" + "=" * 80)
    print("                    策略健壮性测试 (简化版)")
    print("=" * 80)
    print(f"启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"回测区间: {BASE_START_DATE} 至 {BASE_END_DATE}")
    print(f"测试合约: {', '.join(CONTRACTS)}")
    print("-" * 80)
    
    print("\n【第一部分: 无滑点测试 (基准)】")
    no_slippage_results = {}
    
    for contract in CONTRACTS:
        print(f"\n  测试合约: {contract}")
        result = run_single_backtest(
            contract=contract,
            start_dt=BASE_START_DATE,
            end_dt=BASE_END_DATE,
            use_random_slippage=False,
        )
        
        if result['status'] == 'completed':
            no_slippage_results[contract] = result
            print(f"    收益率: {result['total_return_percent']:.2f}%")
            print(f"    交易次数: {result['total_trades']}")
            print(f"    胜率: {result['win_rate']:.2f}%")
    
    print("\n" + "=" * 80)
    print("【第二部分: 随机滑点测试 (1-3 ticks)】")
    
    slippage_results = {}
    num_runs = 2
    
    for contract in CONTRACTS:
        print(f"\n  测试合约: {contract}")
        slippage_results[contract] = []
        
        for i in range(num_runs):
            print(f"    运行 {i+1}/{num_runs}...")
            result = run_single_backtest(
                contract=contract,
                start_dt=BASE_START_DATE,
                end_dt=BASE_END_DATE,
                use_random_slippage=True,
                slippage_min=1,
                slippage_max=3,
            )
            
            if result['status'] == 'completed':
                slippage_results[contract].append(result)
                print(f"      收益率: {result['total_return_percent']:.2f}%, 交易次数: {result['total_trades']}")
    
    print("\n" + "=" * 80)
    print("                    测试结果汇总")
    print("=" * 80)
    
    print(f"\n{'合约':<15} {'无滑点收益率':<15} {'随机滑点平均收益率':<20} {'收益衰减':<15} {'状态':<10}")
    print("-" * 85)
    
    all_ok = True
    total_no_slip = 0.0
    total_slip = 0.0
    
    for contract in CONTRACTS:
        no_slip = no_slippage_results.get(contract, {})
        slips = slippage_results.get(contract, [])
        
        no_slip_return = no_slip.get('total_return_percent', 0)
        total_no_slip += no_slip_return
        
        if slips:
            avg_slip_return = sum(r.get('total_return_percent', 0) for r in slips) / len(slips)
            decay = no_slip_return - avg_slip_return
            total_slip += avg_slip_return
        else:
            avg_slip_return = 0
            decay = 0
        
        is_ok = avg_slip_return >= 0
        if not is_ok:
            all_ok = False
        
        status = "通过" if is_ok else "失败"
        
        print(f"{contract:<15} {no_slip_return:>12.2f}%    {avg_slip_return:>17.2f}%       {decay:>12.2f}%   {status}")
    
    print("-" * 85)
    
    avg_no_slip = total_no_slip / len(CONTRACTS)
    avg_slip = total_slip / len(CONTRACTS)
    print(f"\n组合平均 (无滑点): {avg_no_slip:.2f}%")
    print(f"组合平均 (随机滑点): {avg_slip:.2f}%")
    
    if all_ok:
        print("\n[通过] 所有合约在随机滑点下仍保持正收益！")
    else:
        print("\n[失败] 部分合约在随机滑点下变为负收益")
    
    print("\n" + "=" * 80)
    print("测试完成")
    print("=" * 80 + "\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
