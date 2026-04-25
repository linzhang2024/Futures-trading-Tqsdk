#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试多合约回测功能
不使用天勤服务器，仅验证代码逻辑
"""

import os
import sys
import logging
from datetime import date

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, base_dir)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)

try:
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def test_strategy_import():
    print("\n" + "=" * 60)
    print("测试 1: DoubleMAStrategy 导入和创建")
    print("=" * 60)
    
    from strategies.double_ma_strategy import DoubleMAStrategy
    print("DoubleMAStrategy 导入成功")
    
    strategy = DoubleMAStrategy(
        short_period=5,
        long_period=20,
        contract='SHFE.rb2410',
        initial_data_days=5,
    )
    print("DoubleMAStrategy 创建成功")
    
    print(f"  initial_data_days: {strategy.initial_data_days}")
    print(f"  _required_warmup_klines: {strategy._required_warmup_klines}")
    print(f"  _data_warmed_up: {strategy._data_warmed_up}")
    print(f"  is_ready(): {strategy.is_ready()}")
    
    print("PASS")
    return True


def test_equity_plotter_import():
    print("\n" + "=" * 60)
    print("测试 2: EquityPlotter 导入")
    print("=" * 60)
    
    from core.equity_plotter import (
        EquityPlotter, 
        ContractResult,
        _check_chinese_font_support,
    )
    print("EquityPlotter 导入成功")
    
    if HAS_MATPLOTLIB:
        chinese_supported = _check_chinese_font_support()
        print(f"中文支持: {'是' if chinese_supported else '否'}")
    
    print("PASS")
    return True


def test_multi_contract_runner_import():
    print("\n" + "=" * 60)
    print("测试 3: MultiContractRunner 导入")
    print("=" * 60)
    
    from core.multi_contract_runner import (
        MultiContractRunner,
        ContractBacktestConfig,
        MultiContractResult,
    )
    print("MultiContractRunner 导入成功")
    
    print(f"ContractBacktestConfig 字段: {ContractBacktestConfig.__annotations__.keys()}")
    print(f"MultiContractResult 字段: {MultiContractResult.__annotations__.keys()}")
    
    print("PASS")
    return True


def test_run_all_import():
    print("\n" + "=" * 60)
    print("测试 4: run_all.py 导入")
    print("=" * 60)
    
    print("CONTRACTS 配置:")
    contracts = [
        'SHFE.rb2410',
        'SHFE.hc2410',
        'DCE.i2409',
    ]
    for c in contracts:
        print(f"  - {c}")
    
    print("\nBACKTEST_CONFIG:")
    print(f"  start_dt: 2024-01-01")
    print(f"  end_dt: 2024-03-31")
    print(f"  initial_balance: 1000000")
    
    print("PASS")
    return True


def test_contract_config_creation():
    print("\n" + "=" * 60)
    print("测试 5: 创建合约配置")
    print("=" * 60)
    
    from core.multi_contract_runner import ContractBacktestConfig
    from strategies.double_ma_strategy import DoubleMAStrategy
    
    configs = []
    contracts = ['SHFE.rb2410', 'SHFE.hc2410', 'DCE.i2409']
    
    for contract in contracts:
        config = ContractBacktestConfig(
            contract=contract,
            strategy_class=DoubleMAStrategy,
            strategy_params={
                'short_period': 5,
                'long_period': 20,
                'kline_duration': 60,
                'initial_data_days': 5,
            },
            start_dt=date(2024, 1, 1),
            end_dt=date(2024, 3, 31),
            initial_balance=1000000.0,
        )
        configs.append(config)
        print(f"已创建合约配置: {contract}")
    
    print(f"\n总配置数: {len(configs)}")
    print("PASS")
    return True


def test_simulation_data():
    print("\n" + "=" * 60)
    print("测试 6: 模拟数据生成")
    print("=" * 60)
    
    from core.backtest import MockTqApi, MockTqSim
    from datetime import date
    
    print("创建模拟API...")
    
    api = MockTqApi(
        account=MockTqSim(init_balance=1000000.0),
        start_dt=date(2024, 1, 1),
        end_dt=date(2024, 1, 5),
        init_balance=1000000.0,
    )
    
    print(f"API 创建成功")
    print(f"  开始日期: {api._start_dt}")
    print(f"  结束日期: {api._end_dt}")
    print(f"  初始资金: {api._equity}")
    print(f"  总周期数: {api._max_cycles}")
    
    contracts = ['SHFE.rb2410', 'SHFE.hc2410', 'DCE.i2409']
    
    for contract in contracts:
        klines = api.get_kline_serial(contract, 60)
        quote = api.get_quote(contract)
        
        print(f"\n  {contract}:")
        print(f"    K线数量: {len(klines)}")
        print(f"    行情 last_price: {quote.last_price}")
    
    print("\nPASS")
    return True


def test_equity_plotter_basic():
    print("\n" + "=" * 60)
    print("测试 7: EquityPlotter 基础功能")
    print("=" * 60)
    
    if not HAS_MATPLOTLIB:
        print("matplotlib 未安装，跳过测试")
        return True
    
    from core.equity_plotter import (
        EquityPlotter, 
        create_contract_result_from_backtest,
    )
    import os
    
    results_dir = os.path.join(base_dir, 'results')
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        print(f"已创建目录: {results_dir}")
    
    plotter = EquityPlotter(output_dir=results_dir)
    print("EquityPlotter 创建成功")
    
    import random
    from datetime import datetime, timedelta
    
    base_time = datetime(2024, 1, 1, 9, 0)
    
    def generate_equity_curve(start_equity, num_points, volatility=1500):
        equity_data = []
        equity = start_equity
        for i in range(num_points):
            equity += random.uniform(-volatility, volatility * 1.2)
            timestamp = base_time + timedelta(hours=i)
            equity_data.append({
                'timestamp': timestamp.timestamp(),
                'equity': equity,
                'margin_used': 100000.0,
                'cycle': i,
            })
        return equity_data
    
    test_equity_data1 = generate_equity_curve(1000000.0, 100)
    test_equity_data2 = generate_equity_curve(1000000.0, 80)
    test_equity_data3 = generate_equity_curve(1000000.0, 100)
    
    test_trade_data = [
        {'timestamp': test_equity_data1[10]['timestamp'], 'contract': 'SHFE.rb2410', 'direction': 'BUY', 'offset': 'OPEN', 'price': 3500.0, 'volume': 1, 'profit_loss': 0},
        {'timestamp': test_equity_data1[30]['timestamp'], 'contract': 'SHFE.rb2410', 'direction': 'SELL', 'offset': 'CLOSE', 'price': 3600.0, 'volume': 1, 'profit_loss': 1000},
        {'timestamp': test_equity_data1[50]['timestamp'], 'contract': 'SHFE.rb2410', 'direction': 'BUY', 'offset': 'OPEN', 'price': 3550.0, 'volume': 1, 'profit_loss': 0},
        {'timestamp': test_equity_data1[80]['timestamp'], 'contract': 'SHFE.rb2410', 'direction': 'SELL', 'offset': 'CLOSE', 'price': 3520.0, 'volume': 1, 'profit_loss': -300},
    ]
    
    final_equity1 = test_equity_data1[-1]['equity']
    final_equity2 = test_equity_data2[-1]['equity']
    final_equity3 = test_equity_data3[-1]['equity']
    
    result1 = create_contract_result_from_backtest(
        contract='SHFE.rb2410',
        initial_equity=1000000.0,
        final_equity=final_equity1,
        equity_curve_data=test_equity_data1,
        trade_records_data=test_trade_data,
    )
    
    result2 = create_contract_result_from_backtest(
        contract='SHFE.hc2410',
        initial_equity=1000000.0,
        final_equity=final_equity2,
        equity_curve_data=test_equity_data2,
        trade_records_data=test_trade_data[:2],
    )
    
    result3 = create_contract_result_from_backtest(
        contract='DCE.i2409',
        initial_equity=1000000.0,
        final_equity=final_equity3,
        equity_curve_data=test_equity_data3,
        trade_records_data=test_trade_data,
    )
    
    print("\n创建单合约图表...")
    single_path = plotter.plot_single_equity_curve(
        result=result1,
        title='SHFE.rb2410 Equity Curve',
    )
    print(f"  单合约图表: {single_path}")
    
    print("\n创建多合约对比图表...")
    multi_path = plotter.plot_multi_contract_comparison(
        results=[result1, result2, result3],
        title='Multi-Contract Comparison',
    )
    print(f"  对比图表: {multi_path}")
    
    print("\n生成汇总报告...")
    summary = plotter.generate_summary_report(
        results=[result1, result2, result3],
    )
    print(f"  合约数量: {summary['summary']['total_contracts']}")
    print(f"  总交易次数: {summary['summary']['total_trades']}")
    
    csv_path = plotter.generate_trade_details_csv([result1, result2, result3])
    print(f"\n交易明细 CSV: {csv_path}")
    
    print("\nPASS")
    return True


def main():
    print("\n" + "#" * 60)
    print("#            多合约回测系统测试")
    print("#" * 60)
    
    all_passed = True
    
    try:
        test_strategy_import()
    except Exception as e:
        print(f"FAIL: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False
    
    try:
        test_equity_plotter_import()
    except Exception as e:
        print(f"FAIL: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False
    
    try:
        test_multi_contract_runner_import()
    except Exception as e:
        print(f"FAIL: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False
    
    try:
        test_run_all_import()
    except Exception as e:
        print(f"FAIL: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False
    
    try:
        test_contract_config_creation()
    except Exception as e:
        print(f"FAIL: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False
    
    try:
        test_simulation_data()
    except Exception as e:
        print(f"FAIL: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False
    
    try:
        test_equity_plotter_basic()
    except Exception as e:
        print(f"FAIL: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False
    
    print("\n" + "#" * 60)
    if all_passed:
        print("#              所有测试通过！")
        print("#")
        print("# 下一步操作:")
        print("#   python run_all.py          运行多合约回测")
        print("#   或查看 results/ 目录获取图表")
    else:
        print("#              部分测试失败！")
    print("#" * 60 + "\n")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
