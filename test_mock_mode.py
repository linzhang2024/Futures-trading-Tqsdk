import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datetime import date

from core.backtest import BacktestEngine, ParameterRange
from strategies.double_ma_strategy import DoubleMAStrategy

config = {
    'backtest': {
        'init_balance': 1000000.0,
        'use_mock_data': True,
        'costs': {
            'default_commission_per_lot': 5.0,
            'default_slippage_points': 1.0,
        }
    }
}

print('测试模拟数据模式回测...')
print(f'use_mock_data: {config["backtest"]["use_mock_data"]}')

engine = BacktestEngine(config=config)

print('\n开始测试单参数回测...')
try:
    result = engine.run_backtest(
        strategy_class=DoubleMAStrategy,
        strategy_params={
            'short_period': 5,
            'long_period': 20,
            'contract': 'SHFE.rb2410',
            'kline_duration': 60,
            'use_ema': False,
        },
        start_dt=date(2024, 1, 2),
        end_dt=date(2024, 1, 15),
    )
    
    print(f'\n回测结果:')
    print(f'  状态: {result.status}')
    print(f'  初始权益: {result.initial_equity:,.0f}')
    print(f'  最终权益: {result.final_equity:,.0f}')
    print(f'  收益率: {result.performance.total_return_percent:.2f}%')
    print(f'  交易次数: {result.performance.total_trades}')
    
    if result.status == 'completed':
        print('\n[OK] 模拟数据模式回测成功!')
    else:
        print(f'\n[FAIL] 回测失败: {result.error_message}')
        
except Exception as e:
    print(f'\n[FAIL] 异常: {e}')
    import traceback
    traceback.print_exc()
