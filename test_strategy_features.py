import sys
import os

base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, base_dir)

from strategies.base_strategy import SignalType
from strategies.adaptive_momentum_strategy import AdaptiveMomentumStrategy

print("=" * 60)
print("自适应动量策略核心功能测试")
print("=" * 60)

print("\n1. 测试策略初始化")
print("-" * 40)

strategy = AdaptiveMomentumStrategy(
    short_period=5,
    long_period=10,
    rsi_period=7,
    atr_period=10,
    atr_entry_multiplier=1.5,
    atr_exit_multiplier=2.0,
    risk_per_trade_percent=0.01,
    trailing_stop_atr_multiplier=2.0,
    contract_multiplier=10,
    debug_logging=False,
)

print(f"  短期均线周期: {strategy.short_period}")
print(f"  长期均线周期: {strategy.long_period}")
print(f"  RSI 周期: {strategy.rsi_period}")
print(f"  RSI 阈值: {strategy.rsi_threshold}")
print(f"  ATR 周期: {strategy.atr_period}")
print(f"  ATR 入场过滤倍数: {strategy.atr_entry_multiplier}")
print(f"  ATR 追踪止损启动倍数: {strategy.atr_exit_multiplier}")
print(f"  追踪止损 ATR 倍数: {strategy.trailing_stop_atr_multiplier}")
print(f"  单笔风险比例: {strategy.risk_per_trade_percent * 100}%")
print(f"  合约乘数: {strategy.contract_multiplier}")
print(f"  当前信号: {strategy.signal.value}")
print(f"  当前持仓: {strategy._position}")

print("\n2. 测试价格更新和指标计算")
print("-" * 40)

prices = [
    (3000.0, 3020.0, 2980.0, 2990.0),
    (3010.0, 3030.0, 2990.0, 3000.0),
    (3020.0, 3040.0, 3000.0, 3010.0),
    (3015.0, 3035.0, 2995.0, 3020.0),
    (3025.0, 3045.0, 3005.0, 3015.0),
    (3030.0, 3050.0, 3010.0, 3025.0),
    (3040.0, 3060.0, 3020.0, 3030.0),
    (3050.0, 3070.0, 3030.0, 3040.0),
    (3060.0, 3080.0, 3040.0, 3050.0),
    (3070.0, 3090.0, 3050.0, 3060.0),
    (3080.0, 3100.0, 3060.0, 3070.0),
    (3090.0, 3110.0, 3070.0, 3080.0),
]

for i, (close, high, low, open_p) in enumerate(prices):
    strategy.update_prices(close_price=close, high_price=high, low_price=low, open_price=open_p)
    print(f"  K线 {i+1}: 收盘价={close:.2f}, 短期MA={strategy.short_ma}, 长期MA={strategy.long_ma}, RSI={strategy.rsi}, ATR={strategy.atr}")

print(f"\n  最终短期均线 (MA{strategy.short_period}): {strategy.short_ma}")
print(f"  最终长期均线 (MA{strategy.long_period}): {strategy.long_ma}")
print(f"  最终 RSI: {strategy.rsi}")
print(f"  最终 ATR: {strategy.atr}")
print(f"  当前信号: {strategy.signal.value}")

print("\n3. 测试动态仓位计算")
print("-" * 40)

strategy._total_capital = 1000000.0

test_cases = [
    (50.0, 3000.0, "正常波动"),
    (100.0, 3000.0, "高波动"),
    (25.0, 3000.0, "低波动"),
]

for atr, price, desc in test_cases:
    position_size = strategy._calculate_position_size(atr, price)
    risk_amount = strategy._total_capital * strategy.risk_per_trade_percent
    risk_per_lot = atr * strategy.contract_multiplier
    
    print(f"  {desc}:")
    print(f"    ATR = {atr}, 价格 = {price}")
    print(f"    风险金额 = {risk_amount:.2f}")
    print(f"    每手风险 = {risk_per_lot:.2f}")
    print(f"    计算仓位 = {position_size:.2f} 手")
    print()

print("4. 测试金叉/死叉信号检测")
print("-" * 40)

strategy2 = AdaptiveMomentumStrategy(
    short_period=2,
    long_period=3,
    atr_period=3,
    atr_entry_multiplier=0.1,
    rsi_threshold=0.0,
    debug_logging=False,
)

print("  构造金叉场景:")
print()

golden_cross_prices = [
    (100.0, 102.0, 98.0, 99.0),
    (110.0, 112.0, 108.0, 109.0),
    (105.0, 107.0, 103.0, 104.0),
    (100.0, 102.0, 98.0, 99.0),
    (120.0, 122.0, 118.0, 119.0),
]

prev_short_above = None

for i, (close, high, low, open_p) in enumerate(golden_cross_prices):
    strategy2.update_prices(close_price=close, high_price=high, low_price=low, open_price=open_p)
    
    short_above = None
    if strategy2.short_ma is not None and strategy2.long_ma is not None:
        short_above = strategy2.short_ma > strategy2.long_ma
    
    print(f"  K线 {i+1}:")
    print(f"    收盘价: {close}")
    print(f"    短期MA ({strategy2.short_period}期): {strategy2.short_ma}")
    print(f"    长期MA ({strategy2.long_period}期): {strategy2.long_ma}")
    print(f"    RSI: {strategy2.rsi}")
    print(f"    ATR: {strategy2.atr}")
    print(f"    信号: {strategy2.signal.value}")
    
    if prev_short_above is not None and short_above is not None:
        if not prev_short_above and short_above:
            print(f"    [INFO] 金叉检测: 短期均线上穿长期均线")
        elif prev_short_above and not short_above:
            print(f"    [INFO] 死叉检测: 短期均线下穿长期均线")
    
    prev_short_above = short_above
    print()

print("  构造死叉场景:")
print()

strategy3 = AdaptiveMomentumStrategy(
    short_period=2,
    long_period=3,
    atr_period=3,
    atr_entry_multiplier=0.1,
    rsi_threshold=100.0,
    debug_logging=False,
)

death_cross_prices = [
    (120.0, 122.0, 118.0, 119.0),
    (110.0, 112.0, 108.0, 109.0),
    (115.0, 117.0, 113.0, 114.0),
    (120.0, 122.0, 118.0, 119.0),
    (100.0, 102.0, 98.0, 99.0),
]

prev_short_above = None

for i, (close, high, low, open_p) in enumerate(death_cross_prices):
    strategy3.update_prices(close_price=close, high_price=high, low_price=low, open_price=open_p)
    
    short_above = None
    if strategy3.short_ma is not None and strategy3.long_ma is not None:
        short_above = strategy3.short_ma > strategy3.long_ma
    
    print(f"  K线 {i+1}:")
    print(f"    收盘价: {close}")
    print(f"    短期MA ({strategy3.short_period}期): {strategy3.short_ma}")
    print(f"    长期MA ({strategy3.long_period}期): {strategy3.long_ma}")
    print(f"    RSI: {strategy3.rsi}")
    print(f"    ATR: {strategy3.atr}")
    print(f"    信号: {strategy3.signal.value}")
    
    if prev_short_above is not None and short_above is not None:
        if not prev_short_above and short_above:
            print(f"    [INFO] 金叉检测: 短期均线上穿长期均线")
        elif prev_short_above and not short_above:
            print(f"    [INFO] 死叉检测: 短期均线下穿长期均线")
    
    prev_short_above = short_above
    print()

print("\n5. 测试追踪止损逻辑")
print("-" * 40)

from strategies.strategy_comparison import BacktestSimulator
from datetime import datetime

simulator = BacktestSimulator(
    initial_capital=1000000.0,
    contract_multiplier=10,
)

print("  开多单测试:")
print(f"    初始资金: {simulator.current_capital:.2f}")

simulator.open_position(
    direction="BUY",
    price=3000.0,
    volume=1,
    current_time=datetime.now(),
    reason="test",
)

print(f"    开仓后持仓: {simulator._position}")
print(f"    开仓价格: {simulator._entry_price}")
print()

atr = 50.0

print("  价格上涨，触发追踪止损:")
print()

price_sequence = [3050.0, 3100.0, 3150.0, 3200.0, 3100.0, 3050.0]

for i, price in enumerate(price_sequence):
    should_close, reason = simulator.update_price(
        current_price=price,
        atr=atr,
        atr_exit_multiplier=2.0,
        trailing_stop_multiplier=2.0,
    )
    
    profit_atr = ((price - 3001.0) * 10) / atr
    
    print(f"  第 {i+1} 步: 价格 = {price}")
    print(f"    盈利: {profit_atr:.2f} x ATR")
    print(f"    追踪止损活跃: {simulator._trailing_stop_active}")
    print(f"    追踪止损价格: {simulator._trailing_stop_price}")
    print(f"    最高价: {simulator._highest_price_since_entry}")
    
    if should_close:
        print(f"    [INFO] 追踪止损触发: {reason}")
        simulator.close_position(price, datetime.now(), reason)
        print(f"    平仓后资金: {simulator.current_capital:.2f}")
    print()

print("\n" + "=" * 60)
print("测试完成！")
print("=" * 60)
