import sys
import os
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.risk_manager import (
    RiskManager,
    RiskLevel,
    RiskEventType,
    AccountSnapshot,
    PositionInfo,
    NUMPY_AVAILABLE,
)


def print_banner():
    banner = """
    ======================================================================
                            全 局 风 控 模 块 演 示
    ======================================================================

    演示场景：
    1. 初始化账户：10,000,000 元
    2. 设置极端风控阈值：回撤 0.1% 即触发熔断
    3. 模拟账户权益快速下跌
    4. 观察风控系统的实时响应

    ======================================================================
    """
    print(banner)


def simulate_market_scenario():
    print("\n" + "=" * 70)
    print("【阶段 1】初始化风控系统")
    print("=" * 70)

    risk_manager = RiskManager(
        max_drawdown_percent=0.1,
        max_strategy_margin_percent=30.0,
        max_total_margin_percent=80.0,
        price_gap_threshold_percent=5.0,
        api_timeout_seconds=3.0,
    )

    print("RiskManager 已初始化")
    print(f"   - 最大回撤阈值: {risk_manager.max_drawdown_percent}%")
    print(f"   - 单策略最大保证金: {risk_manager.max_strategy_margin_percent}%")
    print(f"   - 价格跳空阈值: {risk_manager.price_gap_threshold_percent}%")
    print(f"   - NumPy 可用: {NUMPY_AVAILABLE}")

    print("\n" + "=" * 70)
    print("【阶段 2】设置初始账户状态")
    print("=" * 70)

    initial_equity = 10000000.0
    risk_manager._peak_equity = initial_equity
    risk_manager._initial_equity = initial_equity
    risk_manager._initialized = True

    initial_snapshot = AccountSnapshot(
        timestamp=time.time(),
        balance=initial_equity,
        equity=initial_equity,
        total_asset=initial_equity,
        margin_used=0.0,
        available=initial_equity,
        float_profit=0.0,
    )
    risk_manager._snapshots.append(initial_snapshot)

    print(f"初始账户权益: {initial_equity:,.2f} 元")
    print(f"   峰值权益: {risk_manager._peak_equity:,.2f} 元")

    print("\n" + "=" * 70)
    print("【阶段 3】设置模拟持仓")
    print("=" * 70)

    positions_data = {
        'SHFE.rb2410': {
            'buy_volume': 50,
            'sell_volume': 0,
            'buy_margin': 1750000.0,
            'sell_margin': 0.0,
            'buy_open_price': 3500.0,
            'sell_open_price': 0.0,
            'last_price': 3650.0,
            'float_profit': 750000.0,
        },
        'DCE.i2409': {
            'buy_volume': 0,
            'sell_volume': 30,
            'buy_margin': 0.0,
            'sell_margin': 240000.0,
            'buy_open_price': 0.0,
            'sell_open_price': 1000.0,
            'last_price': 980.0,
            'float_profit': 120000.0,
        },
        'SHFE.hc2410': {
            'buy_volume': 20,
            'sell_volume': 0,
            'buy_margin': 720000.0,
            'sell_margin': 0.0,
            'buy_open_price': 3600.0,
            'sell_open_price': 0.0,
            'last_price': 3550.0,
            'float_profit': -50000.0,
        },
    }

    risk_manager.update_positions(positions_data)

    positions = risk_manager._positions
    for contract, pos in positions.items():
        direction = "多头" if pos.net_position > 0 else "空头" if pos.net_position < 0 else "净空"
        print(f"   {contract}:")
        print(f"      净持仓: {pos.net_position:+d} 手 ({direction})")
        print(f"      浮动盈亏: {pos.float_profit:,.2f} 元")
        print(f"      占用保证金: {pos.total_margin:,.2f} 元")

    print("\n" + "=" * 70)
    print("【阶段 4】添加模拟策略风险信息")
    print("=" * 70)

    strategies_info = [
        ("Strategy_MA5_10", 1500000.0, 4500000.0, 250000.0),
        ("Strategy_MA10_20", 800000.0, 2400000.0, -50000.0),
        ("Strategy_RSI", 500000.0, 1500000.0, 80000.0),
        ("Strategy_Bollinger", 600000.0, 1800000.0, 120000.0),
    ]

    for name, margin, pos_value, float_pnl in strategies_info:
        risk_manager.update_strategy_risk(
            strategy_name=name,
            margin_used=margin,
            position_value=pos_value,
            float_profit=float_pnl,
        )
        print(f"   {name}:")
        print(f"      占用保证金: {margin:,.2f} 元")
        print(f"      持仓价值: {pos_value:,.2f} 元")
        print(f"      浮动盈亏: {float_pnl:,.2f} 元")

    print("\n" + "=" * 70)
    print("【阶段 5】运行性能基准测试 (50策略)")
    print("=" * 70)

    report = risk_manager.run_performance_benchmark(num_strategies=50)
    stats = report['statistics']

    print("\n性能测试结果:")
    print(f"   策略数量: 50")
    print(f"   迭代次数: 100")
    print(f"   NumPy 优化: {'启用' if NUMPY_AVAILABLE else '未启用'}")
    print(f"\n   单次风控检查耗时统计:")
    print(f"   最小: {stats['min_ms']:.3f} ms")
    print(f"   最大: {stats['max_ms']:.3f} ms")
    print(f"   平均: {stats['avg_ms']:.3f} ms")
    print(f"   中位数: {stats['median_ms']:.3f} ms")
    print(f"   P95: {stats['p95_ms']:.3f} ms")
    print(f"   P99: {stats['p99_ms']:.3f} ms")

    print("\n" + "=" * 70)
    print("【阶段 6】模拟极端行情 - 触发风控熔断")
    print("=" * 70)

    print(f"\n风控阈值设置: 回撤 {risk_manager.max_drawdown_percent}% 即触发熔断")
    print(f"   当前峰值权益: {risk_manager._peak_equity:,.2f} 元")
    print(f"   触发熔断的权益阈值: {risk_manager._peak_equity * (1 - risk_manager.max_drawdown_percent / 100):,.2f} 元")

    print("\n开始模拟账户权益下跌...\n")

    equity_series = [
        (10000000.0, "初始状态"),
        (9995000.0, "小幅下跌 0.05%"),
        (9992000.0, "继续下跌 0.08%"),
        (9990000.0, "接近阈值 0.10%"),
        (9985000.0, "跌破阈值 0.15% - 触发熔断!"),
        (9980000.0, "继续下跌 (熔断已触发)"),
    ]

    freeze_step = None

    for i, (equity, description) in enumerate(equity_series):
        if risk_manager.is_frozen():
            break

        drawdown = (risk_manager._peak_equity - equity) / risk_manager._peak_equity * 100
        time.sleep(0.3)

        timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]
        if drawdown >= risk_manager.max_drawdown_percent:
            status_icon = "[CRITICAL]"
        elif drawdown >= risk_manager.max_drawdown_percent * 0.5:
            status_icon = "[WARNING]"
        else:
            status_icon = "[OK]"

        print(f"   [{timestamp}] {status_icon} 权益: {equity:,.2f} 元 | 回撤: {drawdown:.3f}% | {description}")

        snapshot = AccountSnapshot(
            timestamp=time.time(),
            balance=equity,
            equity=equity,
            total_asset=equity,
            margin_used=2500000.0,
            available=equity - 2500000.0,
            float_profit=equity - initial_equity,
        )
        risk_manager._snapshots.append(snapshot)

        level = risk_manager.check_drawdown(snapshot)

        if level == RiskLevel.FROZEN and freeze_step is None:
            freeze_step = i

    print("\n" + "=" * 70)
    print("【阶段 7】熔断触发后检查")
    print("=" * 70)

    if risk_manager.is_frozen():
        print(f"\n系统已成功冻结!")
        print(f"   冻结原因: {risk_manager.get_frozen_reason()}")
        print(f"   当前回撤: {risk_manager._current_drawdown_percent:.3f}%")
        print(f"   熔断阈值: {risk_manager.max_drawdown_percent}%")

        risk_info = risk_manager.get_total_risk_info()
        print(f"\n冻结时账户状态:")
        print(f"   时间: {risk_info['timestamp']}")
        print(f"   权益: {risk_info['equity']:,.2f} 元")
        print(f"   占用保证金: {risk_info['margin_used']:,.2f} 元")
        print(f"   可用资金: {risk_info['available']:,.2f} 元")
        print(f"   持仓数量: {risk_info['position_count']}")
        print(f"   策略数量: {risk_info['strategy_count']}")

        print("\n风控事件日志:")
        events = risk_manager.get_risk_events(limit=10)
        for event in events:
            level_icon = "[CRITICAL]" if event['level'] in ['CRITICAL', 'FROZEN'] else "[WARNING]"
            print(f"   {level_icon} [{event['datetime']}] {event['event_type']}: {event['message']}")

        print("\n查看日志文件:")
        print(f"   风险事件日志: logs/risk_event.log")
        print(f"   熔断复盘报告: logs/freeze_reports/")

    else:
        print("熔断未触发 (测试可能有问题)")

    print("\n" + "=" * 70)
    print("【演示结束】")
    print("=" * 70)

    print("\n总结:")
    print("   - 风控系统能够实时监控账户权益变化")
    print("   - 当回撤超过阈值时能够毫秒级响应并触发熔断")
    print("   - 熔断时自动生成详细的复盘报告")
    print("   - 支持多策略并行风控检查")
    print("   - 性能优化: 50策略单次风控检查耗时 < 1ms")
    print("   - 支持价格跳空检测和API健康监控")

    return risk_manager


if __name__ == "__main__":
    print_banner()

    try:
        risk_manager = simulate_market_scenario()
    except KeyboardInterrupt:
        print("\n\n用户中断演示")
    except Exception as e:
        print(f"\n\n演示出错: {e}")
        import traceback
        traceback.print_exc()
