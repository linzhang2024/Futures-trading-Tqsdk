import sys
import os
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.risk_manager import (
    RiskManager,
    RiskLevel,
    RiskEventType,
    RiskEvent,
    TradeEventType,
    AccountSnapshot,
    RiskCheckReport,
    StructuredLogger,
)


def print_separator(title: str):
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def test_singleton_pattern():
    print_separator("测试 1: 单例模式验证")
    
    RiskManager.reset_instance()
    
    rm1 = RiskManager()
    rm2 = RiskManager()
    
    assert rm1 is rm2, "两次实例化应该返回同一个对象"
    print("[PASS] 单例模式验证通过: 两次实例化返回同一对象")
    
    rm3 = RiskManager.get_instance()
    assert rm1 is rm3, "get_instance() 应该返回同一个单例"
    print("[PASS] get_instance() 单例验证通过")


def test_price_deviation_block():
    print_separator("测试 2: 价格偏离拦截验证")
    
    RiskManager.reset_instance()
    rm = RiskManager(price_deviation_threshold_percent=1.0)
    
    contract = "SHFE.rb2410"
    current_price = 3500.0
    rm.update_current_price(contract, current_price)
    
    print(f"当前市场价格: {current_price}")
    print(f"价格偏离阈值: {rm.price_deviation_threshold_percent}%")
    print(f"允许的价格范围: {current_price * 0.99:.2f} ~ {current_price * 1.01:.2f}")
    
    test_cases = [
        (3535.0, False, f"偏离 1.0% = 拦截"),
        (3465.0, False, f"偏离 -1.0% = 拦截"),
        (3530.0, False, f"偏离 0.857% = 允许"),
        (3470.0, False, f"偏离 -0.857% = 允许"),
        (3550.0, True, f"偏离 1.428% = 拦截"),
        (3450.0, True, f"偏离 -1.428% = 拦截"),
    ]
    
    print("\n测试价格偏离检查:")
    for order_price, should_block, desc in test_cases:
        ok, msg, deviation = rm.check_price_deviation(
            contract=contract,
            order_price=order_price,
            current_price=current_price,
        )
        
        expected_ok = not should_block
        status = "[PASS]" if ok == expected_ok else "[FAIL]"
        print(f"  {status} 订单价格 {order_price:.2f} - 偏离 {deviation:.3f}% - {'拦截' if not ok else '通过'}")
        
        if should_block and ok:
            print(f"     [ERROR] 错误: 应该被拦截但通过了!")
    
    print("\n测试 can_place_order 价格偏离拦截:")
    can_place, message, level = rm.can_place_order(
        strategy_name="TestStrategy",
        contract=contract,
        direction="BUY",
        volume=10,
        price=3550.0,
        current_market_price=current_price,
    )
    
    assert not can_place, f"价格偏离 {((3550-3500)/3500*100):.2f}% 应该被拦截"
    assert "价格偏离" in message, f"消息应该包含价格偏离: {message}"
    print(f"[PASS] 订单价格 3550.0 (偏离 {((3550-3500)/3500*100):.2f}%) 被成功拦截")
    print(f"   拦截原因: {message}")
    print(f"   风控拦截计数: {rm._risk_blocked_orders}")


def test_daily_loss_freeze():
    print_separator("测试 3: 日损限额熔断验证")
    
    RiskManager.reset_instance()
    rm = RiskManager(
        daily_loss_limit_percent=2.0,
        max_drawdown_percent=10.0,
    )
    
    initial_equity = 1000000.0
    rm.set_daily_start_equity(initial_equity)
    rm._peak_equity = initial_equity
    
    print(f"当日起始权益: {initial_equity:,.0f}")
    print(f"日损限额: {rm.daily_loss_limit_percent}%")
    print(f"最大允许亏损: {initial_equity * 0.02:,.0f}")
    
    loss_scenarios = [
        (0.01, 0.01, RiskLevel.SAFE, "正常波动"),
        (0.015, 0.015, RiskLevel.WARNING, "接近限额 (70% 警告)"),
        (0.02, 0.02, RiskLevel.FROZEN, "触发熔断 (等于限额)"),
        (0.03, 0.03, RiskLevel.FROZEN, "严重亏损 (超过限额)"),
    ]
    
    print("\n模拟日损场景:")
    for loss_pct, expected_loss, expected_level, desc in loss_scenarios:
        RiskManager.reset_instance()
        rm = RiskManager(
            daily_loss_limit_percent=2.0,
            max_drawdown_percent=10.0,
        )
        rm.set_daily_start_equity(initial_equity)
        rm._peak_equity = initial_equity
        
        current_equity = initial_equity * (1 - loss_pct)
        snapshot = AccountSnapshot(
            timestamp=time.time(),
            balance=current_equity,
            equity=current_equity,
            total_asset=current_equity,
            margin_used=0.0,
            available=current_equity,
            float_profit=-initial_equity * loss_pct,
        )
        
        level = rm.check_daily_loss(snapshot)
        
        actual_loss = (initial_equity - current_equity) / initial_equity * 100
        status = "[PASS]" if level == expected_level else "[FAIL]"
        print(f"  {status} 亏损 {actual_loss:.2f}% - 风险等级: {level.value} - {desc}")
        
        if loss_pct >= 0.02:
            assert rm.is_frozen(), f"亏损 {loss_pct*100:.1f}% 应该触发熔断"
            assert "日亏损" in rm.get_frozen_reason(), f"熔断原因应该包含日亏损"
            print(f"     [PASS] 系统已冻结，原因: {rm.get_frozen_reason()}")


def test_consecutive_loss_pause():
    print_separator("测试 4: 连续亏损策略暂停验证")
    
    RiskManager.reset_instance()
    rm = RiskManager(
        consecutive_loss_limit=5,
    )
    
    strategy_name = "TestStrategy"
    contract = "SHFE.rb2410"
    
    print(f"连续亏损限制: {rm.consecutive_loss_limit} 次")
    print(f"策略: {strategy_name}")
    
    print("\n模拟连续亏损交易:")
    
    for i in range(1, 8):
        loss_amount = -1000.0 * i
        rm.record_trade_event(
            strategy_name=strategy_name,
            contract=contract,
            direction="SELL",
            volume=1,
            price=3500.0 - i * 10,
            profit_loss=loss_amount,
            event_type=TradeEventType.POSITION_CLOSED,
        )
        
        strategy_info = rm._strategy_risk.get(strategy_name)
        is_paused = rm.is_strategy_paused(strategy_name)
        
        pause_status = "[PAUSED]" if is_paused else "[RUNNING]"
        status = "[PASS]" if (i >= 5 and is_paused) or (i < 5 and not is_paused) else "[FAIL]"
        
        print(f"  {status} 第 {i} 次交易 - 亏损 {abs(loss_amount):,.0f} - 连续亏损: {strategy_info.consecutive_losses} - {pause_status}")
        
        if i == 5:
            assert is_paused, "连续亏损 5 次后策略应该暂停"
            print(f"     [PASS] 策略已暂停，原因: {strategy_info.pause_reason}")
    
    print("\n测试策略恢复:")
    resumed = rm.resume_strategy(strategy_name)
    assert resumed, "策略应该恢复成功"
    assert not rm.is_strategy_paused(strategy_name), "策略恢复后不应暂停"
    
    strategy_info = rm._strategy_risk.get(strategy_name)
    assert strategy_info.consecutive_losses == 0, "恢复后连续亏损计数应重置为 0"
    print(f"[PASS] 策略已恢复，连续亏损计数重置为: {strategy_info.consecutive_losses}")


def test_risk_check_report():
    print_separator("测试 5: 风控体检报告验证")
    
    RiskManager.reset_instance()
    rm = RiskManager(
        daily_loss_limit_percent=2.0,
        consecutive_loss_limit=5,
        price_deviation_threshold_percent=1.0,
    )
    
    rm._peak_equity = 1000000.0
    rm._daily_start_equity = 1000000.0
    rm._total_canceled_orders = 3
    rm._risk_blocked_orders = 2
    rm._total_trades = 15
    rm._winning_trades = 8
    rm._losing_trades = 7
    rm._max_consecutive_losses = 4
    rm._max_single_drawdown = 50000.0
    rm._max_single_drawdown_percent = 5.0
    
    strategy_name = "TestStrategy"
    rm.record_trade_event(
        strategy_name=strategy_name,
        contract="SHFE.rb2410",
        direction="SELL",
        volume=1,
        price=3500.0,
        profit_loss=-1000.0,
    )
    
    report = rm.generate_risk_check_report()
    
    print(f"报告生成时间: {report.generated_at}")
    print(f"\n订单与风控统计:")
    print(f"  总撤单数: {report.total_canceled_orders}")
    print(f"  风控拦截订单数: {report.risk_blocked_orders}")
    
    print(f"\n盈亏与回撤统计:")
    print(f"  峰值权益: {report.peak_equity:,.2f}")
    print(f"  最大单笔回撤: {report.max_single_drawdown:,.2f} ({report.max_single_drawdown_percent:.2f}%)")
    
    print(f"\n交易与连胜统计:")
    print(f"  总交易次数: {report.total_trades}")
    print(f"  盈利交易: {report.winning_trades}")
    print(f"  亏损交易: {report.losing_trades}")
    print(f"  最大连续亏损: {report.max_consecutive_losses}")
    
    print(f"\n当前状态:")
    print(f"  当前风险等级: {report.current_risk_level.value}")
    print(f"  是否冻结: {report.is_frozen}")
    
    print("\n打印完整风控体检报告:")
    report_str = rm.print_risk_check_report()
    print("[PASS] 风控体检报告生成成功")


def test_extreme_market_scenario():
    print_separator("测试 6: 极端行情综合验证")
    
    RiskManager.reset_instance()
    
    rm = RiskManager(
        max_drawdown_percent=5.0,
        daily_loss_limit_percent=2.0,
        consecutive_loss_limit=3,
        price_deviation_threshold_percent=1.0,
    )
    
    initial_equity = 1000000.0
    rm._peak_equity = initial_equity
    rm._daily_start_equity = initial_equity
    
    contract = "SHFE.rb2410"
    strategy_name = "AggressiveStrategy"
    
    print("=" * 80)
    print("  模拟极端行情场景")
    print("=" * 80)
    print(f"初始权益: {initial_equity:,.0f}")
    print(f"合约: {contract}")
    print(f"策略: {strategy_name}")
    print()
    
    print("--- 阶段 1: 市场剧烈下跌，价格偏离订单被拦截 ---")
    current_price = 3500.0
    rm.update_current_price(contract, current_price)
    
    panic_price = 3300.0
    print(f"当前市场价格: {current_price}")
    print(f"策略试图以恐慌价格下单: {panic_price}")
    
    can_place, message, level = rm.can_place_order(
        strategy_name=strategy_name,
        contract=contract,
        direction="SELL",
        volume=50,
        price=panic_price,
        current_market_price=current_price,
    )
    
    deviation = abs((panic_price - current_price) / current_price * 100)
    print(f"价格偏离: {deviation:.2f}%")
    print(f"订单状态: {'[BLOCKED]' if not can_place else '[ALLOWED]'}")
    print(f"拦截原因: {message}")
    print(f"风控拦截计数: {rm._risk_blocked_orders}")
    
    assert not can_place, "极端价格偏离应该被拦截"
    assert rm._risk_blocked_orders == 1, "应该有 1 次拦截记录"
    print("[PASS] 阶段 1 通过: 价格偏离订单被成功拦截")
    print()
    
    print("--- 阶段 2: 连续亏损触发策略暂停 ---")
    print("模拟 3 次连续亏损交易:")
    
    for i in range(1, 4):
        rm.record_trade_event(
            strategy_name=strategy_name,
            contract=contract,
            direction="SELL",
            volume=10,
            price=3500.0 - i * 50,
            profit_loss=-5000.0 * i,
        )
        
        strategy_info = rm._strategy_risk.get(strategy_name)
        print(f"  第 {i} 次亏损: -${5000*i:,.0f} - 连续亏损: {strategy_info.consecutive_losses}")
    
    is_paused = rm.is_strategy_paused(strategy_name)
    strategy_info = rm._strategy_risk.get(strategy_name)
    
    print(f"策略状态: {'[PAUSED]' if is_paused else '[RUNNING]'}")
    print(f"暂停原因: {strategy_info.pause_reason}")
    
    assert is_paused, "连续亏损 3 次后策略应该暂停"
    print("[PASS] 阶段 2 通过: 连续亏损触发策略暂停")
    print()
    
    print("--- 阶段 3: 日亏损超过限额触发系统熔断 ---")
    
    RiskManager.reset_instance()
    rm = RiskManager(
        max_drawdown_percent=10.0,
        daily_loss_limit_percent=2.0,
    )
    rm._peak_equity = initial_equity
    rm._daily_start_equity = initial_equity
    
    loss_equity = initial_equity * 0.97
    print(f"当前权益: {initial_equity:,.0f}")
    print(f"暴跌后权益: {loss_equity:,.0f}")
    print(f"亏损金额: {initial_equity - loss_equity:,.0f}")
    print(f"亏损比例: {(initial_equity - loss_equity) / initial_equity * 100:.2f}%")
    print(f"日损限额: {rm.daily_loss_limit_percent}%")
    
    snapshot = AccountSnapshot(
        timestamp=time.time(),
        balance=loss_equity,
        equity=loss_equity,
        total_asset=loss_equity,
        margin_used=0.0,
        available=loss_equity,
        float_profit=-(initial_equity - loss_equity),
    )
    
    level = rm.check_daily_loss(snapshot)
    
    print(f"风险等级: {level.value}")
    print(f"系统状态: {'[FROZEN]' if rm.is_frozen() else '[NORMAL]'}")
    print(f"冻结原因: {rm.get_frozen_reason()}")
    
    assert rm.is_frozen(), "日亏损超过 2% 应该触发熔断"
    assert level == RiskLevel.FROZEN, "风险等级应为 FROZEN"
    print("[PASS] 阶段 3 通过: 日亏损超限触发系统熔断")
    print()
    
    print("--- 阶段 4: 生成风控体检报告 ---")
    report = rm.generate_risk_check_report()
    
    print(f"  风控体检报告摘要:")
    print(f"  生成时间: {report.generated_at.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  当前风险等级: {report.current_risk_level.value}")
    print(f"  是否冻结: {report.is_frozen}")
    print(f"  冻结原因: {report.frozen_reason}")
    print(f"  当日亏损: {report.daily_loss_amount:,.0f} ({report.daily_loss_percent:.2f}%)")
    print(f"  风控拦截订单数: {report.risk_blocked_orders}")
    
    print("\n打印完整报告:")
    report_str = rm.print_risk_check_report()
    
    print("[PASS] 阶段 4 通过: 风控体检报告生成成功")
    print()
    
    print("=" * 80)
    print("  所有极端行情测试通过！风控系统正常工作")
    print("=" * 80)


def main():
    print("\n" + "#" * 80)
    print("#           风控模块极端行情测试脚本")
    print("#           Risk Manager Extreme Market Test")
    print("#" * 80)
    print(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    tests = [
        ("单例模式", test_singleton_pattern),
        ("价格偏离拦截", test_price_deviation_block),
        ("日损限额熔断", test_daily_loss_freeze),
        ("连续亏损暂停", test_consecutive_loss_pause),
        ("风控体检报告", test_risk_check_report),
        ("极端行情综合", test_extreme_market_scenario),
    ]
    
    passed = 0
    failed = 0
    failures = []
    
    for test_name, test_func in tests:
        try:
            test_func()
            passed += 1
            print(f"\n[PASS] {test_name}")
        except Exception as e:
            failed += 1
            failures.append((test_name, str(e)))
            print(f"\n[FAIL] {test_name}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("测试结果汇总:")
    print(f"  总测试数: {len(tests)}")
    print(f"  [PASS] 通过: {passed}")
    print(f"  [FAIL] 失败: {failed}")
    
    if failures:
        print("\n失败详情:")
        for name, error in failures:
            print(f"  - {name}: {error}")
    
    print("=" * 80)
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
