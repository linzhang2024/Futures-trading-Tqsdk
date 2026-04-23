import sys
import os
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import yaml
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

from core.backtest import BacktestEngine, ParameterRange
from strategies.double_ma_strategy import DoubleMAStrategy


def load_config(config_path: str = None) -> dict:
    if config_path is None:
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'config', 'settings.yaml'
        )
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config


def run_high_slippage_backtest():
    print("=" * 80)
    print("           实机复盘演练：高成本环境下的风控测试")
    print("=" * 80)
    print(f"\n测试目的:")
    print(f"  - 设置极高的手续费，模拟极端恶劣的交易环境")
    print(f"  - 验证风控模块是否能在极端行情下识别风险并触发熔断")
    print(f"  - 测试策略在高成本环境下的表现")
    
    config = load_config()
    
    print(f"\n【正常配置】")
    print(f"  最大回撤限制: {config['risk']['max_drawdown_percent']}%")
    print(f"  默认手续费: 5元/手")
    
    high_cost_config = dict(config)
    high_cost_config['risk']['max_drawdown_percent'] = 3.0
    
    high_cost_config['backtest']['costs'] = {
        'default_commission_per_lot': 100.0,
        'default_slippage_points': 50.0,
    }
    
    print(f"\n【测试配置】")
    print(f"  最大回撤限制: {high_cost_config['risk']['max_drawdown_percent']}% (更严格)")
    print(f"  手续费: 100元/手 (极端高)")
    print(f"  滑点: 50点 (极端高 - 每手约500元损失)")
    print(f"  合约: SHFE.rb2410")
    print(f"  回测区间: 2024-01-01 至 2024-03-31")
    
    engine = BacktestEngine(config=high_cost_config)
    
    print(f"\n【策略参数】")
    print(f"  快线周期: 5")
    print(f"  慢线周期: 20")
    print(f"  K线周期: 60秒")
    
    print(f"\n{'='*80}")
    print("开始回测...")
    print("=" * 80)
    
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
            end_dt=date(2024, 1, 10),
        )
        
        print(f"\n{'='*80}")
        print("                    回测结果")
        print("=" * 80)
        
        print(f"\n【基本信息】")
        print(f"  策略名称: {result.strategy_name}")
        print(f"  回测区间: {result.start_dt} 至 {result.end_dt}")
        print(f"  运行状态: {result.status}")
        
        p = result.performance
        print(f"\n【收益指标】")
        print(f"  初始权益: {result.initial_equity:,.2f}")
        print(f"  最终权益: {result.final_equity:,.2f}")
        print(f"  总收益: {p.total_return:,.2f}")
        print(f"  总收益率: {p.total_return_percent:.2f}%")
        
        print(f"\n【风险指标】")
        print(f"  最大回撤: {p.max_drawdown:,.2f}")
        print(f"  最大回撤率: {p.max_drawdown_percent:.2f}%")
        
        print(f"\n【交易统计】")
        print(f"  总交易次数: {p.total_trades}")
        
        print(f"\n【成本统计】")
        print(f"  总手续费: {p.total_commission_cost:,.2f}")
        print(f"  总成本: {p.total_cost:,.2f}")
        
        print(f"\n【风控状态】")
        print(f"  风控触发: {'是' if result.risk_triggered else '否'}")
        print(f"  期间冻结: {'是' if result.frozen_during_backtest else '否'}")
        if result.frozen_reason:
            print(f"  冻结原因: {result.frozen_reason}")
        
        if result.risk_events:
            print(f"\n【风险事件记录】")
            for i, event in enumerate(result.risk_events[:5], 1):
                print(f"  {i}. {event}")
        
        print(f"\n{'='*80}")
        print("                          测试结论")
        print("=" * 80)
        
        if result.frozen_during_backtest:
            print("\n✅ 测试通过：风控模块成功触发了冻结！")
            print(f"   冻结原因: {result.frozen_reason}")
            print(f"   这说明在极端高成本环境下，风控模块能够有效保护账户。")
        else:
            print("\n⚠️  测试结果：风控模块未触发冻结")
            if p.total_trades == 0:
                print("   - 策略在测试期间没有产生任何交易信号")
                print("   - 建议延长回测时间或调整策略参数以产生交易")
            else:
                print(f"   - 交易次数: {p.total_trades}")
                print(f"   - 最大回撤: {p.max_drawdown_percent:.2f}%")
                print(f"   - 回撤限制: {high_cost_config['risk']['max_drawdown_percent']}%")
                if p.max_drawdown_percent >= high_cost_config['risk']['max_drawdown_percent']:
                    print("   - 回撤已超过限制，但未触发冻结，可能存在问题")
                else:
                    print("   - 回撤未超过限制，属于正常情况")
        
        if result.error_message:
            print(f"\n❌ 错误信息: {result.error_message}")
        
    except Exception as e:
        print(f"\n❌ 回测执行出错: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("                    回测报告生成中...")
    print("=" * 80)
    
    try:
        report = engine.generate_report()
        print(f"\n报告已生成:")
        print(f"  - 总回测次数: {report['total_backtests']}")
    except Exception as e:
        print(f"\n报告生成失败: {e}")


def run_normal_backtest_comparison():
    print("\n" + "=" * 80)
    print("           对比测试：正常成本环境下的回测")
    print("=" * 80)
    
    config = load_config()
    
    normal_config = dict(config)
    normal_config['risk']['max_drawdown_percent'] = 10.0
    normal_config['backtest']['costs'] = {
        'default_commission_per_lot': 5.0,
        'default_slippage_points': 1.0,
    }
    
    print(f"\n【正常配置】")
    print(f"  最大回撤限制: {normal_config['risk']['max_drawdown_percent']}%")
    print(f"  手续费: 5元/手")
    print(f"  滑点: 1点")
    
    engine = BacktestEngine(config=normal_config)
    
    print(f"\n开始正常成本回测 (2024-01-02 至 2024-01-10)...")
    
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
            end_dt=date(2024, 1, 10),
        )
        
        p = result.performance
        print(f"\n【正常成本回测结果】")
        print(f"  初始权益: {result.initial_equity:,.2f}")
        print(f"  最终权益: {result.final_equity:,.2f}")
        print(f"  总收益率: {p.total_return_percent:.2f}%")
        print(f"  最大回撤率: {p.max_drawdown_percent:.2f}%")
        print(f"  交易次数: {p.total_trades}")
        print(f"  风控冻结: {'是' if result.frozen_during_backtest else '否'}")
        
    except Exception as e:
        print(f"\n正常成本回测出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_high_slippage_backtest()
    run_normal_backtest_comparison()
