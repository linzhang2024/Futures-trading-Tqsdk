import sys
import os
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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


def run_optimization_demo():
    print("\n" + "=" * 80)
    print("                    参数寻优演示")
    print("=" * 80)
    print(f"\n功能说明:")
    print(f"  - 使用参数网格寻优找到最优参数组合")
    print(f"  - 自动生成可视化分析图表")
    print(f"  - 展示最优参数组合及性能指标")
    print(f"  - 提供一键同步参数到配置文件的选项")
    
    config = load_config()
    
    optimization_config = {
        'short_period': ParameterRange(
            name='short_period',
            min_val=2,
            max_val=5,
            step=1,
            param_type=int
        ),
        'long_period': ParameterRange(
            name='long_period',
            min_val=8,
            max_val=15,
            step=2,
            param_type=int
        ),
        'rsi_period': ParameterRange(
            name='rsi_period',
            min_val=7,
            max_val=21,
            step=7,
            param_type=int
        ),
        'rsi_threshold': ParameterRange(
            name='rsi_threshold',
            min_val=45,
            max_val=55,
            step=5,
            param_type=float
        ),
    }
    
    print(f"\n【寻优参数范围】")
    print(f"  短期均线 (short_period): 2, 3, 4, 5")
    print(f"  长期均线 (long_period): 8, 10, 12, 14")
    print(f"  RSI 周期 (rsi_period): 7, 14, 21")
    print(f"  RSI 阈值 (rsi_threshold): 45, 50, 55")
    print(f"  合约: SHFE.rb2410")
    print(f"  回测区间: 2024-01-02 至 2024-02-01 (30天)")
    
    base_params = {
        'contract': 'SHFE.rb2410',
        'kline_duration': 60,
        'use_ema': False,
        'use_rsi_filter': True,
    }
    
    engine = BacktestEngine(config=config)
    
    print(f"\n{'='*80}")
    print("开始参数寻优...")
    print("=" * 80)
    
    try:
        results = engine.run_optimization(
            strategy_class=DoubleMAStrategy,
            param_ranges=optimization_config,
            base_params=base_params,
            start_dt=date(2024, 1, 2),
            end_dt=date(2024, 2, 1),
            optimize_by='total_return_percent',
        )
        
        print(f"\n{'='*80}")
        print("                    寻优结果汇总")
        print("=" * 80)
        
        print(f"\n【测试统计】")
        print(f"  总测试组合数: {len(results)}")
        print(f"  成功完成数: {len([r for r in results if r.status == 'completed'])}")
        
        best_result = engine.get_best_result()
        
        if best_result:
            print(f"\n{'='*80}")
            print("                    最优参数组合")
            print("=" * 80)
            
            print(f"\n【参数配置】")
            for param_name, param_value in best_result.params.items():
                print(f"  {param_name}: {param_value}")
            
            p = best_result.performance
            print(f"\n【核心性能指标】")
            print(f"  总收益率: {p.total_return_percent:.2f}%")
            print(f"  夏普比率: {p.sharpe_ratio:.2f}")
            print(f"  最大回撤率: {p.max_drawdown_percent:.2f}%")
            
            print(f"\n【详细收益指标】")
            print(f"  初始权益: {best_result.initial_equity:,.2f}")
            print(f"  最终权益: {best_result.final_equity:,.2f}")
            print(f"  总收益: {p.total_return:,.2f}")
            print(f"  年化收益率: {p.annualized_return_percent:.2f}%")
            
            print(f"\n【风险调整指标】")
            print(f"  索提诺比率: {p.sortino_ratio:.2f}")
            print(f"  卡尔玛比率: {p.calmar_ratio:.2f}")
            print(f"  最大回撤金额: {p.max_drawdown:,.2f}")
            
            print(f"\n【交易统计】")
            print(f"  总交易次数: {p.total_trades}")
            print(f"  胜率: {p.win_rate:.2f}%")
            print(f"  盈亏比: {p.profit_factor:.2f}")
            print(f"  平均每笔收益: {p.avg_trade_return:,.2f}")
            
            print(f"\n【成本统计】")
            print(f"  总手续费: {p.total_commission_cost:,.2f}")
            print(f"  总滑点成本: {p.total_slippage_cost:,.2f}")
            print(f"  总成本: {p.total_cost:,.2f}")
            
            print(f"\n{'='*80}")
            print("                    风控拦截统计")
            print("=" * 80)
            
            print_risk_events(best_result)
            
            print(f"\n{'='*80}")
            print("                    报告生成")
            print("=" * 80)
            
            report = engine.generate_report()
            
            if 'chart_path' in report:
                print(f"\n✅ 可视化图表已生成:")
                print(f"   路径: {report['chart_path']}")
                print(f"   图表包含:")
                print(f"   - 收益率热力图")
                print(f"   - 最大回撤热力图")
                print(f"   - 收益-风险散点图")
                print(f"   - Top参数组合排名")
            
            print(f"\n{'='*80}")
            print("                    参数同步选项")
            print("=" * 80)
            
            print(f"\n当前配置文件中的策略参数:")
            strategies = config.get('strategies', [])
            for i, s in enumerate(strategies):
                print(f"\n  策略 [{i+1}]: {s.get('name', 'Unknown')}")
                print(f"    类名: {s.get('class', 'Unknown')}")
                params = s.get('params', {})
                for pn, pv in params.items():
                    print(f"    {pn}: {pv}")
            
            print(f"\n是否要将最优参数同步到配置文件?")
            print(f"  - 这将更新 config/settings.yaml 中的 strategies 段")
            print(f"  - 参数映射: short_period -> fast, long_period -> slow")
            print(f"  - RSI 参数: rsi_period, rsi_threshold, use_rsi_filter")
            
            try:
                user_input = input("\n请输入 'yes' 确认同步，或其他键跳过: ").strip().lower()
                
                if user_input == 'yes':
                    print(f"\n正在同步参数...")
                    success = engine.sync_optimal_params_to_config(
                        param_mapping={
                            'short_period': 'fast',
                            'long_period': 'slow',
                        }
                    )
                    if success:
                        print(f"\n✅ 参数同步成功!")
                    else:
                        print(f"\n❌ 参数同步失败，请查看日志")
                else:
                    print(f"\n跳过参数同步")
            except EOFError:
                print(f"\n跳过参数同步 (非交互式环境)")
        
        else:
            print("\n⚠️  未找到有效的最优结果")
            
    except Exception as e:
        print(f"\n❌ 参数寻优执行出错: {e}")
        import traceback
        traceback.print_exc()


def print_risk_events(result):
    risk_events = result.risk_events
    
    if not risk_events:
        print(f"\n  ✅ 回测期间无风险事件触发")
        print(f"  - 风控状态: 正常")
        print(f"  - 期间冻结: {'是' if result.frozen_during_backtest else '否'}")
        return
    
    print(f"\n  风险事件总数: {len(risk_events)}")
    print(f"  期间冻结: {'是' if result.frozen_during_backtest else '否'}")
    
    if result.frozen_reason:
        print(f"  冻结原因: {result.frozen_reason}")
    
    event_types = {}
    for event in risk_events:
        event_type = event.get('event_type', 'Unknown') if isinstance(event, dict) else str(event)
        event_types[event_type] = event_types.get(event_type, 0) + 1
    
    print(f"\n  按类型统计:")
    for event_type, count in event_types.items():
        print(f"    - {event_type}: {count} 次")
    
    print(f"\n  最近5条风险事件:")
    recent_events = risk_events[-5:] if len(risk_events) > 5 else risk_events
    for i, event in enumerate(recent_events, 1):
        if isinstance(event, dict):
            event_str = (
                f"类型: {event.get('event_type', 'N/A')}, "
                f"级别: {event.get('risk_level', 'N/A')}, "
                f"描述: {event.get('description', 'N/A')[:50]}"
            )
        else:
            event_str = str(event)[:80]
        print(f"    {i}. {event_str}")


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
        print(f"  夏普比率: {p.sharpe_ratio:.2f}")
        
        print(f"\n【交易统计】")
        print(f"  总交易次数: {p.total_trades}")
        
        print(f"\n【成本统计】")
        print(f"  总手续费: {p.total_commission_cost:,.2f}")
        print(f"  总成本: {p.total_cost:,.2f}")
        
        print(f"\n【风控拦截统计】")
        print_risk_events(result)
        
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
        print(f"  夏普比率: {p.sharpe_ratio:.2f}")
        print(f"  最大回撤率: {p.max_drawdown_percent:.2f}%")
        print(f"  交易次数: {p.total_trades}")
        print(f"  风控冻结: {'是' if result.frozen_during_backtest else '否'}")
        
    except Exception as e:
        print(f"\n正常成本回测出错: {e}")
        import traceback
        traceback.print_exc()


def run_rsi_filter_comparison():
    print("\n" + "=" * 80)
    print("           RSI 过滤效果对比测试")
    print("=" * 80)
    
    print(f"\n测试目的:")
    print(f"  - 对比 RSI 过滤开启前后的策略表现")
    print(f"  - 验证 RSI 过滤是否能有效减少交易次数并提升胜率")
    print(f"  - 对比夏普比率和最终权益的变化")
    
    config = load_config()
    
    test_config = dict(config)
    test_config['risk']['max_drawdown_percent'] = 10.0
    test_config['backtest']['costs'] = {
        'default_commission_per_lot': 5.0,
        'default_slippage_points': 1.0,
    }
    
    print(f"\n【测试配置】")
    print(f"  最大回撤限制: {test_config['risk']['max_drawdown_percent']}%")
    print(f"  手续费: 5元/手")
    print(f"  滑点: 1点")
    print(f"  合约: SHFE.rb2410")
    print(f"  回测区间: 2024-01-02 至 2024-02-01 (30天)")
    print(f"  RSI 周期: 14")
    print(f"  RSI 阈值: 50")
    print(f"  短期均线周期: 5")
    print(f"  长期均线周期: 20")
    
    print(f"\n{'='*80}")
    print("开始对比测试...")
    print("=" * 80)
    
    results_no_rsi = None
    results_with_rsi = None
    
    try:
        engine_no_rsi = BacktestEngine(config=test_config)
        
        print(f"\n【测试 1: 无 RSI 过滤】")
        result_no_rsi = engine_no_rsi.run_backtest(
            strategy_class=DoubleMAStrategy,
            strategy_params={
                'short_period': 5,
                'long_period': 20,
                'contract': 'SHFE.rb2410',
                'kline_duration': 60,
                'use_ema': False,
                'use_rsi_filter': False,
            },
            start_dt=date(2024, 1, 2),
            end_dt=date(2024, 2, 1),
        )
        
        engine_with_rsi = BacktestEngine(config=test_config)
        
        print(f"\n【测试 2: 有 RSI 过滤】")
        result_with_rsi = engine_with_rsi.run_backtest(
            strategy_class=DoubleMAStrategy,
            strategy_params={
                'short_period': 5,
                'long_period': 20,
                'contract': 'SHFE.rb2410',
                'kline_duration': 60,
                'use_ema': False,
                'use_rsi_filter': True,
                'rsi_period': 14,
                'rsi_threshold': 50.0,
            },
            start_dt=date(2024, 1, 2),
            end_dt=date(2024, 2, 1),
        )
        
        p_no_rsi = result_no_rsi.performance
        p_with_rsi = result_with_rsi.performance
        
        print(f"\n{'='*80}")
        print("                    对比测试结果")
        print("=" * 80)
        
        print(f"\n【收益对比】")
        print(f"  {'指标':<20} {'无RSI过滤':<20} {'有RSI过滤':<20} {'变化':<15}")
        print(f"  {'-'*75}")
        
        final_eq_no = result_no_rsi.final_equity
        final_eq_with = result_with_rsi.final_equity
        eq_change = ((final_eq_with - final_eq_no) / final_eq_no * 100) if final_eq_no > 0 else 0
        
        print(f"  {'最终权益':<20} {final_eq_no:,.2f}{'':<10} {final_eq_with:,.2f}{'':<10} {eq_change:+.2f}%")
        
        return_no = p_no_rsi.total_return_percent
        return_with = p_with_rsi.total_return_percent
        return_change = return_with - return_no
        print(f"  {'总收益率(%)':<20} {return_no:.2f}%{'':<12} {return_with:.2f}%{'':<12} {return_change:+.2f}%")
        
        sharpe_no = p_no_rsi.sharpe_ratio
        sharpe_with = p_with_rsi.sharpe_ratio
        sharpe_change = sharpe_with - sharpe_no
        print(f"  {'夏普比率':<20} {sharpe_no:.2f}{'':<15} {sharpe_with:.2f}{'':<15} {sharpe_change:+.2f}")
        
        print(f"\n【风险对比】")
        max_dd_no = p_no_rsi.max_drawdown_percent
        max_dd_with = p_with_rsi.max_drawdown_percent
        max_dd_change = max_dd_with - max_dd_no
        print(f"  {'最大回撤率(%)':<20} {max_dd_no:.2f}%{'':<12} {max_dd_with:.2f}%{'':<12} {max_dd_change:+.2f}%")
        
        sortino_no = p_no_rsi.sortino_ratio
        sortino_with = p_with_rsi.sortino_ratio
        sortino_change = sortino_with - sortino_no
        print(f"  {'索提诺比率':<20} {sortino_no:.2f}{'':<15} {sortino_with:.2f}{'':<15} {sortino_change:+.2f}")
        
        print(f"\n【交易统计对比】")
        trades_no = p_no_rsi.total_trades
        trades_with = p_with_rsi.total_trades
        trades_reduction = ((trades_no - trades_with) / trades_no * 100) if trades_no > 0 else 0
        
        if trades_reduction > 0:
            trades_reduction_str = f"-{trades_reduction:.1f}%"
        else:
            trades_reduction_str = "0%"
        
        print(f"  总交易次数            {trades_no:<20} {trades_with:<20} {trades_reduction_str:<15}")
        
        win_rate_no = p_no_rsi.win_rate
        win_rate_with = p_with_rsi.win_rate
        win_rate_change = win_rate_with - win_rate_no
        print(f"  胜率(%)              {win_rate_no:.2f}%{'':<12} {win_rate_with:.2f}%{'':<12} {win_rate_change:+.2f}%")
        
        profit_factor_no = p_no_rsi.profit_factor
        profit_factor_with = p_with_rsi.profit_factor
        profit_factor_change = profit_factor_with - profit_factor_no
        print(f"  盈亏比                {profit_factor_no:.2f}{'':<15} {profit_factor_with:.2f}{'':<15} {profit_factor_change:+.2f}")
        
        avg_trade_no = p_no_rsi.avg_trade_return
        avg_trade_with = p_with_rsi.avg_trade_return
        avg_trade_change = avg_trade_with - avg_trade_no if avg_trade_no is not None and avg_trade_with is not None else 0
        print(f"  平均每笔收益          {avg_trade_no:,.2f}{'':<10} {avg_trade_with:,.2f}{'':<10} {avg_trade_change:+.2f}")
        
        print(f"\n{'='*80}")
        print("                    测试结论")
        print("=" * 80)
        
        conclusions = []
        
        if trades_with < trades_no:
            conclusions.append(f"✅ RSI 过滤成功减少交易次数: {trades_no} -> {trades_with} (减少 {trades_reduction:.1f}%)")
        else:
            conclusions.append(f"⚠️ RSI 过滤未减少交易次数")
        
        if sharpe_with > sharpe_no:
            conclusions.append(f"✅ 夏普比率提升: {sharpe_no:.2f} -> {sharpe_with:.2f}")
        elif sharpe_with < sharpe_no:
            conclusions.append(f"⚠️ 夏普比率下降: {sharpe_no:.2f} -> {sharpe_with:.2f}")
        
        if final_eq_with > final_eq_no:
            conclusions.append(f"✅ 最终权益提升: {final_eq_no:,.2f} -> {final_eq_with:,.2f}")
        elif final_eq_with < final_eq_no:
            conclusions.append(f"⚠️ 最终权益下降: {final_eq_no:,.2f} -> {final_eq_with:,.2f}")
        
        if win_rate_with > win_rate_no:
            conclusions.append(f"✅ 胜率提升: {win_rate_no:.2f}% -> {win_rate_with:.2f}%")
        
        print("\n" + "\n".join(conclusions))
        
        print(f"\n【关键指标总结】")
        print(f"  交易次数下降比例: {trades_reduction:.1f}%")
        print(f"  夏普比率变化: {sharpe_change:+.2f}")
        print(f"  最终权益变化: {eq_change:+.2f}%")
        
    except Exception as e:
        print(f"\n❌ 对比测试执行出错: {e}")
        import traceback
        traceback.print_exc()


def print_menu():
    print("\n" + "=" * 80)
    print("                    回测模块演示菜单")
    print("=" * 80)
    print(f"\n请选择要运行的演示:")
    print(f"\n  1. 参数寻优演示 (推荐)")
    print(f"     - 功能: 自动寻优最优参数组合 (含 RSI 参数)")
    print(f"     - 输出: 最优参数、夏普比率、最大回撤、风控统计")
    print(f"     - 可视化: 自动生成收益热力图和资金曲线图表")
    print(f"     - 同步: 提供一键同步参数到配置文件的选项")
    print(f"\n  2. 高成本风控测试")
    print(f"     - 功能: 测试极端高成本环境下的风控模块")
    print(f"     - 输出: 风控冻结状态、风险事件统计")
    print(f"\n  3. 正常成本对比测试")
    print(f"     - 功能: 正常成本环境下的回测对比")
    print(f"     - 输出: 收益指标、风险指标")
    print(f"\n  4. RSI 过滤效果对比测试")
    print(f"     - 功能: 对比 RSI 过滤开启前后的策略表现")
    print(f"     - 输出: 交易次数变化、夏普比率变化、最终权益变化")
    print(f"\n  5. 运行全部演示")
    print(f"     - 顺序执行所有演示")
    print(f"\n  0. 退出")
    print("\n" + "=" * 80)


def main():
    while True:
        print_menu()
        
        try:
            choice = input("\n请输入选项 (0-5): ").strip()
            
            if choice == '0':
                print("\n感谢使用，再见！")
                break
            elif choice == '1':
                run_optimization_demo()
            elif choice == '2':
                run_high_slippage_backtest()
            elif choice == '3':
                run_normal_backtest_comparison()
            elif choice == '4':
                run_rsi_filter_comparison()
            elif choice == '5':
                run_optimization_demo()
                run_high_slippage_backtest()
                run_normal_backtest_comparison()
                run_rsi_filter_comparison()
            else:
                print(f"\n无效选项: {choice}，请重新输入")
                continue
            
            print(f"\n{'='*80}")
            print("                    演示完成")
            print("=" * 80)
            
            input("\n按回车键返回菜单...")
            
        except EOFError:
            print("\n非交互式环境，运行参数寻优演示...")
            run_optimization_demo()
            break
        except KeyboardInterrupt:
            print("\n\n用户中断，退出程序")
            break


if __name__ == "__main__":
    main()
