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


def run_multi_contract_optimization():
    print("\n" + "=" * 120)
    print("                    多维参数矩阵寻优与多合约交叉验证")
    print("=" * 120)
    print(f"\n功能说明:")
    print(f"  - 多维参数网格寻优（均线、RSI、止盈止损）")
    print(f"  - 多合约交叉验证（螺纹钢、热卷）")
    print(f"  - 稳定性评分（收益率/最大回撤）")
    print(f"  - 自动输出 Top 3 组合并提供同步选项")
    
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
    
    tp_sl_options = [
        {'take_profit_ratio': None, 'stop_loss_ratio': None},
        {'take_profit_ratio': 0.01, 'stop_loss_ratio': 0.01},
        {'take_profit_ratio': 0.02, 'stop_loss_ratio': 0.02},
        {'take_profit_ratio': 0.03, 'stop_loss_ratio': 0.03},
    ]
    
    contracts = ['SHFE.rb2410', 'SHFE.hc2410']
    
    total_short = int((optimization_config['short_period'].max_val - optimization_config['short_period'].min_val) / optimization_config['short_period'].step) + 1
    total_long = int((optimization_config['long_period'].max_val - optimization_config['long_period'].min_val) / optimization_config['long_period'].step) + 1
    total_rsi_period = int((optimization_config['rsi_period'].max_val - optimization_config['rsi_period'].min_val) / optimization_config['rsi_period'].step) + 1
    total_rsi_threshold = int((optimization_config['rsi_threshold'].max_val - optimization_config['rsi_threshold'].min_val) / optimization_config['rsi_threshold'].step) + 1
    total_tp_sl = len(tp_sl_options)
    total_contracts = len(contracts)
    
    total_combinations = total_short * total_long * total_rsi_period * total_rsi_threshold * total_tp_sl * total_contracts
    
    print(f"\n【寻优参数范围】")
    print(f"  短期均线 (short_period): {[optimization_config['short_period'].min_val + i * optimization_config['short_period'].step for i in range(total_short)]}")
    print(f"  长期均线 (long_period): {[optimization_config['long_period'].min_val + i * optimization_config['long_period'].step for i in range(total_long)]}")
    print(f"  RSI 周期 (rsi_period): {[optimization_config['rsi_period'].min_val + i * optimization_config['rsi_period'].step for i in range(total_rsi_period)]}")
    print(f"  RSI 阈值 (rsi_threshold): {[optimization_config['rsi_threshold'].min_val + i * optimization_config['rsi_threshold'].step for i in range(total_rsi_threshold)]}")
    print(f"  止盈止损组合数: {total_tp_sl} 种")
    print(f"  合约: {contracts}")
    print(f"  回测区间: 2024-01-02 至 2024-04-02 (90天)")
    print(f"\n  总测试组合数: {total_combinations}")
    
    all_results = []
    
    for contract in contracts:
        print(f"\n{'='*120}")
        print(f"                    开始回测合约: {contract}")
        print("=" * 120)
        
        for tp_sl_config in tp_sl_options:
            tp_ratio = tp_sl_config['take_profit_ratio']
            sl_ratio = tp_sl_config['stop_loss_ratio']
            
            if tp_ratio is None and sl_ratio is None:
                tp_sl_desc = "无止盈止损"
            else:
                tp_sl_desc = f"止盈={tp_ratio*100:.0f}%, 止损={sl_ratio*100:.0f}%"
            
            print(f"\n  止盈止损配置: {tp_sl_desc}")
            
            base_params = {
                'contract': contract,
                'kline_duration': 60,
                'use_ema': False,
                'use_rsi_filter': True,
                'take_profit_ratio': tp_ratio,
                'stop_loss_ratio': sl_ratio,
            }
            
            engine = BacktestEngine(config=config)
            
            try:
                results = engine.run_optimization(
                    strategy_class=DoubleMAStrategy,
                    param_ranges=optimization_config,
                    base_params=base_params,
                    start_dt=date(2024, 1, 2),
                    end_dt=date(2024, 4, 2),
                    optimize_by='total_return_percent',
                )
                
                for r in results:
                    r.contract = contract
                    r.take_profit_ratio = tp_ratio
                    r.stop_loss_ratio = sl_ratio
                    all_results.append(r)
                
                print(f"    完成: {len(results)} 个参数组合")
                
            except Exception as e:
                print(f"    ❌ 回测出错: {e}")
                import traceback
                traceback.print_exc()
    
    print(f"\n{'='*120}")
    print("                    多合约寻优结果汇总")
    print("=" * 120)
    
    completed_results = [r for r in all_results if r.status == 'completed']
    print(f"\n【测试统计】")
    print(f"  总测试组合数: {len(all_results)}")
    print(f"  成功完成数: {len(completed_results)}")
    
    if not completed_results:
        print("\n❌ 没有成功完成的回测结果")
        return
    
    def calculate_stability_score(result):
        p = result.performance
        return_pct = p.total_return_percent
        max_dd_pct = p.max_drawdown_percent
        
        if max_dd_pct == 0:
            if return_pct > 0:
                return 100.0
            else:
                return 0.0
        
        stability_score = return_pct / abs(max_dd_pct)
        return stability_score
    
    for r in completed_results:
        r.stability_score = calculate_stability_score(r)
    
    def is_valid_result(result):
        p = result.performance
        has_trades = p.total_trades >= 1
        has_non_zero_return = abs(p.total_return_percent) > 0.0001
        return has_trades and has_non_zero_return
    
    def is_recommended_result(result):
        p = result.performance
        has_enough_trades = p.total_trades >= 3
        has_non_zero_return = abs(p.total_return_percent) > 0.0001
        return has_enough_trades and has_non_zero_return
    
    sorted_by_stability = sorted(completed_results, key=lambda x: x.stability_score, reverse=True)
    sorted_by_trades = sorted([r for r in completed_results if is_recommended_result(r)], 
                               key=lambda x: x.stability_score, reverse=True)
    
    valid_results = [r for r in completed_results if is_valid_result(r)]
    print(f"\n【有效性统计】")
    print(f"  有效结果数(有交易且非零收益): {len(valid_results)}")
    print(f"  推荐结果数(交易>=3且非零收益): {len(sorted_by_trades)}")
    
    print(f"\n{'='*120}")
    print("                    按稳定性评分排序 Top 10")
    print("  稳定性评分 = 总收益率(%) / 最大回撤率(%)")
    print("=" * 120)
    
    print(f"\n{'排名':<4} {'合约':<15} {'短期均线':<10} {'长期均线':<10} {'RSI周期':<10} {'RSI阈值':<10} {'止盈止损':<15} {'交易次数':<10} {'稳定性评分':<12} {'收益率(%)':<12} {'最大回撤(%)':<12}")
    print("-" * 120)
    
    for i, r in enumerate(sorted_by_stability[:10], 1):
        p = r.performance
        tp_sl_str = f"TP={r.take_profit_ratio*100:.0f}%,SL={r.stop_loss_ratio*100:.0f}%" if r.take_profit_ratio else "无TP/SL"
        print(f"{i:<4} {r.contract:<15} {r.params.get('short_period', '-'):<10} {r.params.get('long_period', '-'):<10} {r.params.get('rsi_period', '-'):<10} {r.params.get('rsi_threshold', '-'):<10} {tp_sl_str:<15} {p.total_trades:<10} {r.stability_score:<12.2f} {p.total_return_percent:<12.2f} {p.max_drawdown_percent:<12.2f}")
    
    print(f"\n{'='*120}")
    print("                    交易次数 >=3 的组合按稳定性评分排序 Top 3")
    print("=" * 120)
    
    if sorted_by_trades:
        print(f"\n{'排名':<4} {'合约':<15} {'短期均线':<10} {'长期均线':<10} {'RSI周期':<10} {'RSI阈值':<10} {'止盈止损':<15} {'交易次数':<10} {'稳定性评分':<12} {'收益率(%)':<12} {'最大回撤(%)':<12}")
        print("-" * 120)
        
        for i, r in enumerate(sorted_by_trades[:3], 1):
            p = r.performance
            tp_sl_str = f"TP={r.take_profit_ratio*100:.0f}%,SL={r.stop_loss_ratio*100:.0f}%" if r.take_profit_ratio else "无TP/SL"
            print(f"{i:<4} {r.contract:<15} {r.params.get('short_period', '-'):<10} {r.params.get('long_period', '-'):<10} {r.params.get('rsi_period', '-'):<10} {r.params.get('rsi_threshold', '-'):<10} {tp_sl_str:<15} {p.total_trades:<10} {r.stability_score:<12.2f} {p.total_return_percent:<12.2f} {p.max_drawdown_percent:<12.2f}")
    else:
        print("\n  ❌ 没有找到交易次数 >=3 的参数组合")
    
    print(f"\n{'='*120}")
    print("                    按合约汇总统计")
    print("=" * 120)
    
    for contract in contracts:
        contract_results = [r for r in completed_results if r.contract == contract]
        contract_with_trades = [r for r in contract_results if r.performance.total_trades >= 3]
        
        print(f"\n【{contract}】")
        print(f"  总测试组合数: {len(contract_results)}")
        print(f"  交易次数 >=3 的组合数: {len(contract_with_trades)}")
        
        if contract_with_trades:
            best_for_contract = sorted(contract_with_trades, key=lambda x: x.stability_score, reverse=True)[0]
            p = best_for_contract.performance
            print(f"  最优稳定性评分: {best_for_contract.stability_score:.2f}")
            print(f"  最优参数: 短期={best_for_contract.params.get('short_period')}, 长期={best_for_contract.params.get('long_period')}, RSI周期={best_for_contract.params.get('rsi_period')}, RSI阈值={best_for_contract.params.get('rsi_threshold')}")
            print(f"  止盈止损: TP={best_for_contract.take_profit_ratio*100:.0f}%, SL={best_for_contract.stop_loss_ratio*100:.0f}%" if best_for_contract.take_profit_ratio else "无止盈止损")
            print(f"  交易次数: {p.total_trades}")
            print(f"  收益率: {p.total_return_percent:.2f}%")
            print(f"  最大回撤: {p.max_drawdown_percent:.2f}%")
    
    print(f"\n{'='*120}")
    print("                    参数同步选项")
    print("=" * 120)
    
    if sorted_by_trades:
        top_candidates = sorted_by_trades[:3]
        print(f"\n  选择标准: 交易次数 >=3 且非零收益")
    elif valid_results:
        top_candidates = sorted(valid_results, key=lambda x: x.stability_score, reverse=True)[:3]
        print(f"\n  选择标准: 有交易且非零收益（回退模式）")
    else:
        print("\n  ⚠️  没有找到有效的结果（无交易或零收益）")
        print("  建议:")
        print("    1. 延长回测时间")
        print("    2. 调整策略参数（如 RSI 阈值、均线周期等）")
        print("    3. 检查模拟数据是否有足够的波动")
        return
    
    if top_candidates:
        print(f"\n【Top 3 候选参数组合】")
        for i, r in enumerate(top_candidates, 1):
            p = r.performance
            tp_sl_str = f"止盈={r.take_profit_ratio*100:.0f}%, 止损={r.stop_loss_ratio*100:.0f}%" if r.take_profit_ratio else "无止盈止损"
            print(f"\n  候选 {i}:")
            print(f"    合约: {r.contract}")
            print(f"    短期均线: {r.params.get('short_period')}")
            print(f"    长期均线: {r.params.get('long_period')}")
            print(f"    RSI周期: {r.params.get('rsi_period')}")
            print(f"    RSI阈值: {r.params.get('rsi_threshold')}")
            print(f"    止盈止损: {tp_sl_str}")
            print(f"    交易次数: {p.total_trades}")
            print(f"    稳定性评分: {r.stability_score:.2f}")
            print(f"    收益率: {p.total_return_percent:.2f}%")
            print(f"    最大回撤: {p.max_drawdown_percent:.2f}%")
            print(f"    夏普比率: {p.sharpe_ratio:.2f}")
        
        print(f"\n是否要将候选参数同步到配置文件?")
        print(f"  请选择要同步的候选 (1-{len(top_candidates)})，或输入 'no' 跳过:")
        
        try:
            user_input = input("\n请输入选择: ").strip().lower()
            
            if user_input == 'no':
                print("\n跳过参数同步")
            else:
                try:
                    choice = int(user_input)
                    if 1 <= choice <= len(top_candidates):
                        selected = top_candidates[choice - 1]
                        
                        print(f"\n正在同步候选 {choice} 的参数...")
                        
                        config_path = os.path.join(
                            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            'config', 'settings.yaml'
                        )
                        
                        with open(config_path, 'r', encoding='utf-8') as f:
                            current_config = yaml.safe_load(f)
                        
                        strategies = current_config.get('strategies', [])
                        
                        updated = False
                        for s in strategies:
                            if s.get('class') == 'DoubleMAStrategy':
                                s['params']['fast'] = selected.params.get('short_period', 5)
                                s['params']['slow'] = selected.params.get('long_period', 20)
                                s['params']['rsi_period'] = selected.params.get('rsi_period', 14)
                                s['params']['rsi_threshold'] = selected.params.get('rsi_threshold', 50.0)
                                s['params']['use_rsi_filter'] = True
                                if selected.take_profit_ratio:
                                    s['params']['take_profit_ratio'] = selected.take_profit_ratio
                                if selected.stop_loss_ratio:
                                    s['params']['stop_loss_ratio'] = selected.stop_loss_ratio
                                updated = True
                                break
                        
                        if not updated and strategies:
                            strategies[0]['params']['fast'] = selected.params.get('short_period', 5)
                            strategies[0]['params']['slow'] = selected.params.get('long_period', 20)
                            strategies[0]['params']['rsi_period'] = selected.params.get('rsi_period', 14)
                            strategies[0]['params']['rsi_threshold'] = selected.params.get('rsi_threshold', 50.0)
                            strategies[0]['params']['use_rsi_filter'] = True
                        
                        with open(config_path, 'w', encoding='utf-8') as f:
                            yaml.dump(current_config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
                        
                        print(f"\n✅ 参数同步成功!")
                        print(f"  同步的参数:")
                        print(f"    短期均线 (fast): {selected.params.get('short_period')}")
                        print(f"    长期均线 (slow): {selected.params.get('long_period')}")
                        print(f"    RSI周期: {selected.params.get('rsi_period')}")
                        print(f"    RSI阈值: {selected.params.get('rsi_threshold')}")
                        tp_sl_str = f"止盈={selected.take_profit_ratio*100:.0f}%, 止损={selected.stop_loss_ratio*100:.0f}%" if selected.take_profit_ratio else "无"
                        print(f"    止盈止损: {tp_sl_str}")
                    else:
                        print(f"\n无效选择: {choice}，跳过同步")
                except ValueError:
                    print(f"\n无效输入，跳过同步")
                    
        except EOFError:
            print(f"\n跳过参数同步 (非交互式环境)")
    else:
        print("\n❌ 没有候选参数组合可同步")


def run_optimization_demo():
    run_multi_contract_optimization()


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
    
    contract = 'DCE.i2409'
    
    print(f"\n【测试配置】")
    print(f"  最大回撤限制: {test_config['risk']['max_drawdown_percent']}%")
    print(f"  手续费: 5元/手")
    print(f"  滑点: 1点")
    print(f"  合约: {contract} (铁矿石 - 波动率更高)")
    print(f"  回测区间: 2024-01-02 至 2024-04-02 (90天)")
    print(f"  RSI 周期: 7")
    print(f"  RSI 阈值: 50")
    print(f"  短期均线周期: 2")
    print(f"  长期均线周期: 8")
    print(f"  止盈: 2%")
    print(f"  止损: 2%")
    
    print(f"\n{'='*80}")
    print("开始对比测试...")
    print("=" * 80)
    
    try:
        engine_no_rsi = BacktestEngine(config=test_config)
        
        print(f"\n【测试 1: 无 RSI 过滤 + 无止盈止损】")
        result_no_rsi_no_tpsl = engine_no_rsi.run_backtest(
            strategy_class=DoubleMAStrategy,
            strategy_params={
                'short_period': 2,
                'long_period': 8,
                'contract': contract,
                'kline_duration': 60,
                'use_ema': False,
                'use_rsi_filter': False,
            },
            start_dt=date(2024, 1, 2),
            end_dt=date(2024, 4, 2),
        )
        
        engine_no_rsi_tpsl = BacktestEngine(config=test_config)
        
        print(f"\n【测试 2: 无 RSI 过滤 + 有止盈止损】")
        result_no_rsi_with_tpsl = engine_no_rsi_tpsl.run_backtest(
            strategy_class=DoubleMAStrategy,
            strategy_params={
                'short_period': 2,
                'long_period': 8,
                'contract': contract,
                'kline_duration': 60,
                'use_ema': False,
                'use_rsi_filter': False,
                'take_profit_ratio': 0.02,
                'stop_loss_ratio': 0.02,
            },
            start_dt=date(2024, 1, 2),
            end_dt=date(2024, 4, 2),
        )
        
        engine_with_rsi = BacktestEngine(config=test_config)
        
        print(f"\n【测试 3: 有 RSI 过滤 + 有止盈止损】")
        result_with_rsi = engine_with_rsi.run_backtest(
            strategy_class=DoubleMAStrategy,
            strategy_params={
                'short_period': 2,
                'long_period': 8,
                'contract': contract,
                'kline_duration': 60,
                'use_ema': False,
                'use_rsi_filter': True,
                'rsi_period': 7,
                'rsi_threshold': 50.0,
                'take_profit_ratio': 0.02,
                'stop_loss_ratio': 0.02,
            },
            start_dt=date(2024, 1, 2),
            end_dt=date(2024, 4, 2),
        )
        
        p_no_rsi_no_tpsl = result_no_rsi_no_tpsl.performance
        p_no_rsi_with_tpsl = result_no_rsi_with_tpsl.performance
        p_with_rsi = result_with_rsi.performance
        
        print(f"\n{'='*80}")
        print("                    对比测试结果")
        print("=" * 80)
        
        print(f"\n【收益对比】")
        print(f"  {'指标':<20} {'无RSI无TP/SL':<20} {'无RSI有TP/SL':<20} {'有RSI有TP/SL':<20}")
        print(f"  {'-'*85}")
        
        final_eq_1 = result_no_rsi_no_tpsl.final_equity
        final_eq_2 = result_no_rsi_with_tpsl.final_equity
        final_eq_3 = result_with_rsi.final_equity
        
        print(f"  {'最终权益':<20} {final_eq_1:,.2f}{'':<10} {final_eq_2:,.2f}{'':<10} {final_eq_3:,.2f}")
        
        return_1 = p_no_rsi_no_tpsl.total_return_percent
        return_2 = p_no_rsi_with_tpsl.total_return_percent
        return_3 = p_with_rsi.total_return_percent
        print(f"  {'总收益率(%)':<20} {return_1:.2f}%{'':<12} {return_2:.2f}%{'':<12} {return_3:.2f}%")
        
        sharpe_1 = p_no_rsi_no_tpsl.sharpe_ratio
        sharpe_2 = p_no_rsi_with_tpsl.sharpe_ratio
        sharpe_3 = p_with_rsi.sharpe_ratio
        print(f"  {'夏普比率':<20} {sharpe_1:.2f}{'':<15} {sharpe_2:.2f}{'':<15} {sharpe_3:.2f}")
        
        print(f"\n【风险对比】")
        max_dd_1 = p_no_rsi_no_tpsl.max_drawdown_percent
        max_dd_2 = p_no_rsi_with_tpsl.max_drawdown_percent
        max_dd_3 = p_with_rsi.max_drawdown_percent
        print(f"  {'最大回撤率(%)':<20} {max_dd_1:.2f}%{'':<12} {max_dd_2:.2f}%{'':<12} {max_dd_3:.2f}%")
        
        sortino_1 = p_no_rsi_no_tpsl.sortino_ratio
        sortino_2 = p_no_rsi_with_tpsl.sortino_ratio
        sortino_3 = p_with_rsi.sortino_ratio
        print(f"  {'索提诺比率':<20} {sortino_1:.2f}{'':<15} {sortino_2:.2f}{'':<15} {sortino_3:.2f}")
        
        print(f"\n【交易统计对比】")
        trades_1 = p_no_rsi_no_tpsl.total_trades
        trades_2 = p_no_rsi_with_tpsl.total_trades
        trades_3 = p_with_rsi.total_trades
        
        print(f"  总交易次数            {trades_1:<20} {trades_2:<20} {trades_3:<20}")
        
        win_rate_1 = p_no_rsi_no_tpsl.win_rate
        win_rate_2 = p_no_rsi_with_tpsl.win_rate
        win_rate_3 = p_with_rsi.win_rate
        print(f"  胜率(%)              {win_rate_1:.2f}%{'':<12} {win_rate_2:.2f}%{'':<12} {win_rate_3:.2f}%")
        
        profit_factor_1 = p_no_rsi_no_tpsl.profit_factor
        profit_factor_2 = p_no_rsi_with_tpsl.profit_factor
        profit_factor_3 = p_with_rsi.profit_factor
        print(f"  盈亏比                {profit_factor_1:.2f}{'':<15} {profit_factor_2:.2f}{'':<15} {profit_factor_3:.2f}")
        
        avg_trade_1 = p_no_rsi_no_tpsl.avg_trade_return
        avg_trade_2 = p_no_rsi_with_tpsl.avg_trade_return
        avg_trade_3 = p_with_rsi.avg_trade_return
        print(f"  平均每笔收益          {avg_trade_1:,.2f}{'':<10} {avg_trade_2:,.2f}{'':<10} {avg_trade_3:,.2f}")
        
        print(f"\n{'='*80}")
        print("                    测试结论")
        print("=" * 80)
        
        conclusions = []
        
        if trades_3 < trades_2:
            reduction_pct = ((trades_2 - trades_3) / trades_2 * 100) if trades_2 > 0 else 0
            conclusions.append(f"[OK] RSI 过滤成功减少交易次数: {trades_2} -> {trades_3} (减少 {reduction_pct:.1f}%)")
        else:
            conclusions.append(f"[WARN] RSI 过滤未减少交易次数")
        
        if sharpe_3 > sharpe_2:
            conclusions.append(f"[OK] 夏普比率提升: {sharpe_2:.2f} -> {sharpe_3:.2f}")
        elif sharpe_3 < sharpe_2:
            conclusions.append(f"[WARN] 夏普比率下降: {sharpe_2:.2f} -> {sharpe_3:.2f}")
        
        if max_dd_3 < max_dd_2:
            conclusions.append(f"[OK] 最大回撤降低: {max_dd_2:.2f}% -> {max_dd_3:.2f}%")
        
        if win_rate_3 > win_rate_2:
            conclusions.append(f"[OK] 胜率提升: {win_rate_2:.2f}% -> {win_rate_3:.2f}%")
        
        print("\n" + "\n".join(conclusions))
        
        print(f"\n【关键指标总结】")
        print(f"  无RSI无TP/SL - 交易次数: {trades_1}, 收益率: {return_1:.2f}%, 最大回撤: {max_dd_1:.2f}%")
        print(f"  无RSI有TP/SL - 交易次数: {trades_2}, 收益率: {return_2:.2f}%, 最大回撤: {max_dd_2:.2f}%")
        print(f"  有RSI有TP/SL - 交易次数: {trades_3}, 收益率: {return_3:.2f}%, 最大回撤: {max_dd_3:.2f}%")
        
        if trades_1 >= 20:
            print(f"\n[OK] 无过滤模式下产生了 {trades_1} 次交易（目标: >=20）")
        else:
            print(f"\n[WARN] 无过滤模式下只产生了 {trades_1} 次交易（目标: >=20）")
        
    except Exception as e:
        print(f"\n[ERROR] 对比测试执行出错: {e}")
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
