#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
多合约路由与资金池管理测试脚本
支持同时在 3 个以上的品种上实现自动化回测

核心功能：
1. 合约路由：每个合约分配独立的策略状态，确保不同品种的指标和信号互不干扰
2. 动态配资：合约波动率越大，分配的资金权重越低
3. 执行一致性：展示各品种的当前持仓、盈亏及信号状态

使用方法：
  python run_multiproduct.py                          # 使用默认配置
  python run_multiproduct.py --contracts rb2410 hc2410 i2409 CF409  # 指定合约
  python run_multiproduct.py --start 2024-01-01 --end 2024-03-31  # 指定日期
  python run_multiproduct.py --balance 2000000      # 指定总资金
"""

import os
import sys
import logging
from datetime import datetime, date
from typing import Dict, Any, List, Optional

base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, base_dir)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    yaml = None

from core.multi_contract_router import (
    MultiContractRouter,
    CapitalPoolConfig,
    ContractState,
)
from core.multi_contract_runner import (
    MultiContractRunner,
    ContractBacktestConfig,
    run_all_backtests,
)
from strategies.double_ma_strategy import DoubleMAStrategy
from strategies.adaptive_ma_strategy import AdaptiveMAStrategy


DEFAULT_CONTRACTS = [
    'SHFE.rb2410',
    'SHFE.hc2410',
    'DCE.i2409',
    'CZCE.CF409',
]

DEFAULT_BACKTEST_CONFIG = {
    'start_dt': '2024-01-01',
    'end_dt': '2024-03-31',
    'total_capital': 2000000.0,
    'min_weight_per_contract': 0.10,
    'max_weight_per_contract': 0.40,
    'strategy_params': {
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
    },
}


def pre_flight_check(
    strategy_class,
    strategy_params: Dict[str, Any],
    test_contract: str = 'SHFE.rb2410',
) -> bool:
    """
    飞行前检查：尝试实例化策略，检测参数不匹配问题
    
    如果抛出 TypeError 或 KeyError，立即打印清晰的修复建议并退出，
    而不是直接炸出 Traceback。
    
    Args:
        strategy_class: 策略类
        strategy_params: 策略参数字典
        test_contract: 测试用的合约代码
    
    Returns:
        bool: 检查通过返回 True，否则打印错误信息并退出
    """
    logger = logging.getLogger('PreFlightCheck')
    
    print("\n" + "=" * 80)
    print("                    飞行前检查 (Pre-Flight Check)")
    print("=" * 80)
    print(f"策略类: {strategy_class.__name__}")
    print(f"测试合约: {test_contract}")
    print("-" * 80)
    
    try:
        params = strategy_params.copy()
        params['contract'] = test_contract
        params['connector'] = None
        
        logger.info(f"尝试实例化策略: {strategy_class.__name__}")
        
        strategy = strategy_class(**params)
        
        logger.info(f"策略实例化成功: {strategy_class.__name__}")
        print(f"  [OK] 策略实例化成功")
        
        if hasattr(strategy, 'contract'):
            print(f"  [OK] 合约参数设置: {strategy.contract}")
        
        print("-" * 80)
        print("飞行前检查通过！")
        print("=" * 80)
        
        return True
        
    except TypeError as e:
        error_msg = str(e)
        
        print("\n" + "!" * 80)
        print("                    [ERROR] 飞行前检查失败：参数不匹配")
        print("!" * 80)
        print(f"\n错误信息: {error_msg}")
        
        if 'unexpected keyword argument' in error_msg:
            import re
            match = re.search(r"'(\w+)'", error_msg)
            if match:
                unknown_param = match.group(1)
                print(f"\n问题分析:")
                print(f"  - 策略类 {strategy_class.__name__} 不支持参数: '{unknown_param}'")
                print(f"  - 这个参数在 strategy_params 中被传递，但策略的 __init__ 方法不接受")
                
                print(f"\n修复建议:")
                print(f"  方案1: 从 strategy_params 中移除 '{unknown_param}' 参数")
                print(f"  方案2: 在策略类 {strategy_class.__name__} 的 __init__ 方法中添加 **kwargs")
                
                if unknown_param in ['use_rsi_filter', 'take_profit_ratio', 'stop_loss_ratio']:
                    print(f"\n提示:")
                    print(f"  - '{unknown_param}' 是 DoubleMAStrategy 的参数")
                    print(f"  - AdaptiveMAStrategy 使用不同的方式实现类似功能：")
                    if unknown_param == 'use_rsi_filter':
                        print(f"    - 自适应策略默认启用 RSI 过滤，无需显式设置")
                    elif unknown_param in ['take_profit_ratio', 'stop_loss_ratio']:
                        print(f"    - 自适应策略使用 ATR 倍数来实现止盈止损")
                        print(f"    - 相关参数: atr_exit_multiplier, atr_entry_multiplier, trailing_stop_atr_multiplier")
                
        elif 'missing' in error_msg.lower() and 'required' in error_msg.lower():
            print(f"\n问题分析:")
            print(f"  - 缺少必需的参数")
            
        print("\n策略参数详情:")
        for key, value in strategy_params.items():
            print(f"  {key}: {value}")
        
        print("\n" + "!" * 80)
        
        import sys
        sys.exit(1)
        
    except KeyError as e:
        error_msg = str(e)
        
        print("\n" + "!" * 80)
        print("                    [ERROR] 飞行前检查失败：键错误")
        print("!" * 80)
        print(f"\n错误信息: KeyError: {error_msg}")
        
        print(f"\n问题分析:")
        print(f"  - 可能是配置文件或参数字典中缺少必需的键")
        
        print("\n" + "!" * 80)
        
        import sys
        sys.exit(1)
        
    except Exception as e:
        print("\n" + "!" * 80)
        print("                    [ERROR] 飞行前检查失败：未知错误")
        print("!" * 80)
        print(f"\n错误类型: {type(e).__name__}")
        print(f"错误信息: {e}")
        
        import traceback
        print(f"\n详细错误信息:")
        traceback.print_exc()
        
        print("\n" + "!" * 80)
        
        import sys
        sys.exit(1)
    
    return False


def load_config():
    config_path = os.path.join(base_dir, 'config', 'settings.yaml')
    if os.path.exists(config_path) and YAML_AVAILABLE:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        trading_config = config.get('trading', {})
        contracts = trading_config.get('contracts', [])
        if contracts and len(contracts) >= 3:
            global DEFAULT_CONTRACTS
            DEFAULT_CONTRACTS = contracts
        
        backtest_config = config.get('backtest', {})
        if backtest_config:
            if backtest_config.get('start_dt'):
                DEFAULT_BACKTEST_CONFIG['start_dt'] = backtest_config['start_dt']
            if backtest_config.get('end_dt'):
                DEFAULT_BACKTEST_CONFIG['end_dt'] = backtest_config['end_dt']
            if backtest_config.get('init_balance'):
                DEFAULT_BACKTEST_CONFIG['total_capital'] = backtest_config['init_balance']
        
        strategies = config.get('strategies', [])
        if strategies:
            first_strategy = strategies[0]
            params = first_strategy.get('params', {})
            
            param_mapping = {
                'fast': 'short_period',
                'slow': 'long_period',
                'period': 'kline_duration',
            }
            
            for key, value in params.items():
                mapped_key = param_mapping.get(key, key)
                if mapped_key in DEFAULT_BACKTEST_CONFIG['strategy_params']:
                    DEFAULT_BACKTEST_CONFIG['strategy_params'][mapped_key] = value
        
        return config
    
    return None


def print_header():
    print("\n" + "=" * 80)
    print("                    多合约路由与资金池管理系统")
    print("=" * 80)
    print(f"启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"工作目录: {base_dir}")
    print("-" * 80)
    print("【核心功能】")
    print("  1. 合约路由：每个合约分配独立的策略状态")
    print("  2. 动态配资：合约波动率越大，分配的资金权重越低")
    print("  3. 执行一致性：展示各品种的当前持仓、盈亏及信号状态")
    print("-" * 80)


def print_config(contracts: List[str], config: Dict[str, Any]):
    print(f"\n【回测配置】")
    print(f"  回测区间: {config['start_dt']} 至 {config['end_dt']}")
    print(f"  总资金: {config['total_capital']:,.0f}")
    print(f"  最小单合约权重: {config['min_weight_per_contract']*100:.1f}%")
    print(f"  最大单合约权重: {config['max_weight_per_contract']*100:.1f}%")
    print(f"\n  策略参数:")
    for key, value in config['strategy_params'].items():
        print(f"    {key}: {value}")
    print(f"\n  回测合约 ({len(contracts)} 个):")
    for i, contract in enumerate(contracts, 1):
        print(f"    {i}. {contract}")
    print("-" * 80)


def print_capital_allocation_summary(
    total_capital: float,
    contract_weights: Dict[str, float],
    allocated_capitals: Dict[str, float],
):
    print("\n" + "=" * 80)
    print("                    资金分配结果")
    print("=" * 80)
    print(f"总资金: {total_capital:,.0f}")
    print("-" * 80)
    
    for contract, weight in contract_weights.items():
        allocated = allocated_capitals.get(contract, 0.0)
        print(f"  {contract}:")
        print(f"    权重: {weight*100:.2f}%")
        print(f"    分配资金: {allocated:,.0f}")
    
    print("=" * 80)


def print_multi_contract_status(
    contract_states: Dict[str, ContractState],
    title: str = "多合约状态",
):
    print("\n" + "=" * 80)
    print(f"                    {title}")
    print("=" * 80)
    
    for contract, state in contract_states.items():
        print(f"\n【{contract}】")
        print(f"  状态: {state.status.value}")
        print(f"  策略: {state.strategy_name}")
        
        print(f"\n  【持仓信息】")
        print(f"    方向: {state.position_direction}")
        print(f"    数量: {state.position_volume} 手")
        print(f"    开仓价: {state.entry_price:.2f}")
        print(f"    当前价: {state.current_price:.2f}")
        print(f"    浮动盈亏: {state.float_profit:,.2f}")
        print(f"    占用保证金: {state.margin_used:,.0f}")
        
        print(f"\n  【技术指标】")
        print(f"    ATR: {state.atr:.4f}" if state.atr else "    ATR: N/A")
        print(f"    波动率比: {state.volatility_ratio:.4f}")
        
        print(f"\n  【信号与交易】")
        print(f"    当前信号: {state.signal.value}")
        print(f"    最后信号时间: {state.last_signal_time}")
        print(f"    总交易次数: {state.total_trades}")
        print(f"    盈利交易次数: {state.win_trades}")
        print(f"    总盈亏: {state.total_profit:,.2f}")
        
        print(f"\n  【资金分配】")
        print(f"    分配资金: {state.allocated_capital:,.0f}")
        print(f"    资金权重: {state.capital_weight*100:.2f}%")
        
        if state.strategy_health:
            print(f"\n  【健康状态】")
            health_dict = state.strategy_health.to_dict()
            print(f"    状态: {health_dict.get('status', 'UNKNOWN')}")
            print(f"    错误次数: {health_dict.get('error_count', 0)}")
            print(f"    成功率: {health_dict.get('success_rate', 1.0)*100:.1f}%")
    
    print("=" * 80)


def print_backtest_results_summary(result):
    print("\n" + "=" * 80)
    print("                    回测结果汇总")
    print("=" * 80)
    
    if result.comparison_chart_path:
        print(f"\n对比图表已生成: {result.comparison_chart_path}")
    
    if result.summary_report:
        print(f"\n回测汇总:")
        summary = result.summary_report.get('summary', {})
        print(f"  合约数量: {summary.get('total_contracts', 0)}")
        print(f"  组合收益率: {summary.get('combined_return_percent', 0):.2f}%")
        print(f"  总交易次数: {summary.get('total_trades', 0)}")
        print(f"  最大回撤(跨合约): {summary.get('max_drawdown_across_contracts', 0):.2f}%")
    
    print("\n合约详情:")
    for contract, contract_result in result.contract_results.items():
        print(f"\n  {contract}:")
        print(f"    收益率: {contract_result.total_return_percent:.2f}%")
        print(f"    最大回撤: {contract_result.max_drawdown_percent:.2f}%")
        print(f"    交易次数: {contract_result.total_trades}")
    
    print("\n" + "=" * 80)
    print("回测完成，请查看 results/ 目录获取详细图表和报告")
    print("=" * 80 + "\n")


def run_multiproduct_backtest(
    contracts: List[str],
    start_dt: date,
    end_dt: date,
    total_capital: float,
    min_weight: float,
    max_weight: float,
    strategy_params: Dict[str, Any],
    parallel: bool = True,
    use_adaptive_strategy: bool = True,
) -> Dict[str, Any]:
    logger = logging.getLogger('MultiProductTest')
    
    capital_pool_config = CapitalPoolConfig(
        total_capital=total_capital,
        min_weight_per_contract=min_weight,
        max_weight_per_contract=max_weight,
        volatility_lookback_period=20,
        weight_calculation_method="inverse_volatility",
    )
    
    logger.info(f"创建资金池配置: 总资金={total_capital:,.0f}, "
                f"最小权重={min_weight*100:.1f}%, "
                f"最大权重={max_weight*100:.1f}%")
    
    router = MultiContractRouter(
        connector=None,
        capital_pool_config=capital_pool_config,
    )
    
    strategy_class = AdaptiveMAStrategy if use_adaptive_strategy else DoubleMAStrategy
    strategy_name = "AdaptiveMAStrategy" if use_adaptive_strategy else "DoubleMAStrategy"
    
    logger.info(f"使用策略: {strategy_name}")
    
    for contract in contracts:
        params = strategy_params.copy()
        params['contract'] = contract
        
        router.register_contract_strategy(
            contract=contract,
            strategy_class=strategy_class,
            strategy_params=params,
            strategy_name=f"{strategy_name}_{contract}",
        )
        logger.info(f"已注册合约策略: {contract}")
    
    registered_contracts = router.get_registered_contracts()
    logger.info(f"已注册合约数量: {len(registered_contracts)}")
    
    capital_pool = router.get_capital_pool()
    
    for contract in registered_contracts:
        mock_atr = 20.0 + hash(contract) % 30
        mock_price = 3000.0 + hash(contract) % 1000
        
        capital_pool.record_contract_volatility(contract, mock_atr, mock_price)
        logger.debug(f"记录模拟波动率: {contract}, ATR={mock_atr}, 价格={mock_price}")
    
    capital_pool.allocate_capital(registered_contracts)
    
    allocation_summary = capital_pool.get_allocation_summary()
    contract_weights = allocation_summary['contract_weights']
    allocated_capitals = allocation_summary['allocated_capitals']
    
    print_capital_allocation_summary(
        total_capital=total_capital,
        contract_weights=contract_weights,
        allocated_capitals=allocated_capitals,
    )
    
    for contract in registered_contracts:
        state = router.get_contract_state(contract)
        if state:
            state.allocated_capital = allocated_capitals.get(contract, 0.0)
            state.capital_weight = contract_weights.get(contract, 0.0)
            state.atr = capital_pool.get_contract_avg_volatility(contract) * state.current_price if state.current_price > 0 else None
            state.volatility_ratio = capital_pool.get_contract_volatility_ratio(contract)
    
    print_multi_contract_status(
        contract_states=router.get_all_contract_states(),
        title="初始多合约状态",
    )
    
    runner = MultiContractRunner(config=load_config())
    runner._contract_configs = []
    
    for contract in contracts:
        params = strategy_params.copy()
        params['contract'] = contract
        
        contract_config = ContractBacktestConfig(
            contract=contract,
            strategy_class=strategy_class,
            strategy_params=params,
            start_dt=start_dt,
            end_dt=end_dt,
            initial_balance=allocated_capitals.get(contract, total_capital / len(contracts)),
        )
        runner.add_contract(contract_config)
        logger.info(f"已添加合约回测配置: {contract}, 初始资金={contract_config.initial_balance:,.0f}")
    
    logger.info(f"开始执行多合约回测，模式={'并行' if parallel else '串行'}")
    result = runner.run_all(parallel=parallel)
    
    print_backtest_results_summary(result)
    
    multi_contract_status = router.get_multi_contract_status()
    
    return {
        'router': router,
        'backtest_result': result,
        'multi_contract_status': multi_contract_status,
        'capital_allocation': {
            'total_capital': total_capital,
            'contract_weights': contract_weights,
            'allocated_capitals': allocated_capitals,
        },
    }


def parse_date(date_str: str) -> date:
    for fmt in ['%Y-%m-%d', '%Y%m%d', '%Y/%m/%d']:
        try:
            return datetime.strptime(str(date_str), fmt).date()
        except ValueError:
            continue
    return date(2024, 1, 1)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='多合约路由与资金池管理测试工具')
    parser.add_argument('--contracts', '-s', type=str, nargs='+', 
                        default=DEFAULT_CONTRACTS,
                        help='合约列表 (默认: SHFE.rb2410 SHFE.hc2410 DCE.i2409 CZCE.CF409)')
    parser.add_argument('--start', type=str, 
                        default=DEFAULT_BACKTEST_CONFIG['start_dt'],
                        help='开始日期 (YYYY-MM-DD, 默认: 2024-01-01)')
    parser.add_argument('--end', type=str, 
                        default=DEFAULT_BACKTEST_CONFIG['end_dt'],
                        help='结束日期 (YYYY-MM-DD, 默认: 2024-03-31)')
    parser.add_argument('--balance', '-b', type=float, 
                        default=DEFAULT_BACKTEST_CONFIG['total_capital'],
                        help='总资金 (默认: 2000000)')
    parser.add_argument('--min-weight', type=float, 
                        default=DEFAULT_BACKTEST_CONFIG['min_weight_per_contract'],
                        help='最小单合约权重 (默认: 0.10, 即 10%%)')
    parser.add_argument('--max-weight', type=float, 
                        default=DEFAULT_BACKTEST_CONFIG['max_weight_per_contract'],
                        help='最大单合约权重 (默认: 0.40, 即 40%%)')
    parser.add_argument('--serial', action='store_true', 
                        help='串行运行 (默认: 并行)')
    parser.add_argument('--use-double-ma', action='store_true',
                        help='使用双均线策略 (默认: 自适应多因子策略)')
    parser.add_argument('--config', '-c', type=str, 
                        help='配置文件路径')
    
    args = parser.parse_args()
    
    print_header()
    
    config = load_config()
    
    contracts = args.contracts if args.contracts else DEFAULT_CONTRACTS
    
    if len(contracts) < 3:
        print(f"\n警告: 合约数量不足 3 个 (当前: {len(contracts)})")
        print(f"将使用默认合约列表: {DEFAULT_CONTRACTS}")
        contracts = DEFAULT_CONTRACTS
    
    start_dt = parse_date(args.start)
    end_dt = parse_date(args.end)
    
    total_capital = args.balance
    min_weight = args.min_weight
    max_weight = args.max_weight
    
    print_config(contracts, {
        'start_dt': args.start,
        'end_dt': args.end,
        'total_capital': total_capital,
        'min_weight_per_contract': min_weight,
        'max_weight_per_contract': max_weight,
        'strategy_params': DEFAULT_BACKTEST_CONFIG['strategy_params'],
    })
    
    print(f"\n【策略选择】")
    if args.use_double_ma:
        print("  使用策略: DoubleMAStrategy (双均线策略)")
        strategy_class = DoubleMAStrategy
    else:
        print("  使用策略: AdaptiveMAStrategy (自适应多因子策略)")
        print("  特点: ATR波动率过滤 + RSI动量确认 + 动态仓位 + 追踪止损")
        strategy_class = AdaptiveMAStrategy
    
    pre_flight_check(
        strategy_class=strategy_class,
        strategy_params=DEFAULT_BACKTEST_CONFIG['strategy_params'],
        test_contract=contracts[0] if contracts else 'SHFE.rb2410',
    )
    
    print("\n【资金分配逻辑】")
    print("  方法: 逆波动率加权 (Inverse Volatility Weighting)")
    print("  公式: 权重 = 1 / 波动率")
    print("  特点: 合约波动率越大，分配的资金权重越低")
    print("-" * 80)
    
    try:
        result = run_multiproduct_backtest(
            contracts=contracts,
            start_dt=start_dt,
            end_dt=end_dt,
            total_capital=total_capital,
            min_weight=min_weight,
            max_weight=max_weight,
            strategy_params=DEFAULT_BACKTEST_CONFIG['strategy_params'],
            parallel=not args.serial,
            use_adaptive_strategy=not args.use_double_ma,
        )
        
        multi_contract_status = result.get('multi_contract_status', {})
        print(f"\n【多合约状态 API 输出示例】")
        print(f"  生成时间: {multi_contract_status.get('generated_at', 'N/A')}")
        print(f"  总合约数: {multi_contract_status.get('total_contracts', 0)}")
        print(f"\n  可通过以下方式获取多合约状态:")
        print(f"    - router.get_multi_contract_status()")
        print(f"    - 用于 Web 仪表盘展示各品种的持仓、盈亏及信号状态")
        
        return 0
        
    except Exception as e:
        logging.getLogger('MultiProductTest').error(f"执行失败: {e}", exc_info=True)
        print(f"\n错误: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
