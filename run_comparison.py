import sys
import os
import random
import math
import json
from datetime import datetime, date, timedelta
from dataclasses import asdict

base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, base_dir)

from strategies.base_strategy import SignalType
from strategies.adaptive_momentum_strategy import AdaptiveMomentumStrategy
from strategies.double_ma_strategy import DoubleMAStrategy
from strategies.strategy_comparison import (
    StrategyComparator,
    BacktestSimulator,
    PerformanceResult,
    ComparisonReport,
    StrategyType,
)


def generate_price_series(
    initial_price: float,
    num_periods: int,
    volatility: float = 0.02,
    trend: float = 0.0,
    seed: int = 42,
):
    random.seed(seed)
    prices = [initial_price]
    highs = [initial_price * 1.01]
    lows = [initial_price * 0.99]
    opens = [initial_price]
    
    for i in range(num_periods - 1):
        change = random.gauss(trend, volatility)
        new_price = prices[-1] * (1 + change)
        
        open_price = prices[-1] * (1 + random.gauss(0, volatility * 0.5))
        high_price = max(new_price, open_price) * (1 + random.uniform(0, volatility * 0.3))
        low_price = min(new_price, open_price) * (1 - random.uniform(0, volatility * 0.3))
        
        prices.append(new_price)
        opens.append(open_price)
        highs.append(high_price)
        lows.append(low_price)
    
    return prices, opens, highs, lows


def run_comprehensive_test():
    print("=" * 70)
    print("策略对比综合测试")
    print("=" * 70)
    
    contracts = ['SHFE.rb', 'DCE.i']
    initial_capital = 1000000.0
    
    contract_configs = {
        'SHFE.rb': {
            'initial_price': 3500.0,
            'volatility': 0.015,
            'trend': 0.0002,
            'contract_multiplier': 10,
        },
        'DCE.i': {
            'initial_price': 800.0,
            'volatility': 0.02,
            'trend': -0.0001,
            'contract_multiplier': 100,
        },
    }
    
    num_periods = 1000
    
    old_results = {}
    new_results = {}
    improvements = {}
    
    for contract in contracts:
        config = contract_configs[contract]
        
        print(f"\n{'=' * 70}")
        print(f"合约: {contract}")
        print(f"{'=' * 70}")
        
        prices, opens, highs, lows = generate_price_series(
            initial_price=config['initial_price'],
            num_periods=num_periods,
            volatility=config['volatility'],
            trend=config['trend'],
            seed=hash(contract) % 10000,
        )
        
        klines = []
        base_time = datetime(2024, 1, 1, 9, 0)
        for i in range(num_periods):
            klines.append({
                'datetime': base_time + timedelta(hours=i),
                'open': opens[i],
                'high': highs[i],
                'low': lows[i],
                'close': prices[i],
                'volume': 1000,
            })
        
        print(f"\n--- 旧策略: DoubleMAStrategy ---")
        old_result = run_single_strategy_backtest(
            strategy_class=DoubleMAStrategy,
            strategy_params={
                'short_period': 5,
                'long_period': 20,
                'kline_duration': 60,
                'use_ema': False,
                'rsi_period': 14,
                'rsi_threshold': 50.0,
                'use_rsi_filter': False,
                'take_profit_ratio': None,
                'stop_loss_ratio': None,
                'debug_logging': False,
            },
            klines=klines,
            contract=contract,
            initial_capital=initial_capital,
            contract_multiplier=config['contract_multiplier'],
            use_dynamic_position=False,
        )
        
        old_results[contract] = old_result
        
        print(f"\n--- 新策略: AdaptiveMomentumStrategy ---")
        new_result = run_single_strategy_backtest(
            strategy_class=AdaptiveMomentumStrategy,
            strategy_params={
                'short_period': 5,
                'long_period': 20,
                'kline_duration': 60,
                'use_ema': False,
                'rsi_period': 14,
                'rsi_threshold': 50.0,
                'atr_period': 14,
                'atr_entry_multiplier': 1.5,
                'atr_exit_multiplier': 2.0,
                'risk_per_trade_percent': 0.01,
                'trailing_stop_atr_multiplier': 2.0,
                'contract_multiplier': config['contract_multiplier'],
                'debug_logging': False,
            },
            klines=klines,
            contract=contract,
            initial_capital=initial_capital,
            contract_multiplier=config['contract_multiplier'],
            use_dynamic_position=True,
        )
        
        new_results[contract] = new_result
        
        print(f"\n--- 对比结果 ---")
        print(f"  交易次数: 旧策略={old_result.total_trades}, 新策略={new_result.total_trades}")
        print(f"  胜率: 旧策略={old_result.win_rate:.2f}%, 新策略={new_result.win_rate:.2f}%")
        print(f"  收益率: 旧策略={old_result.total_return_percent:.2f}%, 新策略={new_result.total_return_percent:.2f}%")
        print(f"  最大回撤: 旧策略={old_result.max_drawdown_percent:.2f}%, 新策略={new_result.max_drawdown_percent:.2f}%")
        print(f"  夏普比率: 旧策略={old_result.sharpe_ratio:.2f}, 新策略={new_result.sharpe_ratio:.2f}")
        
        if old_result.total_trades > 0 and new_result.total_trades > 0:
            win_imp = new_result.win_rate - old_result.win_rate
            dd_imp = old_result.max_drawdown_percent - new_result.max_drawdown_percent
            sharpe_imp = new_result.sharpe_ratio - old_result.sharpe_ratio
            
            print(f"\n  改善情况:")
            print(f"    胜率提升: {win_imp:+.2f}%")
            print(f"    回撤降低: {dd_imp:+.2f}%")
            print(f"    夏普比率提升: {sharpe_imp:+.2f}")
            
            improvements[contract] = {
                'win_rate_improvement': win_imp,
                'max_drawdown_improvement': dd_imp,
                'sharpe_improvement': sharpe_imp,
            }
        else:
            improvements[contract] = {
                'win_rate_improvement': 0,
                'max_drawdown_improvement': 0,
                'sharpe_improvement': 0,
            }
    
    print(f"\n{'=' * 70}")
    print("综合对比报告")
    print("=" * 70)
    
    overall_win_imp = 0.0
    overall_dd_imp = 0.0
    overall_sharpe_imp = 0.0
    count = 0
    
    report_data = {
        'generated_at': datetime.now().isoformat(),
        'contracts': contracts,
        'initial_capital': initial_capital,
        'old_strategy': 'DoubleMAStrategy',
        'new_strategy': 'AdaptiveMomentumStrategy',
        'results': {},
        'improvements': {},
    }
    
    for contract in contracts:
        old_result = old_results.get(contract)
        new_result = new_results.get(contract)
        imp = improvements.get(contract, {})
        
        if old_result and new_result:
            print(f"\n合约: {contract}")
            print(f"  旧策略 (DoubleMAStrategy):")
            print(f"    交易次数: {old_result.total_trades}")
            print(f"    胜率: {old_result.win_rate:.2f}%")
            print(f"    盈利次数: {old_result.winning_trades}")
            print(f"    亏损次数: {old_result.losing_trades}")
            print(f"    总盈亏: {old_result.total_profit_loss:.2f}")
            print(f"    收益率: {old_result.total_return_percent:.2f}%")
            print(f"    最大回撤: {old_result.max_drawdown_percent:.2f}%")
            print(f"    夏普比率: {old_result.sharpe_ratio:.2f}")
            
            print(f"\n  新策略 (AdaptiveMomentumStrategy):")
            print(f"    交易次数: {new_result.total_trades}")
            print(f"    胜率: {new_result.win_rate:.2f}%")
            print(f"    盈利次数: {new_result.winning_trades}")
            print(f"    亏损次数: {new_result.losing_trades}")
            print(f"    总盈亏: {new_result.total_profit_loss:.2f}")
            print(f"    收益率: {new_result.total_return_percent:.2f}%")
            print(f"    最大回撤: {new_result.max_drawdown_percent:.2f}%")
            print(f"    夏普比率: {new_result.sharpe_ratio:.2f}")
            
            win_imp = imp.get('win_rate_improvement', 0)
            dd_imp = imp.get('max_drawdown_improvement', 0)
            sharpe_imp = imp.get('sharpe_improvement', 0)
            
            print(f"\n  改善情况:")
            print(f"    胜率提升: {win_imp:+.2f}%")
            print(f"    回撤降低: {dd_imp:+.2f}%")
            print(f"    夏普比率提升: {sharpe_imp:+.2f}")
            
            if old_result.total_trades > 0 and new_result.total_trades > 0:
                overall_win_imp += win_imp
                overall_dd_imp += dd_imp
                overall_sharpe_imp += sharpe_imp
                count += 1
            
            report_data['results'][contract] = {
                'old_strategy': {
                    'total_trades': old_result.total_trades,
                    'win_rate': old_result.win_rate,
                    'winning_trades': old_result.winning_trades,
                    'losing_trades': old_result.losing_trades,
                    'total_profit_loss': old_result.total_profit_loss,
                    'total_return_percent': old_result.total_return_percent,
                    'max_drawdown_percent': old_result.max_drawdown_percent,
                    'sharpe_ratio': old_result.sharpe_ratio,
                },
                'new_strategy': {
                    'total_trades': new_result.total_trades,
                    'win_rate': new_result.win_rate,
                    'winning_trades': new_result.winning_trades,
                    'losing_trades': new_result.losing_trades,
                    'total_profit_loss': new_result.total_profit_loss,
                    'total_return_percent': new_result.total_return_percent,
                    'max_drawdown_percent': new_result.max_drawdown_percent,
                    'sharpe_ratio': new_result.sharpe_ratio,
                },
            }
            report_data['improvements'][contract] = {
                'win_rate_improvement': win_imp,
                'max_drawdown_improvement': dd_imp,
                'sharpe_improvement': sharpe_imp,
            }
    
    if count > 0:
        avg_win_imp = overall_win_imp / count
        avg_dd_imp = overall_dd_imp / count
        avg_sharpe_imp = overall_sharpe_imp / count
        
        print(f"\n{'=' * 70}")
        print("平均改善情况")
        print("=" * 70)
        print(f"  平均胜率提升: {avg_win_imp:+.2f}%")
        print(f"  平均回撤降低: {avg_dd_imp:+.2f}%")
        print(f"  平均夏普比率提升: {avg_sharpe_imp:+.2f}")
        
        report_data['overall_summary'] = {
            'average_win_rate_improvement': avg_win_imp,
            'average_max_drawdown_improvement': avg_dd_imp,
            'average_sharpe_improvement': avg_sharpe_imp,
        }
        
        if avg_win_imp > 0 and avg_dd_imp > 0:
            print(f"\n结论: 新策略 (AdaptiveMomentumStrategy) 在胜率和风险控制方面均优于旧策略！")
            report_data['conclusion'] = "新策略在胜率和风险控制方面均优于旧策略"
        elif avg_win_imp > 0:
            print(f"\n结论: 新策略 (AdaptiveMomentumStrategy) 在胜率方面优于旧策略。")
            report_data['conclusion'] = "新策略在胜率方面优于旧策略"
        elif avg_dd_imp > 0:
            print(f"\n结论: 新策略 (AdaptiveMomentumStrategy) 在风险控制方面优于旧策略。")
            report_data['conclusion'] = "新策略在风险控制方面优于旧策略"
        else:
            print(f"\n结论: 新策略 (AdaptiveMomentumStrategy) 在本次测试中未显示出明显优势。")
            print(f"  可能的原因:")
            print(f"  1. 测试数据可能不适合新策略的特点")
            print(f"  2. 新策略的过滤条件可能过于严格，减少了交易机会")
            print(f"  3. 需要调整参数以适应特定的市场环境")
            report_data['conclusion'] = "新策略在本次测试中未显示出明显优势"
            report_data['possible_reasons'] = [
                "测试数据可能不适合新策略的特点",
                "新策略的过滤条件可能过于严格，减少了交易机会",
                "需要调整参数以适应特定的市场环境"
            ]
    
    report_file = os.path.join(base_dir, 'strategy_comparison_report.json')
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report_data, f, ensure_ascii=False, indent=2)
    print(f"\n报告已保存至: {report_file}")
    
    print(f"\n{'=' * 70}")
    print("测试完成！")
    print("=" * 70)
    
    return report_data


def run_single_strategy_backtest(
    strategy_class,
    strategy_params,
    klines,
    contract,
    initial_capital,
    contract_multiplier,
    use_dynamic_position=False,
):
    from datetime import datetime
    
    strategy = strategy_class(**strategy_params)
    
    simulator = BacktestSimulator(
        initial_capital=initial_capital,
        contract_multiplier=contract_multiplier,
    )
    
    long_period = strategy_params.get('long_period', 20)
    rsi_period = strategy_params.get('rsi_period', 14)
    atr_period = strategy_params.get('atr_period', 14)
    warmup_period = max(long_period, rsi_period, atr_period) + 10
    
    if len(klines) <= warmup_period:
        result = PerformanceResult(
            strategy_name=strategy_class.__name__,
            strategy_type=StrategyType.ADAPTIVE_MOMENTUM if strategy_class == AdaptiveMomentumStrategy else StrategyType.DOUBLE_MA,
            contract=contract,
            initial_capital=initial_capital,
        )
        result.status = "error"
        result.error_message = f"K线数量不足"
        return result
    
    for i in range(warmup_period):
        kline = klines[i]
        if strategy_class == DoubleMAStrategy:
            strategy.update_prices(close_price=kline['close'])
        else:
            strategy.update_prices(
                close_price=kline['close'],
                high_price=kline['high'],
                low_price=kline['low'],
                open_price=kline['open'],
            )
    
    for i in range(warmup_period, len(klines)):
        kline = klines[i]
        current_time = kline.get('datetime', i)
        close_price = kline['close']
        high_price = kline['high']
        low_price = kline['low']
        
        atr = getattr(strategy, 'atr', None)
        if atr is None or atr <= 0:
            atr = (high_price - low_price) * 1.5
        
        if simulator._position != 0:
            should_close, close_reason = simulator.update_price(
                current_price=close_price,
                atr=atr,
                atr_exit_multiplier=getattr(strategy, 'atr_exit_multiplier', 2.0),
                trailing_stop_multiplier=getattr(strategy, 'trailing_stop_atr_multiplier', 2.0),
            )
            
            if should_close:
                simulator.close_position(close_price, current_time, close_reason)
        
        if strategy_class == DoubleMAStrategy:
            strategy.update_prices(close_price=close_price)
        else:
            strategy.update_prices(
                close_price=close_price,
                high_price=high_price,
                low_price=low_price,
                open_price=kline['open'],
            )
        
        signal = strategy.signal
        
        if simulator._position == 0:
            if signal == SignalType.BUY:
                position_size = 1
                if use_dynamic_position and hasattr(strategy, '_calculate_position_size'):
                    position_size = strategy._calculate_position_size(atr, close_price)
                    position_size = max(1, int(position_size))
                
                simulator.open_position(
                    direction="BUY",
                    price=close_price,
                    volume=position_size,
                    current_time=current_time,
                    reason="golden_cross",
                )
            
            elif signal == SignalType.SELL:
                position_size = 1
                if use_dynamic_position and hasattr(strategy, '_calculate_position_size'):
                    position_size = strategy._calculate_position_size(atr, close_price)
                    position_size = max(1, int(position_size))
                
                simulator.open_position(
                    direction="SELL",
                    price=close_price,
                    volume=position_size,
                    current_time=current_time,
                    reason="death_cross",
                )
        
        simulator.record_equity(i, current_time)
    
    if simulator._position != 0:
        last_kline = klines[-1]
        last_time = last_kline.get('datetime', len(klines) - 1)
        last_price = last_kline['close']
        simulator.close_position(last_price, last_time, "end_of_backtest")
    
    result = PerformanceResult(
        strategy_name=strategy_class.__name__,
        strategy_type=StrategyType.ADAPTIVE_MOMENTUM if strategy_class == AdaptiveMomentumStrategy else StrategyType.DOUBLE_MA,
        contract=contract,
        initial_capital=initial_capital,
    )
    
    result.equity_curve = simulator._equity_curve
    
    total_trades = len(simulator._trade_records)
    winning_trades = sum(1 for t in simulator._trade_records if t.profit_loss and t.profit_loss > 0)
    losing_trades = sum(1 for t in simulator._trade_records if t.profit_loss and t.profit_loss <= 0)
    total_pnl = sum(t.profit_loss for t in simulator._trade_records if t.profit_loss is not None)
    
    result.total_trades = total_trades
    result.winning_trades = winning_trades
    result.losing_trades = losing_trades
    result.total_profit_loss = total_pnl
    
    if result.total_trades > 0:
        result.win_rate = (result.winning_trades / result.total_trades) * 100
    
    if result.initial_capital > 0:
        result.total_return_percent = (result.total_profit_loss / result.initial_capital) * 100
    
    if len(result.equity_curve) > 0:
        peak = result.initial_capital
        max_dd = 0.0
        
        for point in result.equity_curve:
            equity = point.get('equity', result.initial_capital)
            if equity > peak:
                peak = equity
            if peak > 0:
                dd = (peak - equity) / peak * 100
                if dd > max_dd:
                    max_dd = dd
        
        result.max_drawdown_percent = max_dd
    
    if len(result.equity_curve) > 1:
        returns = []
        prev_equity = result.initial_capital
        
        for point in result.equity_curve:
            equity = point.get('equity', result.initial_capital)
            if prev_equity > 0:
                returns.append((equity - prev_equity) / prev_equity)
            prev_equity = equity
        
        if len(returns) > 0:
            non_zero_returns = [r for r in returns if r != 0]
            if len(non_zero_returns) > 1:
                avg_return = sum(returns) / len(returns)
                variance = sum((r - avg_return) ** 2 for r in returns) / len(returns)
                std_dev = math.sqrt(variance) if variance > 0 else 0.0001
                
                risk_free_rate = 0.02 / 252
                if std_dev > 0:
                    result.sharpe_ratio = (avg_return - risk_free_rate) / std_dev * math.sqrt(252)
    
    result.status = "completed"
    
    return result


if __name__ == "__main__":
    run_comprehensive_test()
