import os
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    plt = None

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False


CHINESE_FONT_AVAILABLE = False
FONT_CHECKED = False


def _check_chinese_font_support() -> bool:
    global CHINESE_FONT_AVAILABLE, FONT_CHECKED
    
    if FONT_CHECKED:
        return CHINESE_FONT_AVAILABLE
    
    if not MATPLOTLIB_AVAILABLE:
        FONT_CHECKED = True
        CHINESE_FONT_AVAILABLE = False
        return False
    
    try:
        from matplotlib.font_manager import findfont, FontProperties
        
        priority_fonts = ['SimHei', 'Microsoft YaHei']
        fallback_fonts = [
            'STSong', 'STKaiti', 'SimSun', 'KaiTi', 'FangSong', 'NSimSun',
            'PingFang SC', 'Hiragino Sans GB', 'Heiti SC',
            'WenQuanYi Micro Hei', 'WenQuanYi Zen Hei',
        ]
        
        all_fonts = priority_fonts + fallback_fonts
        found_font = None
        
        for font_name in all_fonts:
            try:
                font_prop = FontProperties(family=[font_name])
                font_path = findfont(font_prop)
                if font_path and os.path.exists(font_path):
                    from matplotlib import font_manager
                    font_files = font_manager.findSystemFonts(fontpaths=None, fontext='ttf')
                    font_names_lower = [font_name.lower()]
                    if font_name == 'Microsoft YaHei':
                        font_names_lower.append('msyh')
                        font_names_lower.append('microsoftyahei')
                    
                    font_found = False
                    for f in font_files:
                        f_lower = f.lower()
                        for check_name in font_names_lower:
                            if check_name.replace(' ', '') in f_lower or check_name in f_lower:
                                font_found = True
                                found_font = font_name
                                break
                        if font_found:
                            break
                    
                    if not font_found and font_name in priority_fonts:
                        font_prop2 = FontProperties(fname=font_path)
                        if font_prop2.get_name():
                            font_found = True
                            found_font = font_name
                    
                    if font_found:
                        plt.rcParams['font.sans-serif'] = [found_font] + plt.rcParams['font.sans-serif']
                        plt.rcParams['axes.unicode_minus'] = False
                        CHINESE_FONT_AVAILABLE = True
                        logging.getLogger(__name__).info(f"找到中文字体: {found_font}")
                        break
            except Exception:
                continue
        
        if not CHINESE_FONT_AVAILABLE:
            logging.getLogger(__name__).warning("未找到中文字体，图表将使用英文标签")
    
    except Exception as e:
        logging.getLogger(__name__).warning(f"字体检测失败: {e}")
        CHINESE_FONT_AVAILABLE = False
    
    FONT_CHECKED = True
    return CHINESE_FONT_AVAILABLE


def _get_label(zh_label: str, en_label: str) -> str:
    if _check_chinese_font_support():
        return zh_label
    return en_label


@dataclass
class EquityPoint:
    timestamp: float
    equity: float
    margin_used: float = 0.0
    cycle: int = 0


@dataclass
class TradeRecord:
    timestamp: float
    contract: str
    direction: str
    offset: str
    volume: int
    price: float
    profit_loss: float = 0.0
    commission: float = 0.0


@dataclass
class ContractResult:
    contract: str
    initial_equity: float
    final_equity: float
    total_return: float
    total_return_percent: float
    max_drawdown: float
    max_drawdown_percent: float
    total_trades: int
    winning_trades: int
    equity_curve: List[EquityPoint] = field(default_factory=list)
    trade_records: List[TradeRecord] = field(default_factory=list)


class EquityPlotter:
    
    def __init__(self, output_dir: str = None):
        self.logger = logging.getLogger(self.__class__.__name__)
        
        if output_dir is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            output_dir = os.path.join(base_dir, 'results')
        
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        _check_chinese_font_support()
    
    def plot_single_equity_curve(
        self,
        result: ContractResult,
        filename: str = None,
        title: str = None,
    ) -> Optional[str]:
        if not MATPLOTLIB_AVAILABLE:
            self.logger.error("matplotlib 未安装，无法生成图表")
            return None
        
        if not result.equity_curve:
            self.logger.warning(f"合约 {result.contract} 没有权益曲线数据")
            return None
        
        try:
            equities = [p.equity for p in result.equity_curve]
            cycles = list(range(len(equities)))
            initial_equity = result.initial_equity
            
            fig, ax = plt.subplots(figsize=(14, 7))
            
            ax.plot(cycles, equities, 'b-', linewidth=1.5, alpha=0.8, 
                    label=_get_label('权益曲线', 'Equity Curve'))
            
            ax.fill_between(cycles, equities, initial_equity,
                           where=np.array(equities) >= initial_equity,
                           alpha=0.3, color='green', interpolate=True)
            
            ax.fill_between(cycles, equities, initial_equity,
                           where=np.array(equities) < initial_equity,
                           alpha=0.3, color='red', interpolate=True)
            
            ax.axhline(y=initial_equity, color='gray', linestyle='--', alpha=0.5,
                      label=_get_label('初始资金', 'Initial Capital'))
            
            final_eq = equities[-1]
            ax.scatter([len(equities)-1], [final_eq], 
                      c='green' if final_eq >= initial_equity else 'red',
                      s=100, zorder=5, label=_get_label('最终权益', 'Final Equity'))
            
            if result.trade_records and len(result.equity_curve) > 0:
                self._plot_trade_signals(ax, result, cycles, equities)
            
            ax.set_xlabel(_get_label('回测周期', 'Backtest Period'), fontsize=12)
            ax.set_ylabel(_get_label('账户权益', 'Account Equity'), fontsize=12)
            
            if title:
                ax.set_title(title, fontsize=14, fontweight='bold')
            else:
                ax.set_title(
                    _get_label(
                        f'{result.contract} 权益曲线 (含交易信号)',
                        f'{result.contract} Equity Curve (with Signals)'
                    ),
                    fontsize=14, fontweight='bold'
                )
            
            return_pct = result.total_return_percent
            info_text = _get_label(
                f'初始资金: {initial_equity:,.0f}\n'
                f'最终权益: {final_eq:,.0f}\n'
                f'收益率: {return_pct:.2f}%\n'
                f'最大回撤: {result.max_drawdown_percent:.2f}%\n'
                f'交易次数: {result.total_trades}',
                f'Initial: {initial_equity:,.0f}\n'
                f'Final: {final_eq:,.0f}\n'
                f'Return: {return_pct:.2f}%\n'
                f'Max DD: {result.max_drawdown_percent:.2f}%\n'
                f'Trades: {result.total_trades}'
            )
            
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=10,
                   verticalalignment='top', bbox=props)
            
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if filename is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                safe_contract = result.contract.replace('.', '_').replace(' ', '_')
                filename = f'equity_{safe_contract}_{timestamp}.png'
            
            output_path = os.path.join(self.output_dir, filename)
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"权益曲线图表已保存: {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"生成权益曲线图表失败: {e}", exc_info=True)
            return None
    
    def _plot_trade_signals(self, ax, result: ContractResult, cycles: List[int], equities: List[float]):
        if not result.trade_records:
            return
        
        eq_timestamps = [p.timestamp for p in result.equity_curve]
        
        buy_open_cycles = []
        buy_open_equities = []
        buy_open_labels = []
        
        sell_open_cycles = []
        sell_open_equities = []
        sell_open_labels = []
        
        buy_close_cycles = []
        buy_close_equities = []
        buy_close_labels = []
        
        sell_close_cycles = []
        sell_close_equities = []
        sell_close_labels = []
        
        for trade in result.trade_records:
            trade_ts = trade.timestamp
            
            nearest_idx = 0
            min_diff = float('inf')
            for i, eq_ts in enumerate(eq_timestamps):
                diff = abs(eq_ts - trade_ts)
                if diff < min_diff:
                    min_diff = diff
                    nearest_idx = i
            
            equity_at_trade = equities[nearest_idx] if nearest_idx < len(equities) else result.initial_equity
            
            direction = trade.direction
            offset = trade.offset
            
            if direction == 'BUY' and offset == 'OPEN':
                buy_open_cycles.append(cycles[nearest_idx])
                buy_open_equities.append(equity_at_trade)
                buy_open_labels.append(_get_label(f'开多\n{trade.price:.2f}', f'Buy Open\n{trade.price:.2f}'))
            elif direction == 'SELL' and offset == 'OPEN':
                sell_open_cycles.append(cycles[nearest_idx])
                sell_open_equities.append(equity_at_trade)
                sell_open_labels.append(_get_label(f'开空\n{trade.price:.2f}', f'Sell Open\n{trade.price:.2f}'))
            elif direction == 'BUY' and offset == 'CLOSE':
                buy_close_cycles.append(cycles[nearest_idx])
                buy_close_equities.append(equity_at_trade)
                buy_close_labels.append(_get_label(f'平空\n{trade.price:.2f}', f'Buy Close\n{trade.price:.2f}'))
            elif direction == 'SELL' and offset == 'CLOSE':
                sell_close_cycles.append(cycles[nearest_idx])
                sell_close_equities.append(equity_at_trade)
                sell_close_labels.append(_get_label(f'平多\n{trade.price:.2f}', f'Sell Close\n{trade.price:.2f}'))
        
        if buy_open_cycles:
            ax.scatter(buy_open_cycles, buy_open_equities, 
                      c='limegreen', s=120, marker='^', zorder=10, edgecolors='darkgreen',
                      label=_get_label('开多信号 (买入)', 'Buy Open Signal'))
            
            for i, (cycle, equity, label) in enumerate(zip(buy_open_cycles, buy_open_equities, buy_open_labels)):
                if i % max(1, len(buy_open_cycles) // 10) == 0:
                    ax.annotate(label, (cycle, equity),
                               xytext=(5, 15), textcoords='offset points',
                               fontsize=7, alpha=0.8, color='darkgreen',
                               bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7))
        
        if sell_open_cycles:
            ax.scatter(sell_open_cycles, sell_open_equities, 
                      c='crimson', s=120, marker='v', zorder=10, edgecolors='darkred',
                      label=_get_label('开空信号 (卖出)', 'Sell Open Signal'))
            
            for i, (cycle, equity, label) in enumerate(zip(sell_open_cycles, sell_open_equities, sell_open_labels)):
                if i % max(1, len(sell_open_cycles) // 10) == 0:
                    ax.annotate(label, (cycle, equity),
                               xytext=(5, -25), textcoords='offset points',
                               fontsize=7, alpha=0.8, color='darkred',
                               bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7))
        
        if buy_close_cycles:
            ax.scatter(buy_close_cycles, buy_close_equities, 
                      c='deepskyblue', s=80, marker='D', zorder=9, edgecolors='navy',
                      label=_get_label('平空信号', 'Buy Close Signal'))
        
        if sell_close_cycles:
            ax.scatter(sell_close_cycles, sell_close_equities, 
                      c='orange', s=80, marker='s', zorder=9, edgecolors='darkorange',
                      label=_get_label('平多信号', 'Sell Close Signal'))
        
        self.logger.info(f"标注交易信号: 开多={len(buy_open_cycles)}, 开空={len(sell_open_cycles)}, 平空={len(buy_close_cycles)}, 平多={len(sell_close_cycles)}")
    
    def plot_multi_contract_comparison(
        self,
        results: List[ContractResult],
        filename: str = None,
        title: str = None,
    ) -> Optional[str]:
        if not MATPLOTLIB_AVAILABLE:
            self.logger.error("matplotlib 未安装，无法生成图表")
            return None
        
        if not results:
            self.logger.warning("没有合约结果数据")
            return None
        
        try:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            
            if title:
                fig.suptitle(title, fontsize=16, fontweight='bold')
            else:
                fig.suptitle(
                    _get_label('多合约回测结果对比', 'Multi-Contract Backtest Comparison'),
                    fontsize=16, fontweight='bold'
                )
            
            colors = ['blue', 'red', 'green', 'orange', 'purple', 'cyan', 'magenta', 'brown']
            
            ax1 = axes[0, 0]
            
            for idx, result in enumerate(results):
                if not result.equity_curve:
                    continue
                
                equities = [p.equity for p in result.equity_curve]
                cycles = list(range(len(equities)))
                color = colors[idx % len(colors)]
                
                initial = result.initial_equity
                normalized = [(eq / initial - 1) * 100 for eq in equities]
                
                ax1.plot(cycles, normalized, color=color, linewidth=1.5, alpha=0.8,
                        label=f'{result.contract}')
            
            ax1.set_xlabel(_get_label('回测周期', 'Backtest Period'), fontsize=11)
            ax1.set_ylabel(_get_label('收益率 (%)', 'Return (%)'), fontsize=11)
            ax1.set_title(
                _get_label('归一化权益曲线对比', 'Normalized Equity Curve Comparison'),
                fontsize=12, fontweight='bold'
            )
            ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            ax1.legend(loc='best', fontsize=9)
            ax1.grid(True, alpha=0.3)
            
            ax2 = axes[0, 1]
            
            contracts = [r.contract for r in results]
            returns = [r.total_return_percent for r in results]
            bar_colors = ['green' if r >= 0 else 'red' for r in returns]
            
            x_pos = range(len(contracts))
            bars = ax2.bar(x_pos, returns, color=bar_colors, alpha=0.7)
            
            for bar, ret in zip(bars, returns):
                ax2.text(bar.get_x() + bar.get_width()/2, 
                        bar.get_height() + (0.5 if ret >= 0 else -0.8),
                        f'{ret:.2f}%',
                        ha='center', va='bottom' if ret >= 0 else 'top',
                        fontsize=10)
            
            ax2.set_xlabel(_get_label('合约', 'Contract'), fontsize=11)
            ax2.set_ylabel(_get_label('总收益率 (%)', 'Total Return (%)'), fontsize=11)
            ax2.set_title(
                _get_label('各合约收益率对比', 'Return Comparison'),
                fontsize=12, fontweight='bold'
            )
            ax2.set_xticks(x_pos)
            ax2.set_xticklabels(contracts, rotation=45, ha='right')
            ax2.axhline(y=0, color='black', linewidth=0.5)
            ax2.grid(True, alpha=0.3, axis='y')
            
            ax3 = axes[1, 0]
            
            max_dds = [r.max_drawdown_percent for r in results]
            dd_colors = ['red' if dd > 5 else 'orange' if dd > 3 else 'green' for dd in max_dds]
            
            bars_dd = ax3.bar(x_pos, max_dds, color=dd_colors, alpha=0.7)
            
            for bar, dd in zip(bars_dd, max_dds):
                ax3.text(bar.get_x() + bar.get_width()/2, 
                        bar.get_height() + 0.1,
                        f'{dd:.2f}%',
                        ha='center', va='bottom', fontsize=10)
            
            ax3.set_xlabel(_get_label('合约', 'Contract'), fontsize=11)
            ax3.set_ylabel(_get_label('最大回撤率 (%)', 'Max Drawdown (%)'), fontsize=11)
            ax3.set_title(
                _get_label('各合约最大回撤对比', 'Max Drawdown Comparison'),
                fontsize=12, fontweight='bold'
            )
            ax3.set_xticks(x_pos)
            ax3.set_xticklabels(contracts, rotation=45, ha='right')
            ax3.grid(True, alpha=0.3, axis='y')
            
            ax4 = axes[1, 1]
            
            trade_counts = [r.total_trades for r in results]
            win_rates = []
            for r in results:
                if r.total_trades > 0:
                    win_rates.append(r.winning_trades / r.total_trades * 100)
                else:
                    win_rates.append(0)
            
            width = 0.35
            x_pos_array = np.arange(len(contracts))
            
            bars_trade = ax4.bar(x_pos_array - width/2, trade_counts, width, 
                                color='steelblue', alpha=0.7, 
                                label=_get_label('交易次数', 'Trade Count'))
            
            ax4_twin = ax4.twinx()
            bars_winrate = ax4_twin.bar(x_pos_array + width/2, win_rates, width,
                                      color='orange', alpha=0.5,
                                      label=_get_label('胜率 (%)', 'Win Rate (%)'))
            
            ax4.set_xlabel(_get_label('合约', 'Contract'), fontsize=11)
            ax4.set_ylabel(_get_label('交易次数', 'Trade Count'), fontsize=11, color='steelblue')
            ax4_twin.set_ylabel(_get_label('胜率 (%)', 'Win Rate (%)'), fontsize=11, color='orange')
            ax4.set_title(
                _get_label('各合约交易统计', 'Trade Statistics'),
                fontsize=12, fontweight='bold'
            )
            ax4.set_xticks(x_pos_array)
            ax4.set_xticklabels(contracts, rotation=45, ha='right')
            
            lines1, labels1 = ax4.get_legend_handles_labels()
            lines2, labels2 = ax4_twin.get_legend_handles_labels()
            ax4.legend(lines1 + lines2, labels1 + labels2, loc='best', fontsize=9)
            
            plt.tight_layout(rect=[0, 0, 1, 0.97])
            
            if filename is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f'multi_contract_comparison_{timestamp}.png'
            
            output_path = os.path.join(self.output_dir, filename)
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"多合约对比图表已保存: {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"生成多合约对比图表失败: {e}", exc_info=True)
            return None
    
    def generate_summary_report(
        self,
        results: List[ContractResult],
        filename: str = None,
    ) -> Dict[str, Any]:
        summary = {
            'generated_at': datetime.now().isoformat(),
            'contracts': [],
            'summary': {},
        }
        
        total_initial = 0.0
        total_final = 0.0
        total_trades = 0
        total_wins = 0
        total_max_dd = 0.0
        
        for result in results:
            contract_data = {
                'contract': result.contract,
                'initial_equity': result.initial_equity,
                'final_equity': result.final_equity,
                'total_return': result.total_return,
                'total_return_percent': result.total_return_percent,
                'max_drawdown': result.max_drawdown,
                'max_drawdown_percent': result.max_drawdown_percent,
                'total_trades': result.total_trades,
                'winning_trades': result.winning_trades,
                'win_rate': result.winning_trades / result.total_trades * 100 if result.total_trades > 0 else 0.0,
                'equity_points_count': len(result.equity_curve),
                'trade_records_count': len(result.trade_records),
            }
            summary['contracts'].append(contract_data)
            
            total_initial += result.initial_equity
            total_final += result.final_equity
            total_trades += result.total_trades
            total_wins += result.winning_trades
            total_max_dd = max(total_max_dd, result.max_drawdown_percent)
        
        combined_return = total_final - total_initial
        combined_return_pct = (combined_return / total_initial * 100) if total_initial > 0 else 0.0
        
        summary['summary'] = {
            'total_contracts': len(results),
            'combined_initial_equity': total_initial,
            'combined_final_equity': total_final,
            'combined_return': combined_return,
            'combined_return_percent': combined_return_pct,
            'total_trades': total_trades,
            'total_wins': total_wins,
            'overall_win_rate': total_wins / total_trades * 100 if total_trades > 0 else 0.0,
            'max_drawdown_across_contracts': total_max_dd,
        }
        
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'summary_report_{timestamp}.json'
        
        output_path = os.path.join(self.output_dir, filename)
        
        try:
            import json
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
            self.logger.info(f"汇总报告已保存: {output_path}")
        except Exception as e:
            self.logger.error(f"保存汇总报告失败: {e}")
        
        return summary
    
    def generate_trade_details_csv(
        self,
        results: List[ContractResult],
        filename: str = None,
    ) -> Optional[str]:
        if not results:
            return None
        
        import csv
        
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'trade_details_{timestamp}.csv'
        
        output_path = os.path.join(self.output_dir, filename)
        
        try:
            fieldnames = [
                'contract', 'timestamp', 'direction', 'offset', 'volume',
                'price', 'profit_loss', 'commission', 'datetime'
            ]
            
            with open(output_path, 'w', encoding='utf-8-sig', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                
                for result in results:
                    for record in result.trade_records:
                        row = {
                            'contract': result.contract,
                            'timestamp': record.timestamp,
                            'direction': record.direction,
                            'offset': record.offset,
                            'volume': record.volume,
                            'price': record.price,
                            'profit_loss': record.profit_loss,
                            'commission': record.commission,
                            'datetime': datetime.fromtimestamp(record.timestamp).strftime('%Y-%m-%d %H:%M:%S') if record.timestamp > 0 else '',
                        }
                        writer.writerow(row)
            
            self.logger.info(f"交易明细已保存: {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"保存交易明细失败: {e}")
            return None


def create_contract_result_from_backtest(
    contract: str,
    initial_equity: float,
    final_equity: float,
    equity_curve_data: List[Dict[str, Any]] = None,
    trade_records_data: List[Dict[str, Any]] = None,
) -> ContractResult:
    total_return = final_equity - initial_equity
    total_return_percent = (total_return / initial_equity * 100) if initial_equity > 0 else 0.0
    
    equity_points = []
    if equity_curve_data:
        for item in equity_curve_data:
            point = EquityPoint(
                timestamp=item.get('timestamp', 0.0),
                equity=item.get('equity', initial_equity),
                margin_used=item.get('margin_used', 0.0),
                cycle=item.get('cycle', 0),
            )
            equity_points.append(point)
    
    trade_records = []
    winning_trades = 0
    if trade_records_data:
        for item in trade_records_data:
            record = TradeRecord(
                timestamp=item.get('timestamp', 0.0),
                contract=item.get('contract', contract),
                direction=item.get('direction', ''),
                offset=item.get('offset', ''),
                volume=item.get('volume', 0),
                price=item.get('price', 0.0),
                profit_loss=item.get('profit_loss', 0.0),
                commission=item.get('commission', 0.0),
            )
            trade_records.append(record)
            
            if record.profit_loss > 0:
                winning_trades += 1
    
    max_drawdown = 0.0
    max_drawdown_percent = 0.0
    
    if equity_points:
        equities = [p.equity for p in equity_points]
        peak = initial_equity
        for eq in equities:
            if eq > peak:
                peak = eq
            dd = peak - eq
            dd_pct = (dd / peak * 100) if peak > 0 else 0.0
            
            if dd > max_drawdown:
                max_drawdown = dd
            if dd_pct > max_drawdown_percent:
                max_drawdown_percent = dd_pct
    
    return ContractResult(
        contract=contract,
        initial_equity=initial_equity,
        final_equity=final_equity,
        total_return=total_return,
        total_return_percent=total_return_percent,
        max_drawdown=max_drawdown,
        max_drawdown_percent=max_drawdown_percent,
        total_trades=len(trade_records),
        winning_trades=winning_trades,
        equity_curve=equity_points,
        trade_records=trade_records,
    )
