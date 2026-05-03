import os
import sys
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict

from flask import Flask, render_template_string, jsonify, request

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, base_dir)

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import findfont, FontProperties
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    plt = None

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


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('WebDashboard')

app = Flask(__name__)

dashboard_data = {
    'account': {
        'equity': 1000000.0,
        'balance': 1000000.0,
        'margin_used': 0.0,
        'available': 1000000.0,
        'float_profit': 0.0,
    },
    'positions': [],
    'strategies': [],
    'risk_info': {
        'current_drawdown_percent': 0.0,
        'is_frozen': False,
        'frozen_reason': None,
    },
    'status': {
        'is_running': False,
        'start_time': None,
        'cycle_count': 0,
    },
    'multi_contract': {
        'total_capital': 2000000.0,
        'contracts': {},
        'capital_allocation': {
            'contract_weights': {},
            'allocated_capitals': {},
        },
    }
}


@dataclass
class PositionDisplay:
    contract: str
    direction: str
    volume: int
    open_price: float
    current_price: float
    float_profit: float
    margin: float


@dataclass
class StrategyDisplay:
    name: str
    contract: str
    short_ma: Optional[float]
    long_ma: Optional[float]
    short_period: int
    long_period: int
    signal: str
    is_ready: bool
    status: str
    error_count: int


@dataclass
class BacktestReport:
    filename: str
    path: str
    generated_at: str
    strategy_name: str
    initial_equity: float
    final_equity: float
    total_return: float
    total_return_percent: float
    max_drawdown_percent: float
    total_trades: int
    win_rate: float
    status: str


def load_backtest_reports() -> List[BacktestReport]:
    reports_dir = os.path.join(base_dir, 'logs', 'backtest_reports')
    reports = []
    
    if not os.path.exists(reports_dir):
        return reports
    
    for filename in os.listdir(reports_dir):
        if filename.startswith('backtest_report_') and filename.endswith('.json'):
            filepath = os.path.join(reports_dir, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                best_result = data.get('best_result', {})
                performance = best_result.get('performance', {})
                
                report = BacktestReport(
                    filename=filename,
                    path=filepath,
                    generated_at=data.get('generated_at', ''),
                    strategy_name=best_result.get('strategy_name', 'Unknown'),
                    initial_equity=best_result.get('initial_equity', 0.0),
                    final_equity=best_result.get('final_equity', 0.0),
                    total_return=performance.get('total_return', 0.0),
                    total_return_percent=performance.get('total_return_percent', 0.0),
                    max_drawdown_percent=performance.get('max_drawdown_percent', 0.0),
                    total_trades=performance.get('total_trades', 0),
                    win_rate=performance.get('win_rate', 0.0),
                    status=best_result.get('status', 'completed'),
                )
                reports.append(report)
            except Exception as e:
                logger.warning(f"加载回测报告失败 {filename}: {e}")
    
    reports.sort(key=lambda x: x.generated_at, reverse=True)
    return reports


def load_backtest_report_detail(filepath: str) -> Dict[str, Any]:
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"加载回测报告详情失败: {e}")
        return {}


def generate_equity_chart(equity_curve: List[Dict[str, Any]], initial_equity: float) -> Optional[str]:
    if not MATPLOTLIB_AVAILABLE:
        logger.error("matplotlib 未安装，无法生成图表")
        return None
    
    if not equity_curve:
        logger.warning("没有权益曲线数据")
        return None
    
    try:
        _check_chinese_font_support()
        
        equities = [initial_equity] + [p.get('equity', initial_equity) for p in equity_curve]
        cycles = list(range(len(equities)))
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        ax1 = axes[0]
        ax1.plot(cycles, equities, 'b-', linewidth=1.5, alpha=0.8,
                label=_get_label('权益曲线', 'Equity Curve'))
        
        ax1.fill_between(cycles, equities, initial_equity,
                        where=np.array(equities) >= initial_equity if NUMPY_AVAILABLE else [e >= initial_equity for e in equities],
                        alpha=0.3, color='green', interpolate=True)
        
        ax1.fill_between(cycles, equities, initial_equity,
                        where=np.array(equities) < initial_equity if NUMPY_AVAILABLE else [e < initial_equity for e in equities],
                        alpha=0.3, color='red', interpolate=True)
        
        ax1.axhline(y=initial_equity, color='gray', linestyle='--', alpha=0.5,
                   label=_get_label('初始资金', 'Initial Capital'))
        
        ax1.set_xlabel(_get_label('回测周期', 'Backtest Period'), fontsize=12)
        ax1.set_ylabel(_get_label('账户权益', 'Account Equity'), fontsize=12)
        ax1.set_title(_get_label('权益曲线图', 'Equity Curve'), fontsize=14, fontweight='bold')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        
        ax2 = axes[1]
        drawdowns = []
        peak = initial_equity
        
        for eq in equities:
            if eq > peak:
                peak = eq
            dd = (peak - eq) / peak * 100 if peak > 0 else 0
            drawdowns.append(dd)
        
        ax2.fill_between(cycles, drawdowns, 0, color='red', alpha=0.5,
                        label=_get_label('回撤', 'Drawdown'))
        ax2.set_xlabel(_get_label('回测周期', 'Backtest Period'), fontsize=12)
        ax2.set_ylabel(_get_label('回撤率 (%)', 'Drawdown (%)'), fontsize=12)
        ax2.set_title(_get_label('回撤分布图', 'Drawdown Distribution'), fontsize=14, fontweight='bold')
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3)
        ax2.invert_yaxis()
        
        plt.tight_layout()
        
        static_dir = os.path.join(base_dir, 'gui', 'static')
        os.makedirs(static_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        chart_filename = f'equity_chart_{timestamp}.png'
        output_path = os.path.join(static_dir, chart_filename)
        
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"权益曲线图表已保存: {output_path}")
        return chart_filename
        
    except Exception as e:
        logger.error(f"生成权益曲线图表失败: {e}", exc_info=True)
        return None


INDEX_TEMPLATE = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>期货策略监控系统</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            min-height: 100vh;
            color: #333;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }
        header {
            text-align: center;
            padding: 20px 0;
            color: white;
            margin-bottom: 20px;
        }
        header h1 {
            font-size: 2.5rem;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        header p {
            font-size: 1.1rem;
            opacity: 0.9;
        }
        .nav-tabs {
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
        }
        .nav-tab {
            padding: 12px 30px;
            background: rgba(255,255,255,0.1);
            border: none;
            color: white;
            font-size: 1rem;
            cursor: pointer;
            border-radius: 8px;
            transition: all 0.3s ease;
        }
        .nav-tab:hover {
            background: rgba(255,255,255,0.2);
        }
        .nav-tab.active {
            background: white;
            color: #1e3c72;
            font-weight: bold;
        }
        .panel {
            background: white;
            border-radius: 12px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
            padding: 24px;
            margin-bottom: 20px;
        }
        .panel h2 {
            color: #1e3c72;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid #eee;
        }
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }
        .metric-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            border-radius: 12px;
            color: white;
            text-align: center;
        }
        .metric-card.green {
            background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        }
        .metric-card.red {
            background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%);
        }
        .metric-card.orange {
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        }
        .metric-card.blue {
            background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        }
        .metric-value {
            font-size: 2rem;
            font-weight: bold;
            margin-bottom: 5px;
        }
        .metric-label {
            font-size: 0.9rem;
            opacity: 0.9;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 10px;
        }
        th, td {
            padding: 12px 15px;
            text-align: left;
            border-bottom: 1px solid #eee;
        }
        th {
            background: #f8f9fa;
            font-weight: 600;
            color: #1e3c72;
        }
        tr:hover {
            background: #f8f9fa;
        }
        .signal-buy {
            color: #28a745;
            font-weight: bold;
        }
        .signal-sell {
            color: #dc3545;
            font-weight: bold;
        }
        .signal-hold {
            color: #6c757d;
            font-weight: bold;
        }
        .status-running {
            color: #28a745;
        }
        .status-stopped {
            color: #6c757d;
        }
        .status-frozen {
            color: #dc3545;
            font-weight: bold;
        }
        .report-row {
            cursor: pointer;
        }
        .report-row:hover {
            background: #e9ecef;
        }
        .profit-positive {
            color: #28a745;
            font-weight: bold;
        }
        .profit-negative {
            color: #dc3545;
            font-weight: bold;
        }
        .chart-container {
            text-align: center;
            margin-top: 20px;
        }
        .chart-container img {
            max-width: 100%;
            border-radius: 8px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }
        .back-button {
            background: #1e3c72;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 6px;
            cursor: pointer;
            margin-bottom: 15px;
            font-size: 1rem;
        }
        .back-button:hover {
            background: #2a5298;
        }
        .tab-content {
            display: none;
        }
        .tab-content.active {
            display: block;
        }
        .refresh-btn {
            background: #28a745;
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 6px;
            cursor: pointer;
            margin-left: 10px;
        }
        .refresh-btn:hover {
            background: #218838;
        }
        .section-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>📊 期货策略监控系统</h1>
            <p>基于 TqSdk 的实时策略监控与回测分析平台</p>
        </header>
        
        <div class="nav-tabs">
            <button class="nav-tab active" onclick="switchTab('dashboard')">实时监控</button>
            <button class="nav-tab" onclick="switchTab('multi-contract')">多合约状态</button>
            <button class="nav-tab" onclick="switchTab('backtest')">回测分析</button>
        </div>
        
        <div id="dashboard-tab" class="tab-content active">
            <div class="panel">
                <div class="section-header">
                    <h2>💼 账户概览</h2>
                    <button class="refresh-btn" onclick="refreshDashboard()">🔄 刷新</button>
                </div>
                <div class="metrics-grid">
                    <div class="metric-card green" id="equity-card">
                        <div class="metric-value" id="account-equity">¥0</div>
                        <div class="metric-label">账户权益</div>
                    </div>
                    <div class="metric-card blue" id="balance-card">
                        <div class="metric-value" id="account-balance">¥0</div>
                        <div class="metric-label">账户余额</div>
                    </div>
                    <div class="metric-card" id="margin-card">
                        <div class="metric-value" id="margin-used">¥0</div>
                        <div class="metric-label">已用保证金</div>
                    </div>
                    <div class="metric-card orange" id="available-card">
                        <div class="metric-value" id="available-funds">¥0</div>
                        <div class="metric-label">可用资金</div>
                    </div>
                </div>
                <div class="metrics-grid">
                    <div class="metric-card" id="profit-card">
                        <div class="metric-value" id="float-profit">¥0</div>
                        <div class="metric-label">持仓盈亏</div>
                    </div>
                    <div class="metric-card red" id="drawdown-card">
                        <div class="metric-value" id="current-drawdown">0%</div>
                        <div class="metric-label">当前回撤</div>
                    </div>
                    <div class="metric-card blue" id="status-card">
                        <div class="metric-value" id="system-status">停止</div>
                        <div class="metric-label">系统状态</div>
                    </div>
                    <div class="metric-card" id="cycle-card">
                        <div class="metric-value" id="cycle-count">0</div>
                        <div class="metric-label">运行周期</div>
                    </div>
                </div>
            </div>
            
            <div class="panel">
                <h2>📈 持仓列表</h2>
                <table id="positions-table">
                    <thead>
                        <tr>
                            <th>合约</th>
                            <th>方向</th>
                            <th>持仓数量</th>
                            <th>开仓价</th>
                            <th>当前价</th>
                            <th>浮动盈亏</th>
                            <th>占用保证金</th>
                        </tr>
                    </thead>
                    <tbody id="positions-body">
                        <tr>
                            <td colspan="7" style="text-align: center; color: #999;">暂无持仓数据</td>
                        </tr>
                    </tbody>
                </table>
            </div>
            
            <div class="panel">
                <h2>🎯 策略状态</h2>
                <table id="strategies-table">
                    <thead>
                        <tr>
                            <th>策略名称</th>
                            <th>交易合约</th>
                            <th>短期均线</th>
                            <th>长期均线</th>
                            <th>当前信号</th>
                            <th>状态</th>
                            <th>错误次数</th>
                        </tr>
                    </thead>
                    <tbody id="strategies-body">
                        <tr>
                            <td colspan="7" style="text-align: center; color: #999;">暂无策略数据</td>
                        </tr>
                    </tbody>
                </table>
            </div>
        </div>
        
        <div id="multi-contract-tab" class="tab-content">
            <div class="panel">
                <div class="section-header">
                    <h2>💰 资金池概览</h2>
                    <button class="refresh-btn" onclick="refreshMultiContractStatus()">🔄 刷新</button>
                </div>
                <div class="metrics-grid">
                    <div class="metric-card green" id="total-capital-card">
                        <div class="metric-value" id="total-capital">¥0</div>
                        <div class="metric-label">总资金</div>
                    </div>
                    <div class="metric-card blue" id="contract-count-card">
                        <div class="metric-value" id="contract-count">0</div>
                        <div class="metric-label">交易合约数</div>
                    </div>
                    <div class="metric-card" id="total-positions-card">
                        <div class="metric-value" id="total-positions">0</div>
                        <div class="metric-label">当前持仓数</div>
                    </div>
                    <div class="metric-card orange" id="total-float-profit-card">
                        <div class="metric-value" id="total-float-profit">¥0</div>
                        <div class="metric-label">总浮动盈亏</div>
                    </div>
                </div>
            </div>
            
            <div class="panel">
                <h2>📊 资金分配比例</h2>
                <div id="capital-allocation-container">
                    <p style="text-align: center; color: #999;">暂无资金分配数据</p>
                </div>
            </div>
            
            <div class="panel">
                <h2>📋 合约状态详情</h2>
                <div id="contracts-status-container">
                    <p style="text-align: center; color: #999;">暂无合约状态数据</p>
                </div>
            </div>
        </div>
        
        <div id="backtest-tab" class="tab-content">
            <div id="backtest-list">
                <div class="panel">
                    <div class="section-header">
                        <h2>📊 回测报告列表</h2>
                        <button class="refresh-btn" onclick="refreshBacktestList()">🔄 刷新</button>
                    </div>
                    <table>
                        <thead>
                            <tr>
                                <th>生成时间</th>
                                <th>策略名称</th>
                                <th>初始权益</th>
                                <th>最终权益</th>
                                <th>总收益率</th>
                                <th>最大回撤</th>
                                <th>交易次数</th>
                                <th>状态</th>
                            </tr>
                        </thead>
                        <tbody id="reports-body">
                            <tr>
                                <td colspan="8" style="text-align: center; color: #999;">暂无回测报告</td>
                            </tr>
                        </tbody>
                    </table>
                </div>
            </div>
            
            <div id="backtest-detail" style="display: none;">
                <div class="panel">
                    <button class="back-button" onclick="showBacktestList()">← 返回报告列表</button>
                    <h2 id="detail-title">回测详情</h2>
                    
                    <div class="metrics-grid" id="detail-metrics">
                    </div>
                    
                    <div class="chart-container" id="chart-container">
                    </div>
                    
                    <h3 style="margin-top: 30px; color: #1e3c72;">交易记录</h3>
                    <table id="trades-table">
                        <thead>
                            <tr>
                                <th>周期</th>
                                <th>时间</th>
                                <th>权益</th>
                                <th>保证金</th>
                            </tr>
                        </thead>
                        <tbody id="trades-body">
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    </div>

    <script>
        function switchTab(tabName) {
            document.querySelectorAll('.nav-tab').forEach(tab => tab.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(content => content.classList.remove('active'));
            
            event.target.classList.add('active');
            document.getElementById(tabName + '-tab').classList.add('active');
        }

        function formatCurrency(value) {
            if (value >= 0) {
                return '¥' + value.toLocaleString('zh-CN', {minimumFractionDigits: 2, maximumFractionDigits: 2});
            } else {
                return '-¥' + Math.abs(value).toLocaleString('zh-CN', {minimumFractionDigits: 2, maximumFractionDigits: 2});
            }
        }

        function formatPercent(value) {
            return value.toFixed(2) + '%';
        }

        function refreshDashboard() {
            fetch('/api/dashboard')
                .then(response => response.json())
                .then(data => {
                    updateDashboard(data);
                })
                .catch(error => console.error('Error:', error));
        }

        function updateDashboard(data) {
            const account = data.account;
            const risk = data.risk_info;
            const status = data.status;
            
            document.getElementById('account-equity').textContent = formatCurrency(account.equity);
            document.getElementById('account-balance').textContent = formatCurrency(account.balance);
            document.getElementById('margin-used').textContent = formatCurrency(account.margin_used);
            document.getElementById('available-funds').textContent = formatCurrency(account.available);
            
            const profitElement = document.getElementById('float-profit');
            profitElement.textContent = formatCurrency(account.float_profit);
            const profitCard = document.getElementById('profit-card');
            profitCard.className = 'metric-card ' + (account.float_profit >= 0 ? 'green' : 'red');
            
            document.getElementById('current-drawdown').textContent = formatPercent(risk.current_drawdown_percent);
            
            let statusText = '停止';
            let statusClass = 'status-stopped';
            if (risk.is_frozen) {
                statusText = '已冻结';
                statusClass = 'status-frozen';
            } else if (status.is_running) {
                statusText = '运行中';
                statusClass = 'status-running';
            }
            document.getElementById('system-status').textContent = statusText;
            document.getElementById('system-status').className = statusClass;
            
            document.getElementById('cycle-count').textContent = status.cycle_count;
            
            updatePositionsTable(data.positions);
            updateStrategiesTable(data.strategies);
        }

        function updatePositionsTable(positions) {
            const tbody = document.getElementById('positions-body');
            if (!positions || positions.length === 0) {
                tbody.innerHTML = '<tr><td colspan="7" style="text-align: center; color: #999;">暂无持仓数据</td></tr>';
                return;
            }
            
            tbody.innerHTML = positions.map(pos => `
                <tr>
                    <td>${pos.contract}</td>
                    <td>${pos.direction}</td>
                    <td>${pos.volume}</td>
                    <td>${pos.open_price.toFixed(2)}</td>
                    <td>${pos.current_price.toFixed(2)}</td>
                    <td class="${pos.float_profit >= 0 ? 'profit-positive' : 'profit-negative'}">${formatCurrency(pos.float_profit)}</td>
                    <td>${formatCurrency(pos.margin)}</td>
                </tr>
            `).join('');
        }

        function updateStrategiesTable(strategies) {
            const tbody = document.getElementById('strategies-body');
            if (!strategies || strategies.length === 0) {
                tbody.innerHTML = '<tr><td colspan="7" style="text-align: center; color: #999;">暂无策略数据</td></tr>';
                return;
            }
            
            tbody.innerHTML = strategies.map(strat => {
                let signalClass = 'signal-hold';
                if (strat.signal === 'BUY') signalClass = 'signal-buy';
                else if (strat.signal === 'SELL') signalClass = 'signal-sell';
                
                return `
                    <tr>
                        <td>${strat.name}</td>
                        <td>${strat.contract}</td>
                        <td>${strat.short_ma !== null ? strat.short_ma.toFixed(2) : 'N/A'}</td>
                        <td>${strat.long_ma !== null ? strat.long_ma.toFixed(2) : 'N/A'}</td>
                        <td class="${signalClass}">${strat.signal}</td>
                        <td>${strat.status}</td>
                        <td>${strat.error_count}</td>
                    </tr>
                `;
            }).join('');
        }

        function refreshBacktestList() {
            fetch('/api/backtest/reports')
                .then(response => response.json())
                .then(data => {
                    updateBacktestList(data.reports);
                })
                .catch(error => console.error('Error:', error));
        }

        function updateBacktestList(reports) {
            const tbody = document.getElementById('reports-body');
            if (!reports || reports.length === 0) {
                tbody.innerHTML = '<tr><td colspan="8" style="text-align: center; color: #999;">暂无回测报告</td></tr>';
                return;
            }
            
            tbody.innerHTML = reports.map(report => `
                <tr class="report-row" onclick="viewBacktestReport('${report.filename}')">
                    <td>${report.generated_at}</td>
                    <td>${report.strategy_name}</td>
                    <td>${formatCurrency(report.initial_equity)}</td>
                    <td>${formatCurrency(report.final_equity)}</td>
                    <td class="${report.total_return_percent >= 0 ? 'profit-positive' : 'profit-negative'}">${formatPercent(report.total_return_percent)}</td>
                    <td>${formatPercent(report.max_drawdown_percent)}</td>
                    <td>${report.total_trades}</td>
                    <td>${report.status}</td>
                </tr>
            `).join('');
        }

        function viewBacktestReport(filename) {
            fetch('/api/backtest/report/' + encodeURIComponent(filename))
                .then(response => response.json())
                .then(data => {
                    showBacktestDetail(data);
                })
                .catch(error => console.error('Error:', error));
        }

        function showBacktestDetail(data) {
            document.getElementById('backtest-list').style.display = 'none';
            document.getElementById('backtest-detail').style.display = 'block';
            
            const best = data.best_result || {};
            const perf = best.performance || {};
            
            document.getElementById('detail-title').textContent = `回测详情 - ${best.strategy_name || '未知策略'}`;
            
            document.getElementById('detail-metrics').innerHTML = `
                <div class="metric-card green">
                    <div class="metric-value">${formatCurrency(best.initial_equity || 0)}</div>
                    <div class="metric-label">初始权益</div>
                </div>
                <div class="metric-card ${(perf.total_return_percent || 0) >= 0 ? 'green' : 'red'}">
                    <div class="metric-value">${formatCurrency(best.final_equity || 0)}</div>
                    <div class="metric-label">最终权益</div>
                </div>
                <div class="metric-card ${(perf.total_return_percent || 0) >= 0 ? 'green' : 'red'}">
                    <div class="metric-value">${formatPercent(perf.total_return_percent || 0)}</div>
                    <div class="metric-label">总收益率</div>
                </div>
                <div class="metric-card red">
                    <div class="metric-value">${formatPercent(perf.max_drawdown_percent || 0)}</div>
                    <div class="metric-label">最大回撤</div>
                </div>
                <div class="metric-card blue">
                    <div class="metric-value">${perf.total_trades || 0}</div>
                    <div class="metric-label">交易次数</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">${formatPercent(perf.win_rate || 0)}</div>
                    <div class="metric-label">胜率</div>
                </div>
            `;
            
            if (data.chart_url) {
                document.getElementById('chart-container').innerHTML = `<img src="${data.chart_url}" alt="权益曲线图">`;
            } else {
                document.getElementById('chart-container').innerHTML = '<p style="color: #999;">暂无图表数据</p>';
            }
            
            const equityCurve = best.equity_curve || [];
            if (equityCurve.length > 0) {
                document.getElementById('trades-body').innerHTML = equityCurve.slice(0, 20).map(point => `
                    <tr>
                        <td>${point.cycle || '-'}</td>
                        <td>${point.timestamp ? new Date(point.timestamp * 1000).toLocaleString() : '-'}</td>
                        <td>${formatCurrency(point.equity || 0)}</td>
                        <td>${formatCurrency(point.margin_used || 0)}</td>
                    </tr>
                `).join('');
            } else {
                document.getElementById('trades-body').innerHTML = '<tr><td colspan="4" style="text-align: center; color: #999;">暂无交易记录</td></tr>';
            }
        }

        function showBacktestList() {
            document.getElementById('backtest-list').style.display = 'block';
            document.getElementById('backtest-detail').style.display = 'none';
        }

        function refreshMultiContractStatus() {
            fetch('/api/multi-contract')
                .then(response => response.json())
                .then(data => {
                    updateMultiContractStatus(data);
                })
                .catch(error => console.error('Error:', error));
        }

        function updateMultiContractStatus(data) {
            const multiContract = data.multi_contract || {};
            const contracts = multiContract.contracts || {};
            const capitalAllocation = multiContract.capital_allocation || {};
            const contractWeights = capitalAllocation.contract_weights || {};
            const allocatedCapitals = capitalAllocation.allocated_capitals || {};
            
            document.getElementById('total-capital').textContent = formatCurrency(multiContract.total_capital || 0);
            document.getElementById('contract-count').textContent = Object.keys(contracts).length;
            
            let totalPositions = 0;
            let totalFloatProfit = 0;
            for (const contract in contracts) {
                const state = contracts[contract];
                if (state.position_volume > 0) {
                    totalPositions++;
                }
                totalFloatProfit += state.float_profit || 0;
            }
            
            document.getElementById('total-positions').textContent = totalPositions;
            
            const profitElement = document.getElementById('total-float-profit');
            profitElement.textContent = formatCurrency(totalFloatProfit);
            const profitCard = document.getElementById('total-float-profit-card');
            profitCard.className = 'metric-card ' + (totalFloatProfit >= 0 ? 'green' : 'red');
            
            const allocationContainer = document.getElementById('capital-allocation-container');
            if (Object.keys(contractWeights).length > 0) {
                let allocationHtml = '<table><thead><tr><th>合约</th><th>资金权重</th><th>分配资金</th><th>波动率比</th></tr></thead><tbody>';
                
                for (const contract in contractWeights) {
                    const weight = contractWeights[contract];
                    const capital = allocatedCapitals[contract] || 0;
                    const state = contracts[contract] || {};
                    const volRatio = state.volatility_ratio || 1.0;
                    
                    allocationHtml += `
                        <tr>
                            <td>${contract}</td>
                            <td>${(weight * 100).toFixed(2)}%</td>
                            <td>${formatCurrency(capital)}</td>
                            <td>${volRatio.toFixed(4)}x</td>
                        </tr>
                    `;
                }
                
                allocationHtml += '</tbody></table>';
                allocationContainer.innerHTML = allocationHtml;
            } else {
                allocationContainer.innerHTML = '<p style="text-align: center; color: #999;">暂无资金分配数据</p>';
            }
            
            const contractsContainer = document.getElementById('contracts-status-container');
            if (Object.keys(contracts).length > 0) {
                let contractsHtml = '';
                
                for (const contract in contracts) {
                    const state = contracts[contract];
                    
                    const signalClass = state.signal === 'BUY' ? 'signal-buy' : 
                                       state.signal === 'SELL' ? 'signal-sell' : 'signal-hold';
                    
                    const positionDirection = state.position_direction === 'LONG' ? '多单' : 
                                             state.position_direction === 'SHORT' ? '空单' : '无持仓';
                    
                    const profitClass = (state.float_profit || 0) >= 0 ? 'profit-positive' : 'profit-negative';
                    
                    contractsHtml += `
                        <div class="contract-status-card" style="border: 1px solid #eee; border-radius: 8px; padding: 20px; margin-bottom: 20px; background: #fafafa;">
                            <h3 style="margin-top: 0; color: #1e3c72; border-bottom: 2px solid #eee; padding-bottom: 10px;">${contract}</h3>
                            
                            <div class="metrics-grid" style="grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 15px; margin-bottom: 15px;">
                                <div style="background: white; padding: 15px; border-radius: 8px; text-align: center;">
                                    <div style="font-size: 1.5rem; font-weight: bold; color: #1e3c72;">${state.status || 'UNKNOWN'}</div>
                                    <div style="font-size: 0.85rem; color: #666;">状态</div>
                                </div>
                                <div style="background: white; padding: 15px; border-radius: 8px; text-align: center;">
                                    <div style="font-size: 1.5rem; font-weight: bold; ${signalClass === 'signal-buy' ? 'color: #28a745;' : signalClass === 'signal-sell' ? 'color: #dc3545;' : 'color: #6c757d;'}">${state.signal || 'HOLD'}</div>
                                    <div style="font-size: 0.85rem; color: #666;">当前信号</div>
                                </div>
                                <div style="background: white; padding: 15px; border-radius: 8px; text-align: center;">
                                    <div style="font-size: 1.5rem; font-weight: bold; color: #1e3c72;">${state.position_volume || 0}</div>
                                    <div style="font-size: 0.85rem; color: #666;">持仓手数</div>
                                </div>
                                <div style="background: white; padding: 15px; border-radius: 8px; text-align: center;">
                                    <div style="font-size: 1.5rem; font-weight: bold; ${profitClass === 'profit-positive' ? 'color: #28a745;' : 'color: #dc3545;'}">${formatCurrency(state.float_profit || 0)}</div>
                                    <div style="font-size: 0.85rem; color: #666;">浮动盈亏</div>
                                </div>
                            </div>
                            
                            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px;">
                                <div style="background: white; padding: 15px; border-radius: 8px;">
                                    <h4 style="margin-top: 0; color: #1e3c72; font-size: 1rem;">持仓信息</h4>
                                    <p style="margin: 8px 0;"><strong>方向:</strong> ${positionDirection}</p>
                                    <p style="margin: 8px 0;"><strong>开仓价:</strong> ${(state.entry_price || 0).toFixed(2)}</p>
                                    <p style="margin: 8px 0;"><strong>当前价:</strong> ${(state.current_price || 0).toFixed(2)}</p>
                                    <p style="margin: 8px 0;"><strong>占用保证金:</strong> ${formatCurrency(state.margin_used || 0)}</p>
                                </div>
                                
                                <div style="background: white; padding: 15px; border-radius: 8px;">
                                    <h4 style="margin-top: 0; color: #1e3c72; font-size: 1rem;">技术指标</h4>
                                    <p style="margin: 8px 0;"><strong>ATR:</strong> ${state.atr ? state.atr.toFixed(4) : 'N/A'}</p>
                                    <p style="margin: 8px 0;"><strong>波动率比:</strong> ${(state.volatility_ratio || 1.0).toFixed(4)}x</p>
                                    <p style="margin: 8px 0;"><strong>短期均线:</strong> ${state.short_ma ? state.short_ma.toFixed(2) : 'N/A'}</p>
                                    <p style="margin: 8px 0;"><strong>长期均线:</strong> ${state.long_ma ? state.long_ma.toFixed(2) : 'N/A'}</p>
                                    <p style="margin: 8px 0;"><strong>RSI:</strong> ${state.rsi ? state.rsi.toFixed(2) : 'N/A'}</p>
                                </div>
                                
                                <div style="background: white; padding: 15px; border-radius: 8px;">
                                    <h4 style="margin-top: 0; color: #1e3c72; font-size: 1rem;">交易统计</h4>
                                    <p style="margin: 8px 0;"><strong>总交易次数:</strong> ${state.total_trades || 0}</p>
                                    <p style="margin: 8px 0;"><strong>盈利交易次数:</strong> ${state.win_trades || 0}</p>
                                    <p style="margin: 8px 0;"><strong>总盈亏:</strong> ${formatCurrency(state.total_profit || 0)}</p>
                                    <p style="margin: 8px 0;"><strong>胜率:</strong> ${state.total_trades > 0 ? ((state.win_trades || 0) / state.total_trades * 100).toFixed(2) : 0}%</p>
                                    <p style="margin: 8px 0;"><strong>最后信号时间:</strong> ${state.last_signal_time || 'N/A'}</p>
                                </div>
                                
                                <div style="background: white; padding: 15px; border-radius: 8px;">
                                    <h4 style="margin-top: 0; color: #1e3c72; font-size: 1rem;">资金分配</h4>
                                    <p style="margin: 8px 0;"><strong>策略名称:</strong> ${state.strategy_name || 'N/A'}</p>
                                    <p style="margin: 8px 0;"><strong>分配资金:</strong> ${formatCurrency(state.allocated_capital || 0)}</p>
                                    <p style="margin: 8px 0;"><strong>资金权重:</strong> ${((state.capital_weight || 0) * 100).toFixed(2)}%</p>
                                    ${state.health ? `
                                        <p style="margin: 8px 0;"><strong>健康状态:</strong> ${state.health.status || 'UNKNOWN'}</p>
                                        <p style="margin: 8px 0;"><strong>错误次数:</strong> ${state.health.error_count || 0}</p>
                                        <p style="margin: 8px 0;"><strong>成功率:</strong> ${((state.health.success_rate || 1.0) * 100).toFixed(1)}%</p>
                                    ` : ''}
                                </div>
                            </div>
                        </div>
                    `;
                }
                
                contractsContainer.innerHTML = contractsHtml;
            } else {
                contractsContainer.innerHTML = '<p style="text-align: center; color: #999;">暂无合约状态数据</p>';
            }
        }

        document.addEventListener('DOMContentLoaded', function() {
            refreshDashboard();
            refreshBacktestList();
            refreshMultiContractStatus();
            
            setInterval(refreshDashboard, 5000);
            setInterval(refreshMultiContractStatus, 5000);
        });
    </script>
</body>
</html>
"""


@app.route('/')
def index():
    return render_template_string(INDEX_TEMPLATE)


@app.route('/api/dashboard')
def get_dashboard():
    global dashboard_data
    return jsonify({
        'account': dashboard_data['account'],
        'positions': [asdict(p) for p in dashboard_data['positions']],
        'strategies': [asdict(s) for s in dashboard_data['strategies']],
        'risk_info': dashboard_data['risk_info'],
        'status': dashboard_data['status'],
    })


@app.route('/api/backtest/reports')
def get_backtest_reports():
    reports = load_backtest_reports()
    return jsonify({
        'reports': [asdict(r) for r in reports],
        'count': len(reports),
    })


@app.route('/api/backtest/report/<filename>')
def get_backtest_report(filename):
    reports_dir = os.path.join(base_dir, 'logs', 'backtest_reports')
    filepath = os.path.join(reports_dir, filename)
    
    if not os.path.exists(filepath):
        return jsonify({'error': 'Report not found'}), 404
    
    data = load_backtest_report_detail(filepath)
    
    best_result = data.get('best_result', {})
    equity_curve = best_result.get('equity_curve', [])
    initial_equity = best_result.get('initial_equity', 1000000.0)
    
    chart_filename = generate_equity_chart(equity_curve, initial_equity)
    
    if chart_filename:
        data['chart_url'] = f'/static/{chart_filename}'
    
    return jsonify(data)


@app.route('/api/status', methods=['GET'])
def get_status():
    return jsonify({
        'status': 'running',
        'timestamp': datetime.now().isoformat(),
    })


@app.route('/api/multi-contract', methods=['GET'])
def get_multi_contract_status():
    global dashboard_data
    return jsonify({
        'multi_contract': dashboard_data.get('multi_contract', {
            'total_capital': 2000000.0,
            'contracts': {},
            'capital_allocation': {
                'contract_weights': {},
                'allocated_capitals': {},
            },
        }),
        'generated_at': datetime.now().isoformat(),
    })


@app.route('/api/update', methods=['POST'])
def update_dashboard():
    global dashboard_data
    try:
        data = request.get_json()
        
        if 'account' in data:
            dashboard_data['account'].update(data['account'])
        
        if 'positions' in data:
            dashboard_data['positions'] = [
                PositionDisplay(**p) for p in data['positions']
            ]
        
        if 'strategies' in data:
            dashboard_data['strategies'] = [
                StrategyDisplay(**s) for s in data['strategies']
            ]
        
        if 'risk_info' in data:
            dashboard_data['risk_info'].update(data['risk_info'])
        
        if 'status' in data:
            dashboard_data['status'].update(data['status'])
        
        return jsonify({'success': True})
    except Exception as e:
        logger.error(f"更新仪表盘数据失败: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


def init_sample_data():
    global dashboard_data
    
    dashboard_data['account'] = {
        'equity': 1050000.0,
        'balance': 1000000.0,
        'margin_used': 50000.0,
        'available': 950000.0,
        'float_profit': 5000.0,
    }
    
    dashboard_data['positions'] = [
        PositionDisplay(
            contract='SHFE.rb2410',
            direction='多单',
            volume=10,
            open_price=3500.0,
            current_price=3520.0,
            float_profit=2000.0,
            margin=35000.0,
        ),
    ]
    
    dashboard_data['strategies'] = [
        StrategyDisplay(
            name='AdaptiveMAStrategy',
            contract='SHFE.rb2410',
            short_ma=3510.5,
            long_ma=3490.2,
            short_period=5,
            long_period=20,
            signal='BUY',
            is_ready=True,
            status='HEALTHY',
            error_count=0,
        ),
    ]
    
    dashboard_data['risk_info'] = {
        'current_drawdown_percent': 1.5,
        'is_frozen': False,
        'frozen_reason': None,
    }
    
    dashboard_data['status'] = {
        'is_running': False,
        'start_time': None,
        'cycle_count': 0,
    }
    
    dashboard_data['multi_contract'] = {
        'total_capital': 2000000.0,
        'contracts': {
            'SHFE.rb2410': {
                'contract': 'SHFE.rb2410',
                'status': 'ACTIVE',
                'strategy_name': 'AdaptiveMAStrategy_SHFE.rb2410',
                'position_direction': 'LONG',
                'position_volume': 10,
                'entry_price': 3500.0,
                'current_price': 3520.0,
                'float_profit': 2000.0,
                'margin_used': 35000.0,
                'atr': 25.5,
                'volatility_ratio': 0.95,
                'signal': 'BUY',
                'last_signal_time': '2024-03-15T14:30:00',
                'total_trades': 15,
                'win_trades': 9,
                'total_profit': 15000.0,
                'allocated_capital': 600000.0,
                'capital_weight': 0.30,
                'short_ma': 3510.5,
                'long_ma': 3490.2,
                'rsi': 55.2,
                'health': {
                    'status': 'HEALTHY',
                    'error_count': 0,
                    'success_rate': 1.0,
                },
            },
            'SHFE.hc2410': {
                'contract': 'SHFE.hc2410',
                'status': 'ACTIVE',
                'strategy_name': 'AdaptiveMAStrategy_SHFE.hc2410',
                'position_direction': 'SHORT',
                'position_volume': 5,
                'entry_price': 3600.0,
                'current_price': 3580.0,
                'float_profit': 1000.0,
                'margin_used': 18000.0,
                'atr': 30.2,
                'volatility_ratio': 1.15,
                'signal': 'SELL',
                'last_signal_time': '2024-03-15T10:15:00',
                'total_trades': 12,
                'win_trades': 7,
                'total_profit': 8000.0,
                'allocated_capital': 400000.0,
                'capital_weight': 0.20,
                'short_ma': 3595.0,
                'long_ma': 3610.0,
                'rsi': 42.8,
                'health': {
                    'status': 'HEALTHY',
                    'error_count': 0,
                    'success_rate': 1.0,
                },
            },
            'DCE.i2409': {
                'contract': 'DCE.i2409',
                'status': 'ACTIVE',
                'strategy_name': 'AdaptiveMAStrategy_DCE.i2409',
                'position_direction': 'FLAT',
                'position_volume': 0,
                'entry_price': 0.0,
                'current_price': 850.0,
                'float_profit': 0.0,
                'margin_used': 0.0,
                'atr': 12.5,
                'volatility_ratio': 0.75,
                'signal': 'HOLD',
                'last_signal_time': '2024-03-14T16:45:00',
                'total_trades': 8,
                'win_trades': 5,
                'total_profit': 3500.0,
                'allocated_capital': 500000.0,
                'capital_weight': 0.25,
                'short_ma': 848.5,
                'long_ma': 847.0,
                'rsi': 50.5,
                'health': {
                    'status': 'HEALTHY',
                    'error_count': 0,
                    'success_rate': 1.0,
                },
            },
            'CZCE.CF409': {
                'contract': 'CZCE.CF409',
                'status': 'ACTIVE',
                'strategy_name': 'AdaptiveMAStrategy_CZCE.CF409',
                'position_direction': 'LONG',
                'position_volume': 3,
                'entry_price': 15500.0,
                'current_price': 15550.0,
                'float_profit': 750.0,
                'margin_used': 23250.0,
                'atr': 85.0,
                'volatility_ratio': 1.15,
                'signal': 'BUY',
                'last_signal_time': '2024-03-15T09:30:00',
                'total_trades': 6,
                'win_trades': 4,
                'total_profit': 4200.0,
                'allocated_capital': 500000.0,
                'capital_weight': 0.25,
                'short_ma': 15520.0,
                'long_ma': 15480.0,
                'rsi': 58.3,
                'health': {
                    'status': 'HEALTHY',
                    'error_count': 0,
                    'success_rate': 1.0,
                },
            },
        },
        'capital_allocation': {
            'contract_weights': {
                'SHFE.rb2410': 0.30,
                'SHFE.hc2410': 0.20,
                'DCE.i2409': 0.25,
                'CZCE.CF409': 0.25,
            },
            'allocated_capitals': {
                'SHFE.rb2410': 600000.0,
                'SHFE.hc2410': 400000.0,
                'DCE.i2409': 500000.0,
                'CZCE.CF409': 500000.0,
            },
        },
    }


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='期货策略监控 Web 仪表盘')
    parser.add_argument('--port', type=int, default=5000, help='Web 服务端口 (默认: 5000)')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='绑定地址 (默认: 0.0.0.0)')
    parser.add_argument('--debug', action='store_true', help='开启调试模式')
    
    args = parser.parse_args()
    
    init_sample_data()
    
    static_dir = os.path.join(base_dir, 'gui', 'static')
    os.makedirs(static_dir, exist_ok=True)
    
    logger.info(f"启动 Web 仪表盘: http://{args.host}:{args.port}")
    logger.info(f"静态文件目录: {static_dir}")
    
    app.run(
        host=args.host,
        port=args.port,
        debug=args.debug,
        threaded=True,
    )


if __name__ == '__main__':
    main()
