# 期货策略监控系统

一个基于 TqSdk 的实时期货策略监控与回测分析系统。

## 项目简介

本项目是一个功能完善的期货量化交易系统，主要特点包括：

- **实时策略监控**：支持多种均线策略（双均线、自适应多因子策略）
- **回测引擎**：支持历史数据回测、参数寻优
- **风险管理**：内置完善的风控机制，支持回撤控制、保证金监控、价格跳空检测
- **Web 监控看板**：提供可视化 Web 界面，实时监控账户状态和策略运行
- **多合约支持**：支持同时运行多个策略和多个合约

## 目录结构

```
Futures-trading-Tqsdk/
├── config/              # 配置文件目录
│   ├── settings.yaml    # 主配置文件
│   └── local_credentials.yaml  # 凭证配置
├── core/                # 核心模块
│   ├── backtest.py         # 回测引擎
│   ├── manager.py          # 策略管理器
│   ├── risk_manager.py     # 风险管理器
│   ├── equity_plotter.py   # 权益曲线绘图
│   ├── connection.py       # 连接管理
│   ├── realtime_runner.py  # 实时运行器
│   └── multi_contract_runner.py  # 多合约运行器
├── strategies/          # 策略模块
│   ├── base_strategy.py       # 策略基类
│   ├── double_ma_strategy.py  # 双均线策略
│   ├── adaptive_momentum_strategy.py  # 自适应动量策略
│   └── adaptive_ma_strategy.py   # 自适应多因子策略
├── gui/                 # Web 监控看板
│   ├── app.py          # Flask Web 应用
│   └── static/         # 静态文件（自动生成）
├── tests/               # 测试文件
├── logs/                # 日志目录
│   ├── backtest_reports/  # 回测报告
│   └── freeze_reports/    # 熔断报告
├── results/             # 输出结果
├── requirements.txt   # 依赖列表
└── README.md        # 本文档
```

## 安装指南

### Python 版本要求

- Python 3.8 或更高版本
- 推荐使用 Python 3.10+

### 安装步骤

1. **克隆或下载项目**

```bash
cd Futures-trading-Tqsdk
```

2. **创建虚拟环境（推荐）**

```bash
# 使用 conda
conda create -n futures-trading python=3.10
conda activate futures-trading

# 或使用 venv
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

3. **安装依赖**

```bash
pip install -r requirements.txt
```

### 可选依赖

- **TA-Lib**：用于加速技术指标计算
  ```bash
  # 使用 conda 安装（推荐）
  conda install -c conda-forge ta-lib
  
  # 或使用 pip（需要先安装系统依赖）
  pip install TA-Lib
  ```

- **pandas**：用于数据处理
  ```bash
  pip install pandas
  ```

## 配置说明

### 1. 天勤账户配置

在 `config/local_credentials.yaml` 或 `config/settings.yaml` 中配置天勤账户：

```yaml
backtest:
  tq_token: "your_tq_token_here"  # 推荐使用 Token
  # 或使用账号密码
  # tq_account: "your_account"
  # tq_password: "your_password"
```

注册天勤账户：https://account.shinnytech.com/

### 2. 模拟数据模式

如果没有天勤账户，可以使用模拟数据模式：

```yaml
backtest:
  use_mock_data: true
```

### 3. 主要配置项

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| `backtest.init_balance` | 初始资金 | 1,000,000.0 |
| `backtest.start_dt` | 回测开始日期 | 2024-01-01 |
| `backtest.end_dt` | 回测结束日期 | 2024-03-31 |
| `risk.max_drawdown_percent` | 最大回撤阈值 | 5.0% |
| `risk.max_total_margin_percent` | 最大总保证金比例 | 80.0% |

## 快速启动

### 1. 启动 Web 监控看板

```bash
python gui/app.py
```

默认访问地址：http://localhost:5000

Web 看板功能：
- **实时监控**：展示账户总权益、持仓盈亏、均线运行状态
- **回测分析**：读取回测报告，自动生成收益曲线图和回撤分布图

### 2. 运行回测

```bash
python scripts/run_backtest_demo.py
```

回测报告将保存在 `logs/backtest_reports/` 目录。

### 3. 实时运行策略

```bash
python run_realtime.py
```

### 4. 参数寻优

```bash
# 查看示例
```

## 策略说明

### 双均线策略 (DoubleMAStrategy)

经典的均线交叉策略：
- 金叉（短期均线上穿长期均线 → 买入信号
- 死叉（短期均线下穿长期均线 → 卖出信号

### 自适应动量策略 (AdaptiveMomentumStrategy)

在双均线基础上增加：
- RSI 动量确认
- ATR 波动率过滤
- 追踪止损

### 自适应多因子策略 (AdaptiveMAStrategy)

增强型策略，包含：
- **ATR 波动率过滤**：只有当前价格波动 > 1.2倍 ATR 时才允许触发信号
- **动量确认**：开多单时 RSI 必须 > 50；开空单时 RSI 必须 < 50
- **动态仓位**：下单手数 = (总资产 × 1%) / (2 × ATR)
- **智能追踪止损**：
  - 盈利达 1.0×ATR 后，将止损位移至成本价（保本逻辑）
  - 盈利超过 2 倍 ATR，启动追踪止损

## 测试说明

### 运行所有测试

```bash
pytest
```

### 运行指定测试文件

```bash
# 测试策略模块
pytest tests/test_strategy.py -v

# 测试风险管理
pytest tests/test_risk_manager.py -v

# 测试策略管理器
pytest tests/test_manager.py -v
```

### 测试覆盖范围

| 测试文件 | 覆盖范围 |
|----------|----------|
| `test_strategy.py` | 策略基类、双均线策略、均线计算、信号检测 |
| `test_manager.py` | 策略注册、配置加载、数据分发、生命周期管理 |
| `test_risk_manager.py` | 回撤检测、保证金监控、订单验证、熔断机制 |
| `test_backtest.py` | 回测引擎、参数寻优、性能指标计算 |
| `test_connection.py` | 连接管理、重连机制 |
| `test_realtime_runner.py` | 实时运行器、心跳检测 |
| `test_risk_trigger.py` | 风险触发场景测试 |
| `test_risk_extreme_scenarios.py` | 极端市场场景测试 |
| `test_multi_contract.py` | 多合约运行测试 |

### 生成测试覆盖率报告

```bash
pytest --cov=. --cov-report=html
```

报告将生成在 `htmlcov/` 目录。

## API 接口

### Web 看板 API

| 接口 | 方法 | 说明 |
|------|------|------|
| `/` | GET | 主页面 |
| `/api/dashboard` | GET | 获取实时监控数据 |
| `/api/backtest/reports` | GET | 获取回测报告列表 |
| `/api/backtest/report/<filename>` | GET | 获取回测报告详情 |
| `/api/update` | POST | 更新仪表盘数据 |
| `/api/status` | GET | 获取服务状态 |

### 更新仪表盘数据示例

```python
import requests

data = {
    'account': {
        'equity': 1050000.0,
        'balance': 1000000.0,
        'margin_used': 50000.0,
        'available': 950000.0,
        'float_profit': 5000.0,
    },
    'positions': [
        {
            'contract': 'SHFE.rb2410',
            'direction': '多单',
            'volume': 10,
            'open_price': 3500.0,
            'current_price': 3520.0,
            'float_profit': 2000.0,
            'margin': 35000.0,
        }
    ],
    'strategies': [
        {
            'name': 'AdaptiveMAStrategy',
            'contract': 'SHFE.rb2410',
            'short_ma': 3510.5,
            'long_ma': 3490.2,
            'short_period': 5,
            'long_period': 20,
            'signal': 'BUY',
            'is_ready': True,
            'status': 'HEALTHY',
            'error_count': 0,
        }
    ],
    'risk_info': {
        'current_drawdown_percent': 1.5,
        'is_frozen': False,
        'frozen_reason': None,
    },
    'status': {
        'is_running': True,
        'start_time': None,
        'cycle_count': 100,
    }
}

response = requests.post('http://localhost:5000/api/update', json=data)
```

## 注意事项

1. **天勤账户**：运行回测和实时交易需要有效的天勤账户
2. **模拟模式**：调试策略逻辑可以使用模拟数据模式（`use_mock_data: true）
3. **风险管理**：实盘交易前请充分测试风控机制
4. **参数优化**：回测结果仅供参考，过往业绩不代表未来表现

## 许可证

MIT License

## 贡献

欢迎提交 Issue 和 Pull Request。
