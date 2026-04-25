import logging
import os
import sys
import csv
import json
import copy
import shutil
import itertools
import math
import statistics
import asyncio
import warnings
from datetime import date, datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple, Callable, Type, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Manager

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    yaml = None

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

from tqsdk import TqApi, TqSim, TqAuth, TqBacktest
from tqsdk.exceptions import BacktestFinished

from strategies.base_strategy import StrategyBase, SignalType
from core.manager import StrategyManager
from core.risk_manager import RiskManager, RiskLevel, AccountSnapshot


CHINESE_FONT_AVAILABLE = False
FONT_CHECKED = False


def _check_chinese_font_support() -> bool:
    """检查当前环境是否支持中文字体，优先检测 SimHei 和 Microsoft YaHei"""
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
            logging.getLogger(__name__).warning("未找到中文字体 (SimHei/Microsoft YaHei)，图表将强制使用英文标签")
        
    except Exception as e:
        logging.getLogger(__name__).warning(f"字体检测失败: {e}")
        CHINESE_FONT_AVAILABLE = False
    
    FONT_CHECKED = True
    return CHINESE_FONT_AVAILABLE


def _get_label(zh_label: str, en_label: str) -> str:
    """根据字体支持情况返回合适的标签"""
    if _check_chinese_font_support():
        return zh_label
    return en_label


def _safe_close_api(api: TqApi) -> None:
    """安全关闭 TqApi，确保所有协程被清理，避免 coroutine was never awaited 警告"""
    if api is None:
        return
    
    logger = logging.getLogger(__name__)
    
    try:
        if api.is_closed():
            return
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=DeprecationWarning)
            warnings.simplefilter("ignore", category=RuntimeWarning)
            
            try:
                api.close()
                logger.debug("TqApi 已成功关闭")
            except Exception as e1:
                logger.debug(f"直接关闭 API 失败，尝试备用方式: {e1}")
                try:
                    loop = None
                    try:
                        loop = asyncio.get_running_loop()
                    except RuntimeError:
                        pass
                    
                    if loop and loop.is_running():
                        loop.call_soon_threadsafe(api.close)
                    else:
                        new_loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(new_loop)
                        try:
                            api.close()
                        finally:
                            new_loop.close()
                            asyncio.set_event_loop(None)
                except Exception as e2:
                    logger.debug(f"备用关闭方式也失败 (可忽略): {e2}")
    
    except Exception as e:
        logger.debug(f"API 关闭时出现异常 (可忽略): {e}")


def _has_valid_token(tq_token: str = None) -> bool:
    """检查是否有有效的 Token"""
    if not tq_token:
        return False
    if tq_token == '' or tq_token == 'your_token':
        return False
    return True


def _prompt_user_for_credentials(logger: logging.Logger = None) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    提示用户输入天勤凭证
    返回: (account, password, token) - 三者至少一个有效，或都为 None
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    print("\n" + "=" * 80)
    print("                    天勤认证失败")
    print("=" * 80)
    print("\n当前无法连接天勤服务器进行回测。")
    print("\n您有以下选择：")
    print("  1. 输入有效的天勤账号密码")
    print("  2. 输入有效的天勤 Token（推荐）")
    print("  3. 切换到模拟数据模式（不依赖天勤服务器）")
    print("  4. 退出程序")
    print("\n" + "-" * 80)
    
    try:
        choice = input("\n请选择 (1-4): ").strip()
        
        if choice == '1':
            print("\n请输入天勤账号密码：")
            account = input("  账号: ").strip()
            password = input("  密码: ").strip()
            if account and password:
                return (account, password, None)
            else:
                logger.warning("账号或密码为空")
                return (None, None, None)
        
        elif choice == '2':
            print("\n请输入天勤 Token：")
            print("  (Token 可以从 https://www.shinnytech.com/ 获取)")
            token = input("  Token: ").strip()
            if token:
                return (None, None, token)
            else:
                logger.warning("Token 为空")
                return (None, None, None)
        
        elif choice == '3':
            logger.info("用户选择切换到模拟数据模式")
            return ('__MOCK_MODE__', None, None)
        
        elif choice == '4':
            logger.info("用户选择退出程序")
            raise SystemExit("用户选择退出")
        
        else:
            logger.warning(f"无效选择: {choice}")
            return (None, None, None)
    
    except EOFError:
        logger.warning("非交互式环境，无法提示用户输入")
        return (None, None, None)
    except KeyboardInterrupt:
        logger.info("用户中断")
        raise


def _create_tq_api_with_auth_fallback(
    account: TqSim,
    backtest: TqBacktest,
    tq_account: str = None,
    tq_password: str = None,
    tq_token: str = None,
    logger: logging.Logger = None,
    allow_prompt: bool = True,
) -> TqApi:
    """
    创建 TqApi，带认证失败回退逻辑。
    优先级: Token > 账号密码 > 用户提示
    
    Args:
        account: TqSim 账户
        backtest: TqBacktest 回测配置
        tq_account: 天勤账号
        tq_password: 天勤密码
        tq_token: 天勤 Token（推荐）
        logger: 日志记录器
        allow_prompt: 是否允许提示用户输入
    
    Returns:
        TqApi 实例
    
    Raises:
        RuntimeError: 所有认证方式都失败时抛出
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    api = None
    
    has_valid_token = _has_valid_token(tq_token)
    has_valid_creds = tq_account and tq_password and tq_account != 'your_account' and tq_password != 'your_password'
    
    if has_valid_token:
        logger.info("尝试使用 Token 连接天勤...")
        try:
            auth = TqAuth(tq_token)
            api = TqApi(account=account, backtest=backtest, auth=auth)
            logger.info("✓ Token 认证成功")
            return api
        except Exception as e:
            error_str = str(e)
            logger.warning(f"Token 认证失败: {error_str}")
            _safe_close_api(api)
            api = None
    
    if has_valid_creds and not has_valid_token:
        logger.info("尝试使用账号密码连接天勤...")
        try:
            auth = TqAuth(tq_account, tq_password)
            api = TqApi(account=account, backtest=backtest, auth=auth)
            logger.info("✓ 账号密码认证成功")
            return api
        except Exception as e:
            error_str = str(e)
            logger.warning(f"账号密码认证失败: {error_str}")
            _safe_close_api(api)
            api = None
    
    if allow_prompt:
        logger.warning("配置文件中的认证信息无效或缺失")
        user_account, user_password, user_token = _prompt_user_for_credentials(logger)
        
        if user_account == '__MOCK_MODE__':
            logger.info("切换到模拟数据模式")
            raise ValueError("__MOCK_MODE__")
        
        if user_token:
            logger.info("使用用户输入的 Token 连接天勤...")
            try:
                auth = TqAuth(user_token)
                api = TqApi(account=account, backtest=backtest, auth=auth)
                logger.info("✓ Token 认证成功")
                return api
            except Exception as e:
                logger.warning(f"Token 认证失败: {e}")
                _safe_close_api(api)
        
        if user_account and user_password:
            logger.info("使用用户输入的账号密码连接天勤...")
            try:
                auth = TqAuth(user_account, user_password)
                api = TqApi(account=account, backtest=backtest, auth=auth)
                logger.info("✓ 账号密码认证成功")
                return api
            except Exception as e:
                logger.warning(f"账号密码认证失败: {e}")
                _safe_close_api(api)
    
    error_msg = (
        "无法创建 TqApi 连接。\n"
        "请检查以下选项：\n"
        "  1. 在配置文件 config/settings.yaml 中设置有效的 tq_token 或 tq_account/tq_password\n"
        "  2. 或者设置 use_mock_data: true 使用模拟数据模式\n"
        "  3. 注册天勤账户: https://account.shinnytech.com/"
    )
    logger.error(error_msg)
    raise RuntimeError(error_msg)


def _has_valid_tq_credentials(config: Dict[str, Any]) -> bool:
    """检查是否有有效的天勤凭证"""
    tq_config = config.get('backtest', {}) if config else {}
    account = tq_config.get('tq_account', '')
    password = tq_config.get('tq_password', '')
    
    if not account or not password:
        return False
    
    if account == 'your_account' or password == 'your_password':
        return False
    
    if account == '' or password == '':
        return False
    
    return True


def _print_tq_credentials_guide():
    """打印天勤凭证注册引导"""
    guide = """
╔══════════════════════════════════════════════════════════════╗
║                    天勤账户配置指南                              ║
╠══════════════════════════════════════════════════════════════╣
║                                                                  ║
║  当前使用匿名模式运行回测。匿名模式限制：                        ║
║  - 无法获取实时行情数据                                          ║
║  - 回测可能无法正常执行                                          ║
║                                                                  ║
║  如需完整功能，请配置天勤账户：                                  ║
║                                                                  ║
║  方法1：环境变量                                                  ║
║    Windows:                                                       ║
║      set TQ_ACCOUNT=你的天勤账号                                 ║
║      set TQ_PASSWORD=你的天勤密码                                ║
║                                                                  ║
║    Linux/Mac:                                                     ║
║      export TQ_ACCOUNT=你的天勤账号                              ║
║      export TQ_PASSWORD=你的天勤密码                             ║
║                                                                  ║
║  方法2：配置文件 config/local_credentials.yaml                   ║
║    tq_account: 你的天勤账号                                      ║
║    tq_password: 你的天勤密码                                     ║
║                                                                  ║
║  注册天勤账户：https://account.shinnytech.com/                  ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════╝
"""
    print(guide)


class BacktestMode(Enum):
    SINGLE = "single"
    OPTIMIZATION = "optimization"


@dataclass
class PerformanceMetrics:
    total_return: float = 0.0
    total_return_percent: float = 0.0
    annualized_return: float = 0.0
    annualized_return_percent: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_percent: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_trade_return: float = 0.0
    total_commission_cost: float = 0.0
    total_slippage_cost: float = 0.0
    total_cost: float = 0.0


@dataclass
class BacktestResult:
    strategy_name: str
    params: Dict[str, Any]
    start_dt: date
    end_dt: date
    initial_equity: float = 0.0
    final_equity: float = 0.0
    performance: PerformanceMetrics = field(default_factory=PerformanceMetrics)
    risk_triggered: bool = False
    frozen_during_backtest: bool = False
    frozen_reason: Optional[str] = None
    status: str = "completed"
    error_message: Optional[str] = None
    equity_curve: List[Dict[str, Any]] = field(default_factory=list)
    trade_records: List[Dict[str, Any]] = field(default_factory=list)
    risk_events: List[Dict[str, Any]] = field(default_factory=list)
    config_snapshot: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ParameterRange:
    name: str
    min_val: Union[int, float]
    max_val: Union[int, float]
    step: Union[int, float] = 1
    param_type: Type = int


@dataclass
class CostConfig:
    default_commission_per_lot: float = 0.0
    default_slippage_points: float = 0.0
    contract_configs: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    def get_commission(self, symbol: str) -> float:
        if symbol in self.contract_configs:
            return self.contract_configs[symbol].get('commission_per_lot', self.default_commission_per_lot)
        return self.default_commission_per_lot
    
    def get_slippage(self, symbol: str) -> float:
        if symbol in self.contract_configs:
            return self.contract_configs[symbol].get('slippage_points', self.default_slippage_points)
        return self.default_slippage_points


@dataclass
class PerformanceConfig:
    risk_free_rate: float = 0.03
    trading_days_per_year: int = 252


class BacktestEngine:
    def __init__(
        self,
        config: Dict[str, Any] = None,
        on_progress: Callable[[int, int, str], None] = None,
    ):
        self.config = config or {}
        self.on_progress = on_progress
        self.logger = logging.getLogger(self.__class__.__name__)
        
        self._results: List[BacktestResult] = []
        self._best_result: Optional[BacktestResult] = None
        
        backtest_config = self.config.get('backtest', {})
        self._init_balance = backtest_config.get('init_balance', 1000000.0)
        
        self._tq_account = backtest_config.get('tq_account')
        self._tq_password = backtest_config.get('tq_password')
        self._tq_token = backtest_config.get('tq_token')
        
        self._use_mock_data = backtest_config.get('use_mock_data', False)
        self._mock_data_dir = backtest_config.get('mock_data_dir', 'data/mock')
        
        self._cost_config = self._parse_cost_config(backtest_config)
        
        self._performance_config = PerformanceConfig(
            risk_free_rate=backtest_config.get('performance', {}).get('risk_free_rate', 0.03),
            trading_days_per_year=backtest_config.get('performance', {}).get('trading_days_per_year', 252),
        )
        
        optimization_config = backtest_config.get('optimization', {})
        self._max_workers = optimization_config.get('max_workers', os.cpu_count() or 4)
        
        self.logger.info(f"BacktestEngine 初始化完成，初始资金: {self._init_balance}")
        self.logger.info(f"成本配置: 手续费={self._cost_config.default_commission_per_lot}元/手, 滑点={self._cost_config.default_slippage_points}点")
        self.logger.info(f"并行配置: 最大工作进程={self._max_workers}")

    def _parse_cost_config(self, backtest_config: Dict[str, Any]) -> CostConfig:
        costs_config = backtest_config.get('costs', {})
        
        default_commission = costs_config.get('default_commission_per_lot', 0.0)
        default_slippage = costs_config.get('default_slippage_points', 0.0)
        
        contract_configs = {}
        contracts = costs_config.get('contracts', {})
        for symbol, cfg in contracts.items():
            contract_configs[symbol] = {
                'commission_per_lot': cfg.get('commission_per_lot', default_commission),
                'slippage_points': cfg.get('slippage_points', default_slippage),
            }
        
        return CostConfig(
            default_commission_per_lot=default_commission,
            default_slippage_points=default_slippage,
            contract_configs=contract_configs,
        )

    def _create_backtest_api(
        self, 
        start_dt: date, 
        end_dt: date,
        cost_config: CostConfig = None,
    ) -> Tuple[TqApi, TqSim]:
        self.logger.info(f"创建回测 API，时间段: {start_dt} 至 {end_dt}")
        
        if self._use_mock_data:
            self.logger.info("检测到 use_mock_data: true，将使用模拟数据模式")
            self.logger.info("提示: 模拟数据模式不依赖天勤服务器，适合调试策略逻辑")
            raise ValueError("__MOCK_MODE__")
        
        account = TqSim(init_balance=self._init_balance)
        backtest = TqBacktest(start_dt=start_dt, end_dt=end_dt)
        
        if cost_config:
            for symbol, cfg in cost_config.contract_configs.items():
                commission = cfg.get('commission_per_lot', 0.0)
                if commission > 0:
                    account.set_commission(symbol, commission)
                    self.logger.debug(f"设置合约 {symbol} 手续费: {commission}元/手")
        
        has_valid_token = _has_valid_token(self._tq_token)
        has_valid_creds = _has_valid_tq_credentials(self.config)
        
        if not has_valid_token and not has_valid_creds:
            self.logger.warning("=" * 60)
            self.logger.warning("未检测到有效的天勤账户凭证")
            self.logger.warning("可用选项:")
            self.logger.warning("  1. 在 config/settings.yaml 中设置 tq_token")
            self.logger.warning("  2. 或者设置 tq_account 和 tq_password")
            self.logger.warning("  3. 或者设置 use_mock_data: true 使用模拟数据模式")
            self.logger.warning("=" * 60)
        
        api = _create_tq_api_with_auth_fallback(
            account=account,
            backtest=backtest,
            tq_account=self._tq_account,
            tq_password=self._tq_password,
            tq_token=self._tq_token,
            logger=self.logger,
            allow_prompt=False,
        )
        
        return api, account

    def _extract_date_range(self) -> Tuple[date, date]:
        backtest_config = self.config.get('backtest', {})
        
        start_dt_str = backtest_config.get('start_dt')
        end_dt_str = backtest_config.get('end_dt')
        
        if not start_dt_str or not end_dt_str:
            raise ValueError("配置中缺少回测时间段 (start_dt 或 end_dt)")
        
        def parse_date(date_str: str) -> date:
            if isinstance(date_str, date):
                return date_str
            if isinstance(date_str, datetime):
                return date_str.date()
            for fmt in ['%Y-%m-%d', '%Y%m%d', '%Y/%m/%d']:
                try:
                    return datetime.strptime(str(date_str), fmt).date()
                except ValueError:
                    continue
            raise ValueError(f"无法解析日期格式: {date_str}")
        
        start_dt = parse_date(start_dt_str)
        end_dt = parse_date(end_dt_str)
        
        if start_dt >= end_dt:
            raise ValueError(f"开始日期 ({start_dt}) 必须早于结束日期 ({end_dt})")
        
        return start_dt, end_dt

    def _create_strategy_manager(
        self,
        api: TqApi,
        strategy_class: Type[StrategyBase],
        strategy_params: Dict[str, Any],
    ) -> StrategyManager:
        mock_connector = type('MockConnector', (), {
            'get_api': lambda self: api,
            'get_config': lambda self: self.config,
        })()
        
        manager = StrategyManager(connector=mock_connector)
        manager.configure_from_dict(self.config)
        
        strategy = strategy_class(**strategy_params)
        strategy.set_connector(mock_connector)
        
        manager.register_strategy('BacktestStrategy', strategy)
        
        return manager

    def _create_risk_manager(
        self,
        api: TqApi,
    ) -> RiskManager:
        mock_connector = type('MockConnector', (), {
            'get_api': lambda self: api,
        })()
        
        risk_manager = RiskManager(connector=mock_connector)
        
        risk_config = self.config.get('risk', {})
        if risk_config.get('max_drawdown_percent'):
            risk_manager.max_drawdown_percent = float(risk_config['max_drawdown_percent'])
        if risk_config.get('max_strategy_margin_percent'):
            risk_manager.max_strategy_margin_percent = float(risk_config['max_strategy_margin_percent'])
        if risk_config.get('max_total_margin_percent'):
            risk_manager.max_total_margin_percent = float(risk_config['max_total_margin_percent'])
        
        return risk_manager

    def _calculate_returns_from_equity_curve(
        self, 
        equity_curve: List[Dict[str, Any]],
        initial_equity: float,
    ) -> Tuple[List[float], List[float]]:
        if not equity_curve:
            return [], []
        
        returns = []
        log_returns = []
        prev_equity = initial_equity
        
        for point in equity_curve:
            equity = point.get('equity', prev_equity)
            if prev_equity > 0:
                simple_return = (equity - prev_equity) / prev_equity
                log_return = math.log(equity / prev_equity) if equity > 0 else 0
                returns.append(simple_return)
                log_returns.append(log_return)
            prev_equity = equity
        
        return returns, log_returns

    def _calculate_performance_metrics(
        self,
        result: BacktestResult,
        start_dt: date,
        end_dt: date,
        equity_curve: List[Dict[str, Any]],
        initial_equity: float,
        final_equity: float,
        total_trades: int = 0,
    ) -> PerformanceMetrics:
        metrics = PerformanceMetrics()
        
        metrics.total_return = final_equity - initial_equity
        metrics.total_return_percent = (metrics.total_return / initial_equity * 100) if initial_equity > 0 else 0.0
        
        days = (end_dt - start_dt).days
        if days > 0 and initial_equity > 0:
            daily_return = (final_equity / initial_equity) ** (1.0 / days) - 1
            metrics.annualized_return = (1 + daily_return) ** 365 - 1
            metrics.annualized_return_percent = metrics.annualized_return * 100
        
        if equity_curve:
            equities = [initial_equity] + [p.get('equity', initial_equity) for p in equity_curve]
            
            peak = initial_equity
            max_dd = 0.0
            max_dd_percent = 0.0
            
            for eq in equities:
                if eq > peak:
                    peak = eq
                dd = peak - eq
                dd_percent = (dd / peak * 100) if peak > 0 else 0.0
                
                if dd > max_dd:
                    max_dd = dd
                if dd_percent > max_dd_percent:
                    max_dd_percent = dd_percent
            
            metrics.max_drawdown = max_dd
            metrics.max_drawdown_percent = max_dd_percent
            
            returns, log_returns = self._calculate_returns_from_equity_curve(equity_curve, initial_equity)
            
            if returns and len(returns) > 1:
                try:
                    avg_return = statistics.mean(returns)
                    std_return = statistics.stdev(returns)
                    
                    if std_return > 0:
                        daily_rf = (1 + self._performance_config.risk_free_rate) ** (1/365) - 1
                        daily_sharpe = (avg_return - daily_rf) / std_return
                        metrics.sharpe_ratio = daily_sharpe * math.sqrt(252)
                    
                    negative_returns = [r for r in returns if r < 0]
                    if negative_returns and len(negative_returns) > 1:
                        std_negative = statistics.stdev(negative_returns)
                        if std_negative > 0:
                            daily_sortino = (avg_return - daily_rf) / std_negative
                            metrics.sortino_ratio = daily_sortino * math.sqrt(252)
                    
                    if max_dd_percent > 0:
                        metrics.calmar_ratio = metrics.annualized_return / (max_dd_percent / 100)
                        
                except Exception as e:
                    self.logger.debug(f"计算风险调整收益指标时出错: {e}")
        
        metrics.total_trades = total_trades
        
        metrics.total_commission_cost = self._cost_config.default_commission_per_lot * total_trades
        metrics.total_slippage_cost = 0.0
        metrics.total_cost = metrics.total_commission_cost + metrics.total_slippage_cost
        
        if total_trades > 0:
            metrics.avg_trade_return = metrics.total_return / total_trades
        
        return metrics

    def _run_single_backtest_internal(
        self,
        start_dt: date,
        end_dt: date,
        strategy_class: Type[StrategyBase],
        strategy_params: Dict[str, Any],
        strategy_name: str = "BacktestStrategy",
        config: Dict[str, Any] = None,
        cost_config: CostConfig = None,
        performance_config: PerformanceConfig = None,
    ) -> Dict[str, Any]:
        result_dict = {
            'strategy_name': strategy_name,
            'params': strategy_params.copy(),
            'start_dt': start_dt.isoformat(),
            'end_dt': end_dt.isoformat(),
            'initial_equity': 0.0,
            'final_equity': 0.0,
            'performance': {},
            'risk_triggered': False,
            'frozen_during_backtest': False,
            'frozen_reason': None,
            'status': 'completed',
            'error_message': None,
            'equity_curve': [],
            'trade_records': [],
            'risk_events': [],
        }
        
        api = None
        account = None
        manager = None
        risk_manager = None
        
        try:
            from tqsdk import TqApi, TqSim, TqAuth, TqBacktest
            from tqsdk.exceptions import BacktestFinished
            
            init_balance = config.get('backtest', {}).get('init_balance', 1000000.0) if config else 1000000.0
            
            use_mock_data = config.get('backtest', {}).get('use_mock_data', False) if config else False
            
            if use_mock_data:
                logging.getLogger(__name__).info("子进程: 检测到 use_mock_data: true，使用模拟数据模式")
                api = MockTqApi(
                    start_dt=start_dt,
                    end_dt=end_dt,
                    init_balance=init_balance,
                    logger=logging.getLogger(__name__),
                )
                account = api
            else:
                account = TqSim(init_balance=init_balance)
                backtest = TqBacktest(start_dt=start_dt, end_dt=end_dt)
                
                tq_account = config.get('backtest', {}).get('tq_account') if config else None
                tq_password = config.get('backtest', {}).get('tq_password') if config else None
                tq_token = config.get('backtest', {}).get('tq_token') if config else None
                
                if cost_config:
                    for symbol, cfg in cost_config.contract_configs.items():
                        commission = cfg.get('commission_per_lot', 0.0)
                        if commission > 0:
                            account.set_commission(symbol, commission)
                
                api = _create_tq_api_with_auth_fallback(
                    account=account,
                    backtest=backtest,
                    tq_account=tq_account,
                    tq_password=tq_password,
                    tq_token=tq_token,
                    logger=logging.getLogger(__name__),
                    allow_prompt=False,
                )
            
            from strategies.base_strategy import StrategyBase
            from core.manager import StrategyManager
            from core.risk_manager import RiskManager
            
            mock_connector = type('MockConnector', (), {
                'get_api': lambda self: api,
                'get_config': lambda self: config or {},
            })()
            
            manager = StrategyManager(connector=mock_connector)
            manager.configure_from_dict(config or {})
            
            strategy = strategy_class(**strategy_params)
            strategy.set_connector(mock_connector)
            manager.register_strategy('BacktestStrategy', strategy)
            
            risk_manager = RiskManager(connector=mock_connector)
            risk_config = config.get('risk', {}) if config else {}
            if risk_config.get('max_drawdown_percent'):
                risk_manager.max_drawdown_percent = float(risk_config['max_drawdown_percent'])
            manager.set_risk_manager(risk_manager)
            
            manager.initialize(load_saved_states=False)
            
            initial_snapshot = risk_manager.get_account_snapshot()
            result_dict['initial_equity'] = initial_snapshot.equity
            
            equity_curve = []
            risk_events = []
            cycle_count = 0
            
            while True:
                try:
                    api.wait_update()
                    cycle_count += 1
                    
                    if manager._should_run_risk_check():
                        manager.run_risk_checks()
                        
                        if manager.is_risk_frozen():
                            result_dict['risk_triggered'] = True
                            result_dict['frozen_during_backtest'] = True
                            result_dict['frozen_reason'] = risk_manager.get_frozen_reason()
                            break
                    
                    for name, strat in manager._strategies.items():
                        health = manager._strategy_health.get(name)
                        if health and health.can_run() and hasattr(strat, '_on_update'):
                            try:
                                strat._on_update()
                                health.record_success()
                            except Exception as e:
                                health.record_error(str(e))
                    
                    if cycle_count % 50 == 0:
                        snapshot = risk_manager.get_account_snapshot()
                        equity_curve.append({
                            'cycle': cycle_count,
                            'timestamp': snapshot.timestamp,
                            'equity': snapshot.equity,
                            'margin_used': snapshot.margin_used,
                        })
                        
                except BacktestFinished:
                    break
            
            final_snapshot = risk_manager.get_account_snapshot()
            result_dict['final_equity'] = final_snapshot.equity
            result_dict['equity_curve'] = equity_curve
            
            events = risk_manager.get_risk_events(limit=100)
            result_dict['risk_events'] = events
            
            drawdown_info = risk_manager.get_drawdown_info()
            current_dd_percent = drawdown_info.get('current_drawdown_percent', 0.0)
            
            total_trades = 0
            trade_records = []
            try:
                if hasattr(account, 'trades') and account.trades:
                    total_trades = len(account.trades)
                    trade_records = list(account.trades)
                    logging.getLogger(__name__).info(f"获取到 {len(trade_records)} 条交易记录")
            except Exception as e:
                logging.getLogger(__name__).warning(f"获取交易记录失败: {e}")
                pass
            
            result_dict['trade_records'] = trade_records
            
            performance_config = performance_config or PerformanceConfig()
            initial_eq = result_dict['initial_equity']
            final_eq = result_dict['final_equity']
            eq_curve = result_dict['equity_curve']
            
            total_return = final_eq - initial_eq
            total_return_percent = (total_return / initial_eq * 100) if initial_eq > 0 else 0.0
            
            days = (end_dt - start_dt).days
            annualized_return = 0.0
            annualized_return_percent = 0.0
            if days > 0 and initial_eq > 0:
                daily_return = (final_eq / initial_eq) ** (1.0 / days) - 1
                annualized_return = (1 + daily_return) ** 365 - 1
                annualized_return_percent = annualized_return * 100
            
            max_dd = 0.0
            max_dd_percent = 0.0
            if eq_curve:
                equities = [initial_eq] + [p.get('equity', initial_eq) for p in eq_curve]
                peak = initial_eq
                for eq in equities:
                    if eq > peak:
                        peak = eq
                    dd = peak - eq
                    dd_percent = (dd / peak * 100) if peak > 0 else 0.0
                    if dd > max_dd:
                        max_dd = dd
                    if dd_percent > max_dd_percent:
                        max_dd_percent = dd_percent
            
            result_dict['performance'] = {
                'total_return': total_return,
                'total_return_percent': total_return_percent,
                'annualized_return': annualized_return,
                'annualized_return_percent': annualized_return_percent,
                'max_drawdown': max_dd,
                'max_drawdown_percent': max_dd_percent,
                'sharpe_ratio': 0.0,
                'sortino_ratio': 0.0,
                'calmar_ratio': 0.0,
                'total_trades': total_trades,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'avg_trade_return': total_return / total_trades if total_trades > 0 else 0.0,
                'total_commission_cost': cost_config.default_commission_per_lot * total_trades if cost_config else 0.0,
                'total_slippage_cost': 0.0,
                'total_cost': cost_config.default_commission_per_lot * total_trades if cost_config else 0.0,
            }
            
            result_dict['config_snapshot'] = {
                'cost_config': {
                    'default_commission': cost_config.default_commission_per_lot if cost_config else 0.0,
                    'default_slippage': cost_config.default_slippage_points if cost_config else 0.0,
                },
            }
            
        except Exception as e:
            result_dict['status'] = 'error'
            result_dict['error_message'] = str(e)
            logging.getLogger(__name__).error(f"回测执行出错: {e}", exc_info=True)
        finally:
            _cleanup_api_and_loop(api, logging.getLogger(__name__))
        
        return result_dict

    def _run_single_backtest(
        self,
        start_dt: date,
        end_dt: date,
        strategy_class: Type[StrategyBase],
        strategy_params: Dict[str, Any],
        strategy_name: str = "BacktestStrategy",
    ) -> BacktestResult:
        result_dict = self._run_single_backtest_internal(
            start_dt=start_dt,
            end_dt=end_dt,
            strategy_class=strategy_class,
            strategy_params=strategy_params,
            strategy_name=strategy_name,
            config=self.config,
            cost_config=self._cost_config,
            performance_config=self._performance_config,
        )
        
        perf_dict = result_dict.get('performance', {})
        performance = PerformanceMetrics(
            total_return=perf_dict.get('total_return', 0.0),
            total_return_percent=perf_dict.get('total_return_percent', 0.0),
            annualized_return=perf_dict.get('annualized_return', 0.0),
            annualized_return_percent=perf_dict.get('annualized_return_percent', 0.0),
            max_drawdown=perf_dict.get('max_drawdown', 0.0),
            max_drawdown_percent=perf_dict.get('max_drawdown_percent', 0.0),
            sharpe_ratio=perf_dict.get('sharpe_ratio', 0.0),
            sortino_ratio=perf_dict.get('sortino_ratio', 0.0),
            calmar_ratio=perf_dict.get('calmar_ratio', 0.0),
            total_trades=perf_dict.get('total_trades', 0),
            winning_trades=perf_dict.get('winning_trades', 0),
            losing_trades=perf_dict.get('losing_trades', 0),
            win_rate=perf_dict.get('win_rate', 0.0),
            profit_factor=perf_dict.get('profit_factor', 0.0),
            avg_trade_return=perf_dict.get('avg_trade_return', 0.0),
            total_commission_cost=perf_dict.get('total_commission_cost', 0.0),
            total_slippage_cost=perf_dict.get('total_slippage_cost', 0.0),
            total_cost=perf_dict.get('total_cost', 0.0),
        )
        
        def parse_date_str(date_str: str) -> date:
            try:
                return datetime.fromisoformat(date_str).date()
            except Exception:
                return date.today()
        
        result = BacktestResult(
            strategy_name=result_dict['strategy_name'],
            params=result_dict['params'],
            start_dt=parse_date_str(result_dict['start_dt']),
            end_dt=parse_date_str(result_dict['end_dt']),
            initial_equity=result_dict['initial_equity'],
            final_equity=result_dict['final_equity'],
            performance=performance,
            risk_triggered=result_dict['risk_triggered'],
            frozen_during_backtest=result_dict['frozen_during_backtest'],
            frozen_reason=result_dict['frozen_reason'],
            status=result_dict['status'],
            error_message=result_dict['error_message'],
            equity_curve=result_dict.get('equity_curve', []),
            trade_records=result_dict.get('trade_records', []),
            risk_events=result_dict.get('risk_events', []),
            config_snapshot=result_dict.get('config_snapshot', {}),
        )
        
        if result.status == 'error':
            self.logger.error(f"回测失败: {result.error_message}")
        else:
            self.logger.info(
                f"回测完成: 初始权益={result.initial_equity:.2f}, "
                f"最终权益={result.final_equity:.2f}, "
                f"收益率={result.performance.total_return_percent:.2f}%"
            )
        
        return result

    def run_backtest(
        self,
        strategy_class: Type[StrategyBase],
        strategy_params: Dict[str, Any] = None,
        start_dt: date = None,
        end_dt: date = None,
    ) -> BacktestResult:
        if start_dt is None or end_dt is None:
            start_dt, end_dt = self._extract_date_range()
        
        if strategy_params is None:
            strategy_params = {}
        
        self.logger.info(f"开始单参数回测: {strategy_class.__name__}")
        self.logger.info(f"回测时间段: {start_dt} 至 {end_dt}")
        self.logger.info(f"策略参数: {strategy_params}")
        self.logger.info(f"成本配置: 手续费={self._cost_config.default_commission_per_lot}元/手, 滑点={self._cost_config.default_slippage_points}点")
        
        result = self._run_single_backtest(
            start_dt=start_dt,
            end_dt=end_dt,
            strategy_class=strategy_class,
            strategy_params=strategy_params,
            strategy_name=strategy_class.__name__,
        )
        
        self._results = [result]
        self._best_result = result
        
        return result

    def _generate_param_grid(
        self,
        param_ranges: Dict[str, ParameterRange],
    ) -> List[Dict[str, Any]]:
        param_names = list(param_ranges.keys())
        param_values = []
        
        for name, pr in param_ranges.items():
            values = []
            current = pr.min_val
            while current <= pr.max_val:
                values.append(pr.param_type(current))
                current += pr.step
            param_values.append(values)
        
        grids = []
        for combo in itertools.product(*param_values):
            grid = dict(zip(param_names, combo))
            grids.append(grid)
        
        return grids

    def run_optimization(
        self,
        strategy_class: Type[StrategyBase],
        param_ranges: Dict[str, ParameterRange],
        base_params: Dict[str, Any] = None,
        start_dt: date = None,
        end_dt: date = None,
        optimize_by: str = 'total_return_percent',
    ) -> List[BacktestResult]:
        if start_dt is None or end_dt is None:
            start_dt, end_dt = self._extract_date_range()
        
        base_params = base_params or {}
        
        param_grid = self._generate_param_grid(param_ranges)
        total_combinations = len(param_grid)
        
        self.logger.info(f"开始参数寻优: {strategy_class.__name__}")
        self.logger.info(f"回测时间段: {start_dt} 至 {end_dt}")
        self.logger.info(f"参数组合数量: {total_combinations}")
        self.logger.info(f"优化目标: {optimize_by}")
        self.logger.info(f"并行工作进程数: {self._max_workers}")
        
        results = []
        
        if self._max_workers > 1 and total_combinations > 1:
            self.logger.info("使用多进程并行执行回测...")
            
            tasks = []
            for i, params in enumerate(param_grid, 1):
                full_params = base_params.copy()
                full_params.update(params)
                tasks.append((i, full_params))
            
            with ProcessPoolExecutor(max_workers=self._max_workers) as executor:
                future_to_task = {
                    executor.submit(
                        _worker_run_backtest,
                        start_dt,
                        end_dt,
                        strategy_class,
                        full_params,
                        f"{strategy_class.__name__}_Opt_{i}",
                        self.config,
                        self._cost_config,
                        self._performance_config,
                    ): (i, full_params)
                    for i, full_params in tasks
                }
                
                completed = 0
                for future in as_completed(future_to_task):
                    i, params = future_to_task[future]
                    completed += 1
                    
                    progress_msg = f"完成进度: {completed}/{total_combinations}, 参数: {params}"
                    self.logger.info(progress_msg)
                    
                    if self.on_progress:
                        self.on_progress(completed, total_combinations, progress_msg)
                    
                    try:
                        result_dict = future.result()
                        
                        perf_dict = result_dict.get('performance', {})
                        performance = PerformanceMetrics(
                            total_return=perf_dict.get('total_return', 0.0),
                            total_return_percent=perf_dict.get('total_return_percent', 0.0),
                            annualized_return=perf_dict.get('annualized_return', 0.0),
                            annualized_return_percent=perf_dict.get('annualized_return_percent', 0.0),
                            max_drawdown=perf_dict.get('max_drawdown', 0.0),
                            max_drawdown_percent=perf_dict.get('max_drawdown_percent', 0.0),
                            sharpe_ratio=perf_dict.get('sharpe_ratio', 0.0),
                            sortino_ratio=perf_dict.get('sortino_ratio', 0.0),
                            calmar_ratio=perf_dict.get('calmar_ratio', 0.0),
                            total_trades=perf_dict.get('total_trades', 0),
                            winning_trades=perf_dict.get('winning_trades', 0),
                            losing_trades=perf_dict.get('losing_trades', 0),
                            win_rate=perf_dict.get('win_rate', 0.0),
                            profit_factor=perf_dict.get('profit_factor', 0.0),
                            avg_trade_return=perf_dict.get('avg_trade_return', 0.0),
                            total_commission_cost=perf_dict.get('total_commission_cost', 0.0),
                            total_slippage_cost=perf_dict.get('total_slippage_cost', 0.0),
                            total_cost=perf_dict.get('total_cost', 0.0),
                        )
                        
                        def parse_date_str(date_str: str) -> date:
                            try:
                                return datetime.fromisoformat(date_str).date()
                            except Exception:
                                return date.today()
                        
                        result = BacktestResult(
                            strategy_name=result_dict['strategy_name'],
                            params=result_dict['params'],
                            start_dt=parse_date_str(result_dict['start_dt']),
                            end_dt=parse_date_str(result_dict['end_dt']),
                            initial_equity=result_dict['initial_equity'],
                            final_equity=result_dict['final_equity'],
                            performance=performance,
                            risk_triggered=result_dict['risk_triggered'],
                            frozen_during_backtest=result_dict['frozen_during_backtest'],
                            frozen_reason=result_dict['frozen_reason'],
                            status=result_dict['status'],
                            error_message=result_dict['error_message'],
                            equity_curve=result_dict.get('equity_curve', []),
                            trade_records=result_dict.get('trade_records', []),
                            risk_events=result_dict.get('risk_events', []),
                            config_snapshot=result_dict.get('config_snapshot', {}),
                        )
                        
                        results.append(result)
                        
                    except Exception as e:
                        self.logger.error(f"回测任务 {i} 执行失败: {e}")
        else:
            self.logger.info("使用单进程串行执行回测...")
            
            for i, params in enumerate(param_grid, 1):
                full_params = base_params.copy()
                full_params.update(params)
                
                progress_msg = f"正在测试组合 {i}/{total_combinations}: {params}"
                self.logger.info(progress_msg)
                
                if self.on_progress:
                    self.on_progress(i, total_combinations, progress_msg)
                
                result = self._run_single_backtest(
                    start_dt=start_dt,
                    end_dt=end_dt,
                    strategy_class=strategy_class,
                    strategy_params=full_params,
                    strategy_name=f"{strategy_class.__name__}_Opt_{i}",
                )
                
                results.append(result)
                
                self.logger.info(
                    f"组合 {i} 结果: 收益率={result.performance.total_return_percent:.2f}%, "
                    f"最大回撤={result.performance.max_drawdown_percent:.2f}%, "
                    f"状态={result.status}"
                )
        
        def get_optimize_value(r: BacktestResult) -> float:
            if r.performance.total_trades <= 0:
                return float('-inf')
            
            if optimize_by == 'total_return_percent':
                return r.performance.total_return_percent
            elif optimize_by == 'annualized_return_percent':
                return r.performance.annualized_return_percent
            elif optimize_by == 'sharpe_ratio':
                return r.performance.sharpe_ratio
            elif optimize_by == 'sortino_ratio':
                return r.performance.sortino_ratio
            elif optimize_by == 'calmar_ratio':
                return r.performance.calmar_ratio
            elif optimize_by == 'max_drawdown_percent':
                return -r.performance.max_drawdown_percent
            elif optimize_by == 'win_rate':
                return r.performance.win_rate
            elif optimize_by == 'profit_factor':
                return r.performance.profit_factor
            return r.performance.total_return_percent
        
        results.sort(key=get_optimize_value, reverse=True)
        
        self._results = results
        if results:
            self._best_result = results[0]
            self.logger.info(f"最优参数组合: {results[0].params}")
            self.logger.info(
                f"最优结果: 收益率={results[0].performance.total_return_percent:.2f}%, "
                f"年化收益率={results[0].performance.annualized_return_percent:.2f}%, "
                f"最大回撤={results[0].performance.max_drawdown_percent:.2f}%, "
                f"夏普比率={results[0].performance.sharpe_ratio:.2f}"
            )
        
        return results

    def _print_performance_table(self, result: BacktestResult) -> None:
        p = result.performance
        
        print("\n" + "=" * 80)
        print("                          回测绩效报告")
        print("=" * 80)
        
        print(f"\n【基本信息】")
        print(f"  策略名称: {result.strategy_name}")
        print(f"  策略参数: {result.params}")
        print(f"  回测区间: {result.start_dt} 至 {result.end_dt}")
        print(f"  运行状态: {result.status}")
        
        print(f"\n【收益指标】")
        row1 = f"{'初始权益':<15} {result.initial_equity:>15,.2f}  {'最终权益':<15} {result.final_equity:>15,.2f}"
        print(f"  {row1}")
        
        row2 = f"{'总收益':<15} {p.total_return:>15,.2f}  {'总收益率':<15} {p.total_return_percent:>13.2f}%"
        print(f"  {row2}")
        
        row3 = f"{'年化收益':<15} {p.annualized_return_percent:>13.2f}%"
        print(f"  {row3}")
        
        print(f"\n【风险指标】")
        row4 = f"{'最大回撤':<15} {p.max_drawdown:>15,.2f}  {'最大回撤率':<15} {p.max_drawdown_percent:>13.2f}%"
        print(f"  {row4}")
        
        row5 = f"{'夏普比率':<15} {p.sharpe_ratio:>15.2f}  {'索提诺比率':<15} {p.sortino_ratio:>15.2f}"
        print(f"  {row5}")
        
        row6 = f"{'卡尔玛比率':<15} {p.calmar_ratio:>15.2f}"
        print(f"  {row6}")
        
        print(f"\n【交易统计】")
        row7 = f"{'总交易次数':<15} {p.total_trades:>15,d}  {'胜率':<15} {p.win_rate:>13.2f}%"
        print(f"  {row7}")
        
        row8 = f"{'盈亏比':<15} {p.profit_factor:>15.2f}  {'平均每笔收益':<15} {p.avg_trade_return:>15,.2f}"
        print(f"  {row8}")
        
        print(f"\n【成本统计】")
        row9 = f"{'总手续费':<15} {p.total_commission_cost:>15,.2f}  {'总滑点成本':<15} {p.total_slippage_cost:>15,.2f}"
        print(f"  {row9}")
        
        row10 = f"{'总成本':<15} {p.total_cost:>15,.2f}"
        print(f"  {row10}")
        
        print(f"\n【风控状态】")
        row11 = f"{'风控触发':<15} {'是' if result.risk_triggered else '否':<15}  {'期间冻结':<15} {'是' if result.frozen_during_backtest else '否':<15}"
        print(f"  {row11}")
        
        if result.frozen_reason:
            print(f"  冻结原因: {result.frozen_reason}")
        
        if result.error_message:
            print(f"\n【错误信息】")
            print(f"  {result.error_message}")
        
        print("\n" + "=" * 80)

    def _print_optimization_table(self, results: List[BacktestResult], top_n: int = 20) -> None:
        if not results:
            return
        
        print("\n" + "=" * 120)
        print("                                      参数寻优结果排名")
        print("=" * 120)
        
        header = (
            f"{'排名':<4} "
            f"{'参数组合':<25} "
            f"{'总收益率(%)':<12} "
            f"{'年化(%)':<10} "
            f"{'最大回撤(%)':<12} "
            f"{'夏普比率':<10} "
            f"{'交易次数':<10} "
            f"{'风控冻结':<8} "
            f"{'状态':<10}"
        )
        print(header)
        print("-" * 120)
        
        display_count = min(top_n, len(results))
        for i, r in enumerate(results[:display_count], 1):
            params_str = str(r.params)[:23]
            if len(str(r.params)) > 23:
                params_str += "..."
            
            row = (
                f"{i:<4} "
                f"{params_str:<25} "
                f"{r.performance.total_return_percent:<12.2f} "
                f"{r.performance.annualized_return_percent:<10.2f} "
                f"{r.performance.max_drawdown_percent:<12.2f} "
                f"{r.performance.sharpe_ratio:<10.2f} "
                f"{r.performance.total_trades:<10} "
                f"{'是' if r.frozen_during_backtest else '否':<8} "
                f"{r.status:<10}"
            )
            print(row)
        
        if len(results) > display_count:
            print(f"... 等共 {len(results)} 个组合")
        
        print("\n" + "=" * 120)

    def generate_report(self, output_dir: str = None) -> Dict[str, Any]:
        if not self._results:
            self.logger.warning("没有回测结果，无法生成报告")
            return {}
        
        if output_dir is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            output_dir = os.path.join(base_dir, 'logs', 'backtest_reports')
        
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        def result_to_dict(r: BacktestResult) -> Dict[str, Any]:
            return {
                'strategy_name': r.strategy_name,
                'params': r.params,
                'start_dt': r.start_dt.isoformat(),
                'end_dt': r.end_dt.isoformat(),
                'initial_equity': r.initial_equity,
                'final_equity': r.final_equity,
                'performance': asdict(r.performance),
                'risk_triggered': r.risk_triggered,
                'frozen_during_backtest': r.frozen_during_backtest,
                'frozen_reason': r.frozen_reason,
                'status': r.status,
                'error_message': r.error_message,
                'equity_curve': r.equity_curve,
                'trade_records': r.trade_records,
                'risk_events': r.risk_events,
                'config_snapshot': r.config_snapshot,
            }
        
        report = {
            'generated_at': datetime.now().isoformat(),
            'total_backtests': len(self._results),
            'best_result': result_to_dict(self._best_result) if self._best_result else None,
            'all_results': [result_to_dict(r) for r in self._results],
        }
        
        if len(self._results) == 1 and self._best_result:
            self._print_performance_table(self._best_result)
        else:
            if self._best_result:
                self._print_performance_table(self._best_result)
            self._print_optimization_table(self._results)
        
        json_file = os.path.join(output_dir, f'backtest_report_{timestamp}.json')
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2, default=str)
        self.logger.info(f"JSON 报告已保存: {json_file}")
        
        csv_file = os.path.join(output_dir, f'backtest_results_{timestamp}.csv')
        self._write_csv_report(csv_file)
        self.logger.info(f"CSV 报告已保存: {csv_file}")
        
        if self._best_result and self._best_result.equity_curve:
            equity_file = os.path.join(output_dir, f'equity_curve_{timestamp}.csv')
            self._write_equity_curve_csv(equity_file, self._best_result.equity_curve)
            self.logger.info(f"权益曲线 CSV 已保存: {equity_file}")
        
        if len(self._results) > 1:
            chart_path = os.path.join(output_dir, f'optimization_result_{timestamp}.png')
            generated_chart = self.generate_optimization_chart(output_path=chart_path)
            if generated_chart:
                report['chart_path'] = generated_chart
                self.logger.info(f"可视化图表已保存: {generated_chart}")
        
        return report

    def _write_csv_report(self, csv_file: str) -> None:
        if not self._results:
            return
        
        fieldnames = [
            'rank', 'strategy_name', 'params', 'start_dt', 'end_dt',
            'initial_equity', 'final_equity',
            'total_return', 'total_return_percent',
            'annualized_return_percent',
            'max_drawdown', 'max_drawdown_percent',
            'sharpe_ratio', 'sortino_ratio', 'calmar_ratio',
            'total_trades', 'win_rate', 'profit_factor',
            'total_commission_cost', 'total_slippage_cost', 'total_cost',
            'risk_triggered', 'frozen_during_backtest', 'frozen_reason',
            'status', 'error_message'
        ]
        
        with open(csv_file, 'w', encoding='utf-8-sig', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for i, r in enumerate(self._results, 1):
                p = r.performance
                row = {
                    'rank': i,
                    'strategy_name': r.strategy_name,
                    'params': json.dumps(r.params, ensure_ascii=False),
                    'start_dt': r.start_dt,
                    'end_dt': r.end_dt,
                    'initial_equity': r.initial_equity,
                    'final_equity': r.final_equity,
                    'total_return': p.total_return,
                    'total_return_percent': p.total_return_percent,
                    'annualized_return_percent': p.annualized_return_percent,
                    'max_drawdown': p.max_drawdown,
                    'max_drawdown_percent': p.max_drawdown_percent,
                    'sharpe_ratio': p.sharpe_ratio,
                    'sortino_ratio': p.sortino_ratio,
                    'calmar_ratio': p.calmar_ratio,
                    'total_trades': p.total_trades,
                    'win_rate': p.win_rate,
                    'profit_factor': p.profit_factor,
                    'total_commission_cost': p.total_commission_cost,
                    'total_slippage_cost': p.total_slippage_cost,
                    'total_cost': p.total_cost,
                    'risk_triggered': r.risk_triggered,
                    'frozen_during_backtest': r.frozen_during_backtest,
                    'frozen_reason': r.frozen_reason or '',
                    'status': r.status,
                    'error_message': r.error_message or '',
                }
                writer.writerow(row)

    def _write_equity_curve_csv(self, csv_file: str, equity_curve: List[Dict[str, Any]]) -> None:
        if not equity_curve:
            return
        
        fieldnames = ['cycle', 'timestamp', 'equity', 'margin_used']
        
        with open(csv_file, 'w', encoding='utf-8-sig', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for point in equity_curve:
                row = {
                    'cycle': point.get('cycle', 0),
                    'timestamp': point.get('timestamp', 0),
                    'equity': point.get('equity', 0),
                    'margin_used': point.get('margin_used', 0),
                }
                writer.writerow(row)

    def get_results(self) -> List[BacktestResult]:
        return self._results.copy()

    def get_best_result(self) -> Optional[BacktestResult]:
        return self._best_result

    def clear_results(self) -> None:
        self._results = []
        self._best_result = None
        self.logger.info("回测结果已清空")

    def sync_optimal_params_to_config(
        self,
        config_path: str = None,
        strategy_name: str = None,
        param_mapping: Dict[str, str] = None,
    ) -> bool:
        """
        将最优参数同步更新到配置文件的 strategies 段。
        
        Args:
            config_path: 配置文件路径，默认使用项目根目录下的 config/settings.yaml
            strategy_name: 目标策略名称，如果为 None 则尝试匹配第一个同类型策略
            param_mapping: 参数名映射字典，例如 {'short_period': 'fast', 'long_period': 'slow'}
        
        Returns:
            bool: 是否同步成功
        """
        if not self._best_result:
            self.logger.error("没有找到最优参数结果，请先运行参数寻优")
            return False
        
        if config_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            config_path = os.path.join(base_dir, 'config', 'settings.yaml')
        
        if not os.path.exists(config_path):
            self.logger.error(f"配置文件不存在: {config_path}")
            return False
        
        default_mapping = {
            'short_period': 'fast',
            'long_period': 'slow',
        }
        param_mapping = param_mapping or default_mapping
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            strategies = config.get('strategies', [])
            if not strategies:
                self.logger.error("配置文件中没有 strategies 配置")
                return False
            
            target_strategy_idx = None
            best_params = self._best_result.params
            best_strategy_name = self._best_result.strategy_name
            
            if strategy_name:
                for i, s in enumerate(strategies):
                    if s.get('name') == strategy_name:
                        target_strategy_idx = i
                        break
                if target_strategy_idx is None:
                    self.logger.error(f"未找到名为 '{strategy_name}' 的策略配置")
                    return False
            else:
                for i, s in enumerate(strategies):
                    if s.get('class') in best_strategy_name or best_strategy_name in s.get('class', ''):
                        target_strategy_idx = i
                        break
                if target_strategy_idx is None:
                    self.logger.warning(f"未找到匹配的策略，将使用第一个策略")
                    target_strategy_idx = 0
            
            target_strategy = strategies[target_strategy_idx]
            current_params = target_strategy.get('params', {})
            
            self.logger.info(f"当前策略参数: {current_params}")
            self.logger.info(f"最优参数: {best_params}")
            
            updated = False
            for source_param, target_param in param_mapping.items():
                if source_param in best_params and target_param in current_params:
                    old_value = current_params[target_param]
                    new_value = best_params[source_param]
                    if old_value != new_value:
                        current_params[target_param] = new_value
                        self.logger.info(f"更新参数: {target_param}: {old_value} -> {new_value}")
                        updated = True
            
            for param_name, param_value in best_params.items():
                if param_name not in param_mapping and param_name in current_params:
                    old_value = current_params[param_name]
                    if old_value != param_value:
                        current_params[param_name] = param_value
                        self.logger.info(f"更新参数: {param_name}: {old_value} -> {param_value}")
                        updated = True
            
            if updated:
                target_strategy['params'] = current_params
                strategies[target_strategy_idx] = target_strategy
                config['strategies'] = strategies
                
                import shutil
                backup_path = config_path + '.bak'
                shutil.copy2(config_path, backup_path)
                self.logger.info(f"✓ 配置文件已备份: {backup_path}")
                
                with open(config_path, 'w', encoding='utf-8') as f:
                    yaml.dump(config, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
                
                self.logger.info(f"✓ 最优参数已同步到配置文件: {config_path}")
                print(f"\n{'='*80}")
                print("                    参数同步完成")
                print("="*80)
                print(f"原配置已备份: {backup_path}")
                print(f"目标策略: {target_strategy.get('name', 'Unknown')}")
                print(f"更新后的参数: {current_params}")
            else:
                self.logger.info("配置文件参数已是最新，无需更新")
            
            return True
            
        except Exception as e:
            self.logger.error(f"参数同步失败: {e}", exc_info=True)
            return False

    def generate_optimization_chart(
        self,
        output_path: str = None,
        chart_type: str = 'heatmap',
    ) -> Optional[str]:
        """
        生成参数寻优结果的可视化图表。
        
        Args:
            output_path: 输出文件路径，默认保存到 logs/backtest_reports/optimization_result.png
            chart_type: 图表类型，'heatmap' (热力图) 或 'equity' (资金曲线)
        
        Returns:
            Optional[str]: 生成的图表文件路径，失败则返回 None
        """
        if not self._results or len(self._results) < 2:
            self.logger.warning("需要至少2个回测结果才能生成可视化图表")
            return None
        
        if not MATPLOTLIB_AVAILABLE or not NUMPY_AVAILABLE:
            self.logger.error("matplotlib 或 numpy 未安装，请运行: pip install matplotlib numpy")
            return None
        
        _check_chinese_font_support()
        
        if output_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            output_dir = os.path.join(base_dir, 'logs', 'backtest_reports')
            os.makedirs(output_dir, exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = os.path.join(output_dir, f'optimization_result_{timestamp}.png')
        
        try:
            total_trades = sum(r.performance.total_trades for r in self._results)
            
            if chart_type == 'heatmap' and self._best_result:
                param_names = list(self._best_result.params.keys())
                if len(param_names) >= 2:
                    x_param = param_names[0]
                    y_param = param_names[1]
                    
                    x_values = sorted(set(r.params.get(x_param) for r in self._results if x_param in r.params))
                    y_values = sorted(set(r.params.get(y_param) for r in self._results if y_param in r.params))
                    
                    if len(x_values) > 1 and len(y_values) > 1:
                        return self._generate_heatmap(
                            output_path, x_param, y_param, x_values, y_values,
                            self._results, total_trades
                        )
            
            return self._generate_equity_curves(output_path, self._results, self._best_result, total_trades)
            
        except Exception as e:
            self.logger.error(f"生成图表失败: {e}", exc_info=True)
            return None

    def _generate_heatmap(
        self,
        output_path: str,
        x_param: str,
        y_param: str,
        x_values: List[Any],
        y_values: List[Any],
        results: List[BacktestResult],
        total_trades: int = 0,
    ) -> str:
        x_to_idx = {v: i for i, v in enumerate(x_values)}
        y_to_idx = {v: i for i, v in enumerate(y_values)}
        
        heatmap_data = np.full((len(y_values), len(x_values)), np.nan)
        
        for r in results:
            x_val = r.params.get(x_param)
            y_val = r.params.get(y_param)
            if x_val in x_to_idx and y_val in y_to_idx:
                heatmap_data[y_to_idx[y_val], x_to_idx[x_val]] = r.performance.total_return_percent
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(
            _get_label('参数寻优结果分析', 'Parameter Optimization Analysis'), 
            fontsize=16, fontweight='bold'
        )
        
        if total_trades <= 0:
            for i in range(2):
                for j in range(2):
                    ax = axes[i, j]
                    ax.text(0.5, 0.55, 
                           _get_label('No Trades Executed', '无交易执行'),
                           transform=ax.transAxes, ha='center', va='center',
                           fontsize=18, fontweight='bold', color='red')
                    ax.text(0.5, 0.4,
                           _get_label('Please check signal logic or backtest period', 
                                     '请检查信号逻辑或回测区间'),
                           transform=ax.transAxes, ha='center', va='center',
                           fontsize=12, color='darkred')
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    ax.spines['left'].set_visible(False)
                    ax.spines['bottom'].set_visible(False)
            
            plt.tight_layout()
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            self.logger.warning(f"图表已生成但显示无交易提示: {output_path}")
            return output_path
        
        ax1 = axes[0, 0]
        im1 = ax1.imshow(heatmap_data, cmap='RdYlGn', aspect='auto', origin='lower')
        ax1.set_xticks(range(len(x_values)))
        ax1.set_xticklabels(x_values)
        ax1.set_yticks(range(len(y_values)))
        ax1.set_yticklabels(y_values)
        ax1.set_xlabel(x_param)
        ax1.set_ylabel(y_param)
        ax1.set_title(
            _get_label('总收益率热力图 (%)', 'Total Return Heatmap (%)')
        )
        plt.colorbar(im1, ax=ax1)
        
        for i in range(len(y_values)):
            for j in range(len(x_values)):
                if not np.isnan(heatmap_data[i, j]):
                    ax1.text(j, i, f'{heatmap_data[i, j]:.1f}%', 
                            ha='center', va='center', fontsize=8,
                            color='black' if abs(heatmap_data[i, j]) < 10 else 'white')
        
        ax2 = axes[0, 1]
        drawdown_data = np.full((len(y_values), len(x_values)), np.nan)
        for r in results:
            x_val = r.params.get(x_param)
            y_val = r.params.get(y_param)
            if x_val in x_to_idx and y_val in y_to_idx:
                drawdown_data[y_to_idx[y_val], x_to_idx[x_val]] = r.performance.max_drawdown_percent
        
        im2 = ax2.imshow(drawdown_data, cmap='YlOrRd_r', aspect='auto', origin='lower')
        ax2.set_xticks(range(len(x_values)))
        ax2.set_xticklabels(x_values)
        ax2.set_yticks(range(len(y_values)))
        ax2.set_yticklabels(y_values)
        ax2.set_xlabel(x_param)
        ax2.set_ylabel(y_param)
        ax2.set_title(
            _get_label('最大回撤率热力图 (%)', 'Max Drawdown Heatmap (%)')
        )
        plt.colorbar(im2, ax=ax2)
        
        returns = [r.performance.total_return_percent for r in results]
        sharpe_ratios = [r.performance.sharpe_ratio for r in results]
        max_drawdowns = [r.performance.max_drawdown_percent for r in results]
        
        ax3 = axes[1, 0]
        ax3.scatter(returns, sharpe_ratios, c=max_drawdowns, cmap='YlOrRd', alpha=0.7, s=50)
        ax3.set_xlabel(_get_label('总收益率 (%)', 'Total Return (%)'))
        ax3.set_ylabel(_get_label('夏普比率', 'Sharpe Ratio'))
        ax3.set_title(
            _get_label('收益-风险散点图', 'Return-Risk Scatter Plot')
        )
        ax3.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
        ax3.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
        plt.colorbar(
            ax3.collections[0], 
            ax=ax3, 
            label=_get_label('最大回撤率 (%)', 'Max Drawdown (%)')
        )
        
        if self._best_result:
            best_return = self._best_result.performance.total_return_percent
            best_sharpe = self._best_result.performance.sharpe_ratio
            ax3.scatter(
                [best_return], [best_sharpe], 
                c='green', s=200, marker='*', 
                label=_get_label('最优组合', 'Best Combination')
            )
            ax3.legend()
        
        ax4 = axes[1, 1]
        sorted_results = sorted(results, key=lambda r: r.performance.total_return_percent, reverse=True)
        top_n = min(20, len(sorted_results))
        top_returns = [r.performance.total_return_percent for r in sorted_results[:top_n]]
        param_labels = [f"{list(r.params.values())[:2]}" for r in sorted_results[:top_n]]
        
        colors = ['green' if r >= 0 else 'red' for r in top_returns]
        y_pos = range(top_n)
        ax4.barh(y_pos, top_returns, color=colors, alpha=0.7)
        ax4.set_yticks(y_pos)
        ax4.set_yticklabels(param_labels, fontsize=8)
        ax4.set_xlabel(_get_label('总收益率 (%)', 'Total Return (%)'))
        ax4.set_title(
            _get_label(f'Top {top_n} 参数组合收益率排名', f'Top {top_n} Parameter Return Ranking')
        )
        ax4.invert_yaxis()
        ax4.axvline(x=0, color='black', linewidth=0.5)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"✓ 热力图已生成: {output_path}")
        return output_path

    def _generate_equity_curves(
        self,
        output_path: str,
        results: List[BacktestResult],
        best_result: Optional[BacktestResult],
        total_trades: int = 0,
    ) -> str:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(
            _get_label('回测结果综合分析', 'Backtest Result Analysis'), 
            fontsize=16, fontweight='bold'
        )
        
        if total_trades <= 0:
            for i in range(2):
                for j in range(2):
                    ax = axes[i, j]
                    ax.text(0.5, 0.55, 
                           _get_label('No Trades Executed', '无交易执行'),
                           transform=ax.transAxes, ha='center', va='center',
                           fontsize=18, fontweight='bold', color='red')
                    ax.text(0.5, 0.4,
                           _get_label('Please check signal logic or backtest period', 
                                     '请检查信号逻辑或回测区间'),
                           transform=ax.transAxes, ha='center', va='center',
                           fontsize=12, color='darkred')
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    ax.spines['left'].set_visible(False)
                    ax.spines['bottom'].set_visible(False)
            
            plt.tight_layout()
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            self.logger.warning(f"图表已生成但显示无交易提示: {output_path}")
            return output_path
        
        ax1 = axes[0, 0]
        if best_result and best_result.equity_curve:
            initial_equity = best_result.initial_equity
            timestamps = [p.get('timestamp', i) for i, p in enumerate(best_result.equity_curve)]
            equities = [p.get('equity', initial_equity) for p in best_result.equity_curve]
            equities = [initial_equity] + equities
            
            ax1.plot(range(len(equities)), equities, 'b-', linewidth=1.5, 
                     label=_get_label('权益曲线', 'Equity Curve'))
            ax1.fill_between(range(len(equities)), equities, alpha=0.3)
            ax1.set_xlabel(_get_label('时间周期', 'Time Period'))
            ax1.set_ylabel(_get_label('账户权益', 'Account Equity'))
            ax1.set_title(
                _get_label('最优参数组合 - 权益曲线', 'Best Parameters - Equity Curve')
            )
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        ax2 = axes[0, 1]
        returns = [r.performance.total_return_percent for r in results]
        max_drawdowns = [r.performance.max_drawdown_percent for r in results]
        sharpe_ratios = [r.performance.sharpe_ratio for r in results]
        
        scatter = ax2.scatter(returns, max_drawdowns, c=sharpe_ratios, cmap='viridis', alpha=0.7, s=60)
        ax2.set_xlabel(_get_label('总收益率 (%)', 'Total Return (%)'))
        ax2.set_ylabel(_get_label('最大回撤率 (%)', 'Max Drawdown (%)'))
        ax2.set_title(
            _get_label('收益-回撤关系图', 'Return-Drawdown Relationship')
        )
        ax2.grid(True, alpha=0.3)
        ax2.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
        plt.colorbar(
            scatter, 
            ax=ax2, 
            label=_get_label('夏普比率', 'Sharpe Ratio')
        )
        
        if best_result:
            best_return = best_result.performance.total_return_percent
            best_dd = best_result.performance.max_drawdown_percent
            ax2.scatter(
                [best_return], [best_dd], 
                c='red', s=200, marker='*', 
                label=_get_label('最优组合', 'Best Combination')
            )
            ax2.legend()
        
        ax3 = axes[1, 0]
        valid_sharpe = [s for s in sharpe_ratios if s != 0]
        if valid_sharpe:
            ax3.hist(valid_sharpe, bins=15, edgecolor='black', alpha=0.7, color='steelblue')
            ax3.axvline(x=1, color='red', linestyle='--', linewidth=2, 
                        label=_get_label('夏普比率=1', 'Sharpe Ratio=1'))
            ax3.axvline(
                x=np.mean(valid_sharpe) if valid_sharpe else 0, 
                color='green', linestyle='-', linewidth=2, 
                label=_get_label('均值', 'Mean')
            )
            ax3.set_xlabel(_get_label('夏普比率', 'Sharpe Ratio'))
            ax3.set_ylabel(_get_label('频次', 'Frequency'))
            ax3.set_title(
                _get_label('夏普比率分布', 'Sharpe Ratio Distribution')
            )
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        ax4 = axes[1, 1]
        sorted_results = sorted(results, key=lambda r: r.performance.total_return_percent, reverse=True)
        top_n = min(15, len(sorted_results))
        
        return_values = [r.performance.total_return_percent for r in sorted_results[:top_n]]
        drawdown_values = [-r.performance.max_drawdown_percent for r in sorted_results[:top_n]]
        sharpe_values = [r.performance.sharpe_ratio * 10 for r in sorted_results[:top_n]]
        
        x = np.arange(top_n)
        width = 0.25
        
        ax4.bar(x - width, return_values, width, 
                label=_get_label('收益率(%)', 'Return (%)'), alpha=0.8)
        ax4.bar(x, drawdown_values, width, 
                label=_get_label('-最大回撤(%)', '-Max Drawdown (%)'), alpha=0.8)
        ax4.bar(x + width, sharpe_values, width, 
                label=_get_label('夏普比率×10', 'Sharpe Ratio × 10'), alpha=0.8)
        
        ax4.set_xlabel(_get_label('参数组合排名', 'Parameter Ranking'))
        ax4.set_ylabel(_get_label('数值', 'Value'))
        ax4.set_title(
            _get_label(f'Top {top_n} 组合多指标对比', f'Top {top_n} Parameters Comparison')
        )
        ax4.set_xticks(x)
        ax4.set_xticklabels([str(i+1) for i in range(top_n)])
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.axhline(y=0, color='black', linewidth=0.5)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"✓ 分析图表已生成: {output_path}")
        return output_path


def _worker_run_backtest(
    start_dt: date,
    end_dt: date,
    strategy_class: Type[StrategyBase],
    strategy_params: Dict[str, Any],
    strategy_name: str,
    config: Dict[str, Any],
    cost_config: CostConfig,
    performance_config: PerformanceConfig,
) -> Dict[str, Any]:
    engine = BacktestEngine(config=config)
    engine._cost_config = cost_config
    engine._performance_config = performance_config
    
    return engine._run_single_backtest_internal(
        start_dt=start_dt,
        end_dt=end_dt,
        strategy_class=strategy_class,
        strategy_params=strategy_params,
        strategy_name=strategy_name,
        config=config,
        cost_config=cost_config,
        performance_config=performance_config,
    )


def _calculate_drawdown_series(equities: List[float], initial_equity: float) -> List[Tuple[int, float, float]]:
    """
    计算回撤时间序列
    返回: [(周期索引, 回撤金额, 回撤百分比), ...]
    """
    if not equities:
        return []
    
    peak = initial_equity
    drawdown_series = []
    
    for i, eq in enumerate(equities):
        if eq > peak:
            peak = eq
        dd = peak - eq
        dd_percent = (dd / peak * 100) if peak > 0 else 0.0
        drawdown_series.append((i, dd, dd_percent))
    
    return drawdown_series


class BacktestCharts:
    """独立的图表生成类，生成两张独立图表：资金净值走势图 + 最大回撤分布图"""
    
    @staticmethod
    def generate_equity_curve_chart(
        result: BacktestResult,
        output_path: str = None,
    ) -> Optional[str]:
        """
        生成资金净值走势图 (Equity Curve)
        
        Args:
            result: 回测结果
            output_path: 输出文件路径
        
        Returns:
            生成的图表文件路径，失败则返回 None
        """
        if not MATPLOTLIB_AVAILABLE or not NUMPY_AVAILABLE:
            logging.getLogger(__name__).error("matplotlib 或 numpy 未安装")
            return None
        
        if not result or not result.equity_curve:
            logging.getLogger(__name__).warning("没有权益曲线数据，无法生成图表")
            return None
        
        _check_chinese_font_support()
        
        if output_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            output_dir = os.path.join(base_dir, 'logs', 'backtest_reports')
            os.makedirs(output_dir, exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = os.path.join(output_dir, f'equity_curve_{timestamp}.png')
        
        try:
            initial_equity = result.initial_equity
            equity_data = result.equity_curve
            
            timestamps = [p.get('cycle', i) for i, p in enumerate(equity_data)]
            equities = [p.get('equity', initial_equity) for p in equity_data]
            equities = [initial_equity] + equities
            timestamps = [-1] + timestamps
            
            fig, ax = plt.subplots(figsize=(12, 6))
            
            ax.plot(range(len(equities)), equities, 'b-', linewidth=1.5, alpha=0.8)
            ax.fill_between(range(len(equities)), equities, initial_equity, 
                           where=np.array(equities) >= initial_equity,
                           alpha=0.3, color='green', interpolate=True)
            ax.fill_between(range(len(equities)), equities, initial_equity,
                           where=np.array(equities) < initial_equity,
                           alpha=0.3, color='red', interpolate=True)
            
            ax.axhline(y=initial_equity, color='gray', linestyle='--', alpha=0.5, 
                      label=_get_label('初始资金', 'Initial Capital'))
            
            final_eq = equities[-1]
            total_return = final_eq - initial_equity
            return_pct = (total_return / initial_equity * 100) if initial_equity > 0 else 0
            
            ax.scatter([len(equities)-1], [final_eq], c='green' if return_pct >= 0 else 'red', 
                      s=100, zorder=5, label=_get_label('最终权益', 'Final Equity'))
            
            ax.set_xlabel(_get_label('回测周期', 'Backtest Period'), fontsize=12)
            ax.set_ylabel(_get_label('账户权益', 'Account Equity'), fontsize=12)
            ax.set_title(
                _get_label('资金净值走势图', 'Equity Curve'),
                fontsize=14, fontweight='bold'
            )
            
            info_text = _get_label(
                f'初始资金: {initial_equity:,.0f}\n最终权益: {final_eq:,.0f}\n收益率: {return_pct:.2f}%',
                f'Initial: {initial_equity:,.0f}\nFinal: {final_eq:,.0f}\nReturn: {return_pct:.2f}%'
            )
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=10,
                   verticalalignment='top', bbox=props)
            
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            logging.getLogger(__name__).info(f"✓ 资金净值走势图已生成: {output_path}")
            return output_path
            
        except Exception as e:
            logging.getLogger(__name__).error(f"生成资金净值走势图失败: {e}", exc_info=True)
            return None
    
    @staticmethod
    def generate_drawdown_chart(
        result: BacktestResult,
        output_path: str = None,
    ) -> Optional[str]:
        """
        生成最大回撤分布图 (Drawdown Distribution)
        
        Args:
            result: 回测结果
            output_path: 输出文件路径
        
        Returns:
            生成的图表文件路径，失败则返回 None
        """
        if not MATPLOTLIB_AVAILABLE or not NUMPY_AVAILABLE:
            logging.getLogger(__name__).error("matplotlib 或 numpy 未安装")
            return None
        
        if not result or not result.equity_curve:
            logging.getLogger(__name__).warning("没有权益曲线数据，无法生成回撤图表")
            return None
        
        _check_chinese_font_support()
        
        if output_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            output_dir = os.path.join(base_dir, 'logs', 'backtest_reports')
            os.makedirs(output_dir, exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = os.path.join(output_dir, f'drawdown_chart_{timestamp}.png')
        
        try:
            initial_equity = result.initial_equity
            equity_data = result.equity_curve
            
            equities = [p.get('equity', initial_equity) for p in equity_data]
            equities = [initial_equity] + equities
            
            drawdown_series = _calculate_drawdown_series(equities, initial_equity)
            dd_percents = [dd[2] for dd in drawdown_series]
            
            fig, axes = plt.subplots(2, 1, figsize=(12, 10))
            
            ax1 = axes[0]
            ax1.fill_between(range(len(dd_percents)), dd_percents, 0, 
                           color='red', alpha=0.6)
            ax1.plot(range(len(dd_percents)), dd_percents, 'r-', linewidth=0.8, alpha=0.8)
            
            max_dd = max(dd_percents) if dd_percents else 0
            max_dd_idx = dd_percents.index(max_dd) if dd_percents else 0
            
            ax1.scatter([max_dd_idx], [max_dd], c='darkred', s=100, zorder=5,
                       label=_get_label(f'最大回撤: {max_dd:.2f}%', f'Max Drawdown: {max_dd:.2f}%'))
            
            ax1.set_xlabel(_get_label('回测周期', 'Backtest Period'), fontsize=12)
            ax1.set_ylabel(_get_label('回撤率 (%)', 'Drawdown (%)'), fontsize=12)
            ax1.set_title(
                _get_label('回撤时间序列', 'Drawdown Time Series'),
                fontsize=14, fontweight='bold'
            )
            ax1.legend(loc='best')
            ax1.grid(True, alpha=0.3)
            ax1.invert_yaxis()
            
            ax2 = axes[1]
            valid_dds = [dd for dd in dd_percents if dd > 0.01]
            if valid_dds:
                n, bins, patches = ax2.hist(valid_dds, bins=20, edgecolor='black', 
                                            alpha=0.7, color='orange')
                
                for patch, left_bin in zip(patches, bins[:-1]):
                    if left_bin < max_dd * 0.33:
                        patch.set_facecolor('green')
                    elif left_bin < max_dd * 0.66:
                        patch.set_facecolor('orange')
                    else:
                        patch.set_facecolor('red')
                
                ax2.axvline(x=np.mean(valid_dds), color='blue', linestyle='--', linewidth=2,
                           label=_get_label(f'平均回撤: {np.mean(valid_dds):.2f}%', 
                                          f'Avg Drawdown: {np.mean(valid_dds):.2f}%'))
                ax2.axvline(x=max_dd, color='red', linestyle='-', linewidth=2,
                           label=_get_label(f'最大回撤: {max_dd:.2f}%', 
                                          f'Max Drawdown: {max_dd:.2f}%'))
                
                ax2.set_xlabel(_get_label('回撤率区间 (%)', 'Drawdown Range (%)'), fontsize=12)
                ax2.set_ylabel(_get_label('频次', 'Frequency'), fontsize=12)
                ax2.set_title(
                    _get_label('回撤分布直方图', 'Drawdown Distribution Histogram'),
                    fontsize=14, fontweight='bold'
                )
                ax2.legend(loc='best')
                ax2.grid(True, alpha=0.3)
            else:
                ax2.text(0.5, 0.5, _get_label('无显著回撤数据', 'No significant drawdown data'),
                        transform=ax2.transAxes, ha='center', va='center', fontsize=14)
                ax2.set_title(
                    _get_label('回撤分布直方图', 'Drawdown Distribution Histogram'),
                    fontsize=14, fontweight='bold'
                )
            
            plt.tight_layout()
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            logging.getLogger(__name__).info(f"✓ 最大回撤分布图已生成: {output_path}")
            return output_path
            
        except Exception as e:
            logging.getLogger(__name__).error(f"生成最大回撤分布图失败: {e}", exc_info=True)
            return None
    
    @staticmethod
    def generate_both_charts(
        result: BacktestResult,
        output_dir: str = None,
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        生成两张独立图表：资金净值走势图 + 最大回撤分布图
        
        Args:
            result: 回测结果
            output_dir: 输出目录
        
        Returns:
            (权益图表路径, 回撤图表路径)
        """
        if output_dir is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            output_dir = os.path.join(base_dir, 'logs', 'backtest_reports')
        
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        equity_path = os.path.join(output_dir, f'equity_curve_{timestamp}.png')
        drawdown_path = os.path.join(output_dir, f'drawdown_chart_{timestamp}.png')
        
        equity_result = BacktestCharts.generate_equity_curve_chart(result, equity_path)
        drawdown_result = BacktestCharts.generate_drawdown_chart(result, drawdown_path)
        
        return equity_result, drawdown_result


def save_optimization_results_to_csv(
    results: List[BacktestResult],
    output_path: str = None,
) -> Optional[str]:
    """
    将参数寻优结果保存为 CSV 文件
    
    Args:
        results: 回测结果列表
        output_path: 输出文件路径，默认保存到 logs/optimization_results.csv
    
    Returns:
        保存的文件路径，失败则返回 None
    """
    if not results:
        logging.getLogger(__name__).warning("没有寻优结果可保存")
        return None
    
    if output_path is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        output_dir = os.path.join(base_dir, 'logs')
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 'optimization_results.csv')
    
    try:
        fieldnames = [
            'rank', 'strategy_name', 'params', 'start_dt', 'end_dt',
            'initial_equity', 'final_equity',
            'total_return', 'total_return_percent',
            'annualized_return_percent',
            'max_drawdown', 'max_drawdown_percent',
            'sharpe_ratio', 'sortino_ratio', 'calmar_ratio',
            'total_trades', 'win_rate', 'profit_factor',
            'risk_triggered', 'frozen_during_backtest', 'frozen_reason',
            'status', 'error_message'
        ]
        
        with open(output_path, 'w', encoding='utf-8-sig', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for i, r in enumerate(results, 1):
                p = r.performance
                row = {
                    'rank': i,
                    'strategy_name': r.strategy_name,
                    'params': json.dumps(r.params, ensure_ascii=False),
                    'start_dt': r.start_dt.isoformat(),
                    'end_dt': r.end_dt.isoformat(),
                    'initial_equity': r.initial_equity,
                    'final_equity': r.final_equity,
                    'total_return': p.total_return,
                    'total_return_percent': p.total_return_percent,
                    'annualized_return_percent': p.annualized_return_percent,
                    'max_drawdown': p.max_drawdown,
                    'max_drawdown_percent': p.max_drawdown_percent,
                    'sharpe_ratio': p.sharpe_ratio,
                    'sortino_ratio': p.sortino_ratio,
                    'calmar_ratio': p.calmar_ratio,
                    'total_trades': p.total_trades,
                    'win_rate': p.win_rate,
                    'profit_factor': p.profit_factor,
                    'risk_triggered': r.risk_triggered,
                    'frozen_during_backtest': r.frozen_during_backtest,
                    'frozen_reason': r.frozen_reason or '',
                    'status': r.status,
                    'error_message': r.error_message or '',
                }
                writer.writerow(row)
        
        logging.getLogger(__name__).info(f"✓ 参数寻优结果已保存到: {output_path}")
        print(f"\n{'='*80}")
        print("                    参数寻优结果导出完成")
        print("="*80)
        print(f"文件路径: {output_path}")
        print(f"共导出 {len(results)} 条记录")
        
        return output_path
        
    except Exception as e:
        logging.getLogger(__name__).error(f"保存寻优结果 CSV 失败: {e}", exc_info=True)
        return None


def apply_best_params_to_config(
    best_result: BacktestResult,
    config_path: str = None,
    strategy_name: str = None,
    param_mapping: Dict[str, str] = None,
) -> bool:
    """
    一键应用最优参数到配置文件
    
    Args:
        best_result: 最优回测结果
        config_path: 配置文件路径，默认使用 config/settings.yaml
        strategy_name: 目标策略名称，None 则自动匹配
        param_mapping: 参数名映射字典，例如 {'short_period': 'fast', 'long_period': 'slow'}
    
    Returns:
        是否成功
    """
    if not best_result:
        logging.getLogger(__name__).error("没有找到最优参数结果")
        return False
    
    if not YAML_AVAILABLE:
        logging.getLogger(__name__).error("PyYAML 未安装，请运行: pip install pyyaml")
        return False
    
    if config_path is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_path = os.path.join(base_dir, 'config', 'settings.yaml')
    
    if not os.path.exists(config_path):
        logging.getLogger(__name__).error(f"配置文件不存在: {config_path}")
        return False
    
    default_mapping = {
        'short_period': 'fast',
        'long_period': 'slow',
    }
    param_mapping = param_mapping or default_mapping
    
    logger = logging.getLogger(__name__)
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        strategies = config.get('strategies', [])
        if not strategies:
            logger.error("配置文件中没有 strategies 配置")
            return False
        
        target_strategy_idx = None
        best_params = best_result.params
        best_strategy_name = best_result.strategy_name
        
        if strategy_name:
            for i, s in enumerate(strategies):
                if s.get('name') == strategy_name:
                    target_strategy_idx = i
                    break
            if target_strategy_idx is None:
                logger.error(f"未找到名为 '{strategy_name}' 的策略配置")
                return False
        else:
            for i, s in enumerate(strategies):
                if s.get('class') in best_strategy_name or best_strategy_name in s.get('class', ''):
                    target_strategy_idx = i
                    break
            if target_strategy_idx is None:
                logger.warning(f"未找到匹配的策略，将使用第一个策略")
                target_strategy_idx = 0
        
        target_strategy = strategies[target_strategy_idx]
        current_params = target_strategy.get('params', {})
        
        logger.info(f"当前策略参数: {current_params}")
        logger.info(f"最优参数: {best_params}")
        
        updated = False
        for source_param, target_param in param_mapping.items():
            if source_param in best_params and target_param in current_params:
                old_value = current_params[target_param]
                new_value = best_params[source_param]
                if old_value != new_value:
                    current_params[target_param] = new_value
                    logger.info(f"更新参数: {target_param}: {old_value} -> {new_value}")
                    updated = True
        
        for param_name, param_value in best_params.items():
            if param_name not in param_mapping and param_name in current_params:
                old_value = current_params[param_name]
                if old_value != param_value:
                    current_params[param_name] = param_value
                    logger.info(f"更新参数: {param_name}: {old_value} -> {param_value}")
                    updated = True
        
        if updated:
            target_strategy['params'] = current_params
            strategies[target_strategy_idx] = target_strategy
            config['strategies'] = strategies
            
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
            
            logger.info(f"✓ 最优参数已同步到配置文件: {config_path}")
            print(f"\n{'='*80}")
            print("                    参数同步完成")
            print("="*80)
            print(f"目标策略: {target_strategy.get('name', 'Unknown')}")
            print(f"更新后的参数: {current_params}")
            print(f"配置文件: {config_path}")
        else:
            logger.info("配置文件参数已是最新，无需更新")
            print("\n配置文件参数已是最新，无需更新")
        
        return True
        
    except Exception as e:
        logger.error(f"参数同步失败: {e}", exc_info=True)
        return False


class MockKlineRow:
    """模拟 K 线数据行，支持 to_dict() 方法"""
    
    def __init__(self, data: Dict[str, Any]):
        self._data = data or {}
    
    def to_dict(self) -> Dict[str, Any]:
        return self._data
    
    def __getattr__(self, name):
        if name in self._data:
            return self._data[name]
        raise AttributeError(f"'MockKlineRow' object has no attribute '{name}'")


class MockKlineData:
    """模拟 K 线数据，支持类似 pandas 的接口"""
    
    def __init__(self, data: List[Dict[str, Any]] = None):
        self._data = data or []
        self._current_idx = -1
    
    def __len__(self):
        return len(self._data)
    
    def __getitem__(self, key):
        if isinstance(key, slice):
            return MockKlineData(self._data[key])
        elif isinstance(key, int):
            return MockKlineRow(self._data[key])
        return MockKlineData(self._data[key])
    
    @property
    def iloc(self):
        return self
    
    def to_dict(self):
        if self._current_idx >= 0 and self._current_idx < len(self._data):
            return self._data[self._current_idx]
        return self._data[-1] if self._data else {}


class MockTqSim:
    """模拟 TqSim 账户"""
    
    def __init__(self, init_balance: float = 1000000.0):
        self.init_balance = init_balance
        self._balance = init_balance
        self._positions = {}
        self._trades = []
        self._total_close_profit = 0.0
        
        self.account = type('obj', (object,), {
            'balance': init_balance,
            'available': init_balance,
            'frozen': 0.0,
            'margin': 0.0,
            'close_profit': 0.0,
            'position_profit': 0.0,
        })()
        
        self._commissions = {}
    
    def set_commission(self, symbol: str, commission: float):
        self._commissions[symbol] = commission
    
    def get_commission(self, symbol: str) -> float:
        return self._commissions.get(symbol, 5.0)
    
    def _get_or_create_position(self, symbol: str) -> Dict[str, Any]:
        if symbol not in self._positions:
            self._positions[symbol] = {
                'long': 0,
                'short': 0,
                'long_open_price': 0.0,
                'short_open_price': 0.0,
                'long_open_prices': [],
                'short_open_prices': [],
            }
        return self._positions[symbol]


class MockTqApi:
    """
    模拟 TqApi，完全不依赖天勤服务器
    使用本地生成的模拟数据进行回测
    """
    
    def __init__(
        self,
        account: MockTqSim = None,
        start_dt: date = None,
        end_dt: date = None,
        init_balance: float = 1000000.0,
        logger: logging.Logger = None,
        slippage_min: int = 1,
        slippage_max: int = 3,
        use_random_slippage: bool = False,
    ):
        self._closed = False
        self._account = account or MockTqSim(init_balance=init_balance)
        self._start_dt = start_dt or date(2024, 1, 1)
        self._end_dt = end_dt or date(2024, 1, 31)
        self._logger = logger or logging.getLogger(__name__)
        
        self._klines = {}
        self._current_kline_idx = {}
        self._quotes = {}
        
        self._cycle = 0
        self._max_cycles = self._calculate_max_cycles()
        
        self._equity = init_balance
        self._margin_used = 0.0
        self._available = init_balance
        self._float_profit = 0.0
        
        self._slippage_min = slippage_min
        self._slippage_max = slippage_max
        self._use_random_slippage = use_random_slippage
        self._total_slippage_cost = 0.0
        self._slippage_applied = []
        
        self._logger.info(f"✓ 模拟数据模式已启动")
        self._logger.info(f"  回测区间: {self._start_dt} 至 {self._end_dt}")
        self._logger.info(f"  初始资金: {init_balance:,.0f}")
        self._logger.info(f"  总周期数: {self._max_cycles}")
        if self._use_random_slippage:
            self._logger.info(f"  随机滑点: 开启 (范围: {slippage_min}-{slippage_max} ticks)")
        else:
            self._logger.info(f"  固定滑点: 关闭")
    
    def _calculate_max_cycles(self) -> int:
        """计算总回测周期数"""
        days = (self._end_dt - self._start_dt).days
        return max(days * 48, 100)
    
    @property
    def is_closed(self) -> bool:
        return self._closed
    
    def close(self):
        """关闭模拟 API"""
        if not self._closed:
            self._closed = True
            self._logger.debug("MockTqApi 已关闭")
    
    def get_kline_serial(self, symbol: str, duration_seconds: int) -> MockKlineData:
        """获取模拟 K 线数据"""
        if symbol not in self._klines:
            kline_count = max(self._max_cycles + 200, 500)
            self._klines[symbol] = self._generate_mock_klines(
                symbol=symbol,
                duration_seconds=duration_seconds,
                count=kline_count,
            )
            self._current_kline_idx[symbol] = 0
        
        return self._klines[symbol]
    
    def get_quote(self, symbol: str):
        """获取模拟行情数据 - 每次返回当前实际价格"""
        return self._generate_mock_quote(symbol)
    
    def _get_current_price(self, symbol: str) -> float:
        """获取当前 K 线的实际收盘价 - 使用最新的 K 线数据而非 _current_kline_idx"""
        if symbol in self._klines:
            kline_data = self._klines[symbol]
            if len(kline_data._data) > 0:
                latest_idx = len(kline_data._data) - 1
                return kline_data._data[latest_idx].get('close', self._get_base_price(symbol))
        
        return self._get_base_price(symbol)
    
    def get_account(self) -> Dict[str, Any]:
        """获取模拟账户信息"""
        return {
            'balance': self._equity - self._float_profit,
            'equity': self._equity,
            'margin': self._margin_used,
            'available': self._available,
            'float_profit': self._float_profit,
        }
    
    @property
    def trades(self) -> List[Dict[str, Any]]:
        """获取交易记录"""
        return self._account._trades
    
    def get_position(self) -> Dict[str, Any]:
        """获取模拟持仓数据"""
        positions = {}
        for symbol, pos in self._account._positions.items():
            if pos['long'] > 0 or pos['short'] > 0:
                current_price = self._get_current_price(symbol)
                positions[symbol] = {
                    'buy_volume': pos['long'],
                    'sell_volume': pos['short'],
                    'buy_margin': 0.0,
                    'sell_margin': 0.0,
                    'buy_open_price': pos.get('long_open_price', 0.0),
                    'sell_open_price': pos.get('short_open_price', 0.0),
                    'last_price': current_price,
                    'float_profit': 0.0,
                }
        return positions
    
    def _generate_mock_quote(self, symbol: str) -> Any:
        """生成模拟行情数据 - 使用当前实际价格"""
        current_price = self._get_current_price(symbol)
        volatility = current_price * 0.01
        
        quote_data = {
            'last_price': current_price,
            'bid_price1': current_price - volatility * 0.1,
            'ask_price1': current_price + volatility * 0.1,
            'bid_volume1': 100,
            'ask_volume1': 100,
            'high': current_price * 1.02,
            'low': current_price * 0.98,
            'open': current_price,
            'pre_close': current_price * 0.995,
            'volume': 10000,
            'amount': current_price * 10000,
            'open_interest': 50000,
            'underlying_symbol': symbol,
        }
        
        return type('obj', (object,), quote_data)()
    
    def _get_base_price(self, symbol: str) -> float:
        """获取合约基准价格"""
        price_map = {
            'SHFE.rb': 4000.0,
            'SHFE.hc': 4200.0,
            'DCE.i': 900.0,
            'CZCE.CF': 15000.0,
            'SHFE.cu': 70000.0,
            'SHFE.au': 450.0,
        }
        
        for prefix, price in price_map.items():
            if symbol.startswith(prefix):
                return price
        
        return 4000.0
    
    def _generate_mock_klines(
        self,
        symbol: str,
        duration_seconds: int,
        count: int = 200,
    ) -> MockKlineData:
        """生成模拟 K 线数据 - 增加波动率和趋势，确保能产生交易信号"""
        import random
        
        seed_val = hash(symbol) % 10000 + int(self._start_dt.year * 10000 + self._start_dt.month * 100 + self._start_dt.day)
        random.seed(seed_val)
        
        base_price = self._get_base_price(symbol)
        volatility = base_price * 0.03
        
        klines = []
        current_price = base_price * 0.95
        
        trend_direction = 1
        trend_counter = 0
        trend_duration = random.randint(20, 50)
        
        for i in range(count):
            trend_counter += 1
            if trend_counter >= trend_duration:
                trend_direction *= -1
                trend_counter = 0
                trend_duration = random.randint(20, 50)
            
            trend_component = trend_direction * random.uniform(0.3, 0.8)
            noise_component = random.uniform(-0.4, 0.4)
            price_change = (trend_component + noise_component) * volatility
            
            current_price = max(base_price * 0.8, min(base_price * 1.2, current_price + price_change))
            
            open_price = current_price
            bar_volatility = volatility * random.uniform(0.3, 0.7)
            high_price = current_price + bar_volatility * random.uniform(0.5, 1.5)
            low_price = current_price - bar_volatility * random.uniform(0.5, 1.5)
            close_price = current_price + random.uniform(-bar_volatility * 0.3, bar_volatility * 0.3)
            
            high_price = max(high_price, low_price, close_price, open_price)
            low_price = min(high_price, low_price, close_price, open_price)
            
            kline = {
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'volume': random.randint(500, 5000),
                'open_oi': random.randint(10000, 50000),
                'close_oi': random.randint(10000, 50000),
                'datetime': self._cycle + i,
                'cycle': self._cycle + i,
            }
            klines.append(kline)
        
        return MockKlineData(klines)
    
    def _get_price_tick(self, symbol: str) -> float:
        """获取合约最小变动价位"""
        tick_map = {
            'SHFE.rb': 1.0,
            'SHFE.hc': 1.0,
            'DCE.i': 0.5,
            'CZCE.CF': 5.0,
            'SHFE.cu': 10.0,
            'SHFE.au': 0.02,
            'SHFE.ag': 1.0,
            'SHFE.ni': 10.0,
            'SHFE.sn': 10.0,
            'DCE.m': 1.0,
            'DCE.y': 2.0,
            'DCE.p': 2.0,
            'DCE.c': 1.0,
            'CZCE.SR': 1.0,
            'CZCE.FG': 1.0,
            'CZCE.MA': 1.0,
        }
        
        for prefix, tick in tick_map.items():
            if symbol.startswith(prefix):
                return tick
        
        return 1.0
    
    def insert_order(self, quote_or_symbol, direction: str, offset: str, volume: int, limit_price: float = 0):
        """
        模拟下单，兼容 TqSdk 的接口
        支持传入 quote 对象（有 underlying_symbol 属性）或直接传入 symbol 字符串
        使用实际行情价格，记录开仓价格，平仓时计算盈亏
        支持随机滑点模拟
        """
        import time
        import random
        
        if hasattr(quote_or_symbol, 'underlying_symbol'):
            symbol = quote_or_symbol.underlying_symbol
        elif hasattr(quote_or_symbol, 'last_price'):
            if hasattr(quote_or_symbol, '__dict__') and 'contract' in quote_or_symbol.__dict__:
                symbol = quote_or_symbol.contract
            else:
                symbol = None
                for attr in dir(quote_or_symbol):
                    if 'symbol' in attr.lower() or 'contract' in attr.lower():
                        val = getattr(quote_or_symbol, attr, None)
                        if isinstance(val, str) and len(val) > 0:
                            symbol = val
                            break
                if symbol is None:
                    symbol = "SHFE.rb2410"
        elif isinstance(quote_or_symbol, str):
            symbol = quote_or_symbol
        else:
            symbol = "SHFE.rb2410"
        
        order_id = f"mock_order_{self._cycle}_{len(self._account._trades)}"
        
        if limit_price > 0:
            trade_price = limit_price
        elif hasattr(quote_or_symbol, 'last_price') and quote_or_symbol.last_price > 0:
            trade_price = quote_or_symbol.last_price
        else:
            trade_price = self._get_current_price(symbol)
        
        price_tick = self._get_price_tick(symbol)
        slippage_ticks = 0
        slippage_cost = 0.0
        original_price = trade_price
        
        if self._use_random_slippage:
            slippage_ticks = random.randint(self._slippage_min, self._slippage_max)
            
            if direction == 'BUY':
                trade_price = original_price + slippage_ticks * price_tick
                slippage_cost = slippage_ticks * price_tick * volume * 10
            elif direction == 'SELL':
                trade_price = original_price - slippage_ticks * price_tick
                slippage_cost = slippage_ticks * price_tick * volume * 10
            
            self._total_slippage_cost += slippage_cost
            self._slippage_applied.append({
                'cycle': self._cycle,
                'symbol': symbol,
                'direction': direction,
                'offset': offset,
                'volume': volume,
                'original_price': original_price,
                'slippage_price': trade_price,
                'slippage_ticks': slippage_ticks,
                'slippage_cost': slippage_cost,
                'price_tick': price_tick,
            })
        
        commission = self._account.get_commission(symbol)
        
        position = self._account._get_or_create_position(symbol)
        
        profit_loss = 0.0
        
        if direction == 'BUY' and offset == 'OPEN':
            position['long'] += volume
            position['long_open_prices'].append(trade_price)
            position['long_open_price'] = sum(position['long_open_prices']) / len(position['long_open_prices'])
            
        elif direction == 'SELL' and offset == 'OPEN':
            position['short'] += volume
            position['short_open_prices'].append(trade_price)
            position['short_open_price'] = sum(position['short_open_prices']) / len(position['short_open_prices'])
            
        elif direction == 'SELL' and offset == 'CLOSE':
            if position['long'] > 0:
                close_volume = min(volume, position['long'])
                if position['long_open_prices']:
                    avg_open_price = sum(position['long_open_prices'][-close_volume:]) / close_volume
                    profit_loss = (trade_price - avg_open_price) * close_volume * 10
                    
                    for _ in range(close_volume):
                        if position['long_open_prices']:
                            position['long_open_prices'].pop(0)
                    
                    position['long'] -= close_volume
                    if position['long_open_prices']:
                        position['long_open_price'] = sum(position['long_open_prices']) / len(position['long_open_prices'])
                    else:
                        position['long_open_price'] = 0.0
                    
                    self._account._total_close_profit += profit_loss
                    self._equity += profit_loss
                
        elif direction == 'BUY' and offset == 'CLOSE':
            if position['short'] > 0:
                close_volume = min(volume, position['short'])
                if position['short_open_prices']:
                    avg_open_price = sum(position['short_open_prices'][-close_volume:]) / close_volume
                    profit_loss = (avg_open_price - trade_price) * close_volume * 10
                    
                    for _ in range(close_volume):
                        if position['short_open_prices']:
                            position['short_open_prices'].pop(0)
                    
                    position['short'] -= close_volume
                    if position['short_open_prices']:
                        position['short_open_price'] = sum(position['short_open_prices']) / len(position['short_open_prices'])
                    else:
                        position['short_open_price'] = 0.0
                    
                    self._account._total_close_profit += profit_loss
                    self._equity += profit_loss
        
        self._equity -= commission * volume
        self._equity -= slippage_cost
        
        trade = {
            'order_id': order_id,
            'symbol': symbol,
            'direction': direction,
            'offset': offset,
            'volume': volume,
            'price': trade_price,
            'original_price': original_price,
            'slippage_ticks': slippage_ticks,
            'slippage_cost': slippage_cost,
            'profit_loss': profit_loss,
            'commission': commission * volume,
            'cycle': self._cycle,
            'timestamp': time.time(),
        }
        self._account._trades.append(trade)
        
        self._account.account.close_profit = self._account._total_close_profit
        self._account.account.balance = self._equity
        self._account.account.available = self._equity
        
        if self._use_random_slippage:
            self._logger.debug(f"模拟订单: {direction} {offset} {volume}手 {symbol} @ 原价={original_price:.2f}, 滑点后={trade_price:.2f}, 滑点={slippage_ticks}ticks, 滑点成本={slippage_cost:.2f}, 盈亏={profit_loss:.2f}")
        else:
            self._logger.debug(f"模拟订单: {direction} {offset} {volume}手 {symbol} @ {trade_price:.2f}, 盈亏={profit_loss:.2f}")
        
        return type('obj', (object,), {
            'order_id': order_id,
            'status': 'FINISHED',
            'volume_orign': volume,
            'volume_left': 0,
            'trade_price': trade_price,
            'original_price': original_price,
            'slippage_ticks': slippage_ticks,
            'slippage_cost': slippage_cost,
        })()
    
    def cancel_order(self, order):
        """模拟撤单"""
        pass
    
    def wait_update(self, timeout: float = 1.0) -> bool:
        """模拟等待更新，推进时间周期"""
        from tqsdk.exceptions import BacktestFinished
        
        if self._cycle >= self._max_cycles:
            raise BacktestFinished("模拟回测完成")
        
        self._cycle += 1
        
        for symbol in self._klines:
            if symbol in self._current_kline_idx:
                self._current_kline_idx[symbol] = min(
                    self._current_kline_idx[symbol] + 1,
                    len(self._klines[symbol]) - 1
                )
        
        return True


def _force_cleanup_asyncio(logger: logging.Logger = None):
    """
    强制清理 Asyncio 资源，避免 Task was destroyed 警告
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=DeprecationWarning)
            warnings.simplefilter("ignore", category=RuntimeWarning)
            
            loop = None
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                pass
            
            if loop is None:
                try:
                    loop = asyncio.get_event_loop()
                except Exception:
                    loop = None
            
            if loop and not loop.is_closed():
                pending = asyncio.all_tasks(loop) if hasattr(asyncio, 'all_tasks') else []
                
                if pending:
                    logger.debug(f"发现 {len(pending)} 个待处理的任务，尝试清理...")
                    
                    for task in pending:
                        if not task.done():
                            task.cancel()
                    
                    try:
                        if loop.is_running():
                            pass
                        else:
                            loop.run_until_complete(asyncio.sleep(0.1))
                    except Exception as e:
                        logger.debug(f"清理任务时出现异常 (可忽略): {e}")
                
                try:
                    if not loop.is_closed():
                        loop.close()
                        logger.debug("事件循环已关闭")
                except Exception as e:
                    logger.debug(f"关闭事件循环时出现异常 (可忽略): {e}")
            
            asyncio.set_event_loop(None)
            
    except Exception as e:
        logger.debug(f"Asyncio 清理时出现异常 (可忽略): {e}")


def _cleanup_api_and_loop(api: Any, logger: logging.Logger = None):
    """
    统一的 API 和事件循环清理函数
    在异常捕获块中调用，确保资源被正确释放
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    if api is not None:
        try:
            if hasattr(api, 'is_closed') and not api.is_closed:
                if hasattr(api, 'close'):
                    api.close()
                    logger.debug("API 已关闭")
        except Exception as e:
            logger.debug(f"关闭 API 时出现异常 (可忽略): {e}")
    
    _force_cleanup_asyncio(logger)


class MockDataBacktestRunner:
    """
    模拟数据回测运行器
    完全不依赖天勤服务器，使用本地生成的模拟数据进行回测
    """
    
    def __init__(
        self,
        init_balance: float = 1000000.0,
        start_dt: date = None,
        end_dt: date = None,
        logger: logging.Logger = None,
        slippage_min: int = 1,
        slippage_max: int = 3,
        use_random_slippage: bool = False,
    ):
        self.init_balance = init_balance
        self.start_dt = start_dt or date(2024, 1, 1)
        self.end_dt = end_dt or date(2024, 1, 31)
        self.logger = logger or logging.getLogger(__name__)
        
        self.slippage_min = slippage_min
        self.slippage_max = slippage_max
        self.use_random_slippage = use_random_slippage
        
        self.equity_curve: List[Dict[str, Any]] = []
        self.trades: List[Dict[str, Any]] = []
        self.risk_events: List[Dict[str, Any]] = []
        self.frozen_during_backtest = False
        self.frozen_reason = None
        
        self._current_equity = init_balance
        self._cycle = 0
    
    def run_backtest(
        self,
        strategy_class: Type[StrategyBase],
        strategy_params: Dict[str, Any],
        cost_config: CostConfig = None,
    ) -> Dict[str, Any]:
        """
        运行模拟数据回测
        
        Args:
            strategy_class: 策略类
            strategy_params: 策略参数
            cost_config: 成本配置
        
        Returns:
            回测结果字典
        """
        self.logger.info(f"开始模拟数据回测...")
        self.logger.info(f"策略: {strategy_class.__name__}")
        self.logger.info(f"参数: {strategy_params}")
        
        try:
            mock_api = MockTqApi(
                start_dt=self.start_dt,
                end_dt=self.end_dt,
                init_balance=self.init_balance,
                logger=self.logger,
                slippage_min=self.slippage_min,
                slippage_max=self.slippage_max,
                use_random_slippage=self.use_random_slippage,
            )
            
            mock_connector = type('MockConnector', (), {
                'get_api': lambda self: mock_api,
                'get_config': lambda self: {},
            })()
            
            strategy = strategy_class(connector=mock_connector, **strategy_params)
            strategy.initialize()
            strategy.subscribe()
            
            self.logger.info("策略初始化完成，开始回测循环...")
            
            max_cycles = mock_api._max_cycles
            cycles_ran = 0
            last_equity = self.init_balance
            
            while cycles_ran < max_cycles:
                try:
                    has_update = mock_api.wait_update(0.1)
                    if not has_update:
                        break
                    
                    strategy._on_update()
                    
                    cycles_ran += 1
                    
                    if cycles_ran % 10 == 0:
                        trade_count = len(mock_api._account._trades)
                        self._current_equity = self._calculate_equity(
                            mock_api._account._balance,
                            mock_api._account._trades,
                            strategy_params.get('contract', 'SHFE.rb2410'),
                        )
                        
                        equity_point = {
                            'cycle': cycles_ran,
                            'equity': self._current_equity,
                            'trades': trade_count,
                        }
                        self.equity_curve.append(equity_point)
                        
                        if cycles_ran % 50 == 0:
                            self.logger.debug(
                                f"周期 {cycles_ran}/{max_cycles}: "
                                f"权益={self._current_equity:,.0f}, "
                                f"交易次数={trade_count}"
                            )
                    
                    last_equity = self._current_equity
                    
                except KeyboardInterrupt:
                    self.logger.info("用户中断，停止回测")
                    break
            
            final_equity = self._calculate_equity(
                mock_api._account._balance,
                mock_api._account._trades,
                strategy_params.get('contract', 'SHFE.rb2410'),
            )
            self.trades = mock_api._account._trades
            
            self.logger.info(f"回测完成: 总周期={cycles_ran}, 最终权益={final_equity:,.0f}")
            
            _cleanup_api_and_loop(mock_api, self.logger)
            
            return {
                'status': 'completed',
                'initial_equity': self.init_balance,
                'final_equity': final_equity,
                'equity_curve': self.equity_curve,
                'trades': self.trades,
                'risk_events': self.risk_events,
                'frozen_during_backtest': self.frozen_during_backtest,
                'frozen_reason': self.frozen_reason,
                'error_message': None,
            }
            
        except Exception as e:
            self.logger.error(f"模拟数据回测失败: {e}", exc_info=True)
            _cleanup_api_and_loop(None, self.logger)
            return {
                'status': 'error',
                'initial_equity': self.init_balance,
                'final_equity': self.init_balance,
                'equity_curve': [],
                'trades': [],
                'risk_events': [],
                'frozen_during_backtest': False,
                'frozen_reason': None,
                'error_message': str(e),
            }
    
    def _calculate_equity(
        self,
        base_balance: float,
        trades: List[Dict[str, Any]],
        contract: str,
    ) -> float:
        """
        根据交易记录计算权益
        使用简化的 PnL 计算模型
        """
        if not trades:
            return base_balance
        
        total_pnl = 0.0
        total_commission = 0.0
        
        base_price = 4000.0
        if 'rb' in contract:
            base_price = 4000.0
        elif 'i' in contract:
            base_price = 900.0
        elif 'cu' in contract:
            base_price = 70000.0
        elif 'au' in contract:
            base_price = 450.0
        
        contract_multiplier = 10
        if 'au' in contract:
            contract_multiplier = 1000
        
        position = 0
        entry_price = 0.0
        
        for trade in trades:
            direction = trade.get('direction', 'BUY')
            offset = trade.get('offset', 'OPEN')
            volume = trade.get('volume', 1)
            price = trade.get('price', base_price)
            commission = trade.get('commission', 5.0)
            
            total_commission += commission
            
            if offset == 'OPEN':
                if direction == 'BUY':
                    if position == 0:
                        entry_price = price
                    else:
                        entry_price = (entry_price * abs(position) + price * volume) / (abs(position) + volume)
                    position += volume
                else:
                    if position == 0:
                        entry_price = price
                    else:
                        entry_price = (entry_price * abs(position) + price * volume) / (abs(position) + volume)
                    position -= volume
            
            elif offset == 'CLOSE':
                if direction == 'SELL' and position > 0:
                    pnl = (price - entry_price) * min(volume, position) * contract_multiplier
                    total_pnl += pnl
                    position -= min(volume, position)
                elif direction == 'BUY' and position < 0:
                    pnl = (entry_price - price) * min(volume, -position) * contract_multiplier
                    total_pnl += pnl
                    position += min(volume, -position)
            
            elif offset == 'CLOSETODAY':
                if direction == 'SELL' and position > 0:
                    pnl = (price - entry_price) * min(volume, position) * contract_multiplier
                    total_pnl += pnl
                    position -= min(volume, position)
                elif direction == 'BUY' and position < 0:
                    pnl = (entry_price - price) * min(volume, -position) * contract_multiplier
                    total_pnl += pnl
                    position += min(volume, -position)
        
        final_equity = base_balance + total_pnl - total_commission
        
        return max(0.0, final_equity)
