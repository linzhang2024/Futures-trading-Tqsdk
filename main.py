import logging
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.connection import TqConnector
from core.manager import StrategyManager, StrategyHealthStatus
from strategies.base_strategy import SignalType


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )


def main():
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 80)
    logger.info("🚀 启动量化交易框架 (多策略并行模式)")
    logger.info("=" * 80)
    
    connector = None
    manager = None
    
    try:
        with TqConnector() as connector:
            default_contract = connector.get_default_contract()
            env_mode = connector.get_env_mode()
            credentials_source = connector.get_credentials_source()
            local_credentials_path = connector.get_local_credentials_path()
            
            logger.info(f"📋 环境模式: {env_mode}")
            logger.info(f"📋 默认交易合约: {default_contract}")
            
            if credentials_source == "local_file":
                logger.info(f"🔐 凭据来源: 本地配置文件 ({local_credentials_path})")
            elif credentials_source == "environment":
                logger.info("🔐 凭据来源: 系统环境变量 (TQ_ACCOUNT / TQ_PASSWORD)")
            else:
                logger.info("🔐 凭据来源: 无 (使用匿名模式)")
            
            config = connector.get_config()
            
            logger.info("")
            logger.info("📦 正在创建策略管理器...")
            manager = StrategyManager(connector=connector)
            
            logger.info("⚙️  正在应用管理器配置...")
            manager.configure_from_dict(config)
            
            logger.info("")
            logger.info("📋 正在从配置加载策略...")
            manager.load_strategies_from_config(config)
            
            strategies = manager.get_all_strategies()
            if not strategies:
                logger.warning("⚠️  未加载到任何策略，程序退出")
                return
            
            logger.info(f"✅ 已加载 {len(strategies)} 个策略:")
            for name, strategy in strategies.items():
                logger.info(f"")
                logger.info(f"  【{name}】")
                logger.info(f"    类名: {strategy.__class__.__name__}")
                if hasattr(strategy, 'contract'):
                    logger.info(f"    合约: {strategy.contract}")
                if hasattr(strategy, 'short_period') and hasattr(strategy, 'long_period'):
                    logger.info(f"    参数: MA{strategy.short_period} / MA{strategy.long_period}")
                if hasattr(strategy, 'kline_duration'):
                    logger.info(f"    K线周期: {strategy.kline_duration} 秒")
                if hasattr(strategy, 'use_ema'):
                    logger.info(f"    均线类型: {'EMA' if strategy.use_ema else 'SMA'}")
            
            logger.info("")
            logger.info("🔧 正在初始化所有策略...")
            
            manager_config = config.get('manager', {})
            load_saved_states = manager_config.get('load_saved_states', True)
            
            manager.initialize(load_saved_states=load_saved_states)
            
            all_states = manager.get_all_states()
            all_health = manager.get_all_health_status()
            
            logger.info("")
            logger.info("=" * 80)
            logger.info("📊 策略初始化完成，状态汇总:")
            logger.info("-" * 80)
            
            for name, state in all_states.items():
                health = all_health.get(name, {})
                health_status = health.get('status', 'UNKNOWN')
                
                is_ready = state.get('is_ready', False)
                signal = state.get('signal', SignalType.HOLD)
                
                status_icon = "🟢" if health_status == StrategyHealthStatus.HEALTHY.value else "🟡" if health_status == StrategyHealthStatus.DEGRADED.value else "🔴"
                ready_icon = "✅" if is_ready else "⏳"
                
                logger.info(f"")
                logger.info(f"  {status_icon} {name}")
                logger.info(f"    健康状态: {health_status}")
                logger.info(f"    数据就绪: {ready_icon} {'是' if is_ready else '否'}")
                
                if is_ready:
                    ma_values = state.get('ma_values', {})
                    short_period = state.get('short_period', 'N/A')
                    long_period = state.get('long_period', 'N/A')
                    
                    short_ma = ma_values.get(f'ma_{short_period}') if ma_values else None
                    long_ma = ma_values.get(f'ma_{long_period}') if ma_values else None
                    
                    if short_ma is not None:
                        logger.info(f"    MA{short_period}: {short_ma:.2f}")
                    if long_ma is not None:
                        logger.info(f"    MA{long_period}: {long_ma:.2f}")
                    
                    signal_color = "🟢" if signal == SignalType.BUY else "🔴" if signal == SignalType.SELL else "⚪"
                    logger.info(f"    当前信号: {signal_color} {signal.value}")
                else:
                    if hasattr(manager.get_strategy(name), '_all_prices'):
                        strategy = manager.get_strategy(name)
                        price_count = len(strategy._all_prices)
                        required = state.get('long_period', 0)
                        logger.info(f"    数据进度: {price_count}/{required} 条价格数据")
            
            logger.info("")
            logger.info("=" * 80)
            logger.info("🎯 策略管理器开始运行")
            logger.info("")
            logger.info("📋 运行说明:")
            logger.info("   - 每 30 秒自动打印一次策略状态报告")
            logger.info("   - 每次报告后自动保存所有策略状态")
            logger.info("   - 状态文件存储在: data/states/ 目录")
            logger.info("   - 按 Ctrl+C 优雅停止所有策略")
            logger.info("=" * 80)
            logger.info("")
            
            manager.run_all(load_saved_states=False)
                
    except KeyboardInterrupt:
        logger.info("")
        logger.info("=" * 80)
        logger.info("👋 用户中断，正在停止策略管理器...")
        logger.info("=" * 80)
    except Exception as e:
        logger.error(f"💥 策略管理器运行出错: {str(e)}", exc_info=True)
        sys.exit(1)
    finally:
        if manager is not None:
            logger.info("")
            logger.info("💾 正在保存所有策略状态...")
            saved_count = manager.save_all_states()
            logger.info(f"✅ 已保存 {saved_count} 个策略的状态")
            
            logger.info("")
            logger.info("🛑 正在停止所有策略...")
            manager.stop_all()
        
        logger.info("")
        logger.info("=" * 80)
        logger.info("👋 程序退出")
        logger.info("=" * 80)


if __name__ == "__main__":
    main()
