import logging
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.connection import TqConnector
from core.manager import StrategyManager
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
    
    logger.info("=" * 60)
    logger.info("启动量化交易框架 (多策略模式)...")
    logger.info("=" * 60)
    
    connector = None
    manager = None
    
    try:
        with TqConnector() as connector:
            default_contract = connector.get_default_contract()
            env_mode = connector.get_env_mode()
            credentials_source = connector.get_credentials_source()
            local_credentials_path = connector.get_local_credentials_path()
            
            logger.info(f"环境模式: {env_mode}")
            logger.info(f"默认交易合约: {default_contract}")
            
            if credentials_source == "local_file":
                logger.info(f"凭据来源: 本地配置文件 ({local_credentials_path})")
            elif credentials_source == "environment":
                logger.info("凭据来源: 系统环境变量 (TQ_ACCOUNT / TQ_PASSWORD)")
            else:
                logger.info("凭据来源: 无 (使用匿名模式)")
            
            config = connector.get_config()
            
            logger.info("正在创建策略管理器...")
            manager = StrategyManager(connector=connector)
            
            logger.info("正在从配置加载策略...")
            manager.load_strategies_from_config(config)
            
            strategies = manager.get_all_strategies()
            if not strategies:
                logger.warning("未加载到任何策略，程序退出")
                return
            
            logger.info(f"已加载 {len(strategies)} 个策略:")
            for name, strategy in strategies.items():
                logger.info(f"  - {name}: {strategy.__class__.__name__}")
                if hasattr(strategy, 'contract'):
                    logger.info(f"    合约: {strategy.contract}")
                if hasattr(strategy, 'short_period') and hasattr(strategy, 'long_period'):
                    logger.info(f"    参数: 短期周期={strategy.short_period}, 长期周期={strategy.long_period}")
            
            logger.info("正在初始化所有策略...")
            manager.initialize()
            
            all_states = manager.get_all_states()
            logger.info("策略初始化完成，状态如下:")
            for name, state in all_states.items():
                logger.info(f"  - {name}:")
                logger.info(f"    合约: {state.get('contract', 'N/A')}")
                logger.info(f"    信号: {state.get('signal', SignalType.HOLD).value}")
                logger.info(f"    就绪: {state.get('is_ready', False)}")
            
            logger.info("=" * 60)
            logger.info("开始运行策略管理器...")
            logger.info("提示: 按 Ctrl+C 停止所有策略")
            logger.info("=" * 60)
            
            manager.run_all()
                
    except KeyboardInterrupt:
        logger.info("用户中断，停止策略管理器")
    except Exception as e:
        logger.error(f"策略管理器运行出错: {str(e)}", exc_info=True)
        sys.exit(1)
    finally:
        if manager is not None:
            manager.stop_all()
        logger.info("程序退出")


if __name__ == "__main__":
    main()
