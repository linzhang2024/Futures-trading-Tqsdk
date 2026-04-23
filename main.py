import logging
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.connection import TqConnector
from strategies.double_ma_strategy import DoubleMAStrategy, SignalType


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
    
    logger.info("启动量化交易框架...")
    
    try:
        with TqConnector() as connector:
            default_contract = connector.get_default_contract()
            
            logger.info(f"默认交易合约: {default_contract}")
            
            strategy = DoubleMAStrategy(
                connector=connector,
                short_period=5,
                long_period=10,
                contract=default_contract or "SHFE.rb2410",
                kline_duration=60,
                use_ema=False,
            )
            
            strategy.initialize()
            
            logger.info("双均线策略初始化完成")
            logger.info(f"短期均线周期: {strategy.short_period}")
            logger.info(f"长期均线周期: {strategy.long_period}")
            logger.info(f"交易合约: {strategy.contract}")
            
            logger.info("开始运行策略，等待行情数据...")
            
            signal_count = 0
            
            while True:
                api = strategy.get_api()
                if api is None:
                    logger.error("API 为 None，退出循环")
                    break
                
                api.wait_update()
                strategy._on_update()
                
                if strategy.is_ready():
                    ma_values = strategy.get_ma_values()
                    current_signal = strategy.get_signal()
                    
                    if current_signal != SignalType.HOLD and current_signal != strategy.prev_signal:
                        signal_count += 1
                        logger.info(f"信号 #{signal_count}: {current_signal.value}")
                        logger.info(f"  MA{strategy.short_period}: {ma_values[f'ma_{strategy.short_period}']:.2f}")
                        logger.info(f"  MA{strategy.long_period}: {ma_values[f'ma_{strategy.long_period}']:.2f}")
                
    except KeyboardInterrupt:
        logger.info("用户中断，停止策略")
    except Exception as e:
        logger.error(f"策略运行出错: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
