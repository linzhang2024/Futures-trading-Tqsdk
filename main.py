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
    
    connector = None
    strategy = None
    
    try:
        with TqConnector() as connector:
            default_contract = connector.get_default_contract()
            env_mode = connector.get_env_mode()
            
            logger.info(f"环境模式: {env_mode}")
            logger.info(f"默认交易合约: {default_contract}")
            
            strategy = DoubleMAStrategy(
                connector=connector,
                short_period=5,
                long_period=10,
                contract=default_contract or "SHFE.rb2410",
                kline_duration=60,
                use_ema=False,
            )
            
            logger.info("正在初始化策略...")
            strategy.initialize()
            
            if strategy.klines is None:
                raise RuntimeError(f"合约 {strategy.contract} 订阅失败，klines 为 None")
            
            logger.info("合约订阅成功！")
            logger.info(f"订阅合约: {strategy.contract}")
            logger.info(f"K线周期: {strategy.kline_duration} 秒")
            
            logger.info("双均线策略初始化完成")
            logger.info(f"短期均线周期: {strategy.short_period}")
            logger.info(f"长期均线周期: {strategy.long_period}")
            logger.info(f"均线类型: {'EMA' if strategy.use_ema else 'SMA'}")
            
            logger.info("开始运行策略，等待行情数据...")
            logger.info("提示: 需要收集至少 {} 个 K 线数据才能计算均线".format(strategy.long_period))
            
            signal_count = 0
            bar_count = 0
            
            while True:
                api = strategy.get_api()
                if api is None:
                    logger.error("API 为 None，退出循环")
                    break
                
                api.wait_update()
                strategy._on_update()
                
                bar_count += 1
                
                if strategy.is_ready():
                    ma_values = strategy.get_ma_values()
                    current_signal = strategy.get_signal()
                    
                    if current_signal != SignalType.HOLD and current_signal != strategy.prev_signal:
                        signal_count += 1
                        logger.info(f"=" * 50)
                        logger.info(f"信号 #{signal_count}: {current_signal.value}")
                        logger.info(f"  MA{strategy.short_period}: {ma_values[f'ma_{strategy.short_period}']:.2f}")
                        logger.info(f"  MA{strategy.long_period}: {ma_values[f'ma_{strategy.long_period}']:.2f}")
                        logger.info(f"=" * 50)
                else:
                    if bar_count % 10 == 0:
                        logger.debug(f"已收集 {len(strategy._all_prices)} 个价格数据，等待足够数据计算均线...")
                
    except KeyboardInterrupt:
        logger.info("用户中断，停止策略")
    except Exception as e:
        logger.error(f"策略运行出错: {str(e)}", exc_info=True)
        sys.exit(1)
    finally:
        if strategy is not None:
            strategy.stop()
        logger.info("程序退出")


if __name__ == "__main__":
    main()
