import os
import sys
import logging
import time
from datetime import datetime

base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, base_dir)

from core.connection import TqConnector
from core.realtime_runner import RealtimeRunner
from strategies.double_ma_strategy import DoubleMAStrategy, VectorizedMAStrategy


def setup_logging():
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        datefmt=date_format,
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    log_file = os.path.join(base_dir, 'logs', 'realtime.log')
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(log_format, datefmt=date_format))
    logging.getLogger().addHandler(file_handler)
    
    return logging.getLogger('RealtimeMain')


def main():
    logger = setup_logging()
    
    print("\n" + "=" * 80)
    print("                🚀 期货实时交易系统启动")
    print("=" * 80)
    print(f"启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    print("\n【系统说明】")
    print("  1. 本系统将连接天勤 TqSim 模拟账户")
    print("  2. 支持实时行情驱动策略运行")
    print("  3. 实现限价追单策略（30秒未成交自动撤单并重试）")
    print("  4. 本地持仓与交易所持仓每分钟自动对齐")
    print("  5. 可配置 Webhook 通知（微信/飞书）")
    print("\n【运行模式】")
    print("  - 环境: TqSim 本地模拟")
    print("  - 策略: 双均线策略 (VectorizedMAStrategy)")
    print("  - 合约: SHFE.rb2410 (螺纹钢)")
    print("")
    print("提示: 按 Ctrl+C 停止系统")
    print("=" * 80 + "\n")
    
    logger.info("正在初始化实时交易系统...")
    
    try:
        connector = TqConnector()
        logger.info("正在连接天勤 API...")
        api = connector.connect()
        logger.info("天勤 API 连接成功")
        
        runner = RealtimeRunner(connector=connector)
        
        strategy = VectorizedMAStrategy(
            connector=connector,
            short_period=5,
            long_period=20,
            contract="SHFE.rb2410",
            kline_duration=60,
            use_ema=False,
            rsi_period=14,
            rsi_threshold=50.0,
            use_rsi_filter=True,
            take_profit_ratio=0.02,
            stop_loss_ratio=0.01,
            initial_data_days=3,
            debug_logging=True,
        )
        
        runner.register_strategy("MA_Strategy", strategy)
        
        logger.info("系统初始化完成，开始运行实时交易...")
        
        runner.run()
        
    except KeyboardInterrupt:
        logger.info("用户中断，正在停止系统...")
        print("\n用户中断，正在停止系统...")
    except Exception as e:
        logger.error(f"系统运行出错: {e}", exc_info=True)
        print(f"\n系统运行出错: {e}")
    finally:
        try:
            connector.disconnect()
            logger.info("天勤 API 连接已断开")
        except:
            pass
        
        print("\n" + "=" * 80)
        print("                实时交易系统已停止")
        print("=" * 80)
        logger.info("实时交易系统已停止")


if __name__ == "__main__":
    main()
