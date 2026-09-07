from loguru import logger
from src.config import LOGS_DIR
from datetime import datetime



timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
LOGS_DIR.parent.mkdir(exist_ok=True)
logger.remove()



# Console logging
logger.add(
    sink=lambda msg: print(msg, end=""),
    level="INFO",
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
           "<level>{level: <8}</level> | "
           "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
           "<level>{message}</level>",
    colorize=True,
)

# File logging
logger.add(
    LOGS_DIR / f"{timestamp}.log",
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} | "
           "{level: <8} | "
           "{name}:{function}:{line} | "
           "{message}",
    rotation="10 MB",
    retention="30 days",
    compression="zip",
    enqueue=True,
    backtrace=True,
    diagnose=True,
)



 