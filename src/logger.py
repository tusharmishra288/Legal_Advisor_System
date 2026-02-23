import sys
import logging
import warnings
from loguru import logger
from .config import LOG_DIR

class InterceptHandler(logging.Handler):
    """Logs everything from standard logging to Loguru."""
    def emit(self, record):
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())

def setup_logging():
    # 1. Clear Loguru's default handler to avoid double-printing
    logging.root.handlers = []
    logger.remove()
    
    # 2. Console Handler (Filtered for clean output)
    # This keeps your terminal readable
    logger.add(
        sys.stderr, 
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>", 
        level="INFO"
    )
    
    # 3. Comprehensive File Handler (Captures EVERYTHING for debugging)
    logger.add(
        LOG_DIR / "legaladvisorsystem.log", 
        rotation="50 MB", 
        retention="10 days", 
        compression="zip",
        level="DEBUG",
        enqueue=True,
        backtrace=True,
        diagnose=True
    )

    # 4. UNIVERSAL INTERCEPT: Catch all standard logging from any library
    # We set level=0 (NOTSET) so it catches everything, then we filter inside Loguru
    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)

    # 5. SILENCE EXTERNAL NOISE: Explicitly set levels for noisy libraries
    # This stops the 'POST https://... 200 OK' and 'Generated queries' plain-text logs
    noisy_libraries = ["httpx", "httpcore", "qdrant_client", "openai", "psycopg", "urllib3", "nicegui"]
    

    for lib in noisy_libraries:
        lib_logger = logging.getLogger(lib)
        lib_logger.setLevel(logging.WARNING)
        lib_logger.handlers = [InterceptHandler()] # Force them to use our handler
        lib_logger.propagate = False

    # 6. WARNING INTERCEPT: Catch Qdrant/Neon/Postgres UserWarnings
    def _warning_to_loguru(message, category, filename, lineno, file=None, line=None):
        logger.warning(
            f"⚠️ {category.__name__} detected at {filename}:{lineno} -> {message}"
        )

    warnings.showwarning = _warning_to_loguru

    logger.success("🛡️  Universal Logger Active: Capturing all System, API, and Library logs.")
    return logger

log = setup_logging()