import logging
import time
import functools
from sqlalchemy.exc import OperationalError, TimeoutError as SATimeoutError

logger = logging.getLogger(__name__)

RETRYABLE_EXCEPTIONS = (
    OperationalError,
    SATimeoutError,
    ConnectionError,
    TimeoutError,
)

def retry_on_db_error(max_attempts=3, base_delay=0.5):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except RETRYABLE_EXCEPTIONS as e:
                    last_exception = e
                    if attempt < max_attempts:
                        delay = base_delay * (2 ** (attempt - 1))
                        logger.warning(
                            "Retry %d/%d for %s after error: %s",
                            attempt, max_attempts, func.__name__, e
                        )
                        time.sleep(delay)
                    else:
                        logger.error(
                            "All %d attempts failed for %s: %s",
                            max_attempts, func.__name__, e
                        )
            raise last_exception
        return wrapper
    return decorator
