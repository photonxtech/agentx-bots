from tenacity import retry, stop_after_attempt, wait_exponential

from app.utils.logging import get_logger

logger = get_logger(__name__)


def with_retry(max_attempts: int = 3, base_delay: float = 1.0):
    def decorator(func):
        wrapped = retry(
            stop=stop_after_attempt(max_attempts),
            wait=wait_exponential(multiplier=base_delay, min=base_delay, max=base_delay * 10),
            reraise=True,
        )(func)
        return wrapped

    return decorator
