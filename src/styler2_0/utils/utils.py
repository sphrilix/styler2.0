from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")


class TooManyTriesException(Exception):
    """
    If a function is decorated with @retry and does not pass a successful run within
    the specified amount this exception is thrown.
    """


def retry(
    n: int,
    default: T = None,
    exceptions: type[Exception] | tuple[type[Exception], ...] = Exception,
) -> Callable[..., T]:
    """
    Retry decorator, that retries the decorated function n times. If within those n
    steps no successful run occurred it raises a TooManyTriesException or if a
    default is given that will be returned.
    Also, the decorator only catches the exceptions from the given ones.
    :param n: The given tries.
    :param default: The given default.
    :param exceptions: The exceptions to be caught.
    :return: The decorated function.
    """

    def _retry_decorator(func: Callable[..., T]) -> Callable[..., T]:
        def _retry_decorator_wrapper(*args: ..., **kwargs: ...) -> T:
            thrown_exceptions: list[str] = []
            for _ in range(n):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    thrown_exceptions.append(type(e).__name__)
            if default:
                return default
            raise TooManyTriesException(
                f"Within {n} tries no successful run of "
                f"{getattr(func, '__name__', repr(callable))} "
                f"(Exceptions raised: [{', '.join(thrown_exceptions)}])."
            )

        return _retry_decorator_wrapper

    return _retry_decorator
