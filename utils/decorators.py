import time
import logging

# Configure logger
logging.basicConfig(level=logging.DEBUG)


def measure_execution_time(func):
    """
    Measure the execution time of a function. To be used as a decorator

    :param func: The function where the execution time is to be measured
    :return: callable: The decorated function.

    Example Usage:
        @measure_execution_time
        def my_function():
            # Function Implementation Here
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()                   # Start Timer
        result = func(*args, **kwargs)             # Run function
        elapsed_time = time.time() - start_time    # End Timer
        logging.info(f"{func.__name__} took {elapsed_time * 1000:.2f} ms to execute")
        return result
    return wrapper
