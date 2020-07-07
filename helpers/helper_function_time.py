import time

def time_decorator(fn):
    def wrapper_function(*args, **kwargs):
        start_time = time.time()

        # invoking the wrapped function and getting the return value.
        value = fn(*args, **kwargs)
        print("The function execution took:", time.time() - start_time, "seconds")

        # returning the value got after invoking the wrapped function
        return value

    return wrapper_function