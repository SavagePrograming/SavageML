from typing import List, Iterable

import numpy as np
import time


def get_sample_from_iterator(data_iterator):
    try:
        sample = data_iterator.__next__()
        return True, sample
    except StopIteration:
        return False, None


def batch_iterator(iterable, batch_size=1, batch_limit=None):
    batch_array = []
    batch_count = 0
    if batch_limit is not None and batch_count >= batch_limit:
        return batch_array
    for data in iterable:
        batch_array.append(data)
        batch_count += 1
        if batch_limit is not None and batch_count >= batch_limit:
            return batch_array
        if len(batch_array) >= batch_size:
            yield batch_array
            batch_array = []
    if len(batch_array) > 0:
        return batch_array


def merge_iterators(iterables: List[Iterable], iteration_limit=None):
    for index, samples in enumerate(zip(*iterables)):
        if iteration_limit is None or iteration_limit > index:
            step_list = [sample[0] for sample in samples]
            step_array = np.concatenate(step_list, axis=0)
            yield step_array
        else:
            break

def iter_steps(x: np.ndarray, iteration_limit=None):
    limit = min(iteration_limit, x.shape[1]) if iteration_limit else x.shape[1]
    for index in range(limit):
        yield x[:, index, :]


def batch_np_array(array: np.ndarray, batch_size=1):
    for batch_index in range(0, array.shape[0], batch_size):
        yield array[batch_index:min(batch_index + batch_size, array.shape[0])]


def batch_np_array_with_state(array: np.ndarray, state: List[np.ndarray], batch_size=1):
    for batch_index in range(0, array.shape[0], batch_size):
        yield array[batch_index:min(batch_index + batch_size, array.shape[0])], \
              [s[batch_index:min(batch_index + batch_size, s.shape[0]), ...] for s in state]


def binary_search(LIST, value, start_index=0, end_index=None):
    if end_index is None:
        end_index = len(LIST)
    pivot = (start_index + end_index) // 2
    if LIST[pivot] == value:
        return pivot
    elif LIST[pivot] > value:
        end_index = pivot
    else:
        start_index = pivot
    binary_search(LIST, value, start_index, end_index)


def wait_until(eval_function, timeout, period=0.25, *args, **kwargs):
    mustend = time.time() + timeout
    while time.time() < mustend:
        if eval_function(*args, **kwargs): return True
        time.sleep(period)
    return False
