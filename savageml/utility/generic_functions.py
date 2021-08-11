import numpy as np


def get_sample_from_iterator(data_iterator):
    try:
        sample = data_iterator.__next__()
        return True, sample
    except StopIteration:
        return False, None


def batch_iterator(iterable, batch_size=1):
    batch_array = []
    for data in iterable:
        batch_array.append(data)
        if len(batch_array) >= batch_size:
            yield batch_array
            batch_array = []
    if len(batch_array) > 0:
        return batch_array


def batch_np_array(array: np.ndarray, batch_size=1):
    for batch_index in range(0, array.shape[0], batch_size):
        yield array[batch_index:min(batch_index + batch_size, array.shape[0])]


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
