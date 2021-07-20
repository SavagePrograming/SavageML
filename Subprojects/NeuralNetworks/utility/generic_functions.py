def get_x_y_from_iterator(data_iterator):
    try:
        x_sample, y_sample = data_iterator.__next__()
        return True, x_sample, y_sample
    except StopIteration:
        return False, None, None
