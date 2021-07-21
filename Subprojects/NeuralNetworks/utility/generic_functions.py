def get_sample_from_iterator(data_iterator):
    try:
        sample = data_iterator.__next__()
        return True, sample
    except StopIteration:
        return False, None
