import numpy as np

import tensorflow as tf

# ====================================================
# Helper functions
# ====================================================


def get_buckets_20(MAX_VAL):
    """
    creates discretization buckets of size 20
    """
    list_buckets = list(np.linspace(0, MAX_VAL, num=20))
    return list_buckets


def tf_if_null_return_zero(val):
    """
    > a trick to remove NANs post tf2.0
    > this function fills in nans to zeros - sometimes happens in embedding calcs.
    > this will clean the embedding inputs downstream
    """
    return tf.clip_by_value(val, -1e12, 1e12)


def get_arch_from_string(arch_string):
    q = arch_string.replace("]", "")
    q = q.replace("[", "")
    q = q.replace(" ", "")

    return [int(x) for x in q.split(",")]


def _is_chief(task_type, task_id):
    """Check for primary if multiworker training"""
    if task_type == "chief":
        results = "chief"
    else:
        results = None
    return results


# data loading and parsing
def full_parse(data):
    # used for interleave - takes tensors and returns a tf.dataset
    data = tf.data.TFRecordDataset(data)
    return data


# full_parse, get_train_strategy, _is_chief, get_arch_from_string, tf_if_null_return_zero, get_buckets_20, upload_blob
