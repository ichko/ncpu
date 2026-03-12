def normalize_neg1_to_1(x):
    """Normalize pixel values from [0, 255] to [-1, 1]."""
    return x / 128.0 - 1.0
