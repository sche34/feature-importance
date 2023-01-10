import numpy as np
import functions.data_sampling as data_sampling

def get_feature_groups(config):
    x, _, __ = data_sampling.get_full_classes(config["classes"])
    superpixel_width = config["superpixel_width"]

    if superpixel_width <= 0 or type(superpixel_width) != int:
        raise ValueError("superpixel_width should be a positive integer")
    if x[0].shape[0] % superpixel_width != 0:
        raise ValueError("superpixel_width is not a divisor of the image height")
    pixels_per_axis = x[0].shape[0] // superpixel_width
    group_numbers = np.arange(pixels_per_axis ** 2)

    feature_groups = []
    for num in group_numbers:
        row_num = num // pixels_per_axis
        col_num = num % pixels_per_axis
        group = np.zeros_like(x[0])
        group[row_num * superpixel_width:(row_num + 1) * superpixel_width,col_num * superpixel_width:(col_num + 1) * superpixel_width] = 1
        group = group.astype(bool)

        if len(feature_groups) == 0:
            feature_groups = np.array([group])
        else:
            feature_groups = np.concatenate((feature_groups, [group]), axis=0)
    return feature_groups


