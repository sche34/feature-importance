import os
import numpy as np


def get_full_classes(class_numbers):
    data_path = r'Data'
    file_paths, y, x = [], [], []

    for index, folder in enumerate(os.listdir(data_path)):
        if index in class_numbers:
            file_list = os.listdir(os.path.join(data_path, folder))
            for file in file_list:
                img_path = os.path.join(data_path, folder, file)
                img = np.load(img_path)
                if img.shape[0] != img.shape[1]:
                    raise ValueError(f"image has different height and width: {img_path}")
                x.append(img)
                y.append(index)
                file_paths.append(os.path.join(data_path, folder, file))
    x, y = np.array(x), np.array(y)

    if x.shape[0] != y.shape[0]:
        raise ValueError("images and labels have different lengths")
    elif y.ndim != 1:
        raise ValueError("y should be 1 dimensional")

    return x, y, file_paths


def sample_classes(class_numbers, n_data_samples):
    data_path = r'Data'
    file_paths, y, x = [], [],[]
    count_dict = {}

    for index, folder in enumerate(os.listdir(data_path)):
        count_dict[index] = 0
        file_list = os.listdir(os.path.join(data_path, folder))

        if index in class_numbers:
            for file in file_list:
                if count_dict[index] == n_data_samples:
                    break

                img_path = os.path.join(data_path, folder, file)
                img = np.load(img_path)
                if img.shape[0] != img.shape[1]:
                    raise ValueError(f"image has different height and width: {img_path}")
                x.append(img)
                y.append(index)
                file_paths.append(os.path.join(data_path, folder, file))
                count_dict[index] += 1
    x, y = np.array(x), np.array(y)

    if x.shape[0] != y.shape[0]:
        raise ValueError("images and labels have different lengths")
    elif y.ndim != 1:
        raise ValueError("y should be 1 dimensional")
    return x, y, file_paths