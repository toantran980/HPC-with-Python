'''
This file is where you migrate all the functions that you made in data_process1 and data_process2. 
'''
import numpy as np
import pandas as pd
import math
import csv
import time
from typeguard import typechecked


# -----------------------------
# From data_process1
# -----------------------------

@typechecked
def load_dataset(filename: str) -> list[list[float]]:
    with open(filename, 'r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)  # Skip the header row
        data = [[float(value) for value in row] for row in csv_reader]
    return data


@typechecked
def load_dataset_np(filename: str) -> np.ndarray:
    data = np.genfromtxt(filename, delimiter=',', skip_header=1)
    return data


@typechecked
def normalize_array(arr: list[list[float]], out_file: str | None = None) -> int:
    if not arr or not arr[0]:
        return 0  # empty input

    col_count = len(arr[0])
    skip = 1  # first column is ID

    # store stats for each feature column (excluding id + target)
    avg_list, min_list, max_list, std_list = [], [], [], []
    for col in range(skip, col_count - 1):  # stop before last col (target)
        col_vals = [row[col] for row in arr]
        avg = sum(col_vals) / len(col_vals)
        var = sum((x - avg) ** 2 for x in col_vals) / len(col_vals)
        stdev = math.sqrt(var)
        avg_list.append(avg)
        min_list.append(min(col_vals))
        max_list.append(max(col_vals))
        std_list.append(stdev)

    # throw out outliers
    kept_rows = []
    for row in arr:
        ok = True
        for i, col in enumerate(range(skip, col_count - 1)):
            if std_list[i] > 0 and abs(row[col] - avg_list[i]) > 2 * std_list[i]:
                ok = False
                break
        if ok:
            kept_rows.append(row)

    # normalize
    final_rows = []
    for row in kept_rows:
        new_row = [row[0]]  # keep id
        for i, col in enumerate(range(skip, col_count - 1)):
            if max_list[i] == min_list[i]:
                new_row.append(0.0)
            else:
                new_row.append((row[col] - min_list[i]) / (max_list[i] - min_list[i]))
        new_row.append(row[-1])  # keep target unchanged
        final_rows.append(new_row)

    # write file if needed
    if out_file:
        with open(out_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(final_rows)

    return len(final_rows)


@typechecked
def normalize_array_np(arr: np.ndarray, out_file: str | None = None) -> int:
    if arr.size == 0:
        return 0  # no data

    cols = arr.shape[1]
    start = 1  # skip first column (id)

    # stats for each feature column (excluding id + target)
    avg_list, min_list, max_list, std_list = [], [], [], []
    for col in range(start, cols - 1):  # stop before last col
        col_data = arr[:, col]
        avg_list.append(np.mean(col_data))
        min_list.append(np.min(col_data))
        max_list.append(np.max(col_data))
        std_list.append(np.std(col_data))

    # remove outliers
    good_rows = []
    for row in arr:
        ok = True
        for i, col in enumerate(range(start, cols - 1)):
            if std_list[i] > 0 and abs(row[col] - avg_list[i]) > 2 * std_list[i]:
                ok = False
                break
        if ok:
            good_rows.append(row)

    if not good_rows:
        return 0

    good_arr = np.array(good_rows)

    # normalize values
    norm_rows = []
    for row in good_arr:
        new_row = [row[0]]  # keep id
        for i, col in enumerate(range(start, cols - 1)):
            if max_list[i] == min_list[i]:
                new_row.append(0.0)
            else:
                new_row.append((row[col] - min_list[i]) / (max_list[i] - min_list[i]))
        new_row.append(row[-1])  # keep target unchanged
        norm_rows.append(new_row)

    norm_arr = np.array(norm_rows)

    # save file
    if out_file:
        np.savetxt(out_file, norm_arr, delimiter=",", fmt="%.6f")

    return norm_arr.shape[0]


# -----------------------------
# From data_process2
# -----------------------------

@typechecked
def load_dataset_pd(filename: str) -> pd.DataFrame:
    return pd.read_csv(filename)


@typechecked
def split_xy(df: pd.DataFrame, y_axis: int = -1) -> tuple[np.ndarray, np.ndarray]:
    X = df.drop(df.columns[y_axis], axis=1).to_numpy()
    Y = df.iloc[:, y_axis].to_numpy()
    return X, Y


@typechecked
def split_training_test(
    X_data: np.ndarray,
    Y_data: np.ndarray,
    split: float = 0.8
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    split_index = int(len(X_data) * split)
    X_train = X_data[:split_index]
    Y_train = Y_data[:split_index]
    X_test = X_data[split_index:]
    Y_test = Y_data[split_index:]
    return X_train, Y_train, X_test, Y_test
