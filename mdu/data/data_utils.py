from sklearn.model_selection import train_test_split
import numpy as np


def split_dataset(
    X, y, train_ratio=0.5, test_ratio=0.5, calib_ratio=0.5, random_state=42
):
    if train_ratio > 0:
        X_train_main, X_temp, y_train_main, y_temp = train_test_split(
            X, y, train_size=train_ratio, random_state=random_state, stratify=y
        )
    else:
        X_temp = X
        y_temp = y
        X_train_main = None
        y_train_main = None
    X_train_cond, X_temp2, y_train_cond, y_temp2 = train_test_split(
        X_temp,
        y_temp,
        test_size=1 - calib_ratio,
        random_state=random_state,
        stratify=y_temp,
    )
    X_calib, X_test, y_calib, y_test = train_test_split(
        X_temp2,
        y_temp2,
        test_size=test_ratio,
        random_state=random_state,
        stratify=y_temp2,
    )

    return (
        X_train_main,
        X_train_cond,
        X_calib,
        X_test,
        y_train_main,
        y_train_cond,
        y_calib,
        y_test,
    )


def split_dataset_indices(
    X, y, train_ratio=0.5, test_ratio=0.5, calib_ratio=0.5, random_state=42
):
    n_samples = len(X)
    all_indices = np.arange(n_samples)

    if train_ratio > 0:
        train_main_idx, temp_idx = train_test_split(
            all_indices, train_size=train_ratio, random_state=random_state, stratify=y
        )
    else:
        temp_idx = all_indices
        train_main_idx = None

    # For the next split, use the y values corresponding to temp_idx
    y_temp = y[temp_idx]
    train_cond_idx, temp2_idx = train_test_split(
        temp_idx,
        test_size=1 - calib_ratio,
        random_state=random_state,
        stratify=y_temp,
    )

    y_temp2 = y[temp2_idx]
    calib_idx, test_idx = train_test_split(
        temp2_idx,
        test_size=test_ratio,
        random_state=random_state,
        stratify=y_temp2,
    )

    return (
        train_main_idx,
        train_cond_idx,
        calib_idx,
        test_idx,
    )
