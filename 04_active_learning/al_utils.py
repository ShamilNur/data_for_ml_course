import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import train_test_split


def split_pool(
    df, target_col, features_cols, pool_frac=0.8, init_frac=0.05, random_state=42
):
    """
    Разделяет DataFrame на начальный обучающий набор, пул для запросов и тестовый набор.

    Args:
        df (pd.DataFrame): Исходный DataFrame.
        target_col (str): Название целевой колонки.
        features_cols (list): Список названий колонок с признаками.
        pool_frac (float): Доля данных, которая останется для пула и начального набора.
        init_frac (float): Доля от *исходного* размера данных для начального обучающего набора.
        random_state (int): Seed для воспроизводимости.

    Returns:
        tuple: X_pool, y_pool, X_init, y_init, X_test, y_test
               (Признаки и метки для пула, начального набора и теста)
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found.")
    for col in features_cols:
        if col not in df.columns:
            raise ValueError(f"Feature column '{col}' not found.")

    y = df[target_col]
    X = df[features_cols]
    n_total = len(df)

    if pool_frac < 0 or pool_frac > 1:
        raise ValueError("pool_frac должен быть в диапазоне [0, 1]")
    if init_frac < 0 or init_frac > pool_frac:
        raise ValueError("init_frac должен быть в диапазоне [0, pool_frac]")

    test_size = 1.0 - pool_frac
    if test_size > 0:
        X_pool_init, X_test, y_pool_init, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
    else:
        X_pool_init, y_pool_init = X, y
        X_test, y_test = pd.DataFrame(), pd.Series()

    n_pool_init = len(X_pool_init)
    n_init = int(np.round(init_frac * n_total))

    n_init = min(n_init, n_pool_init)

    if n_init == 0 and n_pool_init > 0:
        n_init = 1
        print(
            f"Warning: init_frac ({init_frac}) слишком мал, используем 1 начальный пример."
        )

    if n_init == n_pool_init:
        X_init, y_init = X_pool_init, y_pool_init
        X_pool, y_pool = pd.DataFrame(), pd.Series()
        print(f"Warning: Весь пул ({n_pool_init}) используется как начальный набор.")
    elif n_init > 0:
        init_split_frac = n_init / n_pool_init
        X_pool, X_init, y_pool, y_init = train_test_split(
            X_pool_init,
            y_pool_init,
            test_size=init_split_frac,
            random_state=random_state,
            stratify=y_pool_init,
        )
    else:
        X_init, y_init = pd.DataFrame(), pd.Series()
        X_pool, y_pool = pd.DataFrame(), pd.Series()

    print(
        f"Размеры наборов: init={len(X_init)}, pool={len(X_pool)}, test={len(X_test)}"
    )
    if len(X_init) + len(X_pool) + len(X_test) != n_total:
        print("Warning: Сумма размеров наборов не равна исходному размеру!")

    return X_pool, y_pool, X_init, y_init, X_test, y_test


def calc_kappa(y_true, y_pred):
    """Обёртка над cohen_kappa_score."""
    return cohen_kappa_score(y_true, y_pred)
