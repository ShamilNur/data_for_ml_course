import numpy as np
import pandas as pd


def inject_test_cases(
    df, target_col, condition_func, n_positive=5, n_negative=5, random_state=42
):
    """
    Вставляет контрольные строки в DataFrame для валидации разметки.

    Args:
        df (pd.DataFrame): Исходный DataFrame.
        target_col (str): Колонка, на основе которой определяется "истинная" метка.
        condition_func (callable): Функция, возвращающая True для позитивного класса
                                   (например, lambda x: x >= 6.0).
        n_positive (int): Количество позитивных контрольных примеров.
        n_negative (int): Количество негативных контрольных примеров.
        random_state (int): Seed для воспроизводимости выбора строк.

    Returns:
        pd.DataFrame: DataFrame с добавленными контрольными строками и колонкой '__test_id'.
        pd.Series: Boolean маска, указывающая на контрольные строки (True) и обычные (False).
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame.")

    np.random.seed(random_state)

    # Определяем истинные метки для выбора примеров
    true_labels = df[target_col].apply(condition_func)

    positive_indices = df.index[true_labels]
    negative_indices = df.index[~true_labels]

    if len(positive_indices) < n_positive:
        print(
            f"Warning: Not enough positive samples ({len(positive_indices)} < {n_positive}). Using all available."
        )
        n_positive = len(positive_indices)
    if len(negative_indices) < n_negative:
        print(
            f"Warning: Not enough negative samples ({len(negative_indices)} < {n_negative}). Using all available."
        )
        n_negative = len(negative_indices)

    # Выбираем случайные индексы для дублирования
    chosen_positive_indices = np.random.choice(
        positive_indices, n_positive, replace=False
    )
    chosen_negative_indices = np.random.choice(
        negative_indices, n_negative, replace=False
    )

    # Создаем копии контрольных строк
    positive_tests = df.loc[chosen_positive_indices].copy()
    negative_tests = df.loc[chosen_negative_indices].copy()

    # Добавляем ID для контрольных строк (можно использовать исходный индекс)
    positive_tests["__test_id"] = positive_tests.index
    negative_tests["__test_id"] = negative_tests.index

    # Объединяем контрольные строки
    test_cases = pd.concat([positive_tests, negative_tests], ignore_index=False)
    test_cases["__is_control"] = True  # Добавим явный флаг контрольной строки

    # Добавляем контрольные строки к исходному датафрейму
    df_with_tests = pd.concat([df, test_cases], ignore_index=False)

    # Создаем маску для контрольных строк в итоговом датафрейме
    # Заполняем NaN в __is_control для оригинальных строк значением False
    df_with_tests["__is_control"] = df_with_tests["__is_control"].fillna(False)
    test_mask = df_with_tests["__is_control"].astype(bool)

    # Перемешиваем датафрейм
    df_with_tests = df_with_tests.sample(frac=1, random_state=random_state).reset_index(
        drop=True
    )
    # Обновляем маску после перемешивания
    test_mask = df_with_tests["__is_control"]

    # Заполняем __test_id для не-контрольных строк (например, -1)
    df_with_tests["__test_id"] = df_with_tests["__test_id"].fillna(-1).astype(int)

    print(
        f"Добавлено {len(test_cases)} контрольных строк ({n_positive} позитивных, {n_negative} негативных)."
    )

    return df_with_tests, test_mask
