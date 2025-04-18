# 📊 Модуль 2: Качество и чистка данных

Выполняет глубокий анализ качества данных, полученных в Модуле 1. Идентифицирует и анализирует пропуски (`missingno`), сравнивает 3 стратегии их заполнения (Median, KNN, MICE). Обнаруживает и обрабатывает выбросы (`scikit-learn`) с использованием 2 методов (Clipping, Winsorizing). Оценивает влияние каждой из 5 комбинаций очистки на предсказательную силу простой линейной регрессии (`scikit-learn`), выбирая оптимальную стратегию по метрикам MAE и R².

## 🔄 Процесс работы
| Этап                   | 📥 Вход                       | ⚙️ Обработка                                                              | 📤 Выход                                               |
|------------------------|-------------------------------|---------------------------------------------------------------------------|--------------------------------------------------------|
| 1. Загрузка            | `../data/happiness_gdp.csv` | `pd.read_csv`                                                             | `pd.DataFrame` (raw)                                   |
| 2. Анализ пропусков    | `pd.DataFrame` (raw)        | `df.isnull().sum()`, `msno.matrix()`, `msno.heatmap()`                    | Визуализации, тип пропусков (MCAR/MAR/...)             |
| 3. Импутация           | `pd.DataFrame` (raw)        | `SimpleImputer`, `KNNImputer`, `IterativeImputer`                         | 3 `pd.DataFrame` (median, knn, mice)                   |
| 4. Анализ выбросов     | `pd.DataFrame` (median)     | Z-score, `IsolationForest`, `sns.boxplot`                               | Идентификация выбросов                                 |
| 5. Обработка выбросов  | `pd.DataFrame` (median)     | Clipping (`.clip()`), Winsorizing (`scipy.stats.mstats.winsorize`)         | 2 `pd.DataFrame` (median+clip, median+winsorize)     |
| 6. Оценка стратегий    | 5 `pd.DataFrame`            | `LinearRegression`, `mean_absolute_error`, `r2_score`                   | Таблица/график с MAE/R² для 5 стратегий                |
| 7. Выбор & Сохранение | `pd.DataFrame` (best)       | Выбор лучшей стратегии по метрикам, `df.to_csv`                          | `../data/clean_[best_strategy].csv`, обоснование выбора |

## ✅ Definition of Done
- [x] Проведен анализ пропусков с определением типа (MCAR/MAR/MNAR)
- [x] Реализованы ≥3 метода импутации (здесь: Median, KNN, MICE)
- [x] Применены ≥2 метода обработки выбросов (здесь: Clipping, Winsorizing)
- [x] Выполнено сравнение стратегий очистки по MAE/R² на downstream-задаче (регрессия)
- [x] Выбрана и обоснована лучшая стратегия очистки
- [x] Визуализированы результаты анализа пропусков и выбросов

## 🔗 Связь с другими модулями
⬅️ **Предыдущий**: Модуль 1 (`01_data_collection`) передает `../data/happiness_gdp.csv`.
➡️ **Следующий**: Модуль 3 (`03_data_labeling`) получает на вход `../data/clean_[best_strategy].csv` для разметки. 