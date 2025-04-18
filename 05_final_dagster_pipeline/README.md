# 🚀 Модуль 5: Финальный пайплайн на Dagster

Оркестрирует полный end-to-end цикл data-centric ML, объединяет логику предыдущих модулей в единый воспроизводимый пайплайн с помощью `dagster`. 

Пайплайн состоит из 5 ключевых шагов (ops): 
1. сбор данных, 
2. очистка, 
3. разметка (генерация метки), 
4. активное обучение 
5. и обучение финальной модели. 

Управляет зависимостями между шагами и сохраняет артефакты (данные, модели, отчеты) в директорию (`./storage/`). Позволяет запускать весь процесс одной командой и визуализировать его выполнение через Dagster UI.

## 🔄 Процесс работы (Dagster Job: `data_centric_job`)
| Шаг (Op)               | 📥 Вход                         | ⚙️ Обработка (логика из модуля) | 📤 Выход (артефакт)                      |
|------------------------|---------------------------------|-----------------------------------|------------------------------------------|
| `collect_data_op`      | URLы источников данных         | Модуль 1                          | `storage/raw/happiness_gdp_raw.csv`      |
| `clean_data_op`        | `Path` на raw CSV               | Модуль 2 (best_strategy)          | `storage/clean/clean_*.csv`              |
| `label_data_op`        | `Path` на clean CSV             | Модуль 3                          | `storage/labeled/final_labeled.csv`      |
| `active_learning_op`   | `Path` на labeled CSV           | Модуль 4                          | `storage/models/is_happy_al.pkl`         |
| `train_model_op`       | `Path` на labeled CSV           | Обучение модели на всех данных   | `storage/models/is_happy_full.pkl`       |
|                        |                                 | Генерация отчетов/метрик         | `storage/reports/*`                      |

**Запуск:**
```bash
# Запуск UI (recomment)
cd /путь/к/проекту/data_for_ml_course
export DAGSTER_HOME=$PWD
dagster dev -f 05_final_dagster_pipeline/pipelines.py

# Прямой запуск (CLI)
dagster job execute -f 05_final_dagster_pipeline/pipelines.py -j data_centric_job
```

## ✅ Definition of Done
- [x] Реализован Dagster-пайплайн, включающий 5 ops: collect → clean → label → active_learning → train
- [x] Настроены зависимости и передача артефактов между ops
- [x] Используется структурированное хранилище (`./storage/`) для данных и моделей
- [x] Пайплайн успешно запускается через `dagster dev` и `dagster job execute`
- [ ] Настроены `resources` и `config` в Dagster (если применимо)
- [ ] Подключен CI/CD для проверки пайплайна
- [ ] Подготовлен Dockerfile для контейнеризации

## 🔗 Связь с другими модулями
- ⬅️ **Предыдущий**: Модули 1-4 предоставляют логику, инкапсулированную в `ops`.
- ➡️ **Следующий**: Нет (финальный модуль, результат - обученные модели и артефакты). 