"""
pipelines.py ― End‑to‑End *Data‑Centric ML* demo project.

- Стягивает открытые датасеты World‑Happiness 2024 + Gapminder GDP
- Чистит, формирует метку `is_happy`, проводит 5‑итерационное Active‑Learning
- Сохраняет артефакты в `./storage/` и выводит финальные метрики

Можно запускать:
    python pipelines.py                          # единичный прогон
    dagster dev -f 05_final_dagster_pipeline/pipelines.py   # UI Dagster
    dagster job launch -f 05_final_dagster_pipeline/pipelines.py -j data_centric_job

Зависимости см. requirements.txt (Python ≥3.9).
Проект не требует ключей /API ― только открытые CSV.
"""

from __future__ import annotations

import io
import json
import os
import textwrap
import warnings
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
import requests
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, cohen_kappa_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

# ─────────────── Dagster imports & wrappers ────────────────
from dagster import In, Nothing, Out, job, op, repository
from skactiveml.classifier import SklearnClassifier
from skactiveml.pool import UncertaintySampling
from skactiveml.utils import MISSING_LABEL


# --- Dagster ops wrapping существующие функции ----------------
@op(out=Out(Path, description="CSV with raw merged data"))
def collect_data_op() -> Path:
    return collect_and_merge()


@op(ins={"raw_path": In(Path)}, out=Out(Path, description="Clean CSV"))
def clean_data_op(raw_path: Path) -> Path:
    return clean_dataset(raw_path)


@op(ins={"clean_path": In(Path)}, out=Out(Path, description="Labeled CSV"))
def label_data_op(clean_path: Path) -> Path:
    return label_dataset(clean_path)


@op(ins={"labeled_path": In(Path)}, out=Out(Path, description="AL model path"))
def active_learning_op(labeled_path: Path) -> Path:
    model_path, _ = active_learning(labeled_path)
    return model_path


@op(ins={"labeled_path": In(Path)}, out=Out(Path, description="Full‑model path"))
def train_model_op(labeled_path: Path) -> Path:
    return train_full_model(labeled_path)


# ----------- Dagster job tying everything together -----------
@job(tags={"owner": "data_for_ml_course"})
def data_centric_job():
    raw = collect_data_op()
    clean = clean_data_op(raw)
    labeled = label_data_op(clean)
    _al_model = active_learning_op(labeled)
    _full_model = train_model_op(labeled)


# ───────────────────  Параметры  ────────────────────
RNG = 42
RAW_DIR = Path("storage/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)
CLEAN_DIR = Path("storage/clean")
CLEAN_DIR.mkdir(exist_ok=True)
LABELED_DIR = Path("storage/labeled")
LABELED_DIR.mkdir(exist_ok=True)
MODELS_DIR = Path("storage/models")
MODELS_DIR.mkdir(exist_ok=True)
REPORT_DIR = Path("storage/reports")
REPORT_DIR.mkdir(exist_ok=True)
FIGS_DIR = REPORT_DIR / "figures"
FIGS_DIR.mkdir(exist_ok=True)

TARGET_ACC = 0.95
INIT_FRAC = 0.05  # 5 % init
POOL_FRAC = 0.80  # 80 % init+pool, 20 % test
N_ITER = 5
QUERY_FRAC = 0.05  # 5 % за итерацию


# ───────────────── 1. Сбор данных (модуль 1) ─────────────────
def fetch_csv(urls: list[str]) -> pd.DataFrame:
    """Download first reachable CSV from mirrors and return as DataFrame."""
    for u in urls:
        try:
            txt = requests.get(u, timeout=15).text
            if txt.strip():
                delim = (
                    ","
                    if txt.splitlines()[0].count(",") >= txt.splitlines()[0].count(";")
                    else ";"
                )
                df = pd.read_csv(io.StringIO(txt), sep=delim)
                print(f"✓ Загружен {u}")
                print(f"  Колонки: {df.columns.tolist()}")
                return df
        except Exception as e:
            print(f"  - {u}: {e}")
    raise RuntimeError("Нет доступных зеркал")


def collect_and_merge() -> Path:
    """Collect WHR + GDP, normalize column names, merge into single CSV."""
    whr_urls = [
        "https://raw.githubusercontent.com/pplonski/datasets-for-start/master/world_happiness_report/WHR_2024.csv",
        "https://raw.githubusercontent.com/sash-04/WorldHappiness/main/World-happiness-report-2024.csv",
    ]
    gdp_urls = [
        "https://raw.githubusercontent.com/kelly041222/PowerBI1/main/gapminder_gdp_per_capita_dollars.csv",
    ]
    print("Загрузка данных WHR...")
    whr_raw = fetch_csv(whr_urls)
    whr_raw.columns = whr_raw.columns.str.strip().str.lower()
    print(f"Колонки после нормализации: {whr_raw.columns.tolist()}")

    country_col = None
    for col in ["country name", "country", "nation"]:
        if col in whr_raw.columns:
            country_col = col
            break

    ladder_col = None
    for col in [
        "ladder score",
        "ladder",
        "score",
        "happiness score",
        "happiness_score",
    ]:
        if col in whr_raw.columns:
            ladder_col = col
            break

    if country_col is None or ladder_col is None:
        print(f"Доступные колонки: {whr_raw.columns.tolist()}")
        raise KeyError(f"Не найдены колонки страны или счастья в WHR")

    print(f"Используем колонки: страна='{country_col}', счастье='{ladder_col}'")
    whr = whr_raw[[country_col, ladder_col]].rename(
        columns={country_col: "country", ladder_col: "ladder"}
    )

    print("Загрузка данных GDP...")
    gdp_raw = fetch_csv(gdp_urls)
    if "country" not in gdp_raw.columns:
        gdp_raw = gdp_raw.rename(columns={gdp_raw.columns[0]: "country"})

    gdp = gdp_raw.melt(id_vars="country", var_name="year", value_name="gdp")
    gdp["year"] = gdp["year"].astype(int)

    print("Объединение данных...")
    df = whr.assign(year=2015).merge(
        gdp.query("year==2015"), on=["country", "year"], how="inner"
    )
    print(f"Итоговый датасет: {len(df)} строк, колонки: {df.columns.tolist()}")

    raw_path = RAW_DIR / "happiness_gdp_raw.csv"
    df.to_csv(raw_path, index=False)
    print(f"✔ RAW сохранён: {raw_path}")
    return raw_path


# ──────────────── 2. Чистка данных (модуль 2) ───────────────
def clean_dataset(raw_path: Path) -> Path:
    """Basic cleaning: numeric coercion, 1–99 % clipping, save clean CSV."""
    df = pd.read_csv(raw_path).dropna()
    df["gdp"] = pd.to_numeric(df["gdp"], errors="coerce")
    df = df.dropna()
    q_low, q_hi = df["gdp"].quantile([0.01, 0.99])
    df["gdp"] = df["gdp"].clip(q_low, q_hi)
    clean_path = CLEAN_DIR / "clean_median_imputed_clip.csv"
    df.to_csv(clean_path, index=False)
    print(f"✔ CLEAN сохранён: {clean_path}")
    return clean_path


# ─────────────── 3. Разметка данных (модуль 3) ──────────────
def label_dataset(clean_path: Path) -> Path:
    """Create binary target `is_happy` and save labeled CSV."""
    df = pd.read_csv(clean_path)
    df["is_happy"] = (df["ladder"] >= 6).astype(int)
    labeled_path = LABELED_DIR / "final_labeled.csv"
    df.to_csv(labeled_path, index=False)
    print(f"✔ LABELED сохранён: {labeled_path}")
    return labeled_path


# ──────── 4. Active Learning (модуль 4) + отчёт ─────────────
def active_learning(labeled_path: Path) -> tuple[Path, dict]:
    """Run entropy‑based Active Learning loop and persist model/metrics."""
    df = pd.read_csv(labeled_path)
    df["gdp_log"] = np.log1p(df["gdp"])
    scaler = StandardScaler().fit(df[["gdp_log"]])
    df["gdp_scaled"] = scaler.transform(df[["gdp_log"]])
    joblib.dump(scaler, MODELS_DIR / "scaler_gdp.pkl")

    X = df[["gdp_scaled"]].values
    y = df["is_happy"].values
    X_pool, X_test, y_pool, y_test = train_test_split(
        X, y, test_size=1 - POOL_FRAC, stratify=y, random_state=RNG
    )
    n_init = max(1, int(INIT_FRAC * len(X)))
    init_idx = np.random.RandomState(RNG).choice(len(X_pool), n_init, replace=False)
    X_init, y_init = X_pool[init_idx], y_pool[init_idx]
    X_pool, y_pool = np.delete(X_pool, init_idx, 0), np.delete(y_pool, init_idx, 0)

    clf = SklearnClassifier(
        estimator=LogisticRegression(
            max_iter=2000, solver="liblinear", random_state=RNG
        ),
        classes=np.unique(y),
        random_state=RNG,
    )
    qs = UncertaintySampling(random_state=RNG)

    X_lab, y_lab = X_init, y_init
    n_hist, acc_hist, kappa_hist = [len(X_lab)], [], []
    for iter_num in range(N_ITER):
        clf.fit(X_lab, y_lab)
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        kappa = cohen_kappa_score(y_test, y_pred)
        acc_hist.append(acc)
        kappa_hist.append(kappa)
        print(
            f"Итерация {iter_num+1}: accuracy={acc:.4f}, kappa={kappa:.4f}, меток={len(X_lab)}"
        )

        n_query = min(int(QUERY_FRAC * (len(X_pool) + len(X_lab))), len(X_pool))
        if n_query == 0:
            print("Пул исчерпан, останавливаем обучение")
            break

        y_pool_masked = np.full(len(X_pool), MISSING_LABEL)
        try:
            q_idx = qs.query(X_pool, y_pool_masked, clf=clf, batch_size=n_query)[0]
            if np.isscalar(q_idx):
                q_idx = np.array([q_idx])
            X_lab = np.vstack([X_lab, X_pool[q_idx]])
            y_lab = np.concatenate([y_lab, y_pool[q_idx]])
            X_pool, y_pool = np.delete(X_pool, q_idx, 0), np.delete(y_pool, q_idx, 0)
            n_hist.append(len(X_lab))
        except Exception as e:
            print(f"Ошибка при запросе новых меток: {e}")
            break

    clf.fit(X_lab, y_lab)
    y_pred = clf.predict(X_test)
    acc_fin, kappa_fin = accuracy_score(y_test, y_pred), cohen_kappa_score(
        y_test, y_pred
    )
    acc_hist.append(acc_fin)
    kappa_hist.append(kappa_fin)
    print(
        f"Финальная модель: accuracy={acc_fin:.4f}, kappa={kappa_fin:.4f}, меток={len(X_lab)}"
    )

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(n_hist, acc_hist, "o-", color="tab:red", label="Accuracy")
    ax1.axhline(0.95, ls="--", color="gray")
    ax1.set_xlabel("N labels")
    ax1.set_ylabel("Accuracy")
    ax2 = ax1.twinx()
    ax2.plot(n_hist, kappa_hist, "s:", color="tab:blue", label="Kappa")
    ax2.set_ylabel("Cohen κ")
    ax1.legend(loc="lower right")
    fig.tight_layout()
    curve_path = FIGS_DIR / "learning_curve.png"
    fig.savefig(curve_path)
    plt.close(fig)

    model_path = MODELS_DIR / "is_happy_al.pkl"
    joblib.dump(clf, model_path)
    used_labels = n_hist[-1]
    saving = 1 - used_labels / (len(y_init) + len(X_pool) + used_labels)

    metrics = {
        "accuracy": float(acc_fin),
        "kappa": float(kappa_fin),
        "labels_used": int(used_labels),
        "saving_pct": round(float(saving) * 100, 2),
    }
    with open(REPORT_DIR / "al_metrics.json", "w") as jf:
        json.dump(metrics, jf, indent=2)
    print(f"✔ Active‑Learning модель: {model_path} | метрики: {metrics}")
    return model_path, metrics


# ─────────────── 5. Training full model (опц.) ──────────────
def train_full_model(labeled_path: Path) -> Path:
    """Train baseline logistic regression on full data for comparison."""
    df = pd.read_csv(labeled_path)
    X = df[["gdp"]].values
    y = df["is_happy"].values
    model = LogisticRegression(max_iter=2000, solver="liblinear", random_state=RNG).fit(
        X, y
    )
    pkl = MODELS_DIR / "is_happy_full.pkl"
    joblib.dump(model, pkl)
    return pkl


# ──────────────────────  main  ──────────────────────────────
def main() -> None:
    """Pipeline orchestrator when script executed directly."""
    warnings.filterwarnings("ignore")
    try:
        raw = collect_and_merge()
        clean = clean_dataset(raw)
        lbl = label_dataset(clean)
        model_al, metrics = active_learning(lbl)
        model_full = train_full_model(lbl)

        print("\n==== FINISHED ====")
        print(
            textwrap.dedent(
                f"""
            AL metrics   : {metrics}
            AL model     : {model_al}
            Full model   : {model_full}
            LearningCurve: {FIGS_DIR/'learning_curve.png'}
            Scaler       : {MODELS_DIR/'scaler_gdp.pkl'}
            Все артефакты лежат в ./storage/
        """
            )
        )
    except Exception as e:
        import traceback

        print(f"\n==== ОШИБКА ====\n{e}")
        print(traceback.format_exc())


if __name__ == "__main__":
    main()


@repository
def data_for_ml_repository():
    """Репозиторий с нашим ML пайплайном."""
    return [data_centric_job]
