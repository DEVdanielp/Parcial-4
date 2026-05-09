"""
Carga y analisis exploratorio del dataset Bank Loan / Credit Risk.

El proyecto usa `credit_train.csv` como dataset supervisado porque incluye la
variable objetivo `Loan Status`. El archivo `credit_test.csv` no incluye la
etiqueta y se reserva para simular predicciones sobre usuarios nuevos.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

TARGET_CANDIDATES = ["Loan Status", "Personal Loan", "loan_status", "target", "approved"]
ID_COLUMNS = ["Loan ID", "Customer ID", "ID", "Id"]


def load_dataset(filepath: str | Path) -> pd.DataFrame | None:
    """Carga un CSV y elimina filas completamente vacias."""
    filepath = Path(filepath)
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"Archivo no encontrado: {filepath}")
        return None
    except Exception as exc:
        print(f"No fue posible cargar {filepath}: {exc}")
        return None

    df = df.dropna(how="all").copy()
    df.columns = [str(col).strip() for col in df.columns]
    print(f"Dataset cargado: {filepath} | dimensiones: {df.shape}")
    return df


def infer_target_column(df: pd.DataFrame) -> str | None:
    """Identifica la variable objetivo evitando confundir IDs con target."""
    for candidate in TARGET_CANDIDATES:
        if candidate in df.columns:
            return candidate

    binary_non_id = []
    for col in df.columns:
        if "id" in col.lower():
            continue
        nunique = df[col].nunique(dropna=True)
        if 1 < nunique <= 2:
            binary_non_id.append(col)
    return binary_non_id[0] if binary_non_id else None


def get_id_columns(df: pd.DataFrame) -> list[str]:
    """Detecta columnas identificadoras que no deben alimentar el modelo."""
    ids = [col for col in ID_COLUMNS if col in df.columns]
    for col in df.columns:
        if col in ids:
            continue
        high_cardinality = df[col].nunique(dropna=True) > max(100, 0.5 * len(df))
        if "id" in col.lower() and high_cardinality:
            ids.append(col)
    return ids


def identify_variable_types(
    df: pd.DataFrame,
    target_col: str | None = None,
    exclude_cols: Iterable[str] | None = None,
) -> tuple[list[str], list[str]]:
    """Devuelve variables numericas y categoricas utiles para modelado."""
    exclude = set(exclude_cols or [])
    if target_col:
        exclude.add(target_col)
    feature_df = df.drop(columns=[col for col in exclude if col in df.columns], errors="ignore")
    numeric_cols = feature_df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = feature_df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    return numeric_cols, categorical_cols


def analyze_dataset(df: pd.DataFrame) -> tuple[list[str], list[str], str | None]:
    """Ejecuta un EDA textual breve y retorna columnas relevantes."""
    print("\n" + "=" * 70)
    print("ANALISIS EXPLORATORIO DEL DATASET")
    print("=" * 70)
    print(f"Registros: {df.shape[0]:,}")
    print(f"Variables: {df.shape[1]:,}")
    print("\nTipos de datos:")
    print(df.dtypes)

    target_col = infer_target_column(df)
    id_cols = get_id_columns(df)
    numeric_cols, categorical_cols = identify_variable_types(df, target_col, id_cols)

    print(f"\nColumnas identificadoras excluidas del modelo: {id_cols}")
    print(f"Variable objetivo: {target_col}")
    print(f"Variables numericas ({len(numeric_cols)}): {numeric_cols}")
    print(f"Variables categoricas ({len(categorical_cols)}): {categorical_cols}")

    if target_col:
        print("\nDistribucion de la variable objetivo:")
        print(df[target_col].value_counts(dropna=False))
        print(df[target_col].value_counts(normalize=True, dropna=False).rename("proporcion"))

    return numeric_cols, categorical_cols, target_col


def check_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Reporta valores nulos por columna."""
    missing = df.isna().sum()
    summary = pd.DataFrame({
        "valores_nulos": missing,
        "porcentaje": (missing / len(df) * 100).round(2),
    }).sort_values("valores_nulos", ascending=False)
    print("\nVALORES NULOS")
    print(summary[summary["valores_nulos"] > 0] if summary["valores_nulos"].sum() else "No hay valores nulos.")
    return summary


def check_duplicates(df: pd.DataFrame) -> int:
    """Cuenta registros duplicados exactos."""
    duplicates = int(df.duplicated().sum())
    print(f"\nRegistros duplicados exactos: {duplicates}")
    return duplicates


def detect_outliers(df: pd.DataFrame, numeric_cols: list[str], method: str = "iqr") -> dict[str, int]:
    """Detecta outliers por IQR o z-score en variables numericas."""
    print("\nOUTLIERS")
    outliers: dict[str, int] = {}
    for col in numeric_cols:
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        if series.empty:
            outliers[col] = 0
            continue
        if method == "zscore":
            count = int((np.abs(stats.zscore(series)) > 3).sum())
        else:
            q1, q3 = series.quantile([0.25, 0.75])
            iqr = q3 - q1
            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            count = int(((series < lower) | (series > upper)).sum())
        outliers[col] = count
        print(f"{col}: {count}")
    return outliers


def plot_distributions(df: pd.DataFrame, numeric_cols: list[str], target_col: str | None = None) -> None:
    """Guarda histogramas de variables numericas."""
    if not numeric_cols:
        return
    cols = numeric_cols[:12]
    n_cols = 3
    n_rows = int(np.ceil(len(cols) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    axes = np.array(axes).reshape(-1)
    for ax, col in zip(axes, cols):
        if target_col and target_col in df.columns:
            sns.histplot(data=df, x=col, hue=target_col, bins=30, ax=ax, element="step")
        else:
            sns.histplot(data=df, x=col, bins=30, ax=ax)
        ax.set_title(f"Distribucion de {col}")
    for ax in axes[len(cols):]:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "distributions.png", dpi=180, bbox_inches="tight")
    plt.show()


def plot_correlations(df: pd.DataFrame, numeric_cols: list[str]) -> pd.DataFrame | None:
    """Guarda matriz de correlacion numerica."""
    if len(numeric_cols) < 2:
        return None
    corr = df[numeric_cols].corr(numeric_only=True)
    plt.figure(figsize=(12, 9))
    sns.heatmap(corr, cmap="coolwarm", center=0, linewidths=0.3)
    plt.title("Matriz de correlacion - variables numericas")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "correlation_matrix.png", dpi=180, bbox_inches="tight")
    plt.show()
    return corr


def profile_dataset(filepath: str | Path, make_plots: bool = True):
    """Carga, analiza y opcionalmente grafica el dataset."""
    df = load_dataset(filepath)
    if df is None:
        return None, None, None, None
    numeric_cols, categorical_cols, target_col = analyze_dataset(df)
    check_missing_values(df)
    check_duplicates(df)
    detect_outliers(df, numeric_cols)
    if make_plots:
        plot_distributions(df, numeric_cols, target_col)
        plot_correlations(df, numeric_cols)
    return df, numeric_cols, categorical_cols, target_col


def main(filepath: str | Path):
    """Compatibilidad con el notebook anterior."""
    return profile_dataset(filepath)


if __name__ == "__main__":
    profile_dataset("data/credit_train.csv")
