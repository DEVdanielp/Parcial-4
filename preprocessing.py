"""
Preprocesamiento reproducible para el proyecto Bank Loan.

Este modulo prepara el dataset tabular para convertirlo posteriormente en
imagenes. La idea central es evitar fuga de informacion, controlar el consumo
de memoria y dejar artefactos reutilizables para datos nuevos.

Decisiones principales:
- `Loan Status` se transforma a binario: Fully Paid=1, Charged Off=0.
- `Loan ID` y `Customer ID` se eliminan antes del modelado.
- Las variables numericas se imputan, escalan y se les aplica capping opcional.
- Las variables categoricas se imputan y codifican con one-hot controlado.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from data_loading import get_id_columns, identify_variable_types, infer_target_column

TARGET_MAPPING = {
    "Charged Off": 0,
    "Fully Paid": 1,
    "not_approved": 0,
    "approved": 1,
    0: 0,
    1: 1,
}

CLASS_NAMES = {
    0: "not_approved",
    1: "approved",
}


@dataclass
class PreprocessArtifacts:
    """Objetos necesarios para transformar datos nuevos igual que el train set."""

    preprocessor: ColumnTransformer
    feature_names: list[str]
    numeric_cols: list[str]
    categorical_cols: list[str]
    target_col: str
    id_cols: list[str]
    class_names: dict[int, str]


def normalize_target(y: pd.Series) -> pd.Series:
    """
    Convierte la variable objetivo a 0/1.

    En el dataset usado:
    - Fully Paid representa un prestamo exitoso.
    - Charged Off representa un prestamo problematico.
    """
    mapped = y.map(TARGET_MAPPING)
    if mapped.isna().any():
        unique_values = y.dropna().unique().tolist()
        raise ValueError(f"Valores de target no reconocidos: {unique_values}")
    return mapped.astype("int32")


def clean_dataset(df: pd.DataFrame, target_col: str | None = None) -> pd.DataFrame:
    """Elimina filas vacias, duplicados exactos y etiquetas nulas."""
    df_clean = df.dropna(how="all").drop_duplicates().copy()

    if target_col and target_col in df_clean.columns:
        before = len(df_clean)
        df_clean = df_clean.dropna(subset=[target_col]).copy()
        print(f"Filas sin etiqueta eliminadas: {before - len(df_clean)}")

    print(f"Dataset limpio: {df.shape} -> {df_clean.shape}")
    return df_clean


def cap_outliers(df: pd.DataFrame, numeric_cols: list[str], factor: float = 1.5) -> pd.DataFrame:
    """
    Aplica capping por IQR a variables numericas.

    En vez de eliminar registros extremos, se limitan al rango:
    [Q1 - factor*IQR, Q3 + factor*IQR].
    """
    df_capped = df.copy()

    for col in numeric_cols:
        if col not in df_capped.columns:
            continue

        series = pd.to_numeric(df_capped[col], errors="coerce")
        q1, q3 = series.quantile([0.25, 0.75])
        iqr = q3 - q1

        if pd.isna(iqr) or iqr == 0:
            continue

        lower = q1 - factor * iqr
        upper = q3 + factor * iqr
        df_capped[col] = series.clip(lower, upper)

    return df_capped


def _make_one_hot_encoder() -> OneHotEncoder:
    """
    Crea OneHotEncoder compatible con versiones nuevas y antiguas de sklearn.

    scikit-learn >= 1.2 usa `sparse_output`.
    scikit-learn antiguo usa `sparse`.
    """
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_preprocessor(numeric_cols: list[str], categorical_cols: list[str]) -> ColumnTransformer:
    """Construye el ColumnTransformer para features numericas y categoricas."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", _make_one_hot_encoder()),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def preprocess_pipeline(
    df: pd.DataFrame,
    numeric_cols: list[str] | None = None,
    categorical_cols: list[str] | None = None,
    target_col: str | None = None,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
    cap_numeric_outliers: bool = True,
) -> dict[str, Any]:
    """
    Ejecuta el pipeline completo de preprocesamiento.

    Retorna datasets preprocesados, etiquetas, datos crudos divididos y
    artefactos para transformar usuarios nuevos.
    """
    target_col = target_col or infer_target_column(df)
    if not target_col or target_col not in df.columns:
        raise ValueError("No se encontro variable objetivo. Para este dataset debe existir `Loan Status`.")

    id_cols = get_id_columns(df)
    df_clean = clean_dataset(df, target_col)
    y = normalize_target(df_clean[target_col])

    if numeric_cols is None or categorical_cols is None:
        numeric_cols, categorical_cols = identify_variable_types(df_clean, target_col, id_cols)
    else:
        numeric_cols = [col for col in numeric_cols if col not in id_cols and col != target_col]
        categorical_cols = [col for col in categorical_cols if col not in id_cols and col != target_col]

    feature_cols = numeric_cols + categorical_cols
    X_raw = df_clean[feature_cols].copy()

    if cap_numeric_outliers:
        X_raw = cap_outliers(X_raw, numeric_cols)

    X_train_raw, X_temp_raw, y_train, y_temp = train_test_split(
        X_raw,
        y,
        test_size=test_size + val_size,
        stratify=y,
        random_state=random_state,
    )

    relative_test_size = test_size / (test_size + val_size)
    X_val_raw, X_test_raw, y_val, y_test = train_test_split(
        X_temp_raw,
        y_temp,
        test_size=relative_test_size,
        stratify=y_temp,
        random_state=random_state,
    )

    preprocessor = build_preprocessor(numeric_cols, categorical_cols)
    X_train_arr = preprocessor.fit_transform(X_train_raw).astype("float32")
    X_val_arr = preprocessor.transform(X_val_raw).astype("float32")
    X_test_arr = preprocessor.transform(X_test_raw).astype("float32")
    feature_names = preprocessor.get_feature_names_out().tolist()

    X_train = pd.DataFrame(X_train_arr, columns=feature_names)
    X_val = pd.DataFrame(X_val_arr, columns=feature_names)
    X_test = pd.DataFrame(X_test_arr, columns=feature_names)

    print("\nDivision estratificada:")
    for name, labels in [("train", y_train), ("val", y_val), ("test", y_test)]:
        distribution = labels.value_counts(normalize=True).round(3).to_dict()
        print(f"{name}: {len(labels):,} registros | clases {distribution}")
    print(f"Features finales despues de one-hot controlado: {len(feature_names)}")

    artifacts = PreprocessArtifacts(
        preprocessor=preprocessor,
        feature_names=feature_names,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        target_col=target_col,
        id_cols=id_cols,
        class_names=CLASS_NAMES,
    )

    df_processed = pd.concat(
        [
            pd.DataFrame(np.vstack([X_train_arr, X_val_arr, X_test_arr]), columns=feature_names),
            pd.Series(np.concatenate([y_train, y_val, y_test]), name=target_col),
        ],
        axis=1,
    )

    return {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train.reset_index(drop=True),
        "y_val": y_val.reset_index(drop=True),
        "y_test": y_test.reset_index(drop=True),
        "X_train_raw": X_train_raw.reset_index(drop=True),
        "X_val_raw": X_val_raw.reset_index(drop=True),
        "X_test_raw": X_test_raw.reset_index(drop=True),
        "artifacts": artifacts,
        "preprocessor": preprocessor,
        "feature_names": feature_names,
        "df_processed": df_processed,
    }


def transform_new_data(df_new: pd.DataFrame, artifacts: PreprocessArtifacts) -> pd.DataFrame:
    """
    Transforma datos nuevos sin etiqueta usando artefactos ya entrenados.

    Esta funcion se usa para simular predicciones con `credit_test.csv`.
    """
    cols = artifacts.numeric_cols + artifacts.categorical_cols
    X_new = df_new.drop(columns=[col for col in artifacts.id_cols if col in df_new.columns], errors="ignore")
    X_new = X_new.reindex(columns=cols)
    X_new = cap_outliers(X_new, artifacts.numeric_cols)

    arr = artifacts.preprocessor.transform(X_new).astype("float32")
    return pd.DataFrame(arr, columns=artifacts.feature_names)


def save_artifacts(
    artifacts: PreprocessArtifacts,
    path: str | Path = "models/preprocess_artifacts.joblib",
) -> None:
    """Guarda los artefactos de preprocesamiento en disco."""
    path = Path(path)
    path.parent.mkdir(exist_ok=True)
    joblib.dump(artifacts, path)
    print(f"Artefactos guardados en {path}")


def load_artifacts(path: str | Path = "models/preprocess_artifacts.joblib") -> PreprocessArtifacts:
    """Carga artefactos de preprocesamiento guardados con joblib."""
    return joblib.load(path)


if __name__ == "__main__":
    from data_loading import load_dataset

    df_train = load_dataset("data/credit_train.csv")
    if df_train is not None:
        processed = preprocess_pipeline(df_train)
        save_artifacts(processed["artifacts"])
