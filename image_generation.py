"""
Generacion de imagenes desde registros tabulares preprocesados.

Estructura generada:
images/<metodo>/<split>/<clase>/sample_000001.png

Metodos disponibles: rgb, grayscale, heatmap, gaf, mtf, recurrence.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

CLASS_NAMES = {0: "not_approved", 1: "approved"}


def _safe_minmax(vector: np.ndarray, feature_range: tuple[float, float] = (0.0, 1.0)) -> np.ndarray:
    vector = np.asarray(vector, dtype="float32")
    vector = np.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0)
    v_min, v_max = float(vector.min()), float(vector.max())
    low, high = feature_range
    if np.isclose(v_max, v_min):
        return np.full_like(vector, (low + high) / 2, dtype="float32")
    return (vector - v_min) / (v_max - v_min) * (high - low) + low


class ImageGenerator:
    """Convierte vectores numericos en imagenes listas para CNN."""

    def __init__(self, image_size: tuple[int, int] = (64, 64), save_dir: str | Path = "images"):
        self.image_size = image_size
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)

    def vector_to_grayscale(self, vector: np.ndarray) -> np.ndarray:
        values = (_safe_minmax(vector) * 255).astype("uint8")
        pixels = np.resize(values, self.image_size[0] * self.image_size[1])
        return pixels.reshape(self.image_size)

    def vector_to_rgb(self, vector: np.ndarray) -> np.ndarray:
        gray = self.vector_to_grayscale(vector)
        rolled_1 = np.roll(gray, shift=1, axis=0)
        rolled_2 = np.roll(gray, shift=1, axis=1)
        return np.stack([gray, rolled_1, rolled_2], axis=-1).astype("uint8")

    def vector_to_heatmap(self, vector: np.ndarray) -> np.ndarray:
        gray = self.vector_to_grayscale(vector) / 255.0
        cmap = plt.get_cmap("viridis")
        return (cmap(gray)[..., :3] * 255).astype("uint8")

    def vector_to_gaf(self, vector: np.ndarray) -> np.ndarray:
        scaled = np.clip(_safe_minmax(vector, (-1.0, 1.0)), -1, 1)
        phi = np.arccos(scaled)
        matrix = np.cos(phi[:, None] + phi[None, :])
        return self._matrix_to_rgb(matrix, cmap_name="magma")

    def vector_to_mtf(self, vector: np.ndarray, n_bins: int = 8) -> np.ndarray:
        scaled = _safe_minmax(vector)
        bins = np.linspace(0, 1, n_bins + 1)
        states = np.clip(np.digitize(scaled, bins) - 1, 0, n_bins - 1)
        transition = np.zeros((n_bins, n_bins), dtype="float32")
        for a, b in zip(states[:-1], states[1:]):
            transition[a, b] += 1
        row_sums = transition.sum(axis=1, keepdims=True)
        transition = np.divide(transition, row_sums, out=np.zeros_like(transition), where=row_sums != 0)
        matrix = transition[states[:, None], states[None, :]]
        return self._matrix_to_rgb(matrix, cmap_name="cividis")

    def vector_to_recurrence(self, vector: np.ndarray) -> np.ndarray:
        scaled = _safe_minmax(vector)
        distances = np.abs(scaled[:, None] - scaled[None, :])
        matrix = 1.0 - _safe_minmax(distances)
        return self._matrix_to_rgb(matrix, cmap_name="gray")

    def _matrix_to_rgb(self, matrix: np.ndarray, cmap_name: str) -> np.ndarray:
        normalized = _safe_minmax(matrix.ravel()).reshape(matrix.shape)
        img = Image.fromarray((normalized * 255).astype("uint8"))
        img = img.resize(self.image_size, Image.Resampling.BILINEAR)
        arr = np.asarray(img) / 255.0
        cmap = plt.get_cmap(cmap_name)
        return (cmap(arr)[..., :3] * 255).astype("uint8")

    def generate_image(self, vector: np.ndarray, method: str = "rgb") -> np.ndarray:
        method = method.lower()
        if method == "rgb":
            return self.vector_to_rgb(vector)
        if method == "grayscale":
            return self.vector_to_grayscale(vector)
        if method == "heatmap":
            return self.vector_to_heatmap(vector)
        if method == "gaf":
            return self.vector_to_gaf(vector)
        if method == "mtf":
            return self.vector_to_mtf(vector)
        if method in {"recurrence", "recurrence_plot"}:
            return self.vector_to_recurrence(vector)
        raise ValueError(f"Metodo no soportado: {method}")

    def save_image(self, image: np.ndarray, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        mode = "L" if image.ndim == 2 else "RGB"
        Image.fromarray(image, mode=mode).save(path)

    def generate_split(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        split: str,
        method: str = "rgb",
        max_samples: int | None = None,
    ) -> Path:
        """Genera imagenes para un split y devuelve la carpeta del split."""
        method_dir = self.save_dir / method / split
        n = len(X) if max_samples is None else min(max_samples, len(X))
        y_values = pd.Series(y).reset_index(drop=True)
        X_iter = X.reset_index(drop=True).iloc[:n]
        print(f"Generando {n:,} imagenes | metodo={method} | split={split}")
        for i, (_, row) in enumerate(X_iter.iterrows()):
            label = int(y_values.iloc[i])
            class_name = CLASS_NAMES[label]
            image = self.generate_image(row.to_numpy(dtype="float32"), method)
            self.save_image(image, method_dir / class_name / f"sample_{i:06d}.png")
            if (i + 1) % 1000 == 0:
                print(f"  {i + 1:,}/{n:,}")
        return method_dir

    def show_samples(self, method: str = "rgb", split: str = "train", n_per_class: int = 4) -> None:
        """Muestra y guarda ejemplos del dataset visual."""
        base = self.save_dir / method / split
        fig, axes = plt.subplots(2, n_per_class, figsize=(3 * n_per_class, 6))
        fig.suptitle(f"Ejemplos de imagenes - {method.upper()} ({split})")
        for row_idx, class_name in enumerate(["not_approved", "approved"]):
            files = sorted((base / class_name).glob("*.png"))[:n_per_class]
            for col_idx in range(n_per_class):
                ax = axes[row_idx, col_idx]
                ax.axis("off")
                if col_idx < len(files):
                    img = Image.open(files[col_idx])
                    ax.imshow(img, cmap="gray" if img.mode == "L" else None)
                    ax.set_title(class_name)
        Path("results").mkdir(exist_ok=True)
        plt.tight_layout()
        plt.savefig(Path("results") / f"sample_images_{method}_{split}.png", dpi=180, bbox_inches="tight")
        plt.show()


def create_image_dataset(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    y_test: pd.Series,
    image_size: tuple[int, int] = (64, 64),
    method: str = "rgb",
    save_dir: str | Path = "images",
    max_samples_per_split: int | None = None,
) -> dict[str, Path]:
    """Crea el dataset de imagenes completo para un metodo."""
    generator = ImageGenerator(image_size=image_size, save_dir=save_dir)
    paths = {
        "train": generator.generate_split(X_train, y_train, "train", method, max_samples_per_split),
        "val": generator.generate_split(X_val, y_val, "val", method, max_samples_per_split),
        "test": generator.generate_split(X_test, y_test, "test", method, max_samples_per_split),
    }
    generator.show_samples(method=method, split="train")
    print("\nEstructura creada:")
    print(f"{save_dir}/{method}/train/not_approved y approved")
    print(f"{save_dir}/{method}/val/not_approved y approved")
    print(f"{save_dir}/{method}/test/not_approved y approved")
    return paths


if __name__ == "__main__":
    from data_loading import load_dataset
    from preprocessing import preprocess_pipeline

    df = load_dataset("data/credit_train.csv")
    if df is not None:
        processed = preprocess_pipeline(df)
        create_image_dataset(
            processed["X_train"], processed["X_val"], processed["X_test"],
            processed["y_train"], processed["y_val"], processed["y_test"],
            method="rgb", max_samples_per_split=500,
        )
