"""Evaluacion academica del clasificador de imagenes."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


class ModelEvaluator:
    """Calcula metricas, graficas y simulaciones sobre imagenes de prueba."""

    def __init__(self, model, test_dir, input_shape=(64, 64, 3), batch_size=32):
        import tensorflow as tf

        self.tf = tf
        self.model = model
        self.test_dir = test_dir
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.class_names = None
        self.y_true = None
        self.y_pred = None
        self.y_pred_proba = None
        self._predict_test_set()

    def _predict_test_set(self):
        ds = self.tf.keras.utils.image_dataset_from_directory(
            self.test_dir,
            labels="inferred",
            label_mode="categorical",
            image_size=self.input_shape[:2],
            batch_size=self.batch_size,
            shuffle=False,
        )
        self.class_names = ds.class_names
        y_true, y_pred_proba = [], []
        for x, y in ds:
            pred = self.model.predict(x / 255.0, verbose=0)
            y_pred_proba.append(pred)
            y_true.append(np.argmax(y.numpy(), axis=1))
        self.y_true = np.concatenate(y_true)
        self.y_pred_proba = np.vstack(y_pred_proba)
        self.y_pred = np.argmax(self.y_pred_proba, axis=1)

    def calculate_metrics(self):
        metrics = {
            "accuracy": accuracy_score(self.y_true, self.y_pred),
            "precision": precision_score(self.y_true, self.y_pred, average="weighted", zero_division=0),
            "recall": recall_score(self.y_true, self.y_pred, average="weighted", zero_division=0),
            "f1_score": f1_score(self.y_true, self.y_pred, average="weighted", zero_division=0),
        }
        print("\nMetricas principales")
        for key, value in metrics.items():
            print(f"{key}: {value:.4f}")
        print("\nReporte por clase")
        print(classification_report(self.y_true, self.y_pred, target_names=self.class_names, zero_division=0))
        return metrics

    def plot_confusion_matrix(self, save_path="results/confusion_matrix.png"):
        Path(save_path).parent.mkdir(exist_ok=True)
        cm = confusion_matrix(self.y_true, self.y_pred)
        plt.figure(figsize=(7, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=self.class_names, yticklabels=self.class_names)
        plt.title("Matriz de confusion - prestamos")
        plt.xlabel("Prediccion")
        plt.ylabel("Clase real")
        plt.tight_layout()
        plt.savefig(save_path, dpi=180, bbox_inches="tight")
        plt.show()
        return cm

    def interpret_metrics_for_loan_classification(self, metrics=None):
        metrics = metrics or self.calculate_metrics()
        print("\nInterpretacion academica y de negocio")
        print(f"Accuracy indica la proporcion total de solicitudes correctamente clasificadas: {metrics['accuracy']:.1%}.")
        print(f"Precision mide cuantas aprobaciones/rechazos predichos son confiables: {metrics['precision']:.1%}.")
        print(f"Recall mide cuantas solicitudes de cada clase logra recuperar el modelo: {metrics['recall']:.1%}.")
        print(f"F1-score resume el equilibrio entre precision y recall: {metrics['f1_score']:.1%}.")
        print("En prestamos, los falsos positivos son delicados: aprobar clientes que terminan en incumplimiento puede generar perdida financiera.")

    def simulate_user_testing(self, n_tests=5):
        rng = np.random.default_rng(42)
        indices = rng.choice(len(self.y_true), size=min(n_tests, len(self.y_true)), replace=False)
        print("\nPruebas simuladas con registros de test")
        for i, idx in enumerate(indices, 1):
            pred_idx = int(self.y_pred[idx])
            true_idx = int(self.y_true[idx])
            confidence = float(self.y_pred_proba[idx, pred_idx])
            print(f"Prueba {i}: prediccion={self.class_names[pred_idx]} | confianza={confidence:.1%} | real={self.class_names[true_idx]}")

    def run_complete_evaluation(self):
        metrics = self.calculate_metrics()
        self.plot_confusion_matrix()
        self.interpret_metrics_for_loan_classification(metrics)
        self.simulate_user_testing()
        return metrics


def evaluate_model_performance(model, test_dir, input_shape=(64, 64, 3), batch_size=32):
    evaluator = ModelEvaluator(model, test_dir, input_shape, batch_size)
    return evaluator.run_complete_evaluation()
