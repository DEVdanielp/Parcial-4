"""
Punto 3: Estrategias de reconocimiento con redes convolucionales
Punto 4: Metricas de desempenio y prueba con usuarios nuevos

Estrategias comparadas sobre el dataset de imagenes RGB generado en el Punto 2:
  1. CNN desde cero     (custom_cnn)
  2. Transfer Learning  (MobileNetV2 con base congelada)
  3. Fine Tuning        (MobileNetV2 con ultimas capas descongeladas)

Ejecucion:
    python punto3_punto4.py
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# ─── Modulos del proyecto ─────────────────────────────────────────────────────
from model import CNNModel
from evaluation import ModelEvaluator

# ─── Rutas ────────────────────────────────────────────────────────────────────
IMAGE_ROOT  = Path("images/rgb")
TRAIN_DIR   = IMAGE_ROOT / "train"
VAL_DIR     = IMAGE_ROOT / "val"
TEST_DIR    = IMAGE_ROOT / "test"
MODELS_DIR  = Path("models")
RESULTS_DIR = Path("results")

for folder in [MODELS_DIR, RESULTS_DIR]:
    folder.mkdir(exist_ok=True)

# ─── Hiper-parametros ─────────────────────────────────────────────────────────
EPOCHS_TL        = 20   # epocas transfer learning (fase 1)
EPOCHS_FT        = 10   # epocas adicionales fine tuning (fase 2)
EPOCHS_CNN       = 25   # epocas CNN desde cero
BATCH_SIZE       = 32
PATIENCE         = 5
IMG_SIZE_CNN     = (64, 64)
IMG_SIZE_PRETRAINED = (96, 96)   # balance velocidad / precision
NUM_CLASSES      = 2
UNFREEZE_LAYERS  = 30            # capas a descongelar en fine tuning
LR_FT            = 1e-5          # tasa de aprendizaje para fine tuning

CLASS_NAMES = ["approved", "not_approved"]


# =============================================================================
# ESTRATEGIA 1 – CNN DESDE CERO
# =============================================================================

def entrenar_cnn_scratch() -> tuple[CNNModel, dict]:
    """Construye y entrena una CNN completamente desde cero."""
    print("\n" + "=" * 65)
    print("ESTRATEGIA 1: CNN DESDE CERO")
    print("=" * 65)

    input_shape = IMG_SIZE_CNN + (3,)
    model = CNNModel(input_shape=input_shape, num_classes=NUM_CLASSES, model_type="custom_cnn")
    model.summary()
    model.compile_model(learning_rate=1e-3)
    model.train_model(
        train_dir=str(TRAIN_DIR),
        val_dir=str(VAL_DIR),
        epochs=EPOCHS_CNN,
        batch_size=BATCH_SIZE,
        patience=PATIENCE,
        save_path=str(MODELS_DIR / "cnn_scratch.keras"),
    )
    model.plot_training_history(save_path=str(RESULTS_DIR / "curvas_cnn_scratch.png"))

    results = model.evaluate_model(str(TEST_DIR), batch_size=BATCH_SIZE)
    print(f"\nResultados en test — CNN desde cero: {results}")
    return model, results


# =============================================================================
# ESTRATEGIA 2 – TRANSFER LEARNING (base congelada)
# =============================================================================

def entrenar_transfer_learning() -> tuple[CNNModel, dict]:
    """Carga MobileNetV2 preentrenado en ImageNet y entrena solo el clasificador."""
    print("\n" + "=" * 65)
    print("ESTRATEGIA 2: TRANSFER LEARNING (MobileNetV2 – base congelada)")
    print("=" * 65)

    input_shape = IMG_SIZE_PRETRAINED + (3,)
    model = CNNModel(input_shape=input_shape, num_classes=NUM_CLASSES, model_type="mobilenet")
    model.summary()
    model.compile_model(learning_rate=1e-3)
    model.train_model(
        train_dir=str(TRAIN_DIR),
        val_dir=str(VAL_DIR),
        epochs=EPOCHS_TL,
        batch_size=BATCH_SIZE,
        patience=PATIENCE,
        save_path=str(MODELS_DIR / "transfer_learning_mobilenet.keras"),
    )
    model.plot_training_history(save_path=str(RESULTS_DIR / "curvas_transfer_learning.png"))

    results = model.evaluate_model(str(TEST_DIR), batch_size=BATCH_SIZE)
    print(f"\nResultados en test — Transfer Learning: {results}")
    return model, results


# =============================================================================
# ESTRATEGIA 3 – FINE TUNING (descongelar ultimas capas)
# =============================================================================

def entrenar_fine_tuning(modelo_tl: CNNModel) -> tuple[CNNModel, dict]:
    """Partiendo del modelo TL entrenado, descongela capas y reajusta pesos."""
    print("\n" + "=" * 65)
    print("ESTRATEGIA 3: FINE TUNING (MobileNetV2 – capas descongeladas)")
    print("=" * 65)

    # Descongelar ultimas capas del modelo base y recompilar con lr muy bajo
    modelo_tl.fine_tune(unfreeze_layers=UNFREEZE_LAYERS, learning_rate=LR_FT)

    modelo_tl.train_model(
        train_dir=str(TRAIN_DIR),
        val_dir=str(VAL_DIR),
        epochs=EPOCHS_FT,
        batch_size=BATCH_SIZE,
        patience=PATIENCE,
        save_path=str(MODELS_DIR / "fine_tuning_mobilenet.keras"),
    )
    modelo_tl.plot_training_history(save_path=str(RESULTS_DIR / "curvas_fine_tuning.png"))

    results = modelo_tl.evaluate_model(str(TEST_DIR), batch_size=BATCH_SIZE)
    print(f"\nResultados en test — Fine Tuning: {results}")
    return modelo_tl, results


# =============================================================================
# PUNTO 4 – METRICAS DE DESEMPENIO
# =============================================================================

def evaluar_estrategia(nombre: str, modelo, input_shape: tuple) -> dict:
    """Calcula metricas completas para un modelo sobre el conjunto de test."""
    print(f"\n{'─' * 65}")
    print(f"METRICAS: {nombre}")
    print(f"{'─' * 65}")

    evaluator = ModelEvaluator(
        model=modelo.model,
        test_dir=str(TEST_DIR),
        input_shape=input_shape,
        batch_size=BATCH_SIZE,
    )
    metrics = evaluator.calculate_metrics()
    evaluator.plot_confusion_matrix(
        save_path=str(RESULTS_DIR / f"confusion_{nombre.lower().replace(' ', '_')}.png")
    )
    evaluator.interpret_metrics_for_loan_classification(metrics)
    evaluator.simulate_user_testing(n_tests=8)

    metrics["estrategia"] = nombre
    return metrics


def comparar_estrategias(resultados: list[dict]) -> pd.DataFrame:
    """Genera tabla comparativa y grafico de barras con las metricas."""
    cols = ["estrategia", "accuracy", "precision", "recall", "f1_score"]
    df = pd.DataFrame(resultados)[cols].set_index("estrategia")

    print("\n" + "=" * 65)
    print("TABLA COMPARATIVA DE ESTRATEGIAS")
    print("=" * 65)
    print(df.to_string(float_format=lambda x: f"{x:.4f}"))

    # Grafico de barras agrupadas
    fig, ax = plt.subplots(figsize=(11, 5))
    metricas = ["accuracy", "precision", "recall", "f1_score"]
    etiquetas = ["Accuracy", "Precision", "Recall", "F1-Score"]
    x = np.arange(len(metricas))
    width = 0.25
    colors = ["#4C72B0", "#55A868", "#C44E52"]

    for i, (idx, row) in enumerate(df.iterrows()):
        valores = [row[m] for m in metricas]
        bars = ax.bar(x + i * width, valores, width, label=idx, color=colors[i], alpha=0.85)
        for bar, val in zip(bars, valores):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{val:.3f}",
                ha="center", va="bottom", fontsize=8,
            )

    ax.set_xticks(x + width)
    ax.set_xticklabels(etiquetas)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Valor de la metrica")
    ax.set_title("Comparacion de estrategias CNN – Clasificacion de prestamos")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(RESULTS_DIR / "comparacion_estrategias.png"), dpi=180, bbox_inches="tight")
    plt.show()
    print(f"\nGrafico guardado en {RESULTS_DIR / 'comparacion_estrategias.png'}")

    df.to_csv(str(RESULTS_DIR / "comparacion_estrategias.csv"))
    return df


# =============================================================================
# PRUEBA CON USUARIOS NUEVOS (credit_test.csv)
# =============================================================================

def probar_con_usuarios_nuevos(modelo_ganador, input_shape: tuple, nombre_modelo: str):
    """
    Genera imagenes de los usuarios nuevos (credit_test.csv) y predice
    si su prestamo seria aprobado o rechazado usando el mejor modelo.
    """
    import sys
    from pathlib import Path

    print("\n" + "=" * 65)
    print(f"PRUEBA CON USUARIOS NUEVOS — modelo: {nombre_modelo}")
    print("=" * 65)

    new_users_path = Path("data/credit_test.csv")
    if not new_users_path.exists():
        print("No se encontro data/credit_test.csv, omitiendo prueba con usuarios nuevos.")
        return

    # Importar el pipeline de preprocesamiento e imagen
    import importlib
    import preprocessing
    import image_generation

    importlib.reload(preprocessing)
    importlib.reload(image_generation)

    from preprocessing import transform_new_data
    from image_generation import ImageGenerator

    # Cargar artefactos del preprocesamiento (scaler, encoder, etc.)
    artifacts_path = Path("models/artifacts.joblib")
    if not artifacts_path.exists():
        print("No se encontraron artefactos de preprocesamiento (models/artifacts.joblib).")
        print("Ejecuta el notebook principal para generarlos antes de probar usuarios nuevos.")
        return

    try:
        df_new = pd.read_csv(new_users_path)
        X_new = transform_new_data(df_new, artifacts_path=str(artifacts_path))
    except Exception as exc:
        print(f"Error al preprocesar usuarios nuevos: {exc}")
        return

    gen = ImageGenerator(image_size=input_shape[:2], method="rgb")
    print(f"\nGenerando imagenes para {min(10, len(X_new))} usuarios de prueba...")

    tf = modelo_ganador.tf
    resultados_nuevos = []

    for i, row in enumerate(X_new[:10]):
        img_array = gen.generate_rgb_image(row)
        img_tensor = tf.expand_dims(
            tf.image.resize(img_array[..., :3], input_shape[:2]) / 255.0,
            axis=0,
        )
        proba = modelo_ganador.model.predict(img_tensor, verbose=0)[0]
        idx = int(np.argmax(proba))
        resultados_nuevos.append({
            "usuario": i + 1,
            "prediccion": CLASS_NAMES[idx],
            "confianza": f"{proba[idx]:.1%}",
            "prob_approved": f"{proba[0]:.3f}",
            "prob_not_approved": f"{proba[1]:.3f}",
        })
        estado = "APROBADO" if idx == 0 else "RECHAZADO"
        print(f"  Usuario {i+1:>2}: {estado} | confianza={proba[idx]:.1%}")

    df_resultados = pd.DataFrame(resultados_nuevos)
    out_path = RESULTS_DIR / "predicciones_usuarios_nuevos.csv"
    df_resultados.to_csv(str(out_path), index=False)
    print(f"\nPredicciones guardadas en {out_path}")


# =============================================================================
# PIPELINE PRINCIPAL
# =============================================================================

def main():
    print("\n" + "#" * 65)
    print("# PUNTO 3 & 4 – CNN para clasificacion de prestamos bancarios  #")
    print("#" * 65)
    print(f"  Dataset de imagenes : {IMAGE_ROOT}")
    print(f"  Train : {sum(1 for _ in TRAIN_DIR.rglob('*.png'))} imagenes")
    print(f"  Val   : {sum(1 for _ in VAL_DIR.rglob('*.png'))} imagenes")
    print(f"  Test  : {sum(1 for _ in TEST_DIR.rglob('*.png'))} imagenes")
    print()

    # ── Entrenar las 3 estrategias ────────────────────────────────────────────
    modelo_scratch, _ = entrenar_cnn_scratch()
    modelo_tl, _      = entrenar_transfer_learning()
    modelo_ft, _      = entrenar_fine_tuning(modelo_tl)

    # ── Punto 4: metricas completas por estrategia ────────────────────────────
    metricas_scratch = evaluar_estrategia(
        "CNN desde cero", modelo_scratch, IMG_SIZE_CNN + (3,)
    )
    metricas_tl = evaluar_estrategia(
        "Transfer Learning", modelo_ft,  # modelo_ft guarda la historia TL + FT
        IMG_SIZE_PRETRAINED + (3,),
    )
    # Cargar modelo TL original guardado para evaluarlo por separado
    import tensorflow as tf

    modelo_tl_solo = CNNModel(
        input_shape=IMG_SIZE_PRETRAINED + (3,),
        num_classes=NUM_CLASSES,
        model_type="mobilenet",
    )
    modelo_tl_solo.model = tf.keras.models.load_model(
        str(MODELS_DIR / "transfer_learning_mobilenet.keras")
    )
    metricas_tl_solo = evaluar_estrategia(
        "Transfer Learning", modelo_tl_solo, IMG_SIZE_PRETRAINED + (3,)
    )

    metricas_ft = evaluar_estrategia(
        "Fine Tuning", modelo_ft, IMG_SIZE_PRETRAINED + (3,)
    )

    # ── Tabla comparativa ─────────────────────────────────────────────────────
    df_comparacion = comparar_estrategias([
        metricas_scratch,
        metricas_tl_solo,
        metricas_ft,
    ])

    # ── Determinar mejor modelo ───────────────────────────────────────────────
    mejor = df_comparacion["f1_score"].idxmax()
    print(f"\nMejor estrategia segun F1-Score: {mejor}")

    # ── Punto 5: prueba con usuarios nuevos usando el mejor modelo ────────────
    if mejor == "CNN desde cero":
        mejor_modelo, mejor_shape = modelo_scratch, IMG_SIZE_CNN + (3,)
    elif mejor == "Transfer Learning":
        mejor_modelo, mejor_shape = modelo_tl_solo, IMG_SIZE_PRETRAINED + (3,)
    else:
        mejor_modelo, mejor_shape = modelo_ft, IMG_SIZE_PRETRAINED + (3,)

    probar_con_usuarios_nuevos(mejor_modelo, mejor_shape, mejor)

    print("\n" + "=" * 65)
    print("EJECUCION COMPLETADA")
    print(f"  Modelos guardados en : {MODELS_DIR}/")
    print(f"  Resultados en        : {RESULTS_DIR}/")
    print("=" * 65)


if __name__ == "__main__":
    main()
