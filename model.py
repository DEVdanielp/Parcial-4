"""Modelos CNN para clasificacion de imagenes de prestamos."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


def _tf():
    import tensorflow as tf
    return tf


class CNNModel:
    """Construye, entrena y evalua una CNN o modelo con transfer learning."""

    def __init__(self, input_shape=(64, 64, 3), num_classes=2, model_type="custom_cnn"):
        self.tf = _tf()
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model_type = model_type
        self.base_model = None  # referencia al modelo preentrenado (solo TL/FT)
        self.model = self._build_model()
        self.history = None

    def _build_model(self):
        tf = self.tf
        layers = tf.keras.layers
        if self.model_type == "custom_cnn":
            return tf.keras.Sequential([
                layers.Input(shape=self.input_shape),
                layers.Conv2D(32, 3, padding="same", activation="relu"),
                layers.BatchNormalization(),
                layers.MaxPooling2D(),
                layers.Dropout(0.20),
                layers.Conv2D(64, 3, padding="same", activation="relu"),
                layers.BatchNormalization(),
                layers.MaxPooling2D(),
                layers.Dropout(0.25),
                layers.Conv2D(128, 3, padding="same", activation="relu"),
                layers.BatchNormalization(),
                layers.GlobalAveragePooling2D(),
                layers.Dense(128, activation="relu"),
                layers.Dropout(0.40),
                layers.Dense(self.num_classes, activation="softmax"),
            ])

        apps = {
            "mobilenet": tf.keras.applications.MobileNetV2,
            "resnet50": tf.keras.applications.ResNet50,
            "efficientnet": tf.keras.applications.EfficientNetB0,
        }
        if self.model_type not in apps:
            raise ValueError(f"Modelo no soportado: {self.model_type}")
        base = apps[self.model_type](weights="imagenet", include_top=False, input_shape=self.input_shape)
        base.trainable = False
        self.base_model = base  # guardar referencia para fine tuning
        x = tf.keras.layers.GlobalAveragePooling2D()(base.output)
        x = tf.keras.layers.Dropout(0.35)(x)
        output = tf.keras.layers.Dense(self.num_classes, activation="softmax")(x)
        return tf.keras.Model(base.input, output)

    def summary(self):
        return self.model.summary()

    def compile_model(self, learning_rate=1e-3):
        self.model.compile(
            optimizer=self.tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        print("Modelo compilado: Adam + categorical_crossentropy + accuracy")

    def _dataset_from_directory(self, directory, batch_size=32, shuffle=True):
        return self.tf.keras.utils.image_dataset_from_directory(
            directory,
            labels="inferred",
            label_mode="categorical",
            image_size=self.input_shape[:2],
            batch_size=batch_size,
            shuffle=shuffle,
            seed=42,
        ).map(lambda x, y: (x / 255.0, y))

    def train_model(self, train_dir, val_dir, epochs=15, batch_size=32, patience=5, save_path="models/best_custom_cnn.keras"):
        Path(save_path).parent.mkdir(exist_ok=True)
        train_ds = self._dataset_from_directory(train_dir, batch_size=batch_size, shuffle=True)
        val_ds = self._dataset_from_directory(val_dir, batch_size=batch_size, shuffle=False)
        callbacks = [
            self.tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=patience, restore_best_weights=True),
            self.tf.keras.callbacks.ModelCheckpoint(save_path, monitor="val_accuracy", save_best_only=True),
            self.tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=max(2, patience // 2)),
        ]
        self.history = self.model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks)
        print(f"Mejor modelo guardado en {save_path}")
        return self.history

    def evaluate_model(self, test_dir, batch_size=32):
        test_ds = self._dataset_from_directory(test_dir, batch_size=batch_size, shuffle=False)
        loss, accuracy = self.model.evaluate(test_ds, verbose=1)
        return {"loss": float(loss), "accuracy": float(accuracy)}

    def plot_training_history(self, save_path="results/training_history.png"):
        if self.history is None:
            print("No hay historial de entrenamiento.")
            return
        Path(save_path).parent.mkdir(exist_ok=True)
        hist = self.history.history
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(hist.get("accuracy", []), label="Entrenamiento")
        axes[0].plot(hist.get("val_accuracy", []), label="Validacion")
        axes[0].set_title("Accuracy")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[1].plot(hist.get("loss", []), label="Entrenamiento")
        axes[1].plot(hist.get("val_loss", []), label="Validacion")
        axes[1].set_title("Loss")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=180, bbox_inches="tight")
        plt.show()

    def fine_tune(self, unfreeze_layers=30, learning_rate=1e-5):
        """Descongela las ultimas N capas del modelo base para fine tuning.

        Debe llamarse despues de un entrenamiento inicial con transfer learning
        (base congelada). Recompila el modelo con una tasa de aprendizaje muy
        baja para ajustar los pesos del modelo preentrenado al problema actual.
        """
        if self.model_type == "custom_cnn":
            raise ValueError("Fine tuning solo aplica a modelos preentrenados (mobilenet, resnet50, efficientnet).")

        # Recuperar el modelo base: primero intentar la referencia guardada,
        # si no, buscarlo como sublayer de tipo Model dentro del modelo funcional.
        if self.base_model is not None:
            base_model = self.base_model
        else:
            base_model = next(
                (l for l in self.model.layers if isinstance(l, self.tf.keras.Model)),
                None,
            )
            if base_model is None:
                raise ValueError("No se encontro el modelo base dentro del modelo funcional.")
            self.base_model = base_model

        base_model.trainable = True

        # Congelar todo menos las ultimas N capas
        for layer in base_model.layers[:-unfreeze_layers]:
            layer.trainable = False

        n_trainable = sum(1 for l in base_model.layers if l.trainable)
        print(f"Fine tuning: {n_trainable} capas entrenable en el modelo base (lr={learning_rate})")

        self.model.compile(
            optimizer=self.tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )

    def predict_image(self, image_path):
        import numpy as np

        img = self.tf.keras.utils.load_img(image_path, target_size=self.input_shape[:2])
        arr = self.tf.keras.utils.img_to_array(img)[None, ...] / 255.0
        proba = self.model.predict(arr, verbose=0)[0]
        idx = int(np.argmax(proba))
        class_names = ["approved", "not_approved"]
        return {
            "class_index": idx,
            "class_name": class_names[idx],
            "confidence": float(proba[idx]),
            "probabilities": proba.tolist(),
        }


def train_and_evaluate_model(
    model_type="custom_cnn",
    image_size=(64, 64),
    image_root="images/rgb",
    epochs=15,
    batch_size=32,
):
    input_shape = (224, 224, 3) if model_type in {"mobilenet", "resnet50", "efficientnet"} else image_size + (3,)
    model = CNNModel(input_shape=input_shape, model_type=model_type)
    model.summary()
    model.compile_model()
    model.train_model(f"{image_root}/train", f"{image_root}/val", epochs=epochs, batch_size=batch_size)
    model.plot_training_history(f"results/training_history_{model_type}.png")
    results = model.evaluate_model(f"{image_root}/test", batch_size=batch_size)
    print(results)
    return model, results


if __name__ == "__main__":
    train_and_evaluate_model(epochs=3)
