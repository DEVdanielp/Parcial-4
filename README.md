# Clasificacion de prestamos bancarios usando imagenes generadas desde datos tabulares

Proyecto universitario de ciencia de datos y machine learning para clasificar prestamos bancarios mediante representaciones visuales de registros tabulares.

## Objetivo

Convertir cada solicitud de prestamo en una imagen y entrenar una red neuronal convolucional (CNN) para predecir la clase del prestamo:

- `approved`: prestamo exitoso, equivalente a `Fully Paid`.
- `not_approved`: prestamo problematico, equivalente a `Charged Off`.

## Dataset

La carpeta `data/` contiene:

- `credit_train.csv`: dataset supervisado con `Loan Status`.
- `credit_test.csv`: datos sin etiqueta para simular usuarios nuevos.

Columnas principales:

- Identificadores excluidos del modelo: `Loan ID`, `Customer ID`.
- Variable objetivo: `Loan Status`.
- Numericas: `Current Loan Amount`, `Credit Score`, `Annual Income`, `Monthly Debt`, `Years of Credit History`, entre otras.
- Categoricas: `Term`, `Years in current job`, `Home Ownership`, `Purpose`.

## Estructura

- `main.ipynb`: notebook principal con explicacion academica paso a paso.
- `data_loading.py`: carga, EDA, nulos, duplicados, outliers y graficas.
- `preprocessing.py`: limpieza, target binario, imputacion, escalado, one-hot controlado y split estratificado.
- `image_generation.py`: conversion tabular a imagenes con RGB, grayscale, heatmap, GAF, MTF y recurrence plots.
- `model.py`: CNN basica y soporte para MobileNet, ResNet50 y EfficientNet.
- `evaluation.py`: accuracy, precision, recall, F1-score, matriz de confusion e interpretacion.
- `requirements.txt`: dependencias.

## Como ejecutar

1. Instalar dependencias:

```bash
pip install -r requirements.txt
```

2. Abrir el notebook (Puntos 1 y 2):

```bash
jupyter notebook main.ipynb
```

3. Ejecutar las celdas en orden.

Para una demostracion rapida, el notebook usa `MAX_SAMPLES_PER_SPLIT = 2000`. Para generar el dataset visual completo, cambiarlo a:

```python
MAX_SAMPLES_PER_SPLIT = None
```

4. Ejecutar los Puntos 3 y 4 (CNN + metricas) directamente en Python:

```bash
python punto3_punto4.py
```

## Pipeline

1. Carga de `credit_train.csv`.
2. Analisis exploratorio: tipos, nulos, duplicados, outliers, distribuciones y correlaciones.
3. Limpieza y preprocesamiento:
   - eliminacion de duplicados;
   - exclusion de IDs;
   - `Loan Status` a binario;
   - imputacion de nulos;
   - escalado numerico;
   - one-hot de categoricas;
   - split train/val/test.
4. Generacion de imagenes:
   - `images/<metodo>/train/<clase>/`;
   - `images/<metodo>/val/<clase>/`;
   - `images/<metodo>/test/<clase>/`.
5. Entrenamiento CNN con `Adam` y `categorical_crossentropy`.
6. Evaluacion con metricas y matriz de confusion.
7. Simulacion de predicciones con `credit_test.csv`.

---

## Punto 3: Estrategias de aprendizaje con redes convolucionales

Implementado en `punto3_punto4.py`. Se comparan tres estrategias sobre el dataset de imagenes RGB generado en el Punto 2:

### Estrategia 1 – CNN desde cero

Se disena y entrena completamente una red convolucional sin usar pesos previos. La arquitectura incluye tres bloques Conv2D + BatchNorm + MaxPooling + Dropout, seguidos de GlobalAveragePooling y capas densas. Todos los pesos se inicializan aleatoriamente y se ajustan con el dataset de imagenes del problema.

- Ventaja: control total sobre la arquitectura.
- Desventaja: requiere mas datos y mas epocas para converger.

### Estrategia 2 – Transfer Learning

Se carga MobileNetV2 preentrenado en ImageNet. La base convolucional se **congela** completamente (`trainable = False`) y solo se entrena el clasificador agregado encima (GlobalAveragePooling + Dropout + Dense softmax). Los pesos de ImageNet se usan directamente sin modificarse.

- Ventaja: converge rapido con pocos datos.
- Desventaja: los pesos no se adaptan al dominio especifico del problema.

### Estrategia 3 – Fine Tuning

Partiendo del modelo de Transfer Learning ya entrenado, se **descongela las ultimas 30 capas** del modelo base y se reentrena todo el modelo con una tasa de aprendizaje muy baja (`lr = 1e-5`). Esto permite ajustar los pesos del modelo preentrenado al problema de clasificacion de prestamos.

- Ventaja: combina el conocimiento de ImageNet con adaptacion al dominio.
- Desventaja: puede producir sobreajuste si no se regula bien.

### Comparacion de resultados

El script genera automaticamente:

| Archivo | Contenido |
|---|---|
| `results/curvas_cnn_scratch.png` | Curvas accuracy/loss — CNN desde cero |
| `results/curvas_transfer_learning.png` | Curvas accuracy/loss — Transfer Learning |
| `results/curvas_fine_tuning.png` | Curvas accuracy/loss — Fine Tuning |
| `results/confusion_cnn_desde_cero.png` | Matriz de confusion — CNN desde cero |
| `results/confusion_transfer_learning.png` | Matriz de confusion — Transfer Learning |
| `results/confusion_fine_tuning.png` | Matriz de confusion — Fine Tuning |
| `results/comparacion_estrategias.png` | Grafico comparativo de las 3 estrategias |
| `results/comparacion_estrategias.csv` | Tabla con accuracy, precision, recall, F1 |

---

## Punto 4: Metricas de desempenio

Para cada estrategia se calculan las siguientes metricas sobre el conjunto de test:

| Metrica | Descripcion |
|---|---|
| **Accuracy** | Proporcion total de clasificaciones correctas |
| **Precision** | Confiabilidad de las predicciones positivas |
| **Recall** | Capacidad de recuperar instancias de cada clase |
| **F1-Score** | Media armonica de precision y recall |
| **Matriz de confusion** | Distribucion de TP, TN, FP, FN por clase |

La interpretacion financiera es clave: los **falsos positivos** (aprobar un cliente que incumplira) representan perdida financiera directa, por lo que el recall de la clase `not_approved` es critico.

---

## Nota sobre el error de memoria

El error:

```text
MemoryError: Unable to allocate ... shape (164026, 89786)
```

ocurria porque columnas de alta cardinalidad como `Loan ID` o `Customer ID` podian llegar a codificarse con one-hot. El pipeline actual las detecta y excluye antes de transformar datos. En el dataset actual, el preprocesamiento produce alrededor de 45 features, no decenas de miles.

## Ventajas

- Permite aplicar CNN a datos tabulares mediante representacion visual.
- Incluye varias tecnicas: RGB, heatmap, GAF, MTF y recurrence plots.
- Produce un dataset de imagenes organizado por clases.
- Es adecuado para presentacion academica porque muestra graficas, imagenes, entrenamiento y metricas.

## Limitaciones

- La transformacion de tabla a imagen puede perder informacion.
- Modelos tabulares tradicionales pueden ser mas eficientes.
- La interpretacion financiera requiere cuidado; una CNN sobre imagenes no sustituye analisis de riesgo formal.
- Generar todas las imagenes puede consumir espacio y tiempo.

## Mejoras futuras

1. Comparar rendimiento entre `rgb`, `gaf`, `mtf`, `heatmap` y `recurrence`.
2. Agregar un baseline tabular con Random Forest, XGBoost o Logistic Regression.
3. Balancear clases si el recall de `not_approved` es bajo.
4. Usar Grad-CAM para explicar zonas de la imagen relevantes.
5. Usar SHAP sobre las variables originales para interpretabilidad bancaria.
