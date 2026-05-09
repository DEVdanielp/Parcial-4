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

2. Abrir el notebook:

```bash
jupyter notebook main.ipynb
```

3. Ejecutar las celdas en orden.

Para una demostracion rapida, el notebook usa `MAX_SAMPLES_PER_SPLIT = 2000`. Para generar el dataset visual completo, cambiarlo a:

```python
MAX_SAMPLES_PER_SPLIT = None
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
