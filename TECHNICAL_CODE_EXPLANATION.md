# Explicacion tecnica del codigo

Este documento explica como funciona internamente el proyecto a nivel de codigo. El README resume el proyecto para uso general; este archivo se enfoca en la arquitectura tecnica, los modulos, las funciones principales y el flujo de datos.

## Vision general del flujo

El sistema sigue este pipeline:

1. Cargar el dataset tabular.
2. Analizar columnas, tipos de datos, nulos, duplicados y outliers.
3. Limpiar datos y transformar variables.
4. Dividir el dataset en entrenamiento, validacion y prueba.
5. Convertir cada registro tabular en una imagen.
6. Guardar las imagenes organizadas por clase.
7. Entrenar una CNN con las imagenes.
8. Evaluar el modelo con metricas de clasificacion.
9. Simular predicciones con registros nuevos.

Los archivos principales son:

- `data_loading.py`
- `preprocessing.py`
- `image_generation.py`
- `model.py`
- `evaluation.py`
- `main.ipynb`

## 1. `data_loading.py`

Este modulo se encarga de cargar y analizar el dataset original.

### Funcion `load_dataset(filepath)`

Lee un archivo CSV con `pandas.read_csv`.

Entrada:

```python
filepath: str | Path
```

Salida:

```python
pd.DataFrame | None
```

Comportamiento tecnico:

- Elimina filas completamente vacias con `dropna(how="all")`.
- Limpia espacios en nombres de columnas.
- Maneja errores si el archivo no existe.

### Funcion `infer_target_column(df)`

Detecta la variable objetivo.

Para este dataset, prioriza:

```python
Loan Status
```

La funcion evita confundir columnas identificadoras como `Loan ID` con la variable objetivo.

### Funcion `get_id_columns(df)`

Detecta columnas identificadoras que no deben entrar al modelo.

En este proyecto normalmente devuelve:

```python
["Loan ID", "Customer ID"]
```

Estas columnas se excluyen porque:

- No generalizan a nuevos usuarios.
- Pueden causar fuga de informacion.
- Tienen alta cardinalidad.
- Si se codifican con one-hot, pueden generar miles de columnas y errores de memoria.

### Funcion `identify_variable_types(df, target_col, exclude_cols)`

Separa columnas en:

- numericas;
- categoricas.

Excluye:

- variable objetivo;
- columnas ID.

### Funcion `analyze_dataset(df)`

Imprime informacion academica y tecnica del dataset:

- dimensiones;
- tipos de datos;
- columnas ID;
- variable objetivo;
- variables numericas;
- variables categoricas;
- distribucion de clases.

### Funciones de diagnostico

`check_missing_values(df)`:

- calcula nulos por columna;
- calcula porcentaje de nulos.

`check_duplicates(df)`:

- cuenta duplicados exactos.

`detect_outliers(df, numeric_cols)`:

- usa IQR por defecto;
- cuenta valores por fuera del rango:

```python
[Q1 - 1.5 * IQR, Q3 + 1.5 * IQR]
```

### Funciones de graficacion

`plot_distributions(df, numeric_cols, target_col)`:

- genera histogramas;
- colorea por clase si existe target.

`plot_correlations(df, numeric_cols)`:

- calcula matriz de correlacion;
- guarda un heatmap en `results/correlation_matrix.png`.

## 2. `preprocessing.py`

Este modulo transforma el dataset tabular en una matriz numerica lista para generar imagenes.

## Variable objetivo

La columna `Loan Status` se convierte asi:

```python
Charged Off -> 0 -> not_approved
Fully Paid  -> 1 -> approved
```

Esto se define en:

```python
TARGET_MAPPING = {
    "Charged Off": 0,
    "Fully Paid": 1,
    "not_approved": 0,
    "approved": 1,
    0: 0,
    1: 1,
}
```

## Clase `PreprocessArtifacts`

Es un `dataclass` que guarda los objetos necesarios para transformar datos nuevos de la misma manera que los datos de entrenamiento.

Contiene:

- `preprocessor`: transformador de scikit-learn ya entrenado;
- `feature_names`: nombres finales despues de one-hot;
- `numeric_cols`: columnas numericas originales;
- `categorical_cols`: columnas categoricas originales;
- `target_col`: nombre del target;
- `id_cols`: columnas excluidas;
- `class_names`: nombres de clases.

Esto es importante porque los datos nuevos deben pasar por exactamente las mismas transformaciones.

## Funcion `clean_dataset(df, target_col)`

Realiza limpieza basica:

- elimina filas completamente vacias;
- elimina duplicados exactos;
- elimina filas sin etiqueta si existe target.

## Funcion `cap_outliers(df, numeric_cols)`

Aplica winsorizacion con IQR.

En lugar de eliminar filas con outliers, limita los valores extremos al rango aceptable.

Ventaja:

- conserva registros;
- reduce impacto de valores extremos.

Limitacion:

- puede modificar informacion real si los extremos son validos.

## Funcion `build_preprocessor(numeric_cols, categorical_cols)`

Crea un `ColumnTransformer` con dos pipelines:

Pipeline numerico:

```python
SimpleImputer(strategy="median")
StandardScaler()
```

Pipeline categorico:

```python
SimpleImputer(strategy="most_frequent")
OneHotEncoder(handle_unknown="ignore", sparse_output=False)
```

`handle_unknown="ignore"` es clave porque permite transformar categorias nuevas en datos de prueba sin romper el pipeline.

## Funcion `preprocess_pipeline(...)`

Es la funcion principal de preprocesamiento.

Pasos internos:

1. Detecta target si no fue entregado.
2. Detecta columnas ID.
3. Limpia el dataset.
4. Convierte `Loan Status` a 0/1.
5. Excluye IDs y target de las features.
6. Aplica capping de outliers.
7. Divide en train, validation y test con `train_test_split`.
8. Ajusta el preprocessor solo con entrenamiento.
9. Transforma validation y test usando el mismo preprocessor.
10. Devuelve matrices finales y artefactos.

La division es estratificada:

```python
stratify=y
```

Esto mantiene la proporcion de clases en cada particion.

Salida principal:

```python
{
    "X_train": ...,
    "X_val": ...,
    "X_test": ...,
    "y_train": ...,
    "y_val": ...,
    "y_test": ...,
    "artifacts": ...,
    "feature_names": ...,
    "df_processed": ...,
}
```

## Funcion `transform_new_data(df_new, artifacts)`

Transforma registros nuevos sin etiqueta.

Se usa con `credit_test.csv`.

Pasos:

- elimina IDs;
- reordena columnas segun entrenamiento;
- aplica capping;
- usa el `preprocessor` ya ajustado;
- devuelve matriz numerica lista para imagenes.

## Funciones `save_artifacts` y `load_artifacts`

Guardan y cargan los artefactos con `joblib`.

Archivo por defecto:

```text
models/preprocess_artifacts.joblib
```

## 3. `image_generation.py`

Este modulo convierte vectores numericos en imagenes.

Entrada general:

```python
X_train, X_val, X_test
```

Cada fila de `X` es un vector numerico preprocesado.

Salida general:

```text
images/<metodo>/<split>/<clase>/sample_000001.png
```

Ejemplo:

```text
images/rgb/train/approved/sample_000001.png
images/rgb/train/not_approved/sample_000002.png
```

## Funcion auxiliar `_safe_minmax(vector)`

Normaliza un vector a un rango definido.

Por defecto:

```python
[0, 1]
```

Tambien maneja:

- valores nulos;
- infinitos;
- vectores constantes.

Esto evita divisiones por cero.

## Clase `ImageGenerator`

Es la clase central para crear imagenes.

### Metodo `vector_to_grayscale(vector)`

Convierte el vector en una imagen de escala de grises.

Proceso:

1. Normaliza valores a `[0, 255]`.
2. Repite o recorta el vector para llenar la imagen.
3. Reorganiza el vector como matriz 2D.

### Metodo `vector_to_rgb(vector)`

Crea una imagen RGB usando tres canales derivados del mismo vector.

Canales:

- canal 1: imagen base;
- canal 2: imagen desplazada verticalmente;
- canal 3: imagen desplazada horizontalmente.

Esto genera una representacion simple pero compatible con CNNs que esperan 3 canales.

### Metodo `vector_to_heatmap(vector)`

Usa un mapa de color (`viridis`) para convertir intensidades en una imagen RGB.

Es util para visualizacion academica porque muestra variaciones de magnitud de forma intuitiva.

### Metodo `vector_to_gaf(vector)`

Implementa Gramian Angular Field.

Proceso:

1. Normaliza el vector a `[-1, 1]`.
2. Convierte cada valor a angulo:

```python
phi = arccos(valor)
```

3. Calcula una matriz de relaciones angulares:

```python
cos(phi_i + phi_j)
```

Interpretacion:

- cada pixel representa una relacion entre dos posiciones del vector.

### Metodo `vector_to_mtf(vector)`

Implementa Markov Transition Field.

Proceso:

1. Normaliza el vector.
2. Divide los valores en bins.
3. Calcula transiciones entre estados consecutivos.
4. Expande esas transiciones a una matriz visual.

Interpretacion:

- representa patrones de cambio entre rangos de valores.

### Metodo `vector_to_recurrence(vector)`

Implementa Recurrence Plot.

Proceso:

1. Normaliza el vector.
2. Calcula distancia absoluta entre pares de posiciones.
3. Convierte distancia en similitud visual.

Interpretacion:

- valores similares aparecen como zonas mas intensas.

## Metodo `generate_split(X, y, split, method)`

Genera imagenes para una particion:

- `train`;
- `val`;
- `test`.

Por cada fila:

1. Convierte la fila en `np.ndarray`.
2. Lee su etiqueta.
3. Genera imagen segun metodo.
4. Guarda PNG en carpeta de clase.

## Funcion `create_image_dataset(...)`

Orquesta la generacion completa.

Parametros importantes:

```python
method="rgb"
image_size=(64, 64)
max_samples_per_split=2000
```

`max_samples_per_split` permite hacer demos rapidas sin generar todo el dataset.

Para generar todo:

```python
max_samples_per_split=None
```

## 4. `model.py`

Este modulo define y entrena modelos CNN.

## Clase `CNNModel`

Recibe:

```python
input_shape=(64, 64, 3)
num_classes=2
model_type="custom_cnn"
```

Modelos soportados:

- `custom_cnn`;
- `mobilenet`;
- `resnet50`;
- `efficientnet`.

## Arquitectura `custom_cnn`

La CNN personalizada usa:

```python
Conv2D
BatchNormalization
MaxPooling2D
Dropout
GlobalAveragePooling2D
Dense
Softmax
```

Flujo:

1. `Conv2D`: aprende filtros locales sobre la imagen.
2. `BatchNormalization`: estabiliza distribuciones internas.
3. `MaxPooling2D`: reduce resolucion espacial.
4. `Dropout`: reduce sobreajuste.
5. `GlobalAveragePooling2D`: resume mapas de activacion.
6. `Dense`: combina caracteristicas.
7. `Softmax`: genera probabilidades por clase.

## Metodo `compile_model`

Compila con:

```python
optimizer = Adam
loss = categorical_crossentropy
metrics = ["accuracy"]
```

`categorical_crossentropy` se usa porque las etiquetas de imagen se cargan en formato one-hot por Keras.

## Metodo `_dataset_from_directory`

Carga imagenes desde carpetas con:

```python
tf.keras.utils.image_dataset_from_directory
```

Keras infiere las clases desde los nombres de carpetas.

Ejemplo:

```text
approved
not_approved
```

Tambien normaliza pixeles:

```python
x / 255.0
```

## Metodo `train_model`

Entrena el modelo con:

- dataset de entrenamiento;
- dataset de validacion;
- early stopping;
- model checkpoint;
- reduccion de learning rate.

Callbacks:

```python
EarlyStopping
ModelCheckpoint
ReduceLROnPlateau
```

Esto evita entrenar de mas y guarda el mejor modelo.

## Metodo `plot_training_history`

Grafica:

- accuracy de entrenamiento y validacion;
- loss de entrenamiento y validacion.

Guarda la figura en `results/`.

## Metodo `predict_image`

Recibe una imagen individual y devuelve:

```python
{
    "class_index": ...,
    "class_name": ...,
    "confidence": ...,
    "probabilities": ...
}
```

Se usa para simular predicciones de usuarios nuevos.

## 5. `evaluation.py`

Este modulo evalua el modelo entrenado.

## Clase `ModelEvaluator`

Recibe:

```python
model
test_dir
input_shape
batch_size
```

Carga las imagenes de prueba, obtiene predicciones y calcula metricas.

## Metodo `calculate_metrics`

Calcula:

```python
accuracy
precision
recall
f1_score
classification_report
```

Usa ponderacion:

```python
average="weighted"
```

Esto considera el desbalance de clases.

## Metodo `plot_confusion_matrix`

Genera una matriz de confusion.

Interpretacion:

- diagonal principal: aciertos;
- fuera de diagonal: errores;
- falsos positivos: casos predichos como aprobados cuando no lo eran;
- falsos negativos: casos rechazados por el modelo aunque eran aprobados.

## Metodo `interpret_metrics_for_loan_classification`

Imprime una explicacion de las metricas en contexto bancario.

Punto clave:

En prestamos, los falsos positivos son especialmente importantes porque aprobar un prestamo riesgoso puede generar perdida financiera.

## Metodo `simulate_user_testing`

Selecciona ejemplos aleatorios del test set y muestra:

- prediccion;
- confianza;
- clase real.

Esto sirve como demostracion funcional.

## 6. `main.ipynb`

El notebook integra todos los modulos.

Orden recomendado:

1. Importar librerias y recargar modulos locales.
2. Cargar dataset.
3. Analizar dataset.
4. Preprocesar.
5. Generar imagenes.
6. Entrenar CNN.
7. Evaluar.
8. Probar usuarios nuevos.
9. Leer conclusiones.

La primera celda recarga modulos con:

```python
importlib.reload(...)
```

Esto evita que Jupyter use versiones antiguas de los archivos `.py`.

## 7. Como se evita el error de memoria

El error original ocurria porque se estaba creando una matriz enorme:

```text
shape (164026, 89786)
```

La causa mas probable era codificar identificadores de alta cardinalidad con one-hot.

Ejemplo:

```python
Loan ID
Customer ID
```

Si cada ID es unico, one-hot crea una columna por ID.

Solucion implementada:

1. Detectar IDs con `get_id_columns`.
2. Excluirlos antes de codificar.
3. Codificar solo variables categoricas reales.

Resultado esperado:

```text
Features finales: alrededor de 45
```

## 8. Estructura de carpetas generada

Para el metodo `rgb`, la estructura queda:

```text
images/
  rgb/
    train/
      approved/
      not_approved/
    val/
      approved/
      not_approved/
    test/
      approved/
      not_approved/
```

Keras usa esta estructura para inferir etiquetas automaticamente.

## 9. Decisiones tecnicas importantes

### Por que excluir IDs

Los IDs no describen comportamiento financiero. Son identificadores administrativos.

### Por que escalar variables numericas

Las tecnicas visuales dependen de magnitudes. Escalar evita que variables con rangos enormes dominen la imagen.

### Por que usar one-hot

Las variables categoricas no tienen orden natural. One-hot evita imponer una relacion ordinal falsa.

### Por que usar imagenes RGB

RGB es compatible con CNNs estandar y modelos preentrenados. Tambien permite extender el proyecto a MobileNet, ResNet o EfficientNet.

### Por que usar CNN

Una CNN aprende patrones espaciales en las imagenes generadas. Aunque el origen sea tabular, la imagen codifica relaciones entre variables como estructura visual.

## 10. Limitaciones tecnicas

- La representacion visual depende del orden de las features.
- Puede perderse informacion tabular original.
- El metodo puede ser menos eficiente que modelos tabulares clasicos.
- Entrenar con muchas imagenes puede requerir tiempo y almacenamiento.
- La interpretabilidad financiera no debe depender solo de la CNN.

## 11. Mejoras tecnicas recomendadas

1. Comparar contra un baseline tabular.
2. Probar todos los metodos visuales con la misma particion.
3. Balancear clases con class weights o sobremuestreo.
4. Agregar Grad-CAM para explicar imagenes.
5. Agregar SHAP sobre variables originales.
6. Guardar predicciones y metricas en CSV.
7. Crear un script `run_pipeline.py` para ejecutar todo sin notebook.
