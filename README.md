# 🎯 CONCEPTOS CLAVE EXAMEN — M03 IA
> Solo lo esencial. Formato: qué es → para qué sirve → diferencia/trampa

---

# 1. PYTHON Y PANDAS

## Series vs DataFrame

| | Series | DataFrame |
|---|---|---|
| Dimensiones | 1D | 2D |
| Analogía | Una columna / vector etiquetado | Tabla completa |
| Ejemplo | Precios de NVIDIA por fecha | Tabla con Close, SMA21, EMA50... |

```python
type(nvd)        # → pandas.Series
type(df)         # → pandas.DataFrame
nvd.squeeze()    # DataFrame de 1 columna → Series
```

**Trampa**: `yf.download()` devuelve un DataFrame aunque solo pidas una columna. `squeeze()` lo convierte a Series.

---

## groupby()

Agrupa filas y aplica una operación a cada grupo.

```python
data.groupby("HomeTeam")["points_home"].sum()
# → total de puntos en casa, por equipo
```

**Analogía**: como una tabla dinámica de Excel. Agrupa por equipo y suma/cuenta/promedia.

---

## value_counts()

Cuenta cuántas veces aparece cada valor único en una columna.

```python
data['FTR'].value_counts()
# H: 4523,  A: 3211,  D: 2876
```

---

## pd.concat(axis=1)

Une varios DataFrames por columnas (horizontalmente).

```python
resum = pd.concat([puntos, goles_local, goles_visitante], axis=1)
# axis=0 → apilar filas | axis=1 → añadir columnas
```

---

# 2. MEDIAS MÓVILES

## ¿Qué son?

Técnica de **suavizado** de series temporales. En lugar del valor puntual de cada día, se calcula el promedio de los N días anteriores. Reducen el **ruido** y revelan la **tendencia real**.

## Tipos

| Tipo | Cómo funciona | Cuándo reacciona |
|---|---|---|
| **SMA** (Simple) | Promedio de los N últimos días, mismo peso a todos | Lento |
| **WMA** (Weighted) | Más peso a los días recientes, linealmente | Intermedio |
| **EMA** (Exponential) | Más peso a los recientes de forma exponencial | Rápido |

```python
sma = nvd.rolling(21).mean()              # SMA 21 días
ema = nvd.ewm(span=50, adjust=False).mean()  # EMA 50 días
```

**¿Por qué ventanas de 21, 50, 100, 200?** Representan horizontes de trading: corto, medio, largo y muy largo plazo.

**¿Por qué NaN al inicio?** La SMA200 necesita 200 días anteriores. Hasta que no hay suficiente historial, devuelve NaN.

**¿Cuál reacciona más rápido a cambios?** La EMA. Da más peso a los datos recientes → detecta cambios antes.

---

# 3. MACHINE LEARNING — CONCEPTOS BASE

## Supervisado vs No supervisado

| | Supervisado | No supervisado |
|---|---|---|
| Tiene etiquetas | ✅ Sí | ❌ No |
| Objetivo | Aprender a predecir y/etiquetas | Descubrir estructura en los datos |
| Ejemplos | Clasificación, regresión | Clustering |
| EAC | EAC2 (Iris, péndulo) | EAC2 (KMeans), EAC6 (KMeans Liga) |

## Clasificación vs Regresión vs Clustering

| Tipo | Predice | Ejemplo |
|---|---|---|
| **Regresión** | Valor continuo | Peso de un pingüino, valor de g |
| **Clasificación** | Categoría / etiqueta | Especie de Iris, sentimiento reseña |
| **Clustering** | Grupos sin etiquetas previas | Equipos de fútbol por rendimiento |

---

# 4. MODELOS DE CLASIFICACIÓN (sklearn)

## KNN — K-Nearest Neighbors

**¿Qué es?** Para clasificar un punto nuevo, mira los K puntos más cercanos del dataset y asigna la clase más frecuente entre ellos.

**Analogía**: "Dime con quién andas y te diré quién eres." Miras tus K vecinos más cercanos.

**Hiperparámetro clave**: K (número de vecinos).
- K muy pequeño (K=1) → overfitting, muy sensible al ruido
- K muy grande → underfitting, fronteras demasiado suaves
- Se elige experimentalmente (probando K=1 a K=10)

```python
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=3)
```

**¿Cuándo usarlo?** Datasets pequeños, cuando las clases son geométricamente separables. Sencillo pero lento en datasets grandes.

---

## SVM — Support Vector Machine

**¿Qué es?** Encuentra el **hiperplano** (línea en 2D, plano en 3D...) que mejor separa las clases, maximizando el margen entre ellas.

**Analogía**: Dibuja la línea más ancha posible entre dos grupos de puntos. Los puntos más cercanos a esa línea son los **vectores soporte**.

```python
from sklearn import svm
model = svm.SVC()
```

**¿Cuándo usarlo?** Muy bueno con datos de alta dimensión. Robusto aunque haya pocas muestras. Menos intuitivo de interpretar.

---

## Logistic Regression

**¿Qué es?** A pesar del nombre, es un modelo de **clasificación**. Calcula la probabilidad de pertenecer a una clase usando la función sigmoide.

**Analogía**: Traza una frontera lineal entre clases y asigna probabilidades. "¿Con qué probabilidad esta flor es Iris-setosa?"

```python
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
```

**Trampa del examen**: Se llama "regresión" pero hace **clasificación**. La regresión logística predice probabilidades que luego se convierten en clases.

**¿Cuándo usarlo?** Cuando las clases son linealmente separables. Muy rápido e interpretable.

---

## Decision Tree (Árbol de decisión)

**¿Qué es?** Crea un árbol de preguntas de sí/no sobre las características hasta llegar a una predicción.

**Analogía**: "¿El pétalo mide más de 2.5cm? Si sí → ¿la anchura es mayor de 1.8cm?..." Como un diagrama de flujo de decisiones.

```python
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
```

**¿Cuándo usarlo?** Muy interpretable (puedes visualizar el árbol). Propenso a overfitting si el árbol es muy profundo.

---

## Comparativa de los 4 modelos

| Modelo | Intuición | Ventaja | Desventaja |
|---|---|---|---|
| **KNN** | Vecinos más cercanos | Simple, no asume distribución | Lento en producción, sensible a K |
| **SVM** | Hiperplano de máximo margen | Bueno en alta dimensión | Difícil de interpretar |
| **Logistic Regression** | Frontera lineal con probabilidades | Rápido, interpretable | Solo fronteras lineales |
| **Decision Tree** | Árbol de preguntas | Muy interpretable | Overfitting fácil |

**¿Cuál es mejor?** Depende del dataset. En Iris, todos funcionan bien con pétalos (>0.9 accuracy). Con sépalos, todos bajan (<0.82). La clave es la **calidad de los datos**, no solo el modelo.

---

## train_test_split — ¿Por qué?

```python
train, test = train_test_split(iris, test_size=0.3)  # 70% train, 30% test
```

**¿Por qué separar?** Para evaluar el modelo con datos que **nunca ha visto** durante el entrenamiento. Si evaluaras en train, medirías memorización, no generalización.

**Trampa**: Si entrenas y evalúas en los mismos datos, el modelo puede tener 100% de accuracy pero fallar completamente con datos nuevos (overfitting).

---

## Accuracy vs R²

| Métrica | Para qué | Rango | Cuándo es buena |
|---|---|---|---|
| **Accuracy** | Clasificación | [0, 1] | Próxima a 1 |
| **R²** | Regresión | [0, 1] | Próxima a 1 |
| **MSE / RMSE** | Regresión | [0, ∞) | Próxima a 0 |

**R²**: proporción de variabilidad de y explicada por el modelo. R²=0.98 → el modelo explica el 98% de la variabilidad.

---

# 5. CLUSTERING — KMeans

## ¿Qué es KMeans?

Algoritmo de clustering no supervisado. Agrupa los datos en K grupos (clústeres) minimizando la distancia de cada punto al centro de su clúster.

**¿Cómo funciona?**
1. Elige K centroides aleatoriamente
2. Asigna cada punto al centroide más cercano
3. Recalcula los centroides como la media de cada grupo
4. Repite hasta que los centroides no se muevan

```python
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=4, random_state=42)
kmeans.fit(X_scaled)
labels = kmeans.labels_   # etiqueta de clúster de cada punto
```

**¿Por qué escalar antes de KMeans?** KMeans usa distancias euclidianas. Si una variable tiene rango [0-100.000] y otra [0-1], la primera domina totalmente. `StandardScaler` lleva todo a media=0, desv=1.

## Silhouette Score

Mide la calidad del clustering. Rango [-1, 1]. Próximo a 1 → clústeres bien separados y compactos.

---

# 6. PREPROCESAMIENTO

## fit() vs transform() — La regla más importante

```python
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)  # aprende los parámetros Y transforma
X_test_scaled  = scaler.transform(X_test)       # solo transforma, NO aprende
```

**¿Por qué nunca fit en test?** → **Data leakage**. Si el scaler aprende los parámetros del test, el modelo "ve" información del test antes de evaluarse. Los resultados serían artificialmente buenos pero no representarían la realidad.

## LabelEncoder

Convierte variables categóricas (texto) a números enteros.

```python
encoder = LabelEncoder()
df['sex'] = encoder.fit_transform(df['sex'])
# 'FEMALE' → 0,  'MALE' → 1
```

**Importante**: guardar el encoder para usarlo en inferencia con nuevos datos.

## MinMaxScaler vs StandardScaler

| | MinMaxScaler | StandardScaler |
|---|---|---|
| Resultado | Rango [0, 1] | Media=0, desv=1 |
| Cuándo | Redes neuronales (activaciones tipo sigmoide/relu) | KMeans, SVM, algoritmos basados en distancias |
| Sensible a outliers | Sí (muy) | Menos |

---

# 7. DEEP LEARNING — CONCEPTOS BASE

## Neurona artificial

Recibe entradas, las multiplica por pesos, suma todo y aplica una función de activación. La red aprende ajustando esos pesos.

## Red neuronal (MLP — Multilayer Perceptron)

Capas de neuronas conectadas entre sí. Datos tabulares → MLP. Texto/secuencias → LSTM.

```
Entrada → [Capa oculta 1] → [Capa oculta 2] → Salida
```

## Funciones de activación

| Función | Qué hace | Cuándo |
|---|---|---|
| **ReLU** | f(x) = max(0, x). Devuelve 0 si negativo, x si positivo | Capas ocultas (estándar en DL) |
| **Sigmoide** | Aplasta a [0,1] | Clasificación binaria en la salida |
| **Softmax** | Convierte a probabilidades que suman 1 | Clasificación multiclase en la salida |
| **Sin activación** | El valor sale tal cual | Regresión en la salida |

**Trampa**: En regresión, la capa de salida **no tiene activación**. Si pusiera sigmoide, la salida estaría limitada a [0,1].

## Forward propagation

Los datos pasan de la entrada hacia la salida, capa por capa. Se calcula la predicción.

## Backpropagation

Una vez calculada la pérdida, el error se propaga hacia atrás para calcular cuánto ha contribuido cada peso al error. Los pesos se actualizan para reducir ese error.

## Epoch vs Batch

| Concepto | Definición |
|---|---|
| **Epoch** | Una pasada completa por todo el dataset de entrenamiento |
| **Batch** | Subconjunto del dataset que se procesa en cada iteración |
| **Iteraciones por epoch** | = total_muestras / batch_size |

**¿Por qué batches y no todo a la vez?** El dataset puede no caber en memoria. Además, los batches añaden ruido al gradiente que ayuda a evitar mínimos locales.

## Overfitting vs Underfitting

| | Overfitting | Underfitting |
|---|---|---|
| Qué pasa | El modelo memoriza el train | El modelo no aprende suficiente |
| Train accuracy | Muy alta | Baja |
| Test accuracy | Baja | Baja |
| Solución | Más datos, Dropout, regularización | Más épocas, más capacidad del modelo |

## Dropout

Técnica de regularización: durante el entrenamiento, desactiva aleatoriamente un porcentaje de neuronas en cada paso. Evita que la red dependa demasiado de neuronas concretas → reduce overfitting.

**Importante**: el Dropout solo está activo en **entrenamiento**. En evaluación/inferencia se desactiva (`model.eval()`).

## Loss functions

| Función | Para qué | Cuándo |
|---|---|---|
| **MSELoss** | Regresión | Predecir valores continuos (peso pingüino) |
| **CrossEntropyLoss** | Clasificación multiclase | Predecir categorías (sentimiento, especie) |
| **BCELoss** | Clasificación binaria | Predecir sí/no |

## Optimizadores

| Optimizador | Características |
|---|---|
| **SGD** | Simple. Requiere ajustar learning rate manualmente |
| **Adam** | Adaptativo. Ajusta el learning rate automáticamente. Estándar en DL |

---

# 8. PYTORCH — HERRAMIENTAS CLAVE

## Tensor

Es el equivalente al array de NumPy pero en PyTorch. Puede vivir en GPU.

```python
x = torch.tensor(data, dtype=torch.float32).to(device)
```

## TensorDataset vs DataLoader

**TensorDataset**: agrupa features (X) y labels (y) en un único objeto.  
**DataLoader**: gestiona el acceso a los datos: crea batches, baraja, itera.

```python
dataset = TensorDataset(X_tensor, y_tensor)  # almacén
loader  = DataLoader(dataset, batch_size=16, shuffle=True)  # dispensador
```

**Trampa clásica**: "¿Qué diferencia hay entre TensorDataset y DataLoader?"  
→ TensorDataset **almacena** los datos. DataLoader **distribuye** esos datos en batches durante el entrenamiento.

## Bucle de entrenamiento — Orden obligatorio

```python
for batch_X, batch_y in train_loader:
    outputs = model(batch_X)           # 1. Forward pass
    loss    = criterion(outputs, batch_y)  # 2. Calcular pérdida
    optimizer.zero_grad()              # 3. Resetear gradientes
    loss.backward()                    # 4. Backpropagation
    optimizer.step()                   # 5. Actualizar pesos
```

**¿Por qué `zero_grad()`?** PyTorch acumula gradientes por defecto. Si no se resetean, los del batch anterior se suman a los del batch actual → pesos incorrectos.

## model.eval() y torch.no_grad()

```python
model.eval()              # Desactiva Dropout y BatchNorm (modo inferencia)
with torch.no_grad():     # No calcula gradientes (ahorra memoria)
    outputs = model(X)
```

**¿Por qué los dos?** `eval()` cambia el comportamiento del modelo. `no_grad()` evita calcular gradientes innecesarios. Se usan siempre juntos en evaluación e inferencia.

## Guardar y cargar modelos en PyTorch

```python
torch.save(model.state_dict(), "model.pth")    # Guardar solo los pesos

model = MLP(...)                               # Recrear la arquitectura
model.load_state_dict(torch.load("model.pth")) # Cargar los pesos
```

**¿Por qué state_dict y no el modelo completo?** El state_dict (diccionario de pesos) es más portable y compatible entre versiones. El modelo completo puede fallar si cambia la definición de la clase.

---

# 9. NLP — PROCESAMIENTO DE TEXTO

## Pipeline de texto → tensor

```
Texto crudo
  ↓ Limpiar (minúsculas, eliminar signos, stop words)
  ↓ Tokenizar (dividir en palabras)
  ↓ Crear vocabulario (word → índice)
  ↓ Convertir texto a secuencia de índices
  ↓ Padding (igualar longitudes)
  ↓ Tensor PyTorch
```

## Stop words

Palabras muy frecuentes sin contenido semántico (artículos, preposiciones: "el", "de", "i"...). Se eliminan para reducir ruido y tamaño del vocabulario.

## Vocabulario y tokens especiales

```python
word2idx = {'<PAD>': 0, '<UNK>': 1, 'restaurant': 2, 'bona': 3, ...}
```

| Token | Significado |
|---|---|
| `<PAD>` | Relleno para igualar longitudes de secuencias |
| `<UNK>` | Palabra desconocida (no está en el vocabulario) |

## Padding

Las secuencias tienen longitudes variables pero los tensores necesitan tamaño fijo. Se añaden `<PAD>` al final hasta alcanzar `max_seq_length`.

## Embedding (nn.Embedding)

Convierte un índice entero (palabra) en un vector denso de números reales. Permite que la red aprenda representaciones semánticas: palabras similares → vectores cercanos.

```python
self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
```

`padding_idx=0` → el vector del token `<PAD>` siempre será ceros (no contribuye al aprendizaje).

## LSTM — Long Short-Term Memory

Red neuronal diseñada para **secuencias**. Tiene memoria: puede recordar información de palabras anteriores cuando procesa las siguientes. Ideal para texto.

**MLP vs LSTM**:
- MLP → procesa vectores de características fijas, sin orden
- LSTM → procesa secuencias manteniendo contexto temporal

## LSTM Bidireccional

Lee la secuencia en ambas direcciones (→ y ←). Captura contexto completo. La salida dobla la dimensión (dos estados ocultos concatenados).

---

# 10. AWS Y DESPLIEGUE EN LA NUBE

## AWS Textract

Servicio de OCR inteligente. Extrae texto y datos estructurados de imágenes/PDFs. Modo `FORMS` → detecta pares clave-valor (campo + valor en un formulario/ticket).

```python
response = textract.analyze_document(
    Document={'Bytes': imageBytes},
    FeatureTypes=["FORMS"]
)
```

**¿Por qué usarlo en lugar de OCR local?** Servicio gestionado, sin mantenimiento, escala automáticamente, ya entrenado para múltiples idiomas y formatos.

## boto3

Librería Python para interactuar con los servicios de AWS desde código.

## Gradio

Framework Python para crear interfaces web de IA sin saber frontend. 

**`gr.Interface`** → simple (un input, un output).  
**`gr.Blocks`** → flexible (layouts personalizados, múltiples componentes, eventos).

## Hugging Face Spaces

Plataforma de hosting gratuita para aplicaciones de IA con Gradio o Streamlit. Para desplegar: `app.py` + `requirements.txt`.

**Credenciales como Secrets**: las credenciales AWS nunca van en el código. Se añaden como variables de entorno en la configuración del Space y se leen con `os.environ.get('AWS_KEY')`.

---

# 11. GESTIÓN DE PROYECTOS IA (EAC6)

## Estructura de un proyecto profesional

```
proyecto/
├── src/            ← Código fuente (módulos)
├── tests/          ← Tests unitarios
├── data/           ← Dataset
├── model/          ← Modelos guardados
├── doc/            ← Documentación generada
├── requirements.txt
└── README.md
```

**¿Por qué modularizar?** Separar en archivos/funciones → más legible, testeable, mantenible y colaborativo.

## Entorno virtual (venv)

```bash
python3 -m venv venv          # Crear
source venv/bin/activate      # Activar
pip install -r requirements.txt  # Instalar dependencias
```

**¿Para qué sirve?** Aísla las dependencias del proyecto. Evita conflictos entre proyectos y garantiza que todos los desarrolladores usan las mismas versiones.

**requirements.txt**: lista las librerías necesarias. Se genera con `pip freeze > requirements.txt`.

## Git — Comandos clave

```bash
git init                          # Inicializar repositorio
git add .                         # Añadir todos los cambios
git commit -m "descripción"       # Guardar snapshot
git push                          # Subir a GitHub

git branch -M main                # Renombrar rama
git remote add origin <url>       # Conectar con GitHub
git push -u origin main           # Primera subida
```

**¿Para qué sirve Git?** Historial de cambios, colaboración, backup en la nube, poder volver a versiones anteriores.

## Docstrings y type hints

```python
def fun_total_goals(data: pd.DataFrame) -> (int, int, int):
    """
    Calcula el total de goles.

    Arguments
        data (DataFrame): el dataset

    Returns
        tuple(int, int, int): goles locales, visitantes, totales
    """
```

**Docstring**: documentación de la función. **Type hints** (`data: pd.DataFrame`): indica los tipos esperados. Ambos mejoran la legibilidad y permiten generar documentación automática con `pydoc`.

## Tests unitarios (unittest)

```python
class TestFunTotalGoals(unittest.TestCase):
    def test_basic_case(self):
        # Arrange
        data = pd.DataFrame({'FTHG': [1, 2, 0], 'FTAG': [0, 1, 3]})
        # Act
        home, away, total = fun_total_goals(data)
        # Assert
        self.assertEqual(total, 7)
```

**¿Para qué sirven los tests?** Verifican que las funciones hacen lo que deben. Si se modifica el código y algo se rompe, los tests lo detectan automáticamente.

## Pylint

Analiza el código en busca de errores, código mal formateado, variables no usadas, funciones sin docstring. Puntúa el código de 0 a 10.

## pydoc

Genera documentación HTML automáticamente a partir de los docstrings.

```bash
pydoc3 -w *.py    # genera un .html por cada .py
```

## pickle

Serializa cualquier objeto Python a un archivo binario. Se usa para guardar modelos de sklearn.

```python
# Guardar
with open("model.pkl", "wb") as f:
    pickle.dump({"scaler": scaler, "kmeans": kmeans}, f)

# Cargar
with open("model.pkl", "rb") as f:
    model_data = pickle.load(f)
```

**¿Por qué guardar el scaler junto con el modelo?** Para poder escalar los datos nuevos con exactamente el mismo scaler que se usó en el entrenamiento. Sin él, las predicciones serían incorrectas.

---

# 12. TABLA RESUMEN RÁPIDO

## Librerías → para qué sirven

| Librería | Para qué |
|---|---|
| **pandas** | Manipulación de datos tabulares (DataFrame, Series) |
| **numpy** | Operaciones matemáticas y arrays numéricos |
| **matplotlib** | Visualización de datos (gráficos) |
| **seaborn** | Visualización estadística (encima de matplotlib, más bonita) |
| **yfinance** | Descargar datos de bolsa de Yahoo Finance |
| **sklearn** | Machine learning clásico (modelos, preprocesamiento, métricas) |
| **torch (PyTorch)** | Deep learning (redes neuronales, GPU) |
| **datasets** | Cargar datasets de Hugging Face |
| **nltk** | Procesamiento de lenguaje natural (stop words, tokenización) |
| **boto3** | Cliente Python para servicios AWS |
| **gradio** | Crear interfaces web para aplicaciones de IA |
| **pickle** | Serializar/deserializar objetos Python |

## Modelos → cuándo usar cuál

| Modelo | Tipo | Dataset / Problema |
|---|---|---|
| LinearRegression | Regresión supervisada | Péndulo (EAC2), relación lineal entre variables |
| LogisticRegression | Clasificación supervisada | Iris (EAC2), frontera lineal entre clases |
| KNN | Clasificación supervisada | Iris (EAC2), dataset pequeño |
| SVM | Clasificación supervisada | Iris (EAC2), alta dimensión |
| DecisionTree | Clasificación supervisada | Iris (EAC2), interpretabilidad |
| KMeans | Clustering no supervisado | Grupos de equipos (EAC6), sin etiquetas |
| MLP | Regresión/Clasif. con DL | Pingüinos (EAC3), datos tabulares |
| LSTM | Clasificación secuencias | Reseñas texto (EAC4), NLP |

---

*Preparación examen M03 IAB-5073 · Eric Rodriguez · Mayo 2026*
