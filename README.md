# 📚 APUNTES EXAMEN — M03 Programació d'Intel·ligència Artificial
> Basados en las soluciones oficiales EAC1 a EAC6 · Preparación examen Eric Rodriguez

---

# ══════════════════════════════════════════
# EAC1 — Análisis de series temporales (NVIDIA)
# ══════════════════════════════════════════

## 🧠 Idea general

Se analiza la evolución del precio de la acción de NVIDIA descargando datos de Yahoo Finance.
Se usan herramientas básicas de Python para suavizar la serie temporal, ajustar un polinomio de regresión y extrapolar el comportamiento futuro (de forma didáctica, NO es una predicción real).

**Librerías**: `yfinance`, `pandas`, `numpy`, `matplotlib`

---

## 📦 Series vs DataFrame — CONCEPTO CLAVE

| | **Series** | **DataFrame** |
|---|---|---|
| Dimensiones | 1D (índice → valores) | 2D (filas × columnas) |
| Analogía | Vector etiquetado | Tabla |
| Equivalente | Una columna | Varias columnas |

### squeeze()
Cuando descargas con `yf.download(...)` y accedes a `["Close"]`, el resultado puede ser un DataFrame de una sola columna. `squeeze()` lo convierte a Series.

```python
nvd = yf.download("NVDA", start="2023-01-01", end="2026-02-04", progress=False)
nvd = nvd["Close"].dropna()
nvd = nvd.squeeze()     # DataFrame 1 col → Series (1D)
```

> 🎯 **Pregunta trampa**: "¿Por qué se usa squeeze()?"  
> **Respuesta**: Porque `yf.download` retorna un DataFrame. squeeze() lo convierte a Series 1D para operar con él como serie temporal.

---

## 📈 Medias móviles (Moving Averages)

### ¿Por qué se usan?
Reducen el **ruido** del mercado y revelan la **tendencia real** de la serie.

| Tipo | Cómo funciona | Código |
|---|---|---|
| **SMA** (Simple) | Promedio de los N últimos días, mismo peso a todos | `nvd.rolling(21).mean()` |
| **WMA** (Weighted) | Más peso a los días recientes, linealmente | `rolling().apply(lambda x: np.dot(x, pesos))` |
| **EMA** (Exponential) | Más peso a los recientes, exponencialmente | `nvd.ewm(span=50, adjust=False).mean()` |

### ¿Por qué varias ventanas (21, 50, 100, 200)?
Representan horizontes temporales reales de trading:
- **21 días** → corto plazo (~1 mes)
- **50 días** → tendencia intermedia
- **100 días** → consolidación
- **200 días** → largo plazo

### reindex(idx) — ¿por qué?
Para garantizar que el índice de la media coincide exactamente con el del precio original. Evita desalineación de fechas en el DataFrame final.

```python
nvd_sma21 = nvd.rolling(21).mean().reindex(idx)
```

> 🎯 **Pregunta típica**: "Explica qué hace `nvd.rolling(21).mean().reindex(idx)`"  
> **Respuesta**: Calcula la media móvil de 21 días (cada día coge los 21 últimos precios y hace la media → curva suavizada). `reindex(idx)` asegura que las fechas coinciden exactamente con las del precio original.

---

## 🔗 DataFrame combinado

```python
df = pd.DataFrame({
    "Close": nvd, "SMA21": nvd_sma21, "SMA200": nvd_sma200,
    "WMA200": nvd_wma200, "EMA50": nvd_ema50, "EMA100": nvd_ema100
})
```

**Nota**: Los primeros valores de SMA200 y WMA200 serán `NaN` porque no hay 200 días de historial previo.

---

## 📊 Media de medias

```python
df["MA_Avg"] = df[["SMA21","SMA200","WMA200","EMA50","EMA100"]].mean(axis=1)
# axis=1 → media por filas (horizontal), no por columnas
```

---

## 📐 Regresión polinómica

**¿Qué es?** Una curva matemática que se ajusta a tus datos para describir su forma general. La regresión lineal dibuja una recta. La polinómica dibuja una curva que puede subir, bajar y girar.

**Grado 7 = 7 posibles curvas**. Más grado → más flexible. Menos grado → demasiado rígida.

```python
df["timestamp"] = (df.index.astype(np.int64) // 10**9)  # Fechas → Unix (números)
x = df["timestamp"].values
y = df["MA_Avg"].values

coeffs = np.polyfit(x, y, 7)   # Encuentra los 8 coeficientes de la curva
poly   = np.poly1d(coeffs)     # Crea la función → poly(x_futuro) da predicción
```

### ¿Por qué convertir fechas a timestamp?
`np.polyfit` necesita valores numéricos, no fechas. La conversión a Unix timestamp (segundos desde 1970) da números enteros comparables.

> ⚠️ **Warning "Polyfit may be poorly conditioned"** → normal. Indica inestabilidad numérica fuera del intervalo. No es un error fatal.

> 🎯 **Pregunta trampa**: "¿Un buen ajuste polinómico a los datos históricos implica buena capacidad predictiva?"  
> **Respuesta**: NO. Un polinomio puede ajustarse muy bien al pasado (overfitting) y ser pésimo extrapolando al futuro. Los mercados dependen de factores externos que ningún polinomio puede capturar.

---

## 🐍 Script Python (.py) — Ejercicio 7

El EAC pide pasar el código del notebook a un archivo `.py` ejecutable desde terminal:
- `plt.show()` → `plt.savefig()` (el .py no muestra ventanas interactivas)
- Hay que crear la carpeta `img/` previamente
- Se añade nombre del alumno y fecha de ejecución

---

## 🎯 Preguntas tipo examen EAC1

**Q: ¿Qué diferencia hay entre Series y DataFrame?**  
R: Series es 1D (índice → valor), DataFrame es 2D (filas × columnas). Una Series es equivalente a una columna de un DataFrame.

**Q: ¿Por qué la SMA200 tiene NaN en los primeros 199 días?**  
R: Porque necesita 200 valores anteriores para calcular la primera media. Hasta que no hay suficiente historial, el valor es NaN.

**Q: ¿Qué media reacciona más rápidamente a los cambios de precio?**  
R: La EMA, porque da peso exponencial a los días recientes y se actualiza más rápido que SMA o WMA.

---

# ══════════════════════════════════════════
# EAC2 — Machine Learning Clásico (scikit-learn)
# ══════════════════════════════════════════

## 🧠 Idea general

Tres problemas independientes con scikit-learn:
- **P1**: Regresión lineal (experimento del péndulo → calcular g)
- **P2**: Clasificación multiclase (dataset Iris → 4 modelos)
- **P3**: Clustering (datos sintéticos → KMeans)

---

## 📐 P1 — Regresión Lineal: péndulo y la constante g

```python
from sklearn.linear_model import LinearRegression

X = data[["L"]]   # 2D (requerido por sklearn)
y = data["T2"]    # 1D

model = LinearRegression(fit_intercept=False)  # sin término independiente
model.fit(X, y)

m  = model.coef_[0]              # pendiente
g_estimada = (4 * np.pi**2) / m # calcular g
r2 = model.score(X, y)           # R²
```

### ¿Por qué `fit_intercept=False`?
La física dice que si L=0 → T²=0. No hay término constante. Forzamos la recta a pasar por el origen.

### Métricas
- **R²**: Proporción de variabilidad explicada. R²=1 → ajuste perfecto. R²=0 → no explica nada.
- **Error absoluto**: `|g_estimada - g_real|`
- **Error relativo**: `(error_absoluto / g_real) * 100` en %

---

## 🌸 P2 — Clasificación: Dataset Iris

### Clave: pétalos vs sépalos
Los **pétalos** separan mejor las especies (correlación 0.962 entre longitud y anchura de pétalos). Los sépalos tienen poca correlación (-0.109) → información poco discriminativa.

### Flujo estándar de clasificación

```python
from sklearn.model_selection import train_test_split
from sklearn import metrics

train, test = train_test_split(iris, test_size=0.3)  # 70/30

train_X = train[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]
train_y = train.Species

model.fit(train_X, train_y)
predictions = model.predict(test_X)
accuracy = metrics.accuracy_score(test_y, predictions)
```

### Los 4 modelos
| Modelo | Import |
|---|---|
| SVM | `from sklearn import svm; svm.SVC()` |
| Logistic Regression | `from sklearn.linear_model import LogisticRegression` |
| Decision Tree | `from sklearn.tree import DecisionTreeClassifier` |
| KNN | `from sklearn.neighbors import KNeighborsClassifier(n_neighbors=3)` |

### Función reutilizable (buena práctica)
```python
def train_and_evaluate(model, train_X, train_y, test_X, test_y):
    model.fit(train_X, train_y)
    predictions = model.predict(test_X)
    accuracy = metrics.accuracy_score(test_y, predictions)
    return accuracy, predictions
```

> 🎯 **Pregunta trampa**: "¿Con qué atributos se obtienen mejores resultados en Iris?"  
> **Respuesta**: Con los pétalos (PetalLength + PetalWidth) se obtiene accuracy >0.9 con todos los modelos. Con sépalos, accuracy <0.82.

---

## 🔵 P3 — Clustering: KMeans

```python
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=1.0, random_state=42)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # ¡Escalar siempre antes de KMeans!

kmeans = KMeans(n_clusters=4, random_state=42)
kmeans.fit(X_scaled)

score = silhouette_score(X_scaled, kmeans.labels_)
```

### Conceptos clave
- **KMeans**: Agrupa datos en K clústeres minimizando la distancia intra-clúster
- **StandardScaler**: Escalar SIEMPRE antes de KMeans. Si no, las variables con valores grandes dominan la distancia
- **Silhouette score**: Calidad del clustering. Rango [-1, 1]. Próximo a 1 → buenos clústeres bien separados

> 🎯 **Pregunta típica**: "¿Por qué hay que escalar los datos antes de KMeans?"  
> **Respuesta**: KMeans usa distancias euclidianas. Si una variable tiene rango [0-1000] y otra [0-1], la primera domina totalmente el cálculo. StandardScaler lleva todas las variables a media=0 y desv.típica=1.

---

## 🎯 Preguntas tipo examen EAC2

**Q: ¿Qué diferencia hay entre clasificación y clustering?**  
R: Clasificación es aprendizaje supervisado (tenemos etiquetas). Clustering es no supervisado (no tenemos etiquetas, el algoritmo descubre grupos).

**Q: ¿Por qué se dividen los datos en train y test?**  
R: Para evaluar el modelo en datos que no ha visto durante el entrenamiento. Si se evaluara en train, mediríamos la "memorización", no la capacidad de generalización.

---

# ══════════════════════════════════════════
# EAC3 — Deep Learning con PyTorch: Regresión MLP
# ══════════════════════════════════════════

## 🧠 Idea general

Red neuronal feedforward (MLP) con PyTorch para predecir la **masa corporal de pingüinos** a partir de características morfológicas.

**Tipo de problema**: Regresión (predecir un valor continuo, no una categoría)

---

## 🔄 Pipeline completo

```
Datos brutos (CSV)
    ↓ Preprocessing (LabelEncoder, MinMaxScaler)
    ↓ Split train/test
    ↓ Conversión a tensores PyTorch
    ↓ TensorDataset + DataLoader
    ↓ Definir arquitectura MLP
    ↓ Definir loss (MSELoss) + optimizador (Adam)
    ↓ Bucle de entrenamiento (epochs + batches)
    ↓ Evaluación (MSE, RMSE, MAE, R²)
    ↓ Guardar modelo (torch.save)
    ↓ Inferencia (nuevos datos)
```

---

## 🧹 Preprocesamiento de datos

### LabelEncoder — variables categóricas
Convierte categorías a números enteros.
```python
encoder_sex = LabelEncoder()
penguins['sex'] = encoder_sex.fit_transform(penguins['sex'])
# 'MALE' → 1, 'FEMALE' → 0
```

### MinMaxScaler — normalización
Lleva los valores al rango [0, 1].

```python
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_train_scaled = scaler_X.fit_transform(X_train)   # fit + transform en train
X_test_scaled  = scaler_X.transform(X_test)        # solo transform en test!
```

> ⚠️ **Error crítico**: Hacer `fit_transform` en los datos de test → **data leakage**: la red "ve" información del test durante el preprocesamiento. Siempre: `fit` con train, `transform` con test.

---

## 🔢 Tensores PyTorch

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32).to(device)
```

---

## 📦 TensorDataset y DataLoader — CONCEPTO CLAVE

| | **TensorDataset** | **DataLoader** |
|---|---|---|
| Rol | **Almacén** de datos (X, y juntos) | **Dispensador** de datos en batches |
| Qué hace | Agrupa features y labels | Itera, baraja, crea mini-lotes |

```python
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader  = DataLoader(train_dataset, batch_size=16, shuffle=True)
```

| Parámetro | Significado |
|---|---|
| `batch_size=16` | 16 ejemplos por iteración |
| `shuffle=True` | Baraja los datos cada epoch (evita orden artificial) |

> 🎯 **Pregunta trampa**: "¿Qué diferencia hay entre TensorDataset y DataLoader?"  
> **Respuesta**: TensorDataset **almacena** los datos (pares X, y). DataLoader **gestiona el acceso**: crea batches, baraja, itera. Trabajan juntos: TensorDataset es el "almacén", DataLoader es el "dispensador".

---

## 🏗️ Arquitectura MLP

```python
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1  = nn.Linear(input_size, hidden_size)
        self.fc2  = nn.Linear(hidden_size, hidden_size)
        self.fc3  = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)   # sin activación en la salida → regresión
        return x
```

| Componente | Detalle |
|---|---|
| Capas ocultas | 2 capas de 10 neuronas |
| Activación | ReLU entre capas ocultas |
| Capa de salida | **Sin activación** (regresión → valor real) |

### ¿Por qué ReLU y no sigmoide?
ReLU no se satura para valores positivos grandes → evita el problema del gradiente desapareciendo.

### ¿Por qué NO hay activación en la salida?
Porque es **regresión**. La red tiene que poder retornar cualquier valor real. Sigmoide o softmax limitarían la salida a [0,1] o a probabilidades.

---

## 🔁 Bucle de entrenamiento

```python
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

for epoch in range(num_epochs):
    for batch_X, batch_y in train_loader:
        outputs = model(batch_X)           # Forward pass
        loss    = criterion(outputs, batch_y)

        optimizer.zero_grad()              # ① Resetear gradientes
        loss.backward()                    # ② Backpropagation
        optimizer.step()                   # ③ Actualizar pesos
```

### ¿Por qué `optimizer.zero_grad()`?
PyTorch **acumula** los gradientes por defecto. Si no los reseteamos en cada batch, los gradientes anteriores se acumulan y los pesos se actualizan incorrectamente.

### Orden obligatorio: zero_grad → backward → step

> ⚠️ **Error típico crítico**: Olvidar `optimizer.zero_grad()` o ponerlo en lugar incorrecto.

---

## 📏 Métricas de evaluación (regresión)

```python
model.eval()
with torch.no_grad():
    outputs = model(X_test_tensor)
    y_pred  = outputs.cpu().numpy()

mse  = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)
mae  = mean_absolute_error(y_true, y_pred)
r2   = r2_score(y_true, y_pred)
```

| Métrica | Significado |
|---|---|
| **MSE** | Error cuadrático medio (penaliza mucho los errores grandes) |
| **RMSE** | Raíz de MSE (misma unidad que y) |
| **MAE** | Error absoluto medio (robusto a outliers) |
| **R²** | Proporción de variabilidad explicada [0, 1] |

---

## 💾 Guardar y cargar el modelo

```python
# Guardar
torch.save(model.state_dict(), "model.pth")

# Cargar
loaded_model = MLP(input_size, hidden_size, output_size)  # misma arquitectura
loaded_model.load_state_dict(torch.load("model.pth"))
loaded_model.to(device)
loaded_model.eval()
```

---

## 🔮 Inferencia con nuevos datos

```python
with torch.no_grad():
    outputs          = model(new_data_tensor)
    predicted_scaled = outputs.cpu().numpy()

predicted = scaler_y.inverse_transform(predicted_scaled)  # ¡invertir el escalado!
```

### Pasos para hacer inferencia:
1. Crear DataFrame con misma estructura que train
2. Aplicar el **mismo** LabelEncoder que se usó en train
3. Aplicar el **mismo** scaler_X que se usó en train
4. Convertir a tensor → `.to(device)`
5. `model.eval()` + `torch.no_grad()`
6. Pasar por la red
7. Invertir el escalado con `scaler_y.inverse_transform()`

> ⚠️ **Error típico**: Crear un nuevo scaler para los datos nuevos → valores incorrectos. Hay que usar el **mismo scaler** ajustado (fit) en train.

---

## 🎯 Preguntas tipo examen EAC3

**Q: ¿Por qué se usa `model.eval()` durante la evaluación?**  
R: Desactiva comportamientos específicos del training como Dropout y BatchNorm, que se comportan diferente en inferencia.

**Q: ¿Por qué se usa `torch.no_grad()` durante la inferencia?**  
R: Para no calcular gradientes (no hace falta para inferencia). Ahorra memoria y acelera el cálculo.

**Q: ¿Qué función de pérdida se usa en regresión?**  
R: MSELoss. Penaliza proporcionalmente al cuadrado del error.

---

# ══════════════════════════════════════════
# EAC4 — Deep Learning NLP: Clasificación de Sentimientos (LSTM)
# ══════════════════════════════════════════

## 🧠 Idea general

Modelo LSTM bidireccional para clasificar reseñas de restaurantes (dataset GuiaCat de Hugging Face) en 5 categorías: molt dolent / dolent / regular / bo / molt bo.

**Tipo de problema**: Clasificación multiclase de texto (NLP)

---

## 📥 Cargar datos de Hugging Face

```python
from datasets import load_dataset

ds_train = load_dataset("projecte-aina/GuiaCat", split="train")
df_train = ds_train.to_pandas()
```

---

## ⚠️ Dataset desequilibrado — CONCEPTO IMPORTANTE

Las reseñas positivas ("bo", "molt bo") representan el **85%** del dataset. Consecuencias:
- El modelo aprende a clasificarlo todo como "bo" o "molt bo"
- Las clases minoritarias (negativas) nunca se predicen correctamente
- La accuracy puede ser alta pero el modelo es inútil para detectar negativas
- **Se ve en la matriz de confusión**: clases 0, 1, 2 no se predicen nunca

> 🎯 **Pregunta de reflexión**: "¿Cómo se podría mejorar?"  
> Posibles soluciones: oversampling de la clase minoritaria, undersampling de la mayoritaria, `class_weight` en la loss function, o usando una loss ponderada.

---

## 🧹 Preprocesamiento de texto

```python
def preprocess_string(s):
    s = s.lower()
    s = re.sub(r"[^a-zA-Zàèéíïòóúüç\s]", ' ', s)  # eliminar no-alfabéticos
    stop_words = set(stopwords.words('catalan'))
    s = ' '.join([w for w in s.split() if w not in stop_words])  # stop words
    return ' '.join(s.split())
```

**Stop words**: Palabras muy frecuentes pero sin significado semántico (artículos, preposiciones). Se eliminan para reducir ruido y tamaño del vocabulario.

---

## 📖 Creación del vocabulario

```python
from collections import Counter

tokenizer = Counter()
for text in df_train['text']:
    tokenizer.update(preprocess_string(text).split())

word2idx = {'<PAD>': 0, '<UNK>': 1}
for idx, (word, _) in enumerate(tokenizer.most_common(1000)):
    word2idx[word] = idx + 2
```

| Token especial | Significado |
|---|---|
| `<PAD>` (índice 0) | Padding: relleno para igualar longitudes |
| `<UNK>` (índice 1) | Unknown: palabras fuera del vocabulario |

---

## 🔢 Texto → secuencia de índices (+ padding)

```python
def text_to_sequences(texts, word2idx, max_seq_length=100):
    sequences = []
    for text in texts:
        seq = [word2idx.get(word, word2idx['<UNK>']) for word in preprocess_string(text).split()]
        if len(seq) < max_seq_length:
            seq = seq + [word2idx['<PAD>']] * (max_seq_length - len(seq))  # padding
        else:
            seq = seq[:max_seq_length]   # truncar si es muy largo
        sequences.append(seq)
    return sequences
```

**¿Por qué padding?** Las redes neuronales trabajan con tensores de tamaño fijo. Las reseñas tienen longitudes variables → hay que igualarlas.

---

## 🔄 Mapeo de etiquetas

```python
label_map = {'molt dolent': 0, 'dolent': 1, 'regular': 2, 'bo': 3, 'molt bo': 4}
y_train = df_train['label'].map(label_map).values
```

---

## 🏗️ Arquitectura LSTM Bidireccional

```python
class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_size, num_layers, bidirectional=True, dropout_rate=0.3):
        super(SentimentLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers,
                            batch_first=True, bidirectional=bidirectional,
                            dropout=dropout_rate if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_dim * (2 if bidirectional else 1), output_size)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, (hidden, cell) = self.lstm(embedded)
        if self.lstm.bidirectional:
            last_hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        else:
            last_hidden = hidden[-1,:,:]
        return self.fc(self.dropout(last_hidden))
```

### Componentes explicados

| Componente | Función |
|---|---|
| `nn.Embedding` | Convierte índices de palabras a vectores densos (representación semántica) |
| `nn.LSTM` | Memoria a largo plazo para procesar secuencias |
| `bidirectional=True` | Lee la secuencia en ambas direcciones (→ y ←) |
| `nn.Dropout` | Regularización: desactiva neuronas aleatoriamente durante train (evita overfitting) |
| `nn.Linear` | Capa final de clasificación |

### ¿Por qué `hidden_dim * 2` si es bidireccional?
La LSTM bidireccional produce dos estados ocultos (uno por dirección) que se concatenan, doblando la dimensión.

---

## 🔁 Bucle de entrenamiento con train y validación

```python
criterion = nn.CrossEntropyLoss()   # Para clasificación multiclase
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(num_epochs):
    model.train()    # Modo entrenamiento (Dropout activo)
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    model.eval()     # Modo evaluación (Dropout inactivo)
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, dim=1)  # clase con max probabilidad
```

### CrossEntropyLoss vs MSELoss
- **MSELoss** → Regresión (valor continuo)
- **CrossEntropyLoss** → Clasificación multiclase

### `torch.max(outputs, dim=1)`
Retorna el valor máximo y su índice por cada fila. `dim=1` → máximo a lo largo de las clases (cada ejemplo → predicción de la clase con mayor probabilidad).

---

## 🎯 Preguntas tipo examen EAC4

**Q: ¿Qué diferencia hay entre MLP (EAC3) y LSTM (EAC4)?**  
R: MLP procesa vectores de características fijas (no tiene en cuenta el orden). LSTM procesa secuencias (tiene memoria y puede capturar relaciones secuenciales entre palabras).

**Q: ¿Para clasificación multiclase, qué función de pérdida se usa?**  
R: CrossEntropyLoss. Combina softmax + log-likelihood. Para clasificación binaria sería BCELoss.

**Q: ¿Qué hace nn.Embedding?**  
R: Mapea índices enteros (palabras) a vectores densos de dimensión fija. Permite que la red aprenda representaciones semánticas de las palabras.

**Q: ¿Por qué el modelo del GuiaCat predice mal las reseñas negativas?**  
R: Dataset muy desequilibrado: 85% reseñas positivas. El modelo aprende a predecir siempre "bo" o "molt bo" y obtiene alta accuracy, pero falla en las clases minoritarias.

---

# ══════════════════════════════════════════
# EAC5 — IA en la nube: AWS Textract + Gradio + Hugging Face
# ══════════════════════════════════════════

## 🧠 Idea general

Aplicación web que recibe una imagen de un ticket de compra, la envía a AWS Textract (OCR en la nube), extrae el importe total y acumula importes de múltiples tickets. Se aloja en Hugging Face Spaces.

**Servicios**: AWS Textract (OCR), Gradio (UI), Hugging Face (hosting)

---

## ☁️ ¿Por qué IA en la nube?

| Ventaja | Explicación |
|---|---|
| **Escalabilidad** | Puede procesar muchas peticiones simultáneas sin límite |
| **GPU bajo demanda** | No hace falta tener hardware propio |
| **APIs especializadas** | Servicios de OCR, visión, NLP ya entrenados y listos |
| **Pago por uso** | Solo se paga lo que se consume |

---

## 🔧 AWS Textract — OCR inteligente

Servicio AWS que extrae texto y datos estructurados de imágenes y PDFs. Modo FORMS → detecta pares clave-valor (como "Total: 12.50€").

```python
import boto3

textract = boto3.client(
    'textract',
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key,
    aws_session_token=aws_session_token,
    region_name='us-east-1'
)

with open(ruta_imatge, 'rb') as document:
    imageBytes = bytearray(document.read())

response = textract.analyze_document(
    Document={'Bytes': imageBytes},
    FeatureTypes=["FORMS"]   # Detecta pares clave-valor
)
```

---

## 📄 Parseo de la respuesta + limpieza regex

```python
from trp import Document

def obtenir_total_tiquet(response):
    doc = Document(response)
    for page in doc.pages:
        for field in page.form.fields:
            if field.key and field.value:
                key_text = field.key.text.strip().rstrip(':')
                if key_text.lower() == "total":
                    return field.value.text
                elif 'total' in key_text.lower():
                    total_value = field.value.text
    return total_value

def netejar_sortida(valor):
    valor_netejat = re.sub(r'[^\d,.]', '', valor)
    if ',' in valor_netejat and '.' in valor_netejat:
        valor_netejat = valor_netejat.replace('.', '').replace(',', '.')
    elif ',' in valor_netejat:
        valor_netejat = valor_netejat.replace(',', '.')
    return "{:.2f}".format(float(valor_netejat))
```

---

## 🖥️ Interfaz con Gradio (Blocks)

```python
total_acumulat = 0.0

def obtenir_total(ruta):
    global total_acumulat
    # ... (leer imagen → Textract → extraer total → formatear)
    total_acumulat += float(valor_netejat)
    return "{:.2f}".format(float(valor_netejat)), total_acumulat

with gr.Blocks(title="Import tiquets") as demo:
    gr.Markdown("<h1>Processador de tiquets</h1>")
    with gr.Row():
        tiquet   = gr.Textbox(label="Import del tiquet (€):")
        acumulat = gr.Number(label="Total acumulat (€):")
    imatge = gr.Image(type='filepath')   # ← filepath, no numpy!

    imatge.upload(fn=obtenir_total, inputs=imatge, outputs=[tiquet, acumulat])

demo.launch(debug=True)
```

### ¿Por qué `type='filepath'` y no `type='numpy'`?
Textract necesita los bytes de la imagen. Con `filepath` recibimos la ruta de un archivo temporal y podemos leerlo como bytes. Con `numpy` recibiríamos un array y habría que convertirlo.

---

## 🚀 Despliegue en Hugging Face Spaces

1. Crear un Space en huggingface.co (Gradio SDK)
2. Pasar el código del notebook a un archivo `app.py`
3. Crear `requirements.txt` con las dependencias
4. Subir `app.py` y `requirements.txt` al Space
5. **Añadir las credenciales AWS como Secrets** (nunca en el código)

```python
# MAL: credenciales en el código (¡nunca!)
aws_access_key_id = 'ASIATKZWHQXRRGFZGRAC'

# BIEN: leer de variables de entorno (Secrets del Space)
import os
aws_access_key_id = os.environ.get('AWS_ACCESS_KEY_ID')
```

> ⚠️ **Error crítico de seguridad**: Subir credenciales AWS al código público. Cualquiera puede usarlas y generar costes en AWS. SIEMPRE usar Secrets.

---

## 🎯 Preguntas tipo examen EAC5

**Q: ¿Por qué se usa `FeatureTypes=["FORMS"]` en Textract?**  
R: Para activar la detección de formularios (pares clave-valor). Sin este parámetro, Textract solo hace detección básica de bloques de texto y no estructura los datos en campos.

**Q: ¿Qué diferencia hay entre Gradio Interface y Gradio Blocks?**  
R: `gr.Interface` es simple: define inputs y outputs directamente. `gr.Blocks` permite layouts personalizados, múltiples componentes y eventos personalizados (upload, click, etc.).

**Q: ¿Por qué hay que poner las credenciales AWS como Secrets?**  
R: Por seguridad. Si se ponen en el código, quedan expuestas en el repositorio público. Cualquiera podría usarlas para generar costes o acceder a recursos privados de AWS.

---

# ══════════════════════════════════════════
# EAC6 — Estructuración de proyectos IA: La Liga
# ══════════════════════════════════════════

## 🧠 Idea general

EAC centrada en **cómo se estructura y gestiona un proyecto de IA real**, no solo en el código. Se analiza el dataset de partidos de La Liga desde 1995, y se aplica KMeans para clasificar equipos. El objetivo didáctico principal es aprender: modularización, entornos virtuales, control de versiones, tests, documentación y linting.

**Librerías**: `pandas`, `matplotlib`, `seaborn`, `sklearn`, `pickle`

---

## 📁 Estructura del proyecto — CONCEPTO CLAVE

```
lliga/
├── src/                    ← Código fuente
│   ├── main.py             ← Punto de entrada (orquesta los ejercicios)
│   ├── config.py           ← Configuración compartida (nombre alumno, fecha)
│   ├── exercises/          ← Módulos con las funciones de cada ejercicio
│   │   ├── __init__.py
│   │   ├── ex1.py, ex2.py... ex7.py
│   ├── data/               ← Dataset
│   ├── img/                ← Gráficas generadas
│   └── model/              ← Modelos guardados (.pkl)
├── tests/                  ← Tests unitarios
│   └── tests_ex6.py
├── doc/                    ← Documentación generada
├── screenshots/
├── requirements.txt        ← Dependencias del proyecto
├── README.md               ← Documentación principal
└── LICENSE.txt
```

### ¿Por qué esta estructura?
- **Separación de responsabilidades**: cada archivo tiene una función clara
- **Reproducibilidad**: cualquiera puede clonar y ejecutar sin problemas
- **Escalabilidad**: fácil añadir nuevos ejercicios/módulos
- **Profesionalidad**: estándar en proyectos reales de IA

> 🎯 **Pregunta típica**: "¿Qué contiene cada carpeta del proyecto?"  
> **Respuesta**: `src/` → código fuente. `tests/` → pruebas unitarias. `doc/` → documentación. `data/` → dataset. `model/` → modelos persistidos. `requirements.txt` → dependencias.

---

## ⚙️ config.py — Configuración centralizada

```python
from datetime import datetime

nom_alumne = "nom_alumne"
date_time  = datetime.now().strftime('%Y%m%d_%H%M%S')
```

Se importa en todos los módulos para tener el nombre y la fecha siempre disponibles:
```python
import config
plt.savefig(f"img/grafica_ex1_{config.nom_alumne}_{config.date_time}.png")
```

**¿Por qué?** Centralizar la configuración evita repetirla en cada archivo. Si cambia el nombre, solo se cambia en un sitio.

---

## 🌿 Entorno virtual (venv)

```bash
python3 -m venv venv          # Crear el entorno virtual
source venv/bin/activate      # Activarlo (Linux/Mac)
pip install -r requirements.txt  # Instalar dependencias
```

### ¿Por qué es fundamental?
- **Aísla las dependencias** del proyecto del resto del sistema
- Evita conflictos de versiones entre proyectos
- Hace el proyecto **reproducible**: otro usuario instala exactamente las mismas versiones

### requirements.txt
```
pandas
matplotlib
seaborn
sklearn
pickle
```

> 🎯 **Pregunta típica**: "¿Qué problema resuelve el entorno virtual?"  
> **Respuesta**: Aísla las dependencias del proyecto. Evita que las librerías de un proyecto interfieran con las de otro, y garantiza que todos los desarrolladores usan las mismas versiones.

---

## 🐙 Git y GitHub — Control de versiones

```bash
# Inicializar y primer commit
git init
git add .
git commit -m "Primer commit del projecte lliga"

# Conectar con GitHub y subir
git branch -M main
git remote add origin https://github.com/USUARI/lliga.git
git push -u origin main

# A partir de ahora, para subir cambios:
git add .
git commit -m "Descripció dels canvis"
git push
```

### ¿Por qué usar Git?
- **Historial de cambios**: puedes volver a cualquier versión anterior
- **Colaboración**: varios desarrolladores en el mismo proyecto
- **Backup**: el código está en la nube (GitHub)
- **Buena práctica** en todo proyecto de IA real

---

## 🔍 Pylint — Calidad del código

```bash
cd src/
pylint main.py
pylint exercises/*.py
```

Pylint analiza el código y detecta: errores, código mal formateado, variables no usadas, funciones sin docstring, etc.

### `.pylintrc`
Archivo de configuración que permite añadir excepciones a las reglas de Pylint (por ejemplo, aceptar nombres cortos de variables como `X` en ML).

---

## 📖 Documentación con pydoc

```bash
cd src/
pydoc3 -w `find . -name '*.py'`   # Genera HTML por cada .py
mv *.html ../doc                   # Mueve a la carpeta doc/
```

Genera automáticamente documentación HTML a partir de los **docstrings** de las funciones. Por eso es importante documentar bien cada función con docstring.

### Estructura de un docstring

```python
def fun_total_goals(data: pd.DataFrame) -> (int, int, int):
    """
    Número total de gols marcats com a locals o visitants

    Arguments
    data (DataFrame): el dataset

    Returns
    tuple(int, int, int): gols locals, visitants, totals
    """
    home_goals  = data['FTHG'].sum()
    away_goals  = data['FTAG'].sum()
    total_goals = home_goals + away_goals
    return home_goals, away_goals, total_goals
```

**Type hints** (`data: pd.DataFrame`, `-> (int, int, int)`): indican el tipo esperado de cada argumento y del retorno. Mejoran la legibilidad y permiten detectar errores.

---

## 🧪 Tests unitarios con unittest

```python
import unittest
import sys
sys.path.append("..")   # Para acceder a src/ desde tests/
from src.exercises.ex6 import fun_total_goals

class TestFunTotalGoals(unittest.TestCase):

    def test_basic_case(self):
        data = pd.DataFrame({'FTHG': [1, 2, 0], 'FTAG': [0, 1, 3]})
        home, away, total = fun_total_goals(data)
        self.assertEqual(home, 3)    # 1+2+0
        self.assertEqual(away, 4)    # 0+1+3
        self.assertEqual(total, 7)   # 3+4

    def test_empty_dataframe(self):
        data = pd.DataFrame({'FTHG': [], 'FTAG': []})
        home, away, total = fun_total_goals(data)
        self.assertEqual(total, 0)

if __name__ == '__main__':
    unittest.main()
```

```bash
# Ejecutar un archivo de tests
cd tests/
python3 tests_ex6.py

# Descubrir y ejecutar todos los tests
python3 -m unittest discover -s tests
```

### ¿Por qué hacer tests?
- Verifican que las funciones hacen lo que deben hacer
- Detectan errores cuando cambia el código (regresiones)
- Documentan el comportamiento esperado de cada función
- Práctica estándar en proyectos profesionales

### Estructura Arrange-Act-Assert (AAA)
```python
# Arrange: preparar los datos
data = pd.DataFrame(...)

# Act: ejecutar la función
home, away, total = fun_total_goals(data)

# Assert: verificar el resultado
self.assertEqual(home, 3)
```

> 🎯 **Pregunta típica**: "¿Qué ventaja tienen los tests unitarios?"  
> **Respuesta**: Garantizan que cada función hace lo que se espera. Si más adelante se modifica el código y algo se rompe, los tests lo detectan automáticamente.

---

## 📊 Análisis de datos: La Liga

### Ex1 — Carga y EDA

```python
data = pd.read_csv('data/LaLiga_Matches.csv')
print(data.head())
print(data.describe())
print(data.info())
data.drop(columns=["HTHG", "HTAG", "HTR"], inplace=True)  # eliminar columnas innecesarias
```

**Visualización**: Boxplot para ver la distribución de goles locales vs visitantes.

---

### Ex2 — Total de partidos por equipo

```python
matches_home  = data['HomeTeam'].value_counts()   # partidos como local
matches_away  = data['AwayTeam'].value_counts()   # partidos como visitante
matches_total = pd.DataFrame(matches_home) + pd.DataFrame(matches_away)
matches_total = matches_total.sort_values("count", ascending=False)

# Equipos que siempre han estado en primera (máximo de partidos)
sempre_a_primera = matches_total[matches_total['count'] == matches_total['count'].max()].index
```

---

### Ex3 — Distribución de goles

```python
distr_gols_locals    = data["FTHG"].value_counts().sort_index()
distr_gols_visitants = data["FTAG"].value_counts().sort_index()
```

`value_counts()` cuenta cuántas veces aparece cada valor. `sort_index()` ordena por el número de goles (0, 1, 2, 3...).

---

### Ex4 — FTR (Full Time Result)

```python
ftr = data['FTR'].value_counts()  # H=home wins, A=away wins, D=draw
# Calcular % de victorias locales
pct = 100 * ftr.loc['H','count'] / ftr['count'].sum()
```

---

### Ex5 — Puntos acumulados

```python
# Añadir columnas de puntos (regla de fútbol: W=3, D=1, L=0)
data["points_home"] = data["FTR"].map({"H": 3, "D": 1, "A": 0})
data["points_away"] = data["FTR"].map({"H": 0, "D": 1, "A": 3})

# Sumar por equipo
home_points  = data.groupby("HomeTeam")["points_home"].sum()
away_points  = data.groupby("AwayTeam")["points_away"].sum()
total_points = home_points.add(away_points, fill_value=0)
```

**`groupby` + `sum()`**: agrupa por equipo y suma los puntos. Patrón muy común en análisis de datos.

**`add(fill_value=0)`**: suma dos Series alineando por índice. `fill_value=0` evita NaN cuando un equipo no aparece en ambas Series.

---

### Ex6 — Resumen 1996-2025

```python
# pd.concat para combinar varios DataFrames en uno
resum_1996_2025 = pd.concat(
    [total_points, home_goals_by_team, away_goals_by_team, total_goals_by_team],
    axis=1
)
resum_1996_2025.columns = ["points", "home_goals", "away_goals", "total_goals"]

# Añadir capacidad de estadios (dict → DataFrame → join)
df_stadium_capacity = pd.DataFrame.from_dict(stadium_capacity, orient='index', columns=['stadium_capacity'])
resum_1996_2025 = resum_1996_2025.join(df_stadium_capacity)
```

---

### Ex7 — KMeans + persistencia del modelo

```python
import pickle

def model_clusters(df, num_clusters):
    X = df[["points", "home_goals", "away_goals", "total_goals", "stadium_capacity"]]

    scaler  = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(X_scaled)

    # Guardar modelo con scaler incluido
    model_data = {"scaler": scaler, "num_clusters": num_clusters, "kmeans": kmeans}
    with open(f"model/model_{num_clusters}.pkl", 'wb') as f:
        pickle.dump(model_data, f)

    return kmeans
```

### ¿Por qué guardar también el scaler junto con el KMeans?
Para hacer inferencia con nuevos datos hay que escalarlos con el **mismo scaler** que se usó en el entrenamiento. Si solo guardamos el KMeans, no podemos escalar correctamente los datos nuevos.

### Inferencia con modelo guardado

```python
# Cargar modelo
with open("model/model_4.pkl", "rb") as f:
    model_data = pickle.load(f)

scaler   = model_data["scaler"]
kmeans   = model_data["kmeans"]

# Preparar nuevos datos
X_new        = df_europa[["points", "home_goals", "away_goals", "total_goals", "stadium_capacity"]]
X_new_scaled = scaler.transform(X_new)   # mismo scaler!

cluster = kmeans.predict(X_new_scaled)
```

### pickle vs torch.save
| | `pickle` | `torch.save` |
|---|---|---|
| Para qué | Cualquier objeto Python (sklearn, dicts, etc.) | Modelos PyTorch |
| Cómo | `pickle.dump` / `pickle.load` | `torch.save` / `torch.load` |

---

## 🎯 Preguntas tipo examen EAC6

**Q: ¿Para qué sirve el archivo `requirements.txt`?**  
R: Lista las dependencias del proyecto con sus versiones. Permite instalar exactamente las mismas librerías en cualquier entorno con `pip install -r requirements.txt`, garantizando la reproducibilidad.

**Q: ¿Por qué se usa `sys.path.append("..")` en los tests?**  
R: Porque los tests están en la carpeta `tests/` y necesitan importar código de `src/`. Son carpetas hermanas, no padre-hijo. Se añade el directorio padre al path para que Python pueda encontrar los módulos.

**Q: ¿Qué ventaja tiene separar el código en múltiples archivos (ex1.py, ex2.py...)?**  
R: Modularización: cada archivo tiene una responsabilidad clara, el código es más legible y mantenible, facilita los tests unitarios por función, y varios desarrolladores pueden trabajar en paralelo sin conflictos.

**Q: ¿Por qué se guarda el scaler junto con el modelo KMeans en el .pkl?**  
R: Para hacer inferencia correctamente. Los datos nuevos deben escalarse con el mismo scaler que se usó en el entrenamiento. Si solo se guarda el KMeans, no se puede reproducir el escalado y las predicciones serían incorrectas.

**Q: ¿Qué hace `data.groupby("HomeTeam")["points_home"].sum()`?**  
R: Agrupa las filas del DataFrame por equipo local, y para cada grupo suma los puntos obtenidos en casa. Retorna una Series con el total de puntos como local de cada equipo.

**Q: ¿Para qué sirve pylint?**  
R: Para analizar la calidad del código de forma automática: detecta errores, variables no usadas, funciones sin documentar, código mal formateado. Ayuda a mantener un código limpio y consistente.

---

# ══════════════════════════════════════════
# RESUMEN TRANSVERSAL — Conceptos que atraviesan todas las EACs
# ══════════════════════════════════════════

## 🐍 Buenas prácticas Python (comunes a todo)

| Práctica | Por qué |
|---|---|
| `random_state` / `np.random.seed()` | Reproducibilidad: resultados idénticos cada ejecución |
| Funciones reutilizables | Código limpio, evitar repeticiones (DRY) |
| Comentarios + docstrings | Legibilidad, requerimiento de las EACs |
| Rutas relativas | El profesor puede ejecutar el notebook en su entorno |
| Entornos virtuales (venv) | Aísla dependencias, evita conflictos |
| `Restart + Run All` | Verificar que el notebook se ejecuta limpio de arriba abajo |

---

## 🔄 Flujo estándar de un proyecto ML/DL

```
1. Cargar datos
2. EDA: shape, info(), head(), visualizar
3. Limpiar datos: dropna(), drop_duplicates(), eliminar columnas innecesarias
4. Preprocesar: encoders, scalers, tokenizadores
5. Dividir train/test (o train/val/test)
6. Definir modelo
7. Definir loss + optimizador
8. Entrenamiento (bucle epochs/batches)
9. Evaluación (métricas en test)
10. Guardar modelo
11. Inferencia con nuevos datos
```

---

## ⚡ GPU vs CPU en PyTorch

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = model.to(device)
data   = data.to(device)
```

**Regla**: modelo y datos deben estar en el **mismo dispositivo**.

> ⚠️ **Error típico**: Tener el modelo en GPU y los datos en CPU → RuntimeError

---

## 📦 fit() vs transform() — Regla de oro

| | `fit()` | `transform()` | `fit_transform()` |
|---|---|---|---|
| Qué hace | Aprende los parámetros | Aplica la transformación | Los dos a la vez |
| Cuándo usar | Solo en **train** | En train y **test** | Cómodamente en **train** |

**Nunca** hacer `fit` con test → data leakage.

---

## 🧮 Comparativa métricas

### Regresión (EAC3)
MSE, RMSE, MAE, R²

### Clasificación (EAC2, EAC4)
Accuracy, Matriz de confusión

### Clustering (EAC2, EAC6)
Silhouette score, Inercia (método del codo)

---

## 🏗️ Arquitecturas de redes neuronales

| Arquitectura | Cuándo usar | EAC |
|---|---|---|
| **MLP** | Datos tabulares (vector de features) | EAC3 |
| **LSTM** | Secuencias (texto, series temporales) | EAC4 |
| **LSTM bidireccional** | Texto (contexto completo en ambas direcciones) | EAC4 |

---

## 🔧 Comparativa optimizadores

| Optimizador | Características |
|---|---|
| **SGD** | Simple, requiere ajustar el learning rate manualmente |
| **Adam** | Adaptativo, generalmente el mejor para DL. lr=0.001-0.01 |

---

## 💾 Guardar modelos: torch.save vs pickle

| | `torch.save` / `torch.load` | `pickle.dump` / `pickle.load` |
|---|---|---|
| Para qué | Modelos PyTorch (state_dict) | Cualquier objeto Python (sklearn, dicts...) |
| EAC | EAC3, EAC4 | EAC6 |
| Buena práctica | Guardar `state_dict()`, no el modelo completo | Guardar dict con modelo + scaler juntos |

---

## 📚 Resumen librerías por EAC

| Librería | EAC1 | EAC2 | EAC3 | EAC4 | EAC5 | EAC6 |
|---|---|---|---|---|---|---|
| pandas | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| numpy | ✅ | ✅ | ✅ | ✅ | - | - |
| matplotlib | ✅ | ✅ | ✅ | ✅ | - | ✅ |
| yfinance | ✅ | - | - | - | - | - |
| sklearn | - | ✅ | ✅ | ✅ | - | ✅ |
| seaborn | - | ✅ | - | - | - | ✅ |
| torch / PyTorch | - | - | ✅ | ✅ | - | - |
| datasets (HF) | - | - | - | ✅ | - | - |
| nltk | - | - | - | ✅ | - | - |
| boto3 | - | - | - | - | ✅ | - |
| gradio | - | - | - | - | ✅ | - |
| pickle | - | - | - | - | - | ✅ |

---

*Documento generado automáticamente por Claude a partir de las soluciones oficiales EAC1-EAC6 del módulo IAB-5073 · Curs 2526S2*
