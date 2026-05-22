# 💻 QUÉ HACE EL CÓDIGO — EAC1 a EAC6
> Explicación bloque a bloque de cada EAC · Examen M03 IAB-5073

---

# ══════════════════════════════════════════
# EAC1 — Análisis de series temporales NVIDIA
# ══════════════════════════════════════════

## PASO 1 — Importar librerías

```python
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
```

Se importan las 4 librerías que se usarán en toda la EAC.
`yfinance` es la única que no es estándar y permite descargar datos de Yahoo Finance.

---

## PASO 2 — Descargar datos de NVIDIA

```python
nvd = yf.download("NVDA", start="2023-01-01", end="2026-02-04", progress=False)
nvd = nvd["Close"]
nvd = nvd.dropna()
```

- `yf.download(...)` descarga datos históricos de la bolsa. Retorna un **DataFrame** con columnas OHLCV (Open, High, Low, Close, Volume).
- `nvd["Close"]` se queda solo con la columna del precio de cierre. El resultado puede ser un DataFrame de una sola columna.
- `dropna()` elimina las filas con valores nulos (días sin cotización).

---

## PASO 3 — Convertir a Series con squeeze()

```python
nvd = nvd.squeeze()
idx = nvd.index
```

- `squeeze()` convierte el DataFrame de una columna a Series (1D). Esto es necesario porque muchas operaciones de Pandas esperan una Series, no un DataFrame.
- `idx` guarda el índice de fechas para usarlo más adelante en `reindex(idx)`.

---

## PASO 4 — Calcular las medias móviles

```python
nvd_sma21  = nvd.rolling(21).mean().reindex(idx)
nvd_sma200 = nvd.rolling(200).mean().reindex(idx)

weights = np.arange(1, 201)   # pesos lineales: 1, 2, 3, ..., 200
nvd_wma200 = nvd.rolling(200).apply(
    lambda x: np.dot(x, weights) / weights.sum(), raw=True
).reindex(idx)

nvd_ema50  = nvd.ewm(span=50, adjust=False).mean().reindex(idx)
nvd_ema100 = nvd.ewm(span=100, adjust=False).mean().reindex(idx)
```

- `.rolling(N).mean()` → SMA: promedio de los N días anteriores, mismo peso a todos.
- `.rolling(200).apply(lambda x: np.dot(x, weights) / weights.sum())` → WMA: producto escalar entre precios y pesos lineales, dividido por la suma de pesos. Los días recientes pesan más.
- `.ewm(span=N).mean()` → EMA: media exponencial, los días recientes pesan exponencialmente más.
- `.reindex(idx)` → asegura que el índice de fechas coincide exactamente con el del precio original. Evita desalineaciones.

---

## PASO 5 — Construir el DataFrame combinado

```python
df = pd.DataFrame({
    "Close": nvd,
    "SMA21": nvd_sma21,
    "SMA200": nvd_sma200,
    "WMA200": nvd_wma200,
    "EMA50": nvd_ema50,
    "EMA100": nvd_ema100
})
```

Junta el precio original y todas las medias en una sola tabla. Los primeros 200 días tendrán NaN en SMA200 y WMA200 (no hay suficiente historial).

---

## PASO 6 — Graficar precio + medias

```python
plt.figure(figsize=(14,7))
plt.plot(df.index, df["Close"], label="NVIDIA Close", color='black')
plt.plot(df.index, df["SMA21"], label="SMA 21", alpha=0.7)
# ... (mismo para cada media)
plt.legend()
plt.show()
```

Gráfico con todas las líneas superpuestas. El precio negro es el más volátil, las medias son curvas más suaves.

---

## PASO 7 — Media de medias

```python
df["MA_Avg"] = df[["SMA21","SMA200","WMA200","EMA50","EMA100"]].mean(axis=1)
```

`axis=1` → calcula la media horizontalmente (por filas), no por columnas. Cada día obtiene la media de los 5 indicadores. Resultado: curva muy suavizada que representa el "consenso" de todas las medias.

---

## PASO 8 — Regresión polinómica

```python
df["timestamp"] = (df.index.astype(np.int64) // 10**9)

x = df["timestamp"].values
y = df["MA_Avg"].values

coeffs = np.polyfit(x, y, 7)
poly   = np.poly1d(coeffs)
```

- Las fechas se convierten a Unix timestamp (segundos enteros desde 1970) porque `np.polyfit` necesita números, no fechas.
- `np.polyfit(x, y, 7)` calcula los 8 coeficientes del polinomio de grado 7 que mejor se ajusta a los puntos (x, y) por mínimos cuadrados.
- `np.poly1d(coeffs)` crea un objeto función: `poly(x)` evalúa el polinomio en cualquier punto.

---

## PASO 9 — Extrapolación al futuro

```python
future_days = 100
last_date   = df.index[-1]
dates_future = pd.date_range(start=last_date, periods=future_days)

timestamps_future = (dates_future.astype(np.int64) // 10**9).values
pred_future = poly(timestamps_future)

max_idx_future = np.argmax(pred_future)
max_date  = dates_future[max_idx_future]
max_price = pred_future[max_idx_future]
```

- Se generan 100 fechas futuras a partir del último día del dataset.
- Se convierten a timestamps y se evalúa el polinomio en esas fechas → predicción futura.
- `np.argmax(pred_future)` devuelve el índice donde el polinomio tiene su máximo → el día que el modelo predice el precio más alto.

---

## PASO 10 — Script .py (Ejercicio 7)

El código del notebook se pasa a un archivo `.py`:
- Se añade cabecera con nombre y fecha.
- `plt.show()` se reemplaza por `plt.savefig("img/grafica_NOM_FECHA.png")`.
- Se ejecuta con `python EAC1_nom.py` desde la terminal.

---

# ══════════════════════════════════════════
# EAC2 — Machine Learning Clásico (sklearn)
# ══════════════════════════════════════════

## P1 — Regresión lineal: péndulo

### PASO 1 — Generar datos simulados

```python
np.random.seed(42)
g_real = 9.81
L = np.linspace(0.2, 1.5, 20)          # 20 longitudes entre 0.2 y 1.5m
T = 2 * np.pi * np.sqrt(L / g_real)    # período teórico
noise = np.random.normal(0, 0.02, size=T.shape)
T_noisy = T * (1 + noise)              # período con 2% de ruido
T2 = T_noisy ** 2                      # variable linealizada
```

- `np.random.seed(42)` → reproducibilidad: siempre los mismos números aleatorios.
- Se añade ruido del ±2% al período para simular error experimental.
- Se usa T² porque la relación entre L y T² es lineal (la ecuación del péndulo linealizada).

### PASO 2 — Modelo de regresión

```python
X = data[["L"]]   # 2D obligatorio para sklearn
y = data["T2"]

model = LinearRegression(fit_intercept=False)
model.fit(X, y)
```

- `X` debe ser 2D (por eso `[["L"]]` y no `["L"]`). sklearn espera matrices, no vectores.
- `fit_intercept=False` → la recta pasa por el origen (si L=0, T²=0, por la física).
- `model.fit(X, y)` calcula el mejor coeficiente (pendiente) por mínimos cuadrados.

### PASO 3 — Calcular g y métricas

```python
m  = model.coef_[0]
g_estimada = (4 * np.pi**2) / m
r2 = model.score(X, y)
error_absoluto = abs(g_estimada - g_real)
error_relativo = (error_absoluto / g_real) * 100
```

- De la ecuación T² = (4π²/g)·L, la pendiente m = 4π²/g → despejando: g = 4π²/m.
- `model.coef_[0]` → el primer (y único) coeficiente del modelo.
- `model.score(X, y)` devuelve R² directamente.

---

## P2 — Clasificación: Iris

### PASO 1 — Cargar y limpiar

```python
iris = pd.read_csv("data/Iris.csv")
iris.drop('Id', axis=1, inplace=True)
```

Se elimina la columna 'Id' porque es un identificador sin información útil.

### PASO 2 — Visualización exploratoria

```python
# Scatter sépalos vs scatter pétalos
iris[iris.Species=='Iris-setosa'].plot(kind='scatter', x='SepalLengthCm', y='SepalWidthCm', ...)
```

Se hacen dos scatter plots: uno con sépalos y otro con pétalos. La conclusión es que los pétalos separan mucho mejor las 3 especies visualmente.

```python
iris.select_dtypes(include='number').corr()
sns.heatmap(iris.select_dtypes(include='number').corr(), annot=True)
```

- `.corr()` calcula la matriz de correlación entre todas las variables numéricas.
- El heatmap la muestra visualmente. Se observa que PetalLength y PetalWidth tienen correlación 0.962 → muy relacionadas. SepalLength y SepalWidth tienen correlación -0.109 → casi independientes.
- `select_dtypes(include='number')` filtra solo columnas numéricas (excluye 'Species' que es texto).

```python
sns.violinplot(x='Species', y='PetalLengthCm', data=iris)
```

Muestra la distribución de cada variable por especie. Las Iris-setosa tienen pétalos claramente más cortos → las medidas de pétalos son discriminativas.

### PASO 3 — Separar train/test y features/labels

```python
train, test = train_test_split(iris, test_size=0.3)

train_X = train[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]
train_y = train.Species
test_X  = test[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]
test_y  = test.Species
```

- 70% train, 30% test.
- Se separan features (X, 4 columnas) de labels (y, la especie) tanto en train como en test.

### PASO 4 — Función reutilizable + entrenar 4 modelos

```python
def train_and_evaluate(model, train_X, train_y, test_X, test_y):
    model.fit(train_X, train_y)
    predictions = model.predict(test_X)
    accuracy = metrics.accuracy_score(test_y, predictions)
    return accuracy, predictions

# Se llama 4 veces con cada modelo:
model = svm.SVC()
model = LogisticRegression()
model = DecisionTreeClassifier()
model = KNeighborsClassifier(n_neighbors=3)
```

La función encapsula el flujo: entrenar → predecir → evaluar. Se llama con cada modelo. Permite comparar los 4 con el mismo código limpio.

### PASO 5 — KNN con K variable

```python
for i in range(1, 11):
    model = KNeighborsClassifier(n_neighbors=i)
    accuracy, preds = train_and_evaluate(model, ...)
    a.append(accuracy)
plt.plot(range(1,11), a)
```

Se prueban K=1 hasta K=10 y se grafica la accuracy de cada uno. Permite elegir el K óptimo visualmente.

### PASO 6 — Comparar pétalos vs sépalos

```python
petal = iris[['PetalLengthCm','PetalWidthCm','Species']]
sepal = iris[['SepalLengthCm','SepalWidthCm','Species']]

# train_test_split y entrenamiento de los 4 modelos dos veces:
# una con pétalos y otra con sépalos
```

Se separa el dataset en dos subsets: solo pétalos y solo sépalos. Se entrena cada modelo dos veces. Resultado: con pétalos >0.9 de accuracy en todos los modelos, con sépalos <0.82. Confirma que los pétalos son más discriminativos.

---

## P3 — Clustering: KMeans

### PASO 1 — Generar datos sintéticos

```python
X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=1.0, random_state=42)
```

`make_blobs` genera 300 puntos alrededor de 4 centroides. `y_true` contiene las etiquetas reales (no se usan en el clustering, solo para comparar después).

### PASO 2 — Crear y entrenar KMeans

```python
kmeans = KMeans(n_clusters=4, init='k-means++', n_init=10, random_state=42)
kmeans.fit(X)
labels = kmeans.labels_
```

- `init='k-means++'` → inicialización inteligente de centroides (mejor que aleatoria).
- `n_init=10` → ejecuta el algoritmo 10 veces y se queda con el mejor resultado.
- `kmeans.labels_` → array con el número de clúster de cada punto.

### PASO 3 — Evaluar con Silhouette Score

```python
sil_score = silhouette_score(X, labels)
```

Mide la calidad del clustering: qué tan compactos y separados están los clústeres. Rango [-1, 1].

### PASO 4 — Visualización

```python
plt.scatter(X[:,0], X[:,1], c=labels, cmap='viridis', s=50)
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1],
            c='red', s=200, marker='X', label='Centroids')
```

- `c=labels` → colorea cada punto según su clúster.
- `kmeans.cluster_centers_` → las coordenadas de los centroides (marcados con X roja).

### PASO 5 — Widget interactivo (ipywidgets)

```python
def plot_clusters(std_dev):
    X, _ = make_blobs(..., cluster_std=std_dev)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    kmeans = KMeans(n_clusters=4, ...).fit(X_scaled)
    # ... graficar

interactive_plot = interactive(plot_clusters, std_dev=widgets.FloatSlider(min=1.0, max=5.0))
display(interactive_plot)
```

Un slider permite cambiar la desviación típica de los clústeres de 1 a 5. A mayor desviación, los clústeres se solapan más y el silhouette score baja. Muestra visualmente cómo afecta la dispersión de los datos al clustering.

---

# ══════════════════════════════════════════
# EAC3 — Deep Learning PyTorch: Regresión MLP
# ══════════════════════════════════════════

## PASO 1 — Importar y detectar GPU

```python
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

Se detecta si hay GPU disponible. Todo el modelo y los datos se moverán a ese dispositivo.

---

## PASO 2 — Cargar y limpiar datos

```python
url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/penguins.csv"
penguins = pd.read_csv(url)
penguins = penguins[['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'sex', 'body_mass_g']]
penguins = penguins.dropna()
```

- Se seleccionan solo 5 columnas: 3 medidas morfológicas, sexo y el peso (objetivo).
- `dropna()` elimina los pingüinos con datos faltantes.

---

## PASO 3 — Codificar la variable categórica

```python
encoder_sex = LabelEncoder()
penguins['sex'] = encoder_sex.fit_transform(penguins['sex'])
# 'FEMALE' → 0,  'MALE' → 1
```

Las redes neuronales no pueden trabajar con texto. LabelEncoder convierte los strings a enteros. **Importante**: guardar el encoder para poder usarlo exactamente igual en la inferencia.

---

## PASO 4 — Separar features y target, dividir train/test

```python
X = penguins.drop('body_mass_g', axis=1)   # features: 4 columnas
y = penguins['body_mass_g']                 # target: peso en gramos

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

80% para entrenamiento, 20% para test. `random_state=42` garantiza reproducibilidad.

---

## PASO 5 — Escalar con MinMaxScaler

```python
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_train_scaled = scaler_X.fit_transform(X_train)       # fit + transform en train
y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1))

X_test_scaled = scaler_X.transform(X_test)             # solo transform en test
y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1))
```

- Dos scalers separados: uno para X y otro para y (target). Se necesita el de y para invertir la escala al hacer predicciones.
- `.reshape(-1, 1)` convierte el vector y a matriz 2D, necesario para MinMaxScaler.
- `fit_transform` solo en train. `transform` en test → evita data leakage.

---

## PASO 6 — Convertir a tensores PyTorch

```python
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32).to(device)
X_test_tensor  = torch.tensor(X_test_scaled,  dtype=torch.float32).to(device)
y_test_tensor  = torch.tensor(y_test_scaled,  dtype=torch.float32).to(device)
```

Los arrays de NumPy se convierten a tensores. `dtype=torch.float32` es el tipo numérico estándar para redes neuronales. `.to(device)` los mueve a GPU o CPU.

---

## PASO 7 — Crear TensorDataset y DataLoader

```python
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader  = DataLoader(train_dataset, batch_size=16, shuffle=True)
```

- `TensorDataset` agrupa X e y en un único objeto iterable (pares X, y).
- `DataLoader` gestiona el acceso: divide en batches de 16, baraja al principio de cada epoch.

---

## PASO 8 — Definir la arquitectura MLP

```python
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1  = nn.Linear(input_size, hidden_size)  # 4 → 10
        self.fc2  = nn.Linear(hidden_size, hidden_size) # 10 → 10
        self.fc3  = nn.Linear(hidden_size, output_size) # 10 → 1
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))  # capa 1 + activación
        x = self.relu(self.fc2(x))  # capa 2 + activación
        x = self.fc3(x)             # capa salida SIN activación (regresión)
        return x

input_size  = X_train_tensor.shape[1]  # = 4
hidden_size = 10
output_size = 1

model = MLP(input_size, hidden_size, output_size).to(device)
```

- `nn.Linear(a, b)` → capa totalmente conectada con `a` entradas y `b` salidas.
- `forward()` define el flujo de datos de entrada a salida.
- Sin activación en la última capa porque es **regresión** → necesitamos cualquier valor real.
- `input_size = X_train_tensor.shape[1]` → número de features (columnas) del dataset.

---

## PASO 9 — Definir loss y optimizador

```python
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
```

- `MSELoss` → función de pérdida para regresión. Calcula el error cuadrático medio entre predicción y valor real.
- `Adam` → optimizador que ajusta los pesos. `model.parameters()` le dice qué parámetros optimizar.

---

## PASO 10 — Bucle de entrenamiento

```python
loss_list = []
num_epochs = 100

for epoch in range(num_epochs):
    epoch_loss = 0
    for batch_X, batch_y in train_loader:        # iterar por batches
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)

        outputs    = model(batch_X)              # 1. Forward pass → predicción
        loss       = criterion(outputs, batch_y) # 2. Calcular error
        epoch_loss += loss.item()

        optimizer.zero_grad()                    # 3. Resetear gradientes acumulados
        loss.backward()                          # 4. Calcular gradientes (backprop)
        optimizer.step()                         # 5. Actualizar pesos

    epoch_loss /= len(train_loader)              # pérdida media del epoch
    loss_list.append(epoch_loss)
```

- Cada epoch = una pasada completa por todo el dataset, dividida en batches.
- El orden `zero_grad → backward → step` es obligatorio e invariable.
- `loss.item()` extrae el valor numérico del tensor (un escalar Python).
- Se grafica `loss_list` al final para ver cómo baja la pérdida con el entrenamiento.

---

## PASO 11 — Evaluar el modelo

```python
model.eval()
with torch.no_grad():
    outputs = model(X_test_tensor)
    y_pred  = outputs.cpu().numpy()
    y_true  = y_test_tensor.cpu().numpy()

mse  = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)
mae  = mean_absolute_error(y_true, y_pred)
r2   = r2_score(y_true, y_pred)
```

- `model.eval()` → modo evaluación (desactiva Dropout).
- `torch.no_grad()` → no calcula gradientes (ahorra memoria, no hacen falta).
- `.cpu().numpy()` → mueve de GPU a CPU y convierte a array NumPy para sklearn.
- Las métricas se calculan sobre datos normalizados [0,1].

---

## PASO 12 — Guardar y cargar el modelo

```python
torch.save(model.state_dict(), "model.pth")

# Cargar:
loaded_model = MLP(input_size, hidden_size, output_size)
loaded_model.load_state_dict(torch.load("model.pth"))
loaded_model.to(device)
loaded_model.eval()
```

`state_dict()` → diccionario con todos los pesos y sesgos de la red. Es la forma recomendada de persistir modelos PyTorch.

---

## PASO 13 — Inferencia con datos nuevos

```python
new_data = [[37.7, 18.7, 180, 'MALE'], [46.4, 18.6, 190, 'FEMALE'], [48.5, 14.1, 220, 'MALE']]
new_df   = pd.DataFrame(new_data, columns=X.columns)

new_df['sex'] = encoder_sex.transform(new_df['sex'])   # mismo encoder que en train
new_data_scaled = scaler_X.transform(new_df)            # mismo scaler que en train
new_data_tensor = torch.tensor(new_data_scaled, dtype=torch.float32).to(device)

with torch.no_grad():
    outputs         = model(new_data_tensor)
    predicted_scaled = outputs.cpu().numpy()

predicted = scaler_y.inverse_transform(predicted_scaled)  # volver a gramos
```

Pasos críticos:
1. Mismo encoder → convierte 'MALE'/'FEMALE' igual que en training.
2. Mismo scaler_X → escala con los mismos parámetros.
3. `inverse_transform` con scaler_y → convierte la predicción normalizada de vuelta a gramos reales.

---

# ══════════════════════════════════════════
# EAC4 — Deep Learning NLP: Clasificación de sentimientos LSTM
# ══════════════════════════════════════════

## PASO 1 — Cargar datos desde Hugging Face

```python
from datasets import load_dataset

ds_train = load_dataset("projecte-aina/GuiaCat", split="train")
ds_val   = load_dataset("projecte-aina/GuiaCat", split="validation")
ds_test  = load_dataset("projecte-aina/GuiaCat", split="test")

df_train = ds_train.to_pandas()
df_val   = ds_val.to_pandas()
df_test  = ds_test.to_pandas()

df_train = df_train[['text', 'label']]
```

El dataset ya viene dividido en train/val/test. Se convierte a Pandas para trabajar con él fácilmente. Se conservan solo las columnas 'text' (la reseña) y 'label' (el sentimiento).

---

## PASO 2 — Exploración: dataset desequilibrado

```python
label_counts = df_train['label'].value_counts()
label_counts.plot(kind='bar')
```

El gráfico muestra que 'bo' y 'molt bo' representan el 85% de los datos. Esto significa que el modelo tenderá a predecir siempre positivo. Las clases negativas quedarán sin aprender.

---

## PASO 3 — Limpiar el texto

```python
def preprocess_string(s):
    s = s.lower()
    s = re.sub(r"[^a-zA-Zàèéíïòóúüç\s]", ' ', s)   # eliminar números y signos
    stop_words = set(stopwords.words('catalan'))
    s = ' '.join([w for w in s.split() if w not in stop_words])  # eliminar stop words
    s = ' '.join(s.split())   # limpiar espacios múltiples
    return s
```

Cada reseña pasa por 4 transformaciones: minúsculas → eliminar no-alfabéticos → eliminar stop words → limpiar espacios. El resultado es texto limpio listo para tokenizar.

---

## PASO 4 — Construir el vocabulario

```python
tokenizer = Counter()
for text in df_train['text']:
    clean_text = preprocess_string(text)
    tokenizer.update(clean_text.split())   # contar frecuencias de cada palabra
```

`Counter` cuenta cuántas veces aparece cada palabra en todo el corpus de entrenamiento.

```python
maxim_nombre_paraules = 1000
word2idx = {'<PAD>': 0, '<UNK>': 1}
for idx, (word, _) in enumerate(tokenizer.most_common(1000)):
    word2idx[word] = idx + 2
```

Se crea el diccionario `word2idx`: las 1000 palabras más frecuentes del corpus obtienen un índice numérico único. `<PAD>` es 0 y `<UNK>` es 1 (reservados). A partir de 2 van las palabras reales.

---

## PASO 5 — Convertir texto a secuencias de índices + padding

```python
def text_to_sequences(texts, word2idx, max_seq_length=100):
    sequences = []
    for text in texts:
        clean_text = preprocess_string(text)
        seq = [word2idx.get(word, word2idx['<UNK>']) for word in clean_text.split()]
        if len(seq) < max_seq_length:
            seq = seq + [word2idx['<PAD>']] * (max_seq_length - len(seq))  # rellenar
        else:
            seq = seq[:max_seq_length]   # truncar
        sequences.append(seq)
    return sequences
```

Cada texto se convierte a una lista de 100 enteros exactamente:
- Cada palabra → su índice. Si no está en el vocabulario → índice de `<UNK>`.
- Si el texto tiene <100 palabras → se rellena con 0s (`<PAD>`).
- Si tiene >100 palabras → se trunca.

---

## PASO 6 — Mapear etiquetas a números

```python
label_map = {'molt dolent': 0, 'dolent': 1, 'regular': 2, 'bo': 3, 'molt bo': 4}
y_train = df_train['label'].map(label_map).values
```

Los textos de etiquetas se convierten a enteros 0-4 para la red neuronal.

---

## PASO 7 — Tensores y DataLoaders

```python
X_train_tensor = torch.tensor(X_train).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader  = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader    = DataLoader(val_dataset, batch_size=64, shuffle=False)  # val no se baraja
```

- `dtype=torch.long` (enteros largos) para las etiquetas de clasificación. Diferente de EAC3 donde se usaba `float32`.
- El val_loader tiene `shuffle=False` → en validación el orden no importa y así los resultados son reproducibles.

---

## PASO 8 — Definir la arquitectura LSTM

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
        embedded = self.embedding(x)                        # índices → vectores
        lstm_out, (hidden, cell) = self.lstm(embedded)      # procesar secuencia
        if self.lstm.bidirectional:
            last_hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)  # concatenar ambas direcciones
        else:
            last_hidden = hidden[-1,:,:]
        return self.fc(self.dropout(last_hidden))           # clasificar
```

Flujo de datos:
1. `Embedding` → convierte cada índice (palabra) a un vector de 64 números. Aprende representaciones semánticas.
2. `LSTM bidireccional` → lee la secuencia de izquierda a derecha Y de derecha a izquierda. Produce un estado oculto final que resume toda la reseña.
3. `Concatenación` → junta el estado final de las dos direcciones. Si hidden_dim=128 → vector de 256.
4. `Dropout` → regularización: apaga el 30% de neuronas aleatoriamente.
5. `Linear` → capa final que mapea el vector de 256 a 5 valores (uno por clase).

Parámetros usados:
```python
vocab_size    = len(word2idx)  # 1002 (1000 + PAD + UNK)
embedding_dim = 64
hidden_dim    = 128
output_size   = 5              # 5 clases de sentimiento
num_layers    = 2
```

---

## PASO 9 — Entrenar con train y validar en cada epoch

```python
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(num_epochs):
    # === ENTRENAMIENTO ===
    model.train()                                   # activa Dropout
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    # === VALIDACIÓN ===
    model.eval()                                    # desactiva Dropout
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, dim=1)  # clase con max probabilidad
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

accuracy = accuracy_score(all_labels, all_predictions)
```

- `CrossEntropyLoss` → combina softmax + log-likelihood. La función correcta para clasificación multiclase.
- `torch.max(outputs, dim=1)` → devuelve el valor máximo y su índice por cada fila. El `_` descarta el valor (no interesa), `predicted` son los índices (las clases predichas).
- Se alterna entre `model.train()` y `model.eval()` en cada epoch.

---

## PASO 10 — Guardar, cargar e inferencia

```python
torch.save(model.state_dict(), "sentiment_lstm_model.pth")

# Cargar:
loaded_model = SentimentLSTM(vocab_size, embedding_dim, hidden_dim, output_size, num_layers, bidirectional)
loaded_model.load_state_dict(torch.load(model_path))
loaded_model.eval()
```

```python
# Inferencia:
X_test_sample = text_to_sequences(sample_texts, word2idx)   # mismo procesamiento
X_test_tensor = torch.tensor(X_test_sample).to(device)

with torch.no_grad():
    outputs = loaded_model(X_test_tensor)
    _, predicted_indices = torch.max(outputs, dim=1)

idx_to_label = {v: k for k, v in label_map.items()}   # invertir el mapa
predicted_labels = [idx_to_label[idx.item()] for idx in predicted_indices]
```

`{v: k for k, v in label_map.items()}` invierte el diccionario: de `{'molt dolent': 0}` a `{0: 'molt dolent'}`. Permite convertir los índices predichos de vuelta a texto legible.

---

# ══════════════════════════════════════════
# EAC5 — AWS Textract + Gradio + Hugging Face
# ══════════════════════════════════════════

## PASO 1 — Instalar librerías y crear cliente AWS

```python
!pip install boto3 gradio amazon-textract-response-parser

import boto3
textract = boto3.client(
    'textract',
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key,
    aws_session_token=aws_session_token,
    region_name='us-east-1'
)
```

`boto3.client('textract', ...)` crea un objeto que representa la conexión con el servicio Textract de AWS. Las credenciales autentican la sesión.

---

## PASO 2 — Leer la imagen y enviarla a Textract

```python
with open(ruta_imatge, 'rb') as document:
    imageBytes = bytearray(document.read())

response = textract.analyze_document(
    Document={'Bytes': imageBytes},
    FeatureTypes=["FORMS"]
)
```

- La imagen se lee como bytes en modo binario (`'rb'`).
- `analyze_document` con `FeatureTypes=["FORMS"]` activa la detección de pares clave-valor (como "Total: 12.50€"). Sin este parámetro solo detectaría bloques de texto.
- `response` es un JSON enorme con toda la información extraída.

---

## PASO 3 — Parsear la respuesta y encontrar el "Total"

```python
from trp import Document

def obtenir_total_tiquet(response):
    doc = Document(response)
    total_value = None

    for page in doc.pages:
        for field in page.form.fields:
            if field.key and field.value and field.key.text:
                key_text = field.key.text.strip().rstrip(':').rstrip()

                if key_text.lower() == "total":         # coincidencia exacta
                    return field.value.text
                elif 'total' in key_text.lower():       # coincidencia parcial
                    total_value = field.value.text

    return total_value
```

- `trp.Document` convierte el JSON complejo de Textract en una estructura Python fácil de iterar.
- Se itera por páginas → campos de formulario. Cada campo tiene `.key.text` y `.value.text`.
- Se busca primero coincidencia exacta ("total"), luego parcial (por si el ticket dice "TOTAL EUROS:" o "TOTAL A PAGAR").
- `.rstrip(':')` elimina los dos puntos del final de la clave (común en tickets).

---

## PASO 4 — Limpiar el valor con regex

```python
def netejar_sortida(valor):
    valor_netejat = re.sub(r'[^\d,.]', '', valor)   # eliminar todo excepto dígitos, coma, punto
    if ',' in valor_netejat and '.' in valor_netejat:
        valor_netejat = valor_netejat.replace('.', '').replace(',', '.')  # 1.234,56 → 1234.56
    elif ',' in valor_netejat:
        valor_netejat = valor_netejat.replace(',', '.')  # 12,50 → 12.50
    return "{:.2f}".format(float(valor_netejat))
```

Textract puede devolver "12,50 €" o "TOTAL: 12.50". El regex elimina todo excepto los dígitos y separadores decimales. Luego se normaliza al formato float de Python (punto como decimal).

---

## PASO 5 — Crear la interfaz con Gradio Blocks

```python
import gradio as gr

total_acumulat = 0.0

def obtenir_total(ruta):      # ruta = archivo temporal de la imagen subida
    global total_acumulat

    with open(ruta, 'rb') as document:
        imageBytes = bytearray(document.read())

    response = textract.analyze_document(Document={'Bytes': imageBytes}, FeatureTypes=["FORMS"])
    # ... (parsear + limpiar) ...
    total_acumulat += float(valor_netejat)
    return "{:.2f}".format(float(valor_netejat)), total_acumulat

with gr.Blocks(title="Import tiquets") as demo:
    gr.Markdown("<h1>Processador de tiquets</h1>")
    with gr.Column():
        with gr.Row():
            tiquet   = gr.Textbox(value='0', label="Import del tiquet (€):")
            acumulat = gr.Number(value=0, label="Total acumulat (€):")
        imatge = gr.Image(type='filepath', ...)  # recibe ruta, no array

    imatge.upload(fn=obtenir_total, inputs=imatge, outputs=[tiquet, acumulat])

demo.launch(debug=True)
```

- `gr.Blocks` → layout personalizado (vs `gr.Interface` que es más simple).
- `gr.Row` / `gr.Column` → organizan los componentes horizontalmente/verticalmente.
- `imatge.upload(fn=...)` → cada vez que el usuario sube una imagen, se llama automáticamente a `obtenir_total`.
- `type='filepath'` → Gradio pasa la ruta del archivo temporal, no el array de píxeles. Necesario para poder leer los bytes y enviárselos a Textract.
- `global total_acumulat` → variable que persiste entre subidas de imágenes en la misma sesión.
- La función retorna 2 valores → se asignan a los 2 outputs [tiquet, acumulat].

---

## PASO 6 — Desplegar en Hugging Face Spaces

1. Crear `app.py` con todo el código (sin `!pip install`, sin credenciales).
2. Crear `requirements.txt`:
```
boto3
gradio
amazon-textract-response-parser
```
3. En el Space de HF → Settings → Variables and Secrets → añadir `AWS_ACCESS_KEY_ID`, etc.
4. En `app.py` leer las credenciales de forma segura:
```python
import os
aws_access_key_id = os.environ.get('AWS_ACCESS_KEY_ID')
```

---

# ══════════════════════════════════════════
# EAC6 — Proyecto estructurado: La Liga
# ══════════════════════════════════════════

## PASO 1 — Cargar y hacer EDA (ex1.py)

```python
def load_and_eda(file: str) -> pd.DataFrame:
    data = pd.read_csv(f'{file}')
    print(data.head())
    print(data.describe())
    print(data.info())
    data.drop(columns=["HTHG", "HTAG", "HTR"], inplace=True)  # eliminar columnas de medio tiempo
    return data
```

Carga el CSV, muestra las 5 primeras filas, estadísticas y tipos. Elimina las columnas de half-time (HTHG=Half Time Home Goals) porque solo interesa el resultado final.

```python
def plot_home_away_goals(data):
    plt.boxplot(data['FTHG'])   # Full Time Home Goals
    plt.savefig(f"img/grafica_ex1_{config.nom_alumne}_{config.date_time}.png")
```

`plt.savefig` en lugar de `plt.show()` porque el script .py guarda imágenes en disco.

---

## PASO 2 — Total de partidos por equipo (ex2.py)

```python
def total_matches(data):
    matches_home  = data['HomeTeam'].value_counts()   # partidos como local
    matches_away  = data['AwayTeam'].value_counts()   # partidos como visitante
    matches_total = pd.DataFrame(matches_home) + pd.DataFrame(matches_away)
    matches_total = matches_total.sort_values("count", ascending=False)
    return matches_total
```

`value_counts()` cuenta partidos por equipo. Se suman los de local y visitante. El resultado muestra qué equipos han jugado más temporadas en Primera (Real Madrid, Barcelona, etc. tienen el máximo).

---

## PASO 3 — Distribución de goles (ex3.py)

```python
def goal_distribution(data):
    distr_gols_locals    = data["FTHG"].value_counts().sort_index()  # ¿cuántos partidos acabaron 0-X, 1-X, 2-X...?
    distr_gols_visitants = data["FTAG"].value_counts().sort_index()
    return distr_gols_locals, distr_gols_visitants
```

`sort_index()` ordena por número de goles (0, 1, 2, 3...) en lugar de por frecuencia. El gráfico de barras muestra que lo más frecuente es 1 gol local y 0-1 gol visitante.

---

## PASO 4 — Full Time Result (ex4.py)

```python
def FTR(data):
    ftr = data['FTR'].value_counts()   # H=home wins, A=away wins, D=draw
    ftr = pd.DataFrame(ftr)
    return ftr

# En main.py:
print(f"% ganan locales: {100*ftr.loc['H','count'] / ftr['count'].sum():.2f} %")
```

La columna FTR contiene 'H' (gana local), 'A' (gana visitante) o 'D' (empate). `value_counts()` cuenta cuántas veces aparece cada resultado en toda la historia de La Liga.

---

## PASO 5 — Puntos acumulados (ex5.py)

```python
def add_points(data):
    data["points_home"] = data["FTR"].map({"H": 3, "D": 1, "A": 0})   # puntos local
    data["points_away"] = data["FTR"].map({"H": 0, "D": 1, "A": 3})   # puntos visitante
    return data

def fun_total_points(data):
    home_points  = data.groupby("HomeTeam")["points_home"].sum()
    away_points  = data.groupby("AwayTeam")["points_away"].sum()
    total_points = home_points.add(away_points, fill_value=0)   # suma alineando por equipo
    df_total_points = total_points.to_frame(name="points")
    return total_points, df_total_points
```

- `.map({"H": 3, "D": 1, "A": 0})` → aplica el sistema de puntos del fútbol a cada partido.
- `groupby("HomeTeam").sum()` → suma todos los puntos de local para cada equipo.
- `.add(fill_value=0)` → suma dos Series alineando por el nombre del equipo. `fill_value=0` evita NaN si un equipo solo ha jugado como local o visitante.

---

## PASO 6 — Resumen 1996-2025 + estadios (ex6.py)

```python
def fun_resum_1996_2025(total_points, home_goals, away_goals, total_goals):
    resum = pd.concat(
        [total_points, home_goals, away_goals, total_goals],
        axis=1   # unir por columnas
    )
    resum.columns = ["points", "home_goals", "away_goals", "total_goals"]
    return resum

def add_stadium_capacity(resum, stadium_capacity):
    df_stadium = pd.DataFrame.from_dict(stadium_capacity, orient='index', columns=['stadium_capacity'])
    resum = resum.join(df_stadium)
    return resum
```

- `pd.concat(axis=1)` → une varias Series/DataFrames horizontalmente, compartiendo el índice (nombre del equipo).
- `pd.DataFrame.from_dict(orient='index')` → convierte el diccionario de estadios (equipo → capacidad) a DataFrame.
- `.join()` → une dos DataFrames por el índice común (nombre del equipo).

---

## PASO 7 — KMeans + guardar modelo con pickle (ex7.py)

```python
def model_clusters(df, num_clusters):
    X = df[["points", "home_goals", "away_goals", "total_goals", "stadium_capacity"]]

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(X_scaled)

    # Guardar modelo + scaler juntos
    model_data = {"scaler": scaler, "num_clusters": num_clusters, "kmeans": kmeans}
    with open(f"model/model_{num_clusters}.pkl", 'wb') as f:
        pickle.dump(model_data, f)

    return kmeans
```

Se escalan los datos antes de KMeans (fundamental). Se guardan scaler Y kmeans en el mismo pickle porque para hacer inferencia se necesitan los dos.

```python
def assignacio_clusters(df, kmeans):
    df["cluster"] = kmeans.labels_    # añadir etiqueta de clúster a cada equipo
    return df
```

---

## PASO 8 — Inferencia con el modelo guardado

```python
# En main.py:
with open("model/model_4.pkl", "rb") as f:
    model_data = pickle.load(f)

scaler  = model_data["scaler"]
kmeans  = model_data["kmeans"]

X_new        = df_europa[["points", "home_goals", "away_goals", "total_goals", "stadium_capacity"]]
X_new_scaled = scaler.transform(X_new)   # mismo scaler, solo transform

cluster = kmeans.predict(X_new_scaled)
```

Se carga el pickle completo (diccionario). Se extrae el scaler y el modelo. Se transforma el nuevo equipo con el mismo scaler (no fit, solo transform) y se predice a qué clúster pertenece.

---

## PASO 9 — Tests unitarios (tests_ex6.py)

```python
import sys
sys.path.append("..")   # añadir la carpeta padre para encontrar src/

from src.exercises.ex6 import fun_total_goals

class TestFunTotalGoals(unittest.TestCase):

    def test_basic_case(self):
        # Arrange: preparar datos de prueba controlados
        data = pd.DataFrame({'FTHG': [1, 2, 0], 'FTAG': [0, 1, 3]})

        # Act: ejecutar la función
        home, away, total = fun_total_goals(data)

        # Assert: verificar que el resultado es el esperado
        self.assertEqual(home, 3)    # 1+2+0 = 3
        self.assertEqual(away, 4)    # 0+1+3 = 4
        self.assertEqual(total, 7)   # 3+4 = 7

    def test_empty_dataframe(self):
        data = pd.DataFrame({'FTHG': [], 'FTAG': []})
        home, away, total = fun_total_goals(data)
        self.assertEqual(total, 0)   # con 0 partidos, 0 goles

if __name__ == '__main__':
    unittest.main()
```

- `sys.path.append("..")` → necesario porque tests/ y src/ son carpetas hermanas. Sin esto Python no encontraría el módulo.
- Patrón **Arrange-Act-Assert**: datos controlados → ejecutar → verificar.
- Se prueba el caso normal y el caso borde (DataFrame vacío).

---

*Explicación del código de todas las EACs · M03 IAB-5073 · Eric Rodriguez · Mayo 2026*
