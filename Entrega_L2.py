import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, silhouette_score
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from matplotlib.colors import ListedColormap
from sklearn.decomposition import PCA

# Instanciar base de datos
data = pd.read_csv('datos.csv')  # Cargar el dataset

X = data.iloc[:, :-1]  # Características: todas las columnas excepto la última
y = data.iloc[:, -1]  # Etiquetas: última columna

# Escalado de características
escalador = MinMaxScaler()
X = escalador.fit_transform(X)

resultados_modelos = {}
resultados_rforest = {}

def equilibrarClase(X, y):
    y_int = y.astype(int)
    X_equilibrado, y_equilibrado = SMOTE().fit_resample(X, y_int)
    return X_equilibrado, y_equilibrado

def maquinaSoporteVectorial_lineal(X_entrenamiento, X_prueba, y_entrenamiento, y_prueba):
    print("\nMÁQUINA DE SOPORTE VECTORIAL (Kernel Lineal)")
    
    # Máquina de soporte vectorial con kernel lineal
    svc_lineal = SVC(kernel='linear', random_state=42)
    svc_lineal.fit(X_entrenamiento, y_entrenamiento)

    y_prediccion = svc_lineal.predict(X_prueba)

    precision = accuracy_score(y_prueba, y_prediccion)
    matriz_confusion = confusion_matrix(y_prueba, y_prediccion)
    resultados_modelos['SVM Lineal'] = {'precision': precision, 'matriz_confusion': matriz_confusion}

    print(f"Precisión: {precision * 100:.2f}%")
    print("Matriz de confusión:\n", matriz_confusion)
    
    return svc_lineal

def maquinaSoporteVectorial_polynomial(X_entrenamiento, X_prueba, y_entrenamiento, y_prueba):
    print("\nMÁQUINA DE SOPORTE VECTORIAL (Kernel Polinomial)")
    
    # Máquina de soporte vectorial con kernel polinomial
    svc_polynomial = SVC(kernel='poly', degree=3, random_state=42)  # Grado 3 del polinomio
    svc_polynomial.fit(X_entrenamiento, y_entrenamiento)

    y_prediccion = svc_polynomial.predict(X_prueba)

    precision = accuracy_score(y_prueba, y_prediccion)
    matriz_confusion = confusion_matrix(y_prueba, y_prediccion)
    resultados_modelos['SVM Polinomial'] = {'precision': precision, 'matriz_confusion': matriz_confusion}

    print(f"Precisión: {precision * 100:.2f}%")
    print("Matriz de confusión:\n", matriz_confusion)
    
    return svc_polynomial

def maquinaSoporteVectorial_rbf(X_entrenamiento, X_prueba, y_entrenamiento, y_prueba):
    print("\nMÁQUINA DE SOPORTE VECTORIAL (Kernel Radial)")
    
    # Máquina de soporte vectorial con kernel radial (RBF)
    svc_rbf = SVC(kernel='rbf', random_state=42)
    svc_rbf.fit(X_entrenamiento, y_entrenamiento)

    y_prediccion = svc_rbf.predict(X_prueba)

    precision = accuracy_score(y_prueba, y_prediccion)
    matriz_confusion = confusion_matrix(y_prueba, y_prediccion)
    resultados_modelos['SVM Radial'] = {'precision': precision, 'matriz_confusion': matriz_confusion}

    print(f"Precisión: {precision * 100:.2f}%")
    print("Matriz de confusión:\n", matriz_confusion)
    
    return svc_rbf

def redNeuronalArtificial_v1(X_entrenamiento, X_prueba, y_entrenamiento, y_prueba):
    print("\nRED NEURONAL ARTIFICIAL v1 (Simple)")
    
    # Red neuronal con una configuración simple
    mlp = MLPClassifier(hidden_layer_sizes=(10,),  # 1 capa oculta con 10 neuronas
                        activation='relu', 
                        solver='adam', 
                        random_state=42, 
                        max_iter=1000)
    
    mlp.fit(X_entrenamiento, y_entrenamiento)

    y_prediccion = mlp.predict(X_prueba)

    precision = accuracy_score(y_prueba, y_prediccion)
    matriz_confusion = confusion_matrix(y_prueba, y_prediccion)
    resultados_modelos['Red Neuronal Artificial v1'] = {'precision': precision, 'matriz_confusion': matriz_confusion}

    print(f"Precisión: {precision * 100:.2f}%")
    print("Matriz de confusión:\n", matriz_confusion)
    
    return mlp

def redNeuronalArtificial_v2(X_entrenamiento, X_prueba, y_entrenamiento, y_prueba):
    print("\nRED NEURONAL ARTIFICIAL v2 (Mediana)")
    
    # Red neuronal con una configuración mediana
    mlp = MLPClassifier(hidden_layer_sizes=(50, 50),  # 2 capas ocultas con 50 neuronas cada una
                        activation='tanh', 
                        solver='adam', 
                        random_state=42, 
                        max_iter=2000)
    
    mlp.fit(X_entrenamiento, y_entrenamiento)

    y_prediccion = mlp.predict(X_prueba)

    precision = accuracy_score(y_prueba, y_prediccion)
    matriz_confusion = confusion_matrix(y_prueba, y_prediccion)
    resultados_modelos['Red Neuronal Artificial v2'] = {'precision': precision, 'matriz_confusion': matriz_confusion}

    print(f"Precisión: {precision * 100:.2f}%")
    print("Matriz de confusión:\n", matriz_confusion)
    
    return mlp

def redNeuronalArtificial_v3(X_entrenamiento, X_prueba, y_entrenamiento, y_prueba):
    print("\nRED NEURONAL ARTIFICIAL v3 (Compleja)")
    
    # Red neuronal con una configuración más compleja
    mlp = MLPClassifier(hidden_layer_sizes=(100, 100, 50),  # 3 capas ocultas con 100, 100 y 50 neuronas
                        activation='relu', 
                        solver='adam', 
                        random_state=42, 
                        max_iter=3000)
    
    mlp.fit(X_entrenamiento, y_entrenamiento)

    y_prediccion = mlp.predict(X_prueba)

    precision = accuracy_score(y_prueba, y_prediccion)
    matriz_confusion = confusion_matrix(y_prueba, y_prediccion)
    resultados_modelos['Red Neuronal Artificial v3'] = {'precision': precision, 'matriz_confusion': matriz_confusion}

    print(f"Precisión: {precision * 100:.2f}%")
    print("Matriz de confusión:\n", matriz_confusion)
    
    return mlp


#Metodo de K-MEANS

# Normalización de los datos (manteniendo el formato de DataFrame)
data_normalizado = (X- X.min()) / (X.max() - X.min())

# Aplicar PCA para reducir las dimensiones a 2 componentes (manteniendo los nombres de las columnas)
pca = PCA(n_components=2)
X_reducido = pd.DataFrame(pca.fit_transform(data_normalizado), columns=['Componente_1', 'Componente_2'])

# Búsqueda de la cantidad óptima de clusters
def encontrarNumeroGrupos(X):
    inercias = []
    for i in range(3, 12):
        kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
        kmeans.fit(X)
        inercias.append(kmeans.inertia_)
    return inercias

# Graficar el codo de Jambú
inercia = encontrarNumeroGrupos(data_normalizado)
plt.plot(range(3, 12), inercia)
plt.title("Codo de Jambú")
plt.xlabel("Número de Clusters")
plt.ylabel("Inercias")  # Es un indicador de qué tan similares son los individuos dentro de los clusters
plt.show()


#K-MEANS Sin las clases 
# Función para ejecutar K-Means
ng = 5
def K_Means2(num_Grupos, X):
    # Entrenar el modelo K-Means
    kmeans = KMeans(n_clusters=num_Grupos, init='k-means++', random_state=42)
    kmeans.fit(X)
    y_prediccion = kmeans.labels_ + 1  # Añadiendo 1 para empezar las etiquetas desde 1

    # Calcular coeficiente de silueta e inercia
    silhouette_avg = silhouette_score(X, kmeans.labels_)
    inercia = kmeans.inertia_

    print(f"K-MEANS CON {num_Grupos} GRUPOS")
    print("Coeficiente de silueta: ", silhouette_avg)
    print("Inercia: ", inercia)

    # Si se proporcionan etiquetas verdaderas (y), calcular precisión y matriz de confusión
    if y is not None and num_Grupos == 5:
        precision = accuracy_score(y, y_prediccion)
        matriz_confusion = confusion_matrix(y, y_prediccion)
        print("Accuracy Score:", precision * 100, "%")
        print("Matriz de Confusión:\n", matriz_confusion)
        
        # Guardar los resultados en un diccionario
        resultados_modelos['K Means 2'] = {'precision': precision, 'matriz_confusion': matriz_confusion}

    return kmeans, silhouette_avg
 
# Mostrar los atributos y clases (ajustar según el archivo datos.csv)
atributos = list(data.columns)[:-1]
print("Atributos:")
for i, atributo in enumerate(atributos, 1):
    print(f"{i}. {atributo}")

# Información de las clases (esto puede necesitar ajustes según el archivo datos.csv)
print("\nClases:")
for clase in np.unique(y):
    print(f"{int(clase)}. Clase {int(clase)}")

# Equilibrar las clases y dividir los datos en conjuntos de entrenamiento y prueba
X_equilibrado, y_equilibrado = equilibrarClase(X, y)
X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = train_test_split(X_equilibrado, y_equilibrado, test_size=0.2, random_state=42)

# Mostrar conteo de clases en los conjuntos de entrenamiento y prueba
print("Número de datos de entrenamiento: ", len(y_entrenamiento))
print(pd.Series(y_entrenamiento).value_counts(), "\n")
print("Número de datos de pruebas: ", len(y_prueba))
print(pd.Series(y_prueba).value_counts(), "\n")

# Entrenar y evaluar las tres variantes de SVM
svm_lineal = maquinaSoporteVectorial_lineal(X_entrenamiento, X_prueba, y_entrenamiento, y_prueba)
svm_polynomial = maquinaSoporteVectorial_polynomial(X_entrenamiento, X_prueba, y_entrenamiento, y_prueba)
svm_rbf = maquinaSoporteVectorial_rbf(X_entrenamiento, X_prueba, y_entrenamiento, y_prueba)

# Entrenar y evaluar las tres versiones de la red neuronal
mlp_v1 = redNeuronalArtificial_v1(X_entrenamiento, X_prueba, y_entrenamiento, y_prueba)
mlp_v2 = redNeuronalArtificial_v2(X_entrenamiento, X_prueba, y_entrenamiento, y_prueba)
mlp_v3 = redNeuronalArtificial_v3(X_entrenamiento, X_prueba, y_entrenamiento, y_prueba)

n = encontrarNumeroGrupos(X_equilibrado)
n = np.argmax(n) + 3

# #SVM LINEAL 
svc_lineal = maquinaSoporteVectorial_lineal(X_entrenamiento, X_prueba, y_entrenamiento, y_prueba)

# Reducir las dimensiones a 2D con PCA para visualizar
X_entrenamiento_2d = pca.fit_transform(X_entrenamiento)
X_prueba_2d = pca.transform(X_prueba)  # Aplicamos la misma transformación a los datos de prueba

# Crear una malla de puntos para cubrir el espacio de los datos
x_min, x_max = X_entrenamiento_2d[:, 0].min() - 1, X_entrenamiento_2d[:, 0].max() + 1
y_min, y_max = X_entrenamiento_2d[:, 1].min() - 1, X_entrenamiento_2d[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

# Predecir sobre cada punto de la malla
Z = svc_lineal.predict(pca.inverse_transform(np.c_[xx.ravel(), yy.ravel()]))
Z = Z.reshape(xx.shape)

# Crear una figura para las regiones de decisión
plt.figure(figsize=(10, 6))

# Definir colores para las diferentes clases
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF', '#FFFFAA', '#AAFFFF', '#FFAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#00FFFF', '#FF00FF'])

# Dibujar las regiones de decisión
plt.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.6)

# Dibujar los puntos de entrenamiento
plt.scatter(X_entrenamiento_2d[:, 0], X_entrenamiento_2d[:, 1], c=y_entrenamiento, cmap=cmap_bold, edgecolor='k', s=40)
plt.scatter(X_prueba_2d[:, 0], X_prueba_2d[:, 1], c=y_prueba, cmap=cmap_bold, edgecolor='k', s=100, marker='*')

plt.title('Máquina de Soporte Vectorial Lineal - Regiones de Decisión con PCA')
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.show()

# #SVM POLINOMIAL 
# Definir y entrenar la máquina de soporte vectorial con kernel polinomial
svc_polynomial = maquinaSoporteVectorial_polynomial(X_entrenamiento, X_prueba, y_entrenamiento, y_prueba)

# Predecir sobre cada punto de la malla polinomial
Z = svc_polynomial.predict(pca.inverse_transform(np.c_[xx.ravel(), yy.ravel()]))
Z = Z.reshape(xx.shape)

# Crear una figura para las regiones de decisión
plt.figure(figsize=(10, 6))

# Definir colores para las diferentes clases
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF', '#FFFFAA', '#AAFFFF', '#FFAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#00FFFF', '#FF00FF'])

# Dibujar las regiones de decisión
plt.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.6)

# Dibujar los puntos de entrenamiento
plt.scatter(X_entrenamiento_2d[:, 0], X_entrenamiento_2d[:, 1], c=y_entrenamiento, cmap=cmap_bold, edgecolor='k', s=40)
plt.scatter(X_prueba_2d[:, 0], X_prueba_2d[:, 1], c=y_prueba, cmap=cmap_bold, edgecolor='k', s=100, marker='*')

plt.title('Máquina de Soporte Vectorial Polinomial - Regiones de Decisión con PCA')
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.show()

# #SVM CIRCULAR
svc_rbf = maquinaSoporteVectorial_rbf(X_entrenamiento, X_prueba, y_entrenamiento, y_prueba)

# Predecir sobre cada punto de la malla rbf
Z = svc_rbf.predict(pca.inverse_transform(np.c_[xx.ravel(), yy.ravel()]))
Z = Z.reshape(xx.shape)

# Crear una figura para las regiones de decisión
plt.figure(figsize=(10, 6))

# Definir colores para las diferentes clases
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF', '#FFFFAA', '#AAFFFF', '#FFAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#00FFFF', '#FF00FF'])

# Dibujar las regiones de decisión
plt.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.6)

# Dibujar los puntos de entrenamiento
plt.scatter(X_entrenamiento_2d[:, 0], X_entrenamiento_2d[:, 1], c=y_entrenamiento, cmap=cmap_bold, edgecolor='k', s=40)
plt.scatter(X_prueba_2d[:, 0], X_prueba_2d[:, 1], c=y_prueba, cmap=cmap_bold, edgecolor='k', s=100, marker='*')

plt.title('Máquina de Soporte Vectorial Radial - Regiones de Decisión con PCA')
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.show()

#RED NEURONAL SIMPLE
mlp = redNeuronalArtificial_v1(X_entrenamiento, X_prueba, y_entrenamiento, y_prueba)
# Crear una figura para las regiones de decisión

plt.figure(figsize=(10, 6))
# Definir colores para las diferentes clases
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF', '#FFFFAA', '#AAFFFF', '#FFAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#00FFFF', '#FF00FF'])

# Dibujar las regiones de decisión
plt.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.6)

# Dibujar los puntos de entrenamiento
plt.scatter(X_entrenamiento_2d[:, 0], X_entrenamiento_2d[:, 1], c=y_entrenamiento, cmap=cmap_bold, edgecolor='k', s=40)
plt.scatter(X_prueba_2d[:, 0], X_prueba_2d[:, 1], c=y_prueba, cmap=cmap_bold, edgecolor='k', s=100, marker='*')

plt.title('Red Neuronal Artificial Con Una Capa')
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.show()

#RED NEURONAL CON DOS CAPAS
mlp = redNeuronalArtificial_v2(X_entrenamiento, X_prueba, y_entrenamiento, y_prueba)


# Crear una figura para las regiones de decisión
plt.figure(figsize=(10, 6))

# Definir colores para las diferentes clases
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF', '#FFFFAA', '#AAFFFF', '#FFAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#00FFFF', '#FF00FF'])

# Dibujar las regiones de decisión
plt.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.6)

# Dibujar los puntos de entrenamiento
plt.scatter(X_entrenamiento_2d[:, 0], X_entrenamiento_2d[:, 1], c=y_entrenamiento, cmap=cmap_bold, edgecolor='k', s=40)
plt.scatter(X_prueba_2d[:, 0], X_prueba_2d[:, 1], c=y_prueba, cmap=cmap_bold, edgecolor='k', s=100, marker='*')

plt.title('Red Neuronal Artificial Con Dos Capas')
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.show()

#RED NEURONAL CON TRES CAPAS
# Definir y entrenar la red neuronal con la nueva función v3
mlp = redNeuronalArtificial_v3(X_entrenamiento, X_prueba, y_entrenamiento, y_prueba)


# Crear una figura para las regiones de decisión
plt.figure(figsize=(10, 6))

# Predecir sobre cada punto de la malla
Z = mlp.predict(pca.inverse_transform(np.c_[xx.ravel(), yy.ravel()]))
Z = Z.reshape(xx.shape)

# Dibujar las regiones de decisión
plt.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.6)

# Dibujar los puntos de prueba
plt.scatter(X_prueba_2d[:, 0], X_prueba_2d[:, 1], c=y_prueba, cmap=cmap_bold, edgecolor='k', s=100, marker='*')


plt.title('Red Neuronal Artificial Con Tres Capas')
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.show()

# Función para graficar los resultados de K-Means
def graficarKMeans2(kmeans, X, num_Grupos):
    # Predicciones y centroides
    y_prediccion = kmeans.labels_
    centroides = kmeans.cluster_centers_

    # Usar las componentes reducidas si X tiene más de 2 dimensiones
    if X.shape[1] > 2:
        pca = PCA(n_components=2)
        X_reducido = pca.fit_transform(X)
        centroides = pca.transform(centroides)
    else:
        X_reducido = X

    colores = ['r', 'g', 'b', 'c', 'm', 'y', 'k'][:num_Grupos]  # Colores limitados al número de grupos

    plt.figure(figsize=(8, 6))
    for i in range(num_Grupos):
        plt.scatter(X_reducido[y_prediccion == i, 0], X_reducido[y_prediccion == i, 1], 
                    s=50, c=colores[i], label=f'Grupo {i+1}', alpha=0.6)
    plt.scatter(centroides[:, 0], centroides[:, 1], s=300, c='black', marker='x', label='Centroides')
    plt.title(f"K-Means con {num_Grupos} Grupos ")
    plt.xlabel("Componente 1")
    plt.ylabel("Componente 2")
    plt.legend()
    plt.grid(True)
    plt.show()

# Ejecutar K-Means y graficar
kmeans_model, _ = K_Means2(ng, data_normalizado)
graficarKMeans2(kmeans_model, data_normalizado, ng)


#PRECISION DE LOS MODELOS
modelos = list(resultados_modelos.keys())
precisiones = [resultado['precision'] for resultado in resultados_modelos.values()]

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(modelos, precisiones)

for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, yval, f"{yval:.4f}", 
            ha='center', va='bottom')

ax.set_title('Precisión de los modelos')
ax.set_xlabel('Modelos')
ax.set_ylabel('Precisión')
plt.xticks(rotation=45)
plt.subplots_adjust(bottom=0.24, top=0.95)
plt.show()

# Visualización de las matrices de confusión con los valores
num_modelos = len(resultados_modelos)
nrows, ncols = 2, int(np.ceil(num_modelos / 2))
fig, axs = plt.subplots(nrows, ncols, figsize=(10, 5))  
axs = axs.flatten()

for i, (modelo, resultado) in enumerate(resultados_modelos.items()):
    matriz_confusion = resultado['matriz_confusion']
    clases = np.unique(y)
    ax = axs[i]
    img = ax.imshow(matriz_confusion, cmap='Blues', interpolation='nearest')
    
    # Añadir los números de la matriz de confusión dentro de las celdas
    for (j, k), valor in np.ndenumerate(matriz_confusion):
        ax.text(k, j, f'{valor}', ha='center', va='center', color='black')

    ax.set_xticks(np.arange(len(clases)))
    ax.set_yticks(np.arange(len(clases)))
    ax.set_xticklabels(clases)
    ax.set_yticklabels(clases)
    ax.set_xlabel('Clase predicha')
    ax.set_ylabel('Clase verdadera')
    ax.set_title(f'{modelo}')

fig.tight_layout()
plt.subplots_adjust(wspace=0.5)
plt.show()



