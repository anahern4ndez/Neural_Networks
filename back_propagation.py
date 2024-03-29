'''
    Ana Lucia Hernandez - 17138. 
    8 de abril del 2020.
    Módulo con definición de funciones necesarias para la implementación de una red neuronal. 

'''
import numpy as np

# prediction: matriz con las predicciones resultantes de feed_forward
# theorical: vector que contiene los valores teoricos  
def get_accuracy(prediction, theorical):
    hits, data = 0, theorical.shape[0]
    for i in range(data):
        if(theorical[i] == np.argmax(prediction[i])):
            hits += 1
    return hits / data

# prediction: matriz con las predicciones resultantes de feed_forward
# theorical: vector que contiene los valores teoricos
# @return: array (dim 10) conteniendo el accuracy por clase 
def get_accuracy_by_class(prediction, theorical):
    hits, instances, data = np.zeros((10,)), np.zeros((10,)), theorical.shape[0]
    for i in range(data):
        instances[theorical[i]] += 1.0
        if(theorical[i] == np.argmax(prediction[i])):
            hits[theorical[i]] += 1.0
    return np.ndarray.tolist(hits / instances)

# z: matmul entre vector x y matriz theta
def sigmoid(z):
    return 1.0/(1 + np.exp(-z))


# thetas: lista de matrices de transicion
# @return: array de matrices aplanadas y concatenadas y lista de tuplas de las shapes de matrices originales
def flatten_matrixes(thetas):
    flat_thetas = []
    for matrix in thetas:
        flat_thetas = [*flat_thetas, *(matrix.flatten())]
    return np.array(flat_thetas).flatten()

# flat matrixes: array con matrices aplastadas y concatenadas
# shape: lista de tuplas de dimensiones de cada matriz original
# @return: lista de matrices infladas
def inflate_matrixes(flat_matrixes, shapes):
    inflated_matrixes = []
    sizes = [ shape[0] * shape [1] for shape in shapes ]
    step = 0
    for i in range(len(sizes)):
        inflated_matrixes.append(
            np.array(flat_matrixes[step : step + sizes[i]]).reshape(shapes[i])
        )
        step = np.array(sizes[0 : i+1]).sum()
    return inflated_matrixes

# thetas: lista de todas las matrices de transicion
# x: matriz de training values (primera capa)
# @return: lista de vectores de activacion de cada capa
def feed_forward(thetas, X):
    a = [X] # [2.1], primera activacion
    for i in range(len(thetas)): #iteracion sobre capas
        a.append(
            sigmoid(
                np.matmul(
                    np.hstack((
                        np.ones(len(a[i])).reshape(len(a[i]), 1),
                        a[i]
                    )),
                    thetas[i].T
                )
            )
        )
    return a 

# x: matriz de los valores de entrada de la primera capa
# f_thetas: lista de matrices de transicion (flattened) de capa 0 a capa 1
# L: cantidad de layers
# y: vector de valores reales (teoricos)
# @return: lista de gradientes 
def back_propagation(f_thetas, shapes, X, Y):
    m, layers = len(X), len(shapes) + 1
    thetas = inflate_matrixes(f_thetas, shapes)
     # [2.2]
    a = feed_forward(thetas, X)
    # lista de errores [2.3]
    deltas = [ *range(layers -1), a[-1] - Y ] 
    # [2.4]
    for i in range(layers-2, 0, -1): # loop desde la ultima capa hasta la segunda (en reversa)
        deltas[i] = np.matmul(
            deltas[i+1],
            (thetas[i])[:, 1:] # a theta se le quita el bias
        ) * (a[i] * (1 - a[i]))
    # [2.5] 
    gradient = []
    for i in range(layers-1): # loop de capa 0 a capa L-1
        gradient.append((np.matmul(
            deltas[i+1].T, 
            np.hstack(( # se agrega bias a 
                        np.ones(len(a[i])).reshape(len(a[i]), 1),
                        a[i]
                    ))
        )) / m)
    return flatten_matrixes(gradient)

#   Funcion costo 
# flat_thetas: lista de matrices de transicion aplanadas en una dimension
# shapes: lista de tuplas de shapes de cada matriz original
# X: matriz de datos de entrenamiento (entrada de NN)
# Y: matriz que contiene los valores reales de cada neurona (para cada dato) en la capa final. Tiene forma m(#datos)x n(#clases/neuronas en capa final)
def nn_cost(flat_thetas, shapes, X, Y):
    # obtener predicciones 
    a = feed_forward(
        inflate_matrixes(flat_thetas, shapes),
        X
    )
    return -(Y * np.log(a[-1]) + (1 - Y) * np.log(1 - a[-1])).sum() / len(X)