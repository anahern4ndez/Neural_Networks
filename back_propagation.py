import numpy as np

# z: matmul entre vector x y matriz theta
def sigmoid(z):
    return (1 + np.exp(-z))**-1

# thetas: lista de matrices de transicion
# @return: lista de matrices aplanadas (arrays)
def flatten_matrixes(thetas):
    flat_thetas = []
    shapes = [] # para guardar las shapes originales
    for matrix in thetas:
        shapes.append(matrix.shape)
        flat_thetas.append(matrix.flatten())
    return flat_thetas, shapes

# flat matrixes: lista de arrays de una dimensión
# shape: lista de tuplas de dimensiones de cada matriz original
# @return: lista de matrices infladas
def inflate_matrixes(flat_matrixes, shapes):
    inflated_matrixes = []
    for i in range(len(flat_matrixes)):
        mat = flat_matrixes[i]
        m, n = shapes[i]
        inflated_matrixes.append(
            np.reshape(mat, (m ,n))
        )
    return inflated_matrixes

# thetas: lista de todas las matrices de transicion
def feed_forward(thetas, x):
    a = [x]
    for i in range(len(thetas) +1 ): #iteracion sobre capas
        a.append(
            sigmoid(
                np.matmul(
                    np.hstack((
                        np.ones(len(x)).reshape(len(x), 1),
                        a[i]
                    )),
                    thetas[i].T
                )
            )
        )
    return a #lista de vectores de activacion de cada capa

# x: matriz de los valores de entrada de la primera capa
# theta: lista de matrices de transicion (infladas) de capa 0 a capa 1
# L: cantidad de layers
# y: vector de valores reales (teoricos)

def back_propagation(X, thetas, L, Y):
    m, layers = len(X), len()
    #crear matriz Delta proporcional al gradiente 
    Delta = np.zeros_like(thetas) #mismo shape que la matriz theta de entrada
    z, delta = [], []
    a = feed_forward(thetas, X) #z es matriz, mxn. m: 
    delta = [*range(lay)]
        #2.4, empieza de L-1 a la segunda capa (la primera no se hace). Theta NO DEBE TENER EL BIAS
        

#debe devolver resultados flat (matrices aplanadas)

#dividir entradas /3, el resultada en una sola capa oculta. 

# overfit -> se quita neuronas

#   Funcion costo 
# flat_thetas: lista de matrices de transicion aplanadas en una dimension
# shapes: lista de tuplas de shapes de cada matriz original
# X: matriz de datos de entrenamiento (entrada de NN)
# Y: vector que contiene los vectores reales de cada neurona en la capa final
def nn_cost(flat_thetas, shapes, X, Y):
    # obtener predicciones 
    a = feed_forward(
        inflate_matrixes(flat_thetas, shapes),
        X
    )
    return -(Y * np.log(a[-1]) + (1 - Y) * np.log(1 - a[-1])).sum() / len (X)