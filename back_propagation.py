import numpy as np

# z: matmul entre vector x y matriz theta
def sigmoid(z):
    return (1 + np.exp(-z))**-1

# thetas: lista de matrices de transicion
# @return: array de matrices aplanadas y concatenadas y lista de tuplas de las shapes de matrices originales
def flatten_matrixes(thetas):
    flat_thetas = []
    # shapes = [] # para guardar las shapes originales
    for matrix in thetas:
        # shapes.append(matrix.shape)
        flat_thetas = [*flat_thetas, *(matrix.flatten())]
    return np.array(flat_thetas).flatten()

# 
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
def feed_forward(thetas, x):
    a = [x] # [2.1], primera activacion
    for i in range(len(thetas)): #iteracion sobre capas
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
    return a 

# x: matriz de los valores de entrada de la primera capa
# f_thetas: lista de matrices de transicion (flattened) de capa 0 a capa 1
# L: cantidad de layers
# y: vector de valores reales (teoricos)
# @return: lista de gradientes 
def back_propagation(f_thetas, shapes, X, Y):
    m, layers = len(X), len(shapes) + 1
    print(layers)
    thetas = inflate_matrixes(f_thetas, shapes)
     # [2.2]
    a = feed_forward(thetas, X)
    # lista de errores [2.3]
    deltas = [ *range(layers -1), a[-1] - Y ] 
    # print(deltas)
    # for d in deltas:
    #     print(d.shape)
    print("thetas shapes\n")
    for t in thetas:
        print(t.shape)
    # [2.4]
    for i in range(layers-2, 0, -1): # loop desde la ultima capa hasta la segunda (en reversa)
        deltas[i] = np.matmul(
            deltas[i+1],
            (thetas[i])[:, :thetas[i].shape[1]-1] # a theta se le quita el bias
        ) * (a[i] * (1 - a[i]))
    # [2.5] 
    print("deltas\n")
    for d in deltas:
        try:
            print(d.shape)
        except:
            print('')
    print("a_s:\n")
    for a_s in a:
        print(a_s.shape)
    gradient = []
    for i in range(layers-2, -1, -1): # loop de capa 0 a capa L-1
        print(i)
        print("d[i+1] shape ", deltas[i+1].shape)
        # print("theta[i] shape ", thetas[i][:, :thetas[i].shape[1]-1].shape)
        print("a[i] shape ", a[i].shape)
        gradient.append((np.matmul(
            deltas[i+1].T, 
            np.hstack((
                        np.ones(len(a[i])).reshape(len(a[i]), 1),
                        a[i]
                    ))
        ))/m)
    #return flatten_matrixes(gradient)
    return gradient

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
    return -(Y * np.log(a[-1]) + (1 - Y) * np.log(1 - a[-1])).sum() / len (X)