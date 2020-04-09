'''
    Ana Lucia Hernandez - 17138. 
    Programa main para ejecucion de red neuronal usando el dataset "fashion-mnist"

'''
from back_propagation import *
import numpy as np
import mnist_reader

# data import 
X_train, y_train = mnist_reader.load_mnist('data/fashion', kind='train')
X_test, y_test = mnist_reader.load_mnist('data/fashion', kind='t10k')

m,n = X_train.shape
HIDDEN_LAYER = 130 # Cantidad de neuronas en la capa oculta
FINAL_LAYER = 10 # cantidad de neuronas en la capa final 

theta_shapes = np.array([
    [ HIDDEN_LAYER, n+1 ],
    [ FINAL_LAYER, HIDDEN_LAYER +1 ]
])

# generacion de matrices de transicion (aplanadas)
flat_thetas = flatten_matrixes([
    np.random.rand(*theta_shape)
    for theta_shape in theta_shapes
])