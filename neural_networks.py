'''
    Ana Lucia Hernandez - 17138. 
    Programa main para ejecucion de red neuronal usando el dataset "fashion-mnist"

'''
from back_propagation import *
import numpy as np

# generacion de matrices de transicion (aplanadas)
flat_thetas = flatten_matrixes([
    np.random.rand(*theta_shape)
    for theta_shape in theta_shapes
])