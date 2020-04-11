'''
    Ana Lucia Hernandez - 17138. 
    Programa main para ejecucion de red neuronal usando el dataset "mnist"
    Prueba.

'''

from back_propagation import *
import numpy as np
import mnist_reader
import scipy.optimize as optimize
import time
import csv 
import pickle


start = time.time()
# load model 
# model_file = open(model_filename, 'rb')
# flat_thetas = pickle.load(open(model_filename, 'rb'))
# model_file.close()
model_filename = 'data/trained_model_mnist'

# data import 
print('Loading data...')
train = np.loadtxt(open("mnist/mnist_train.csv", "rb"), delimiter=",", skiprows=1)
test = np.loadtxt(open("mnist/mnist_test.csv", "rb"), delimiter=",", skiprows=1)
x_train = train[:, 1:]
y_train = train[:, 0]
x_test = train[:, 1:]
y_test = train[:, 0]
print(x_train, '\n\n', y_train)
print('Loaded!')

print('y_train shape: ', y_train.shape)
print('y_train: ', y_train)

X = x_train
m,n = X.shape
print('x_train shape: ', X.shape)
HIDDEN_LAYER = 90 # Cantidad de neuronas en la capa oculta
FINAL_LAYER = 10 # cantidad de neuronas en la capa final 

# Y reshape into matrix
Y = np.zeros((X.shape[0], FINAL_LAYER))
for i in range(m):
    Y[i][int(y_train[i])] = 1

print(Y)

theta_shapes = np.array([
    [ HIDDEN_LAYER, n+1 ],
    [ FINAL_LAYER, HIDDEN_LAYER+1 ]
])
theta_shapes_list = [(HIDDEN_LAYER, n+1), (FINAL_LAYER, HIDDEN_LAYER+1)]

# generacion de matrices de transicion (aplanadas) random 
flat_thetas = flatten_matrixes([
    np.random.rand(*theta_shape)
    for theta_shape in theta_shapes
])


print("Optimizing...")
result = optimize.minimize(
    fun=nn_cost,
    x0=flat_thetas,
    args=(theta_shapes, X, Y),
    method='L-BFGS-B',
    jac=back_propagation,
    options={
        'disp': True, 
        'maxiter': 3000, 
    }
)
print("Optimized!")


# calculo de porcentaje de error  
# result_thetas = inflate_matrixes(flat_thetas, theta_shapes_list)
# prediction = feed_forward(result_thetas, X)[-1]
# percentage = error_percentage(prediction, y_train)

result_thetas = inflate_matrixes(result.x, theta_shapes_list)
print('res thetas: \n', result_thetas)
print('len res thetas: ', len(result_thetas))
print('shapes res thetas: ', result_thetas[0].shape, result_thetas[1].shape)
prediction = feed_forward(result_thetas, X)[-1]
print('prediction: \n', prediction)
percentage = error_percentage(prediction, y_train)

print("ACCURACY: ", percentage)

model_file = open(model_filename, 'wb')
# writing model 
pickle.dump(np.asarray(result.x), model_file)
model_file.close()

end = time.time()
print("TIEMPO DE EJECUCION: {}s".format(end-start))