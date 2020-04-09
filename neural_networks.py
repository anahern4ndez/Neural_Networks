'''
    Ana Lucia Hernandez - 17138. 
    Programa main para ejecucion de red neuronal usando el dataset "fashion-mnist"

'''
from back_propagation import *
import numpy as np
import mnist_reader
import scipy.optimize as optimize
import time
import csv 
import pickle

start = time.time()

# data import 
X_train, y_train = mnist_reader.load_mnist('data/', kind='train')
#X_test, y_test = mnist_reader.load_mnist('data/fashion', kind='t10k')

X = X_train / 1000.0
m,n = X_train.shape
HIDDEN_LAYER = 130 # Cantidad de neuronas en la capa oculta
FINAL_LAYER = 10 # cantidad de neuronas en la capa final 

# y_train.reshape(m, 1)
# Y = (y_train == np.array(range(10))).astype(int)
# Y reshape into matrix
Y = np.zeros((X_train.shape[0], FINAL_LAYER))
for i in range(m):
    Y[i][y_train[i]] = 1

# y_csv = open('y_train.csv', 'w')
# with y_csv:
#     writer = csv.writer(y_csv)
#     writer.writerows(Y)

# x_csv = open('x_train.csv', 'w')
# with x_csv:
#     writer = csv.writer(x_csv)
#     writer.writerows(X)

theta_shapes = np.array([
    [ HIDDEN_LAYER, n+1 ],
    [ FINAL_LAYER, HIDDEN_LAYER+1 ]
])
print(theta_shapes)

# generacion de matrices de transicion (aplanadas)
flat_thetas = flatten_matrixes([
    np.random.rand(*theta_shape)
    for theta_shape in theta_shapes
])

# print("Optimizing...")
# result = optimize.minimize(
#     fun=nn_cost,
#     x0=flat_thetas,
#     args=(theta_shapes, X, Y),
#     method='L-BFGS-B',
#     jac=back_propagation,
#     options={'disp': True, 'maxiter': 1300}
# )
# print("Optimized!")

print("test: \n")
res = back_propagation(flat_thetas, theta_shapes, X, Y)
print(res)
for matrix in res:
    print("shape: ", matrix.shape)
    print("values:\n", matrix)

# test1 = open('test1.csv', 'w')
# with test1:
#     writer = csv.writer(test1)
#     writer.writerows(res)

# model_file = open('trained_model', 'wb')
# # writing model 
# pickle.dump(result.x, model_file)
# model_file.close()

end = time.time()
print("Tiempo: {}s".format(end-start))