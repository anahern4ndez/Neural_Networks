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

model_filename = 'data/trained_model'

start = time.time()
# load model 
model_file = open(model_filename, 'rb')
flat_thetas = pickle.load(open(model_filename, 'rb'))
model_file.close()

# data import 
X_train, y_train = mnist_reader.load_mnist('data/', kind='train')
X_test, y_test = mnist_reader.load_mnist('data/', kind='t10k')

print('y_train shape: ', y_train.shape)

X = X_train / 1000.0
m,n = X.shape
HIDDEN_LAYER = 100 # Cantidad de neuronas en la capa oculta
FINAL_LAYER = 10 # cantidad de neuronas en la capa final 

# Y reshape into matrix
Y = np.zeros((X.shape[0], FINAL_LAYER))
for i in range(m):
    Y[i][y_train[i]] = 1


theta_shapes = np.array([
    [ HIDDEN_LAYER, n+1 ],
    [ FINAL_LAYER, HIDDEN_LAYER+1 ]
])
theta_shapes_list = [(HIDDEN_LAYER, n+1), (FINAL_LAYER, HIDDEN_LAYER+1)]

# # generacion de matrices de transicion (aplanadas) random 
# flat_thetas = flatten_matrixes([
#     np.random.rand(*theta_shape)
#     for theta_shape in theta_shapes
# ])


# print("Optimizing...")
# result = optimize.minimize(
#     fun=nn_cost,
#     x0=flat_thetas,
#     args=(theta_shapes, X, Y),
#     method='L-BFGS-B',
#     jac=back_propagation,
#     options={
#         'disp': True, 
#         'maxiter': 3000, 
#     }
# )
# print("Optimized!")

#print(result)
# print("flat shapes: ", flat_thetas.reshape(1,flat_thetas.shape[0]).shape)
# thetas = inflate_matrixes(flat_thetas.reshape(1, flat_thetas.shape[0]), theta_shapes_list)
# for matrix in thetas:
#     print("shape: ", matrix.shape)
#     print("values:\n", matrix)

# #result = feed_forward()
# print("test: \n")
# res = back_propagation(flat_thetas, theta_shapes, X, Y)
# print(res)
# for matrix in res:
#     print("shape: ", matrix.shape)
#     print("values:\n", matrix)


# calculo de porcentaje de error  
# print("result len:", len(flat_thetas))
result_thetas = inflate_matrixes(flat_thetas, theta_shapes_list)
# print("thetas list len:", len(result_thetas))
# # print("result_thetas shape: ", result_thetas.shape)
# print("result_thetas value: \n", result_thetas)
prediction = feed_forward(result_thetas, X)[-1]
percentage = error_percentage(prediction, y_train)

print("ACCURACY: ", percentage)

model_file = open(model_filename, 'wb')
# writing model 
pickle.dump(np.asarray(flat_thetas), model_file)
model_file.close()

end = time.time()
print("TIEMPO DE EJECUCION: {}s".format(end-start))