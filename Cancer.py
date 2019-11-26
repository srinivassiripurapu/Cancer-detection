import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import DataFrame



# My Neural Network model
"""
    ---> [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID
"""



# Dimensions of each layer
"""
    --> 2 hidden layers with 5 and 2 neuron units, 
    --> A input layer with 9 input neuron units
    --> Output layer with a single unit which gives either '0' or '1'
"""
layers_dims = [9, 5, 2, 1] 



# Reading Data
"""
    --> Reading data from the given excel sheet
    --> Division of given data into training set(80%) and test set(20%)
"""
path = ('Cancer.xlsx')  # The data is read as a data frame from the given excel sheet
df = pd.read_excel(path)
temp = df.as_matrix()   # conversion of data frame into a numpy array

test_x = temp[0:140, 1:10]
test_x = test_x.T
train_x = temp[140:698, 1:10]
train_x = train_x.T

test_y = temp[0:140, [10]]
test_y = test_y.T
test_y = (test_y/2)-1   # Because the output column in data set is given in terms of 2 and 4.
train_y = temp[140:698, [10]]
train_y = train_y.T
train_y = (train_y/2)-1



# Common functions used in artificial neural networks
"""
    --> Basic sigmoid function
"""
def sigmoid(z):

    s = 1/(1 + np.exp(-z))    
    return s, z



"""
    --> Basic relu function
"""
def relu(z):

    s = z*(z > 0)
    return s, z



"""
    --> Function for backward propagation
"""
def sigmoid_backward(dA, cache):

    Z = cache
    
    s = 1/(1+np.exp(-Z))
    dZ = dA * s * (1-s)
    
    assert (dZ.shape == Z.shape)
    
    return dZ



"""
    --> Function for backward propagation
"""
def relu_backward(dA, cache):
    
    Z = cache

    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    
    assert (dZ.shape == Z.shape)
    
    return dZ



"""
    --> Initialising parameters
"""
def initialize_parameters_deep(layer_dims):
    """
    Input Arguments:
    layer_dims - A python list containing the dimensions of each layer in my neural network
    
    Function Returns:
    parameters - python dictionary containing my network parameters "W1", "b1", ..., "WL", "bL"
    """
    L = len(layer_dims) 
    parameters = {}       

    for l in range(1, L):
       
        np.random.seed(1);
        # Random Initialisation
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * 0.1
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
        assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))
        assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))
   
    return parameters



"""
    --> Function for forward propagation
"""
def linear_forward(A, W, b):
    """
    Input Arguments:
    A - Activations from previous layer
    W & b are weight matrix and bias vector respectively

    Function Returns:
    Z - Pre-activation parameter 
    cache -- A python dictionary containing "A", "W" and "b"
    It is stored for computing the backward propagation
    """

    Z = np.dot(W, A) + b
    assert(Z.shape == (W.shape[0], A.shape[1]))

    cache = (A, W, b)
    return Z, cache



"""
    --> Function for forward propagation
"""
def linear_activation_forward(A_prev, W, b, activation):
    """
    Input Arguments:
    A_prev - activations from previous layer
    W & b are weight matrix and bias vector respectively
    activation - "sigmoid" or "relu" (to tell whether it is sigmoid or relu)

    Function Returns:
    A - Post-activation value 
    cache - A python dictionary containing "linear_cache" and "activation_cache";
    It is stored for computing the backward pass efficiently
    """
    
    if activation == "sigmoid":

        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)

    elif activation == "relu":

        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)

    assert (A.shape == (W.shape[0], A_prev.shape[1]))
    cache = (linear_cache, activation_cache)

    return A, cache



"""
    --> Main function for forward propagation 
"""
def L_model_forward(X, parameters):
    """
    Input Arguments:
    parameters - Current parameters which is a dictionary cantaining weight(W) matrices and bias(b) vectors of all layers
    
    Function Returns:
    AL - last layers' post-activation value
    """

    caches = []
    A = X
    L = len(parameters) // 2         
    
    # [LINEAR -> RELU]*(L-1)
    for l in range(1, L):
        A_prev = A
        A, cache = linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], 'relu')
        caches.append(cache)
    
    # [LINEAR -> SIGMOID] ---->last layer
    AL, cache = linear_activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)], 'sigmoid')
    caches.append(cache)
    
    assert(AL.shape == (1,X.shape[1]))
            
    return AL, caches



"""
    --> Cross-Entropy cost function
"""
def compute_cost(AL, Y):
    """
    Input Arguments:
    AL - last layer's activations
    Y - original output predictions

    Function Returns:
    cost - cross-entropy cost
    """
    
    m = Y.shape[1]

    cost = (-1/m) * np.sum(np.dot(Y, (np.log(AL)).T) + np.dot(1-Y, (np.log(1-AL)).T))
    
    cost = np.squeeze(cost)    
    assert(cost.shape == ())
    
    return cost



"""
    --> Function for backward propagation
"""
def linear_backward(dZ, cache):
    """
    Input Arguments:
    dZ - Gradient of the cost with respect to the linear output of current layer l
    cache - tuple of values (A_prev, W, b) coming from the forward propagation in the current layer

    Function Returns:
    dA_prev - Gradient of the cost with respect to the activation of the previous layer l-1
    dW - Gradient of the cost with respect to W for current layer l
    db - Gradient of the cost with respect to b for current layer l
    """
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = (1/m) * np.dot(dZ, A_prev.T)
    db = (1/m) * np.sum(dZ, axis = 1, keepdims = True)
    dA_prev = np.dot(W.T, dZ)
    
    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)
    
    return dA_prev, dW, db



"""
    --> Function for backward propagation
"""
def linear_activation_backward(dA, cache, activation):
    """
    Input Arguments:
    dA - post-activation gradient for current layer l 
    cache - tuple of values (linear_cache, activation_cache)
    
    Function Returns:
    dA_prev - Gradient of the cost with respect to the activation of the previous layer l-1
    dW -- Gradient of the cost with respect to W for current layer l
    db -- Gradient of the cost with respect to b for current layer l
    """

    linear_cache, activation_cache = cache
    
    if activation == "relu":
   
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
     
    elif activation == "sigmoid":

        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    
    return dA_prev, dW, db



"""
    --> Main function for backward propagation
"""
def L_model_backward(AL, Y, caches):
    """
    Function Returns:
    grads - A dictionary with the gradients containing dA, dW, db
    """
    grads = {}
    L = len(caches) 
    m = AL.shape[1]
    
    # Initializing the backpropagation
    dAL = -(np.divide(Y, AL) - np.divide(1-Y, 1-AL))
    
    # Lth layer (SIGMOID -> LINEAR) gradients
    current_cache = caches[L-1]
    grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache, 'sigmoid')
    
    # Loop from l = L-2 to l = 0
    for l in reversed(range(L-1)):
        # lth layer: (RELU -> LINEAR) gradients. 
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads['dA' + str(l+1)], current_cache, 'relu')
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads



"""
    --> for updating parameters at each gradient step
"""
def update_parameters(parameters, grads, learning_rate):
    """
    Input Arguments:
    parameters - python dictionary containing all parameters 
    grads - python dictionary containing your gradients, output of L_model_backward
    """
    
    L = len(parameters) // 2 

    # Update rule for each parameter
    for l in range(L):
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * grads["dW" + str(l+1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * grads["db" + str(l+1)]
    return parameters



"""
    --> Final model
"""
def L_layer_model(X, Y, layers_dims, learning_rate = 0.0075, num_iterations = 3000, print_cost=False):
    
    # keeping track of cost
    costs = []                     
    
    # Parameters initialization
    parameters = initialize_parameters_deep(layers_dims)
    
    # Gradient descent
    for i in range(0, num_iterations):

        # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID
        AL, caches = L_model_forward(X, parameters)
        
        # Compute cost
        cost = compute_cost(AL, Y)
    
        # Backward propagation
        grads = L_model_backward(AL, Y, caches)
 
        # Update parameters
        parameters = update_parameters(parameters, grads, learning_rate)
                
        # Printing the cost every 100 training example
        if print_cost and i % 1000 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
        if print_cost and i % 1000 == 0:
            costs.append(cost)
            
    # plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    
    return parameters



"""
    --> calling for the main function to get the final parameters
"""
parameters = L_layer_model(train_x, train_y, layers_dims, num_iterations = 10000, print_cost = True)



"""
    --> for predicting the output and calculating accuracy
"""
def predict(X, Y, parameters):
    
    # Computes probabilities using forward propagation, and classifies to 0/1 using 0.5 as the threshold.
 
    AL, cache = L_model_forward(X, parameters)
    m = Y.shape[1]
    predictions = (AL >= 0.5)
    accuracy = ((np.dot(Y,predictions.T) + np.dot(1-Y,1-predictions.T))/(Y.size)*100)
    
    return np.squeeze(accuracy), predictions



"""
    --> printing predicted values for training set
"""
print("-----------------------------------------------------------------------------\n")
train_accuracy, pred_train = predict(train_x, train_y, parameters)
print("The predicted values for training set---\n")
print(pred_train)
print("-----------------------------------------------------------------------------\n")



"""
    --> printing predicted values for test set
"""
print("-----------------------------------------------------------------------------\n")
test_accuracy, pred_test = predict(test_x, test_y, parameters)
print("The predicted values for test set---\n")
print(pred_test)
print("-----------------------------------------------------------------------------\n")



"""
    --> printing train_accuracy and test_accuracy
"""
print("The Accuracy for the training set is : ")
print(train_accuracy)
print("-----------------------------------------------------------------------------\n")
print("The Accuracy for the test set is : ")
print(test_accuracy)
print("-----------------------------------------------------------------------------\n")