import numpy as np
#AARON'S NEURAL NETWORK SUCKS(TRUE).
A = np.array([[1,2,3],[4,5,6],[7,8,9]])

def sigmoid(Z):
    #assuming theta is a row vector
    return 1/(1+np.exp(-Z))

def sigmoidGradient(Z):
    return Z * (1-Z)


"""
INPUTS
X           : Test Feature Vector
Theta_list  : List of numpy array coefficients of each activation layer
depth       : Number of hidden layers

OUTPUT
y           : Binarized outputs
"""
def feedForward(x,Theta_list,depth):
    Theta = Theta_list[0]
    # appending column vector on ones to the beginning of the input matrix
    x = np.extend(np.ones((x.shape[0],1)),x)
    ##a = np.ones((Theta.shape[0],1))
    # iteration variable

    a = sigmoid(x.dot(np.transpose(Theta)))

    if depth > 0:
        del(Theta_list[0])
        return feedForward(a,Theta_list,depth-1)
    elif depth == 0:
        return a
    else:
        print("Depth cannot be less than 0")


"""
INPUTS
X           : Feature Vector
y           : BINARIZED Target Variable
depth       : Number of hidden layers (Excluding input and output layers)
num_nodes   : Number of nodes per layer
reg_coeff   : Regularization Coefficient

OUTPUTS
thetas      : List of matrices of coefficients of each layer
cost        : Final cost function
grad        : Cost gradients (To check whether backprop is working correctly)
"""
def backPropogation(X,y,depth,Thetas,num_nodes,reg_coeff = 1):
    m = X.shape[0]

    activation_layers = []
    #Thetas = []

    """
    STEP 1 : Initialize thetas randomly
    """
    """
    prev = X.shape[1]
    i = 0
    while(i < depth):
        theta = np.random.rand(num_nodes,prev + 1)
        Thetas.append(theta)
        prev = num_nodes
        i += 1
        print(theta.shape)
    Thetas.append(np.random.rand(y.shape[1],prev + 1))
    print(Thetas[2].shape)
    """

    """
    STEP 2 
    Run feedForward once but retain all activation layer values
    NOTE: Levels start from zero NOT one. (Input layer is the zeroeth
          layer
    """
    activation = [X]
    reg_theta_sum =[]
    reg_theta = []

    layer = 0
    for theta in Thetas:
        # Inserts column of ones to the previous layer in the network
        activation[layer] = np.concatenate((np.ones((activation[layer].shape[0],1)),
                                            activation[layer]),axis = 1)
        a = sigmoid(activation[layer].dot(np.transpose(theta)))
        print(a.shape)
        activation.append(a)
        # Stores the sum of the sqaures of the coefficients
        # This list will be used when regularize the cost function
        r_theta = theta[:,1:]**2
        reg_theta.append(r_theta)
        reg_theta_sum.append(r_theta.sum())
        layer += 1

    # Without Regularization
    cost = (-1/m)*(sum(y*np.log(activation[-1])) + sum((1-y)*np.log(1 - activation[-1])))

    # With Regularization
    error = (reg_coeff/(2*m))*(sum(reg_theta_sum))
    cost =  cost + error

    """
    STEP 3
    Coding the core of the neural network. A.K.A the backpropogation steps
    """
    d = [0 for x in range(len(activation))]
    deltas = [0 for x in range(len(Thetas))]

    for i in range(m):
        d[-1] = activation[-1][i,:] - y[i,:]
        #print(d[-1].shape, Thetas[-1].shape)

        for layer in range(len(d)-2,0,-1):
            #print(layer,Thetas[layer].shape,d[layer+1].shape,)
            d[layer] = (np.transpose(Thetas[layer])).dot(np.transpose(d[layer+1])) * \
                       sigmoidGradient(activation[layer][i,:])
            d[layer] = d[layer][1:]
            print(d[layer].shape)

        for layer in range(len(deltas)):
            d_temp = d[layer+1].reshape(d[layer+1].shape[0],1)
            activation_temp = activation[layer][i,:].reshape(activation[layer][i,:].shape[0],1)
            print(layer,d[layer+1].shape,np.transpose(activation[layer][i,:].shape))
            deltas[layer] = deltas[layer] + d_temp.dot(np.transpose(activation_temp))

        csize = activation[-2][i,:].shape[0]
        deltas[-1] = deltas[-1] + np.transpose(d[-1]).dot(activation[-2][i,:].reshape(1,csize))

    Theta_grad = [0 for x in range(len(Thetas))]
    print(deltas)

    for i in range(len(Thetas)):
        Theta_grad[i] = 1/m * deltas[i] + (reg_coeff/m)* np.concatenate((np.zeros((Thetas[i].shape[0],1)),
                                                reg_theta[i]), axis = 1)

    print(Theta_grad)
    return cost, Theta_grad


X = np.ones((50,4))
y = np.ones((50,1))

depth = 2
num_nodes = 25

Thetas = []
prev = X.shape[1]
i = 0
while(i < depth):
    theta = np.random.rand(num_nodes,prev + 1)
    Thetas.append(theta)
    prev = num_nodes
    i += 1
    print(theta.shape)
Thetas.append(np.random.rand(y.shape[1],prev + 1))
print(Thetas[2].shape)

print(backPropogation(X,y,depth = 2,num_nodes = 25,reg_coeff = 1,Thetas = Thetas ));