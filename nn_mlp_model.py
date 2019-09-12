"""

_author: Bruno Carvalho <bgcarvalho gmail com>
_copyright: 2018

"""

import numpy as np
import warnings
np.seterr(all='raise')


def relu(Z):
    """

    :param Z: linear output to be activated
    :return:
    """
    return np.fmax(0, Z)

def relu_deriv(Z):
    Z[Z < 0] = 0
    Z[Z >= 1] = 1
    return Z


def relu_deriv_hidden(m, A1, W, dZ, A):
    """
    relu_deriv_output
    relu_deriv_hidden
    **kwargs

    :param X:
    :param Y:
    :param W:
    :param b:
    :param Z:
    :param A:
    :return:
    """
    dZa = np.dot(W.T, dZ) * (1 - np.power(A, 2))
    dW = np.dot(dZa, A1.T) / m
    db = np.sum(dZa, axis=1, keepdims=True) / m
    dA = np.dot(W.T, dZa)
    return dW, db, dA, dZa


def relu_deriv_hidden2(m, dW, dZ, Z, A):
    """
    relu_deriv_output
    relu_deriv_hidden
    **kwargs

    :param X:
    :param Y:
    :param W:
    :param b:
    :param Z:
    :param A:
    :return:
    """
    dZ_ = np.dot(dW, dZ) * relu_deriv(Z)
    dW = np.dot(dZ_, A.T) / m
    db = np.sum(dZ_, axis=1, keepdims=True) / m
    dA = np.zeros((9,9))
    return dW, db, dA, dZ_


def relu_deriv_output(m, W, A, Y):
    """
    For ReLU activation, the output would be a regression problem.
    A possible cost function is: C = (1/2)*(AL - Y)ˆ2

    :param Y:
    :param A1:
    :param A2:
    :return:
    """
    dZ = A - Y
    dW = np.dot(dZ, A.T) / m
    db = np.sum(dZ, axis=1, keepdims=True) / m
    dA = np.dot(W.T, dZ)
    return dW, db, dA, dZ


def sigmoid(Z):
    """
    Sigmoid
    :param Z:
    :return:
    """
    Z = np.squeeze(Z)
    try:
        return 1.0 / (1.0 + 1.0 * np.exp(-Z))
    except:
        return np.inf


def sigmoid_deriv_output(A, Y):
    return - (np.divide(Y, A) - np.divide(1 - Y, 1 - A))


def sigmoid_deriv(A):
    #return A * (1 - A)
    return sigmoid(A) * (1 - sigmoid(A))


def sigmoid_deriv_hidden(Z):
    deriv = sigmoid(Z) * (1 - sigmoid(Z))
    pass


def softmax(Z):
    return np.exp(Z) / np.sum(np.exp(Z))


class Layer:
    pass


class MLP:
    """Neural Network Model

    """
    _layer = {
        'name': '',
        'layer': 0,
        'W': np.array([]),
        'b': np.array([]),
        'dW': np.array([]),
        'db': np.array([]),
        'activation': None,
        'a_fn': None,  # activation
        'd_fn': None,  # derivative
        'z_fn': None,  # sum
        'A': np.array([]),
        'Z': np.array([]),
        'dA': np.array([]),
        'dZ': np.array([]),
        'type': None,  # input, hidden, output
        'input': 0,
        'output': 0,
        'size': 0,
    }
    _layers = []
    _X = None
    _Y = None
    _input = None
    _lambda = 1e8
    _epoch = 0
    _iterations = []
    _costs = []
    _AL = None

    batch_size = 32
    alpha = 0.01
    gamma = 1e6
    beta1 = 0.9
    beta2 = 0.01
    n_iterations = 1
    epochs = 10
    batch = 0
    batch_cost = 0.0

    def __init__(self, input_shape):
        """Simple MLP network"""
        self._input = input_shape

    def add_fc(self, nodes, activation='relu'):
        """
        Fully connected layer
        """
        nl = len(self._layers)
        layer = self._layer.copy()
        layer['name'] = 'fully_connected_' + str(nl + 1)
        layer['layer'] = nl
        layer['activation'] = activation

        if nl == 0:
            layer['input'] = self._input
            layer['type'] = 'input'
        else:
            layer['input'] = self._layers[nl - 1]['output']

        if activation == 'relu':
            layer['a_fn'] = relu
            #layer['d_fn'] = None

        elif activation == 'sigmoid':
            layer['a_fn'] = sigmoid
            #layer['d_fn'] = None

        elif activation == 'softmax':
            layer['a_fn'] = softmax
            #layer['d_fn'] = None

        layer['z_fn'] = self.z_fully
        layer['output'] = nodes

        self._layers.append(layer)


    def build(self):
        """

        :return: void
        """
        self._L = len(self._layers)
        self.init_parameters()

    def init_parameters(self):
        """

        """
        for lay in range(0, self._L):
            # weights
            self._layers[lay]['W'] = 0.1 * np.random.randn(
                self._layers[lay]['input'][1],
                self._layers[lay]['output']
            )
            # bias
            self._layers[lay]['b'] = np.zeros((self._layers[lay]['output'], 1))

            # ReLU
            if self._layers[lay]['activation'] == 'relu':
                if lay == self._L:
                    # output layer
                    self._layers[lay]['d_fn'] = relu_deriv_output
                else:
                    # intermediate
                    self._layers[lay]['d_fn'] = relu_deriv_hidden

            # sigmoid
            elif self._layers[lay]['activation'] == 'sigmoid':
                if lay == self._L:
                    # output layer
                    self._layers[lay]['d_fn'] = sigmoid_deriv_output
                else:
                    # intermediate
                    self._layers[lay]['d_fn'] = sigmoid_deriv_hidden

            # Softmax
            elif self._layers[lay]['activation'] == 'softmax':
                if lay == self._L:
                    pass
                else:
                    pass
            self._layers[lay]['size'] = self._layers[lay]['W'].size + \
                                        self._layers[lay]['b'].size + \
                                        self._layers[lay]['A'].size

    def train(self, X, Y):
        """

        """
        self._X = np.array(X).T
        self._Y = np.array(Y).T
        #update_params = self.update_params_default()
        cost_b = 0.0
        Yb = self._Y

        for i in range(self.epochs):

            # batch
            for b in range(self.n_iterations):
                xi = X[b, :]
                # forward
                Ab = self.forward(xi)

                # cost
                #cost_b = self.cost_logistic(Ab, Yb)
                cost_b = self.cost_regression(Ab, Yb)

                # backward
                self.backward(Ab, Yb)

        return cost_b

    def forward(self, X):
        """

        """
        X = np.expand_dims(X, axis=0)
        lay = 0
        W = self._layers[lay]['W']
        b = self._layers[lay]['b']
        Z = self._layers[lay]['z_fn'](X, W, b)
        A = self._layers[lay]['a_fn'](Z)
        self._layers[lay]['Z'] = Z
        self._layers[lay]['A'] = A

        for lay in range(1, self._L):
            W = self._layers[lay]['W']
            b = self._layers[lay]['b']
            Z = self._layers[lay]['z_fn'](A, W, b)
            A = self._layers[lay]['a_fn'](Z)
            self._layers[lay]['Z'] = Z
            self._layers[lay]['A'] = A

        self._AL = A

        return A

    def z_fully(self, X, W, b):
        """

        :param X:
        :param W:
        :param b:
        :return:
        """
        # return np.dot(W.T, X) + b
        return np.dot(X, W) + b

    def cost(self, A, Y):
        """

        """
        return A - Y

    def cost_logistic(self, A, Y):
        """
        Cross-entropy error/cost
        :param A:
        :param Y:
        :return:
        """
        m = Y.shape[0]
        cost = np.dot(np.log(A), Y.T) + np.dot(np.log(1 - A), (1 - Y).T)
        cost = np.squeeze(- np.sum(cost) / m)

        return cost

    def cost_regression(self, A, Y):
        """

        :param A:
        :param Y:
        :return:
        """
        self.batch_cost = 0.5 * np.sum(np.square(A - Y))
        self._iterations.append(self.batch_cost)
        #self._iterations[self._epoch] = cost
        return self.batch_cost

    def backward(self, A, Y):
        """

        """
        m = Y.shape[0]

        # output layer
        dW, db, dA, dZ = self._layers[self._L]['d_fn'](A, Y)

        A1 = A
        for lay in range(len(self._layers) - 1, 1):
            A1 = self._layers[lay - 1]['A']
            A2 = self._layers[lay]['A']
            W = self._layers[lay]['W']
            b = self._layers[lay]['b']
            #dZ = self._layers[lay - 1]['dZ']
            m = A1.shape[0]

            # recalculate
            dW, db, dA, dZ = self._layers[lay]['d_fn'](m, A1, W, dZ, A2)

            # save for further use
            self._layers[lay]['dW'] = dW
            self._layers[lay]['db'] = db
            self._layers[lay]['dA'] = dA
            self._layers[lay]['dZ'] = dZ

            # update weights and bias!
            self._layers[lay]['W'] = W - self.alpha * self._layers[lay]['dW']
            self._layers[lay]['b'] = b - self.alpha * self._layers[lay]['db']

    def update_params_default(self):
        """

        :return:
        """
        for lay in range(0, len(self._layers)):
            self._layers[lay]['W'] = self._layers[lay]['W'] - self.alpha * self._layers[lay]['dW']
            self._layers[lay]['b'] = self._layers[lay]['b'] - self.alpha * self._layers[lay]['db']

    def update_params_reg(self):
        """

        :return:
        """
        for lay in range(0, len(self._layers)):
            self._layers[lay]['W'] = self._layers[lay]['W'] - self.alpha * self._layers[lay]['dW']
            self._layers[lay]['b'] = self._layers[lay]['b'] - self.alpha * self._layers[lay]['db']

    def loss(self):
        """

        """
        pass

    def predict(self):
        """

        """
        pass

    def get_layers(self):
        """

        """
        return self._layers

    def save(self):
        """

        """
        pass

    def load(self):
        """

        """
        pass

    def count_parameters(self, layer=None):
        n = 0
        if layer is None:
            for lay in range(0, self._L):
                n += self._layers[lay]['W'].size
                n += self._layers[lay]['b'].size
                n += self._layers[lay]['A'].size
        return n

    def get_info(self):
        """

        :return:
        """
        msg = ''
        for lay in range(0, len(self._layers)):
            pass


np.random.seed(5)
data = np.array([
    [1, 0, 1],
    [1, 1, 1],
    [0, 1, 1],
    [0, 0, 0],
    [0, 0, 0],
])

net = MLP((1, 2))
net.add_fc(1, activation='sigmoid')
net.build()

# feed forward
y_h = net.forward(data[0, :2])

# error function (MSE)
net.cost_regression(y_h, data[0, -1])

"""
for ln, lay in enumerate(net.get_layers()):
    print(f'Layer {ln}')
    for key in lay:
        print('\t', key)
        print('\t', lay[key])
"""

print('Weights')
print(net.get_layers()[-1]['W'])
print('Bias')
print(net.get_layers()[-1]['b'])
print('Sum')
print(net.get_layers()[-1]['Z'])
print('Output')
print(net.get_layers()[-1]['A'])
print('Cost')
print(net.batch_cost)