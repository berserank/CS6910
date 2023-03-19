import numpy as np
import matplotlib.pyplot as plt
import wandb

"""###**Helper Functions - 1**"""

#One Hot Encoding for y
def one_hot_encode(y):
  encoded_array = np.zeros((y.size, y.max()+1), dtype=int)
  encoded_array[np.arange(y.size),y] = 1 
  return encoded_array

#Converts softmax probabilities to labels
def softmax_to_label(softmax_output):
  max_index = np.argmax(softmax_output, axis = 0)
  return max_index


"""### **Activation Functions**

Here are the 4 activation functions I used for my network. 

While dealing with MNIST/ Fashion-MNIST , output layer will have softmax as it's activation function inorder to output a vector of probabilities.
"""

def sigmoid(z):
    #print(-z)
    return 1 / (1 + np.exp(-(z)))

def tanh(z):
    return np.tanh(z)

def relu(z):
    return (z>0)*(z) + ((z<0)*0)

def leakyRelu(z):
    return (z>0)*(z) + ((z<0)*(z)*0.01)

def softmax(x):
    # x = np.float128(x)
    temp = np.exp(x-np.max(x, axis = 0))
    fin = temp/temp.sum(axis = 0)
    return fin

"""### **Weight Initialisation**

Attaching the reference I used for implementing Xavier initialisation: [Xavier Initialisation](https://cs230.stanford.edu/section/4/)

This initialises a dictionary of parameters of length = *hidden layer size + 1*


> **Parameters:** { 'W1' : $W_{(n[0],X)}$ , 'b1' : $b_{(n[0],1)}$ , 'W2' : $W_{(n[1],n[0])}$ , 'b2' : $b_{(n[1],1)}$ , 'W2' : $W_{(n[2],n[1])}$, 'b2' : $b_{(n[2],1)}$ , . . .  .  .  }




where n is the vector of hidden layer sizes.

While dealing with MNIST/ Fashion-MNIST , input size is 784 and output layer size is 10


"""

def initialize_parameters(input_size, n, output_size, initialisation):
  if (initialisation == 'Random'):
    parameters = {}
    parameters['W'+str(1)] = np.random.randn(n[0],input_size)*0.01
    parameters['b'+str(1)] = np.random.randn(n[0],1)
    for i in range(1,len(n)):
        parameters['W'+str(i+1)] = np.random.randn(n[i],n[i-1])*0.01
        parameters['b'+str(i+1)] = np.random.randn(n[i],1)
    parameters['W'+str(len(n)+1)] = np.random.randn(output_size,n[-1])*0.01
    parameters['b'+str(len(n)+1)] = np.random.randn(output_size,1)

  elif (initialisation == 'Xavier'):
    parameters = {}
    m = np.sqrt(6)/(input_size+n[0])
    parameters['W'+str(1)] = np.random.uniform(-m,m, (n[0],input_size))
    parameters['b'+str(1)] = np.random.randn(n[0],1)
    for i in range(1,len(n)):
        m = np.sqrt(6)/(n[i-1]+n[i])
        parameters['W'+str(i+1)] = np.random.uniform(-m,m, (n[i],n[i-1]) )
        parameters['b'+str(i+1)] = np.random.randn(n[i],1)
    m = np.sqrt(6)/(output_size+n[-1])
    parameters['W'+str(len(n)+1)] = np.random.uniform(-m,m,(output_size,n[-1]))
    parameters['b'+str(len(n)+1)] = np.random.randn(output_size,1)

  return parameters

def initialize_parameters_zeros(input_size, n, output_size):
    parameters = {}
    parameters['W'+str(1)] = np.zeros((n[0],input_size))
    parameters['b'+str(1)] = np.zeros((n[0],1))
    for i in range(1,len(n)):
        parameters['W'+str(i+1)] = np.zeros((n[i],n[i-1]))
        parameters['b'+str(i+1)] = np.zeros((n[i],1))
    parameters['W'+str(len(n)+1)] = np.zeros((output_size,n[-1]))
    parameters['b'+str(len(n)+1)] = np.zeros((output_size,1))
    return parameters

"""### **Forward Propagation**

This module takes the parameters dictionary, sequence of activation functions, input vector as inputs and outputs a dictionary of layer wise outputs.



> **Layer Wise Outputs:** { 'h1' : $h_{n[0]}$ ,  'a1' : $a_{n[0]}$ , 'h2' : $h_{n[1]}$ ,  'a2' : $a_{n[1]}$ ,  'h3' : $h_{n[2]}$ ,  'a3' : $a_{n[2]}$ , . . .  .  .  }




  Where, h is pre-activation output and a is post-activation output of a particular layer. If g is the activation function, this module basically does,



>$h_i = W_ia_{i-1} + b_i$   

> $g(h_i) = a_i$
"""

def linear(W, X, b, activation_func):
    #print(f"W Shape = {W.shape}, X Shape= {X.shape}, W= {W}, X = {X}, b = {b} " )
    h = np.matmul(W,X)+b
    if activation_func == 'sigmoid':
        #print(h)
        a = sigmoid(h)
    elif activation_func == 'relu':
        a = relu(h)
    elif activation_func == 'leakyRelu':
        a = leakyRelu(h)
    elif activation_func == 'tanh':
        a = tanh(h)
    elif activation_func == 'softmax':
        a = softmax(h)
    return h,a

def ForwardPropagation(X, parameters, activation_func):
    layer_wise_outputs = {}
    layer_wise_outputs['h1'], layer_wise_outputs['a1'] = linear(parameters['W1'], X, parameters['b1'], activation_func[0])
    for i in range(1, (len(parameters)//2)):
        layer_wise_outputs['h'+str(i+1)], layer_wise_outputs['a'+str(i+1)] = linear(parameters['W'+str(i+1)],layer_wise_outputs['a'+str(i)],parameters['b'+str(i+1)], activation_func[i])
    return layer_wise_outputs

"""### **Loss, Cost, Accuracy Functions**

Below is the code for mean square error loss, cross entropy loss, cost function, accuarcy score 

Notice that accuracy score takes it's *true values* as labels and *a softmax vector* as predicted values.
"""

def MSELoss(Y, Y_pred):
    MSE = np.mean((Y - Y_pred) ** 2, axis = 1)
    MSE = np.mean(MSE)
    return MSE

def CrossEntropyLoss(Y, Y_pred):
    CE = [-Y[i] * np.log(Y_pred[i]) for i in range(len(Y_pred))]
    crossEntropy = np.mean(CE)
    return crossEntropy

def cost(Y, Y_pred, loss_func):
    if (loss_func == 'MSE'):
        return (MSELoss(Y, Y_pred))
    elif (loss_func == 'CE'):
        return (CrossEntropyLoss(Y, Y_pred))

def accuracy_score(y_true, y_pred):
    pred_labels = np.argmax(y_pred, axis=0)
    return np.sum(pred_labels == y_true) / len(y_true)

"""
 **Back Propagation**

I have implemented backpropagation with 2 modules in a for loop. First module backpropagates through post-actiavtion outputs(`ActivationBackward`) and 
second one back propagates through pre-activation outputs(`LayerBackward`). Later the `BackPropagate` function helps us creating the 
`gradients` dictionary which has all the corresponding gradients of the loss function with respect to each weight and bias.
"""

def ActivationBackward(dA, Z, activation_func) :
    
    if (activation_func == 'sigmoid'):
        grad = sigmoid(Z)*(1-sigmoid(Z))
       
    elif (activation_func == 'relu'):
        grad = np.where(Z>0, 1, 0)

    elif (activation_func == 'leakyRelu'):
        grad = np.where(Z>0, 1, 0.01)

    elif (activation_func == 'tanh'):
        grad = 1 - tanh(Z)**2
    elif (activation_func == 'softmax'):
        grad = softmax(Z) * (1-softmax(Z))
    dZ = dA * grad
    return dZ

def softmax_derivative(x):
    return softmax(x) * (1-softmax(x))        
    
def LayerBackward(dZl, Wl, bl, A_prev):
    
    m = A_prev.shape[1]
    # print(m)
    dWl = (1/m) * np.matmul(dZl, A_prev.T)
    dbl = (1/m)* np.sum(dZl, axis=1, keepdims=True)
    dA_prev = np.matmul(Wl.T,dZl)
    
    assert (dA_prev.shape == A_prev.shape)
    assert (dWl.shape == Wl.shape)
    assert (dbl.shape == bl.shape)
    return dWl, dbl, dA_prev
   
def BackPropagate(parameters, layer_wise_outputs,X, Y, activation_func, loss):
    gradients = {}
    l = len(layer_wise_outputs)//2
    m = Y.shape[1]
    AL = layer_wise_outputs['a'+str(l)]
    HL = layer_wise_outputs['h'+str(l)]
    
    if (loss == 'CE' or loss == 'cross_entropy'):
        gradients['dh'+str(l)] = AL-Y
    elif (loss == 'MSE' or loss == 'mean_squared_error'):
        gradients['dh'+str(l)] = (AL-Y) * softmax_derivative(HL)
        
    for i in range(l-1,0,-1):
        gradients['dW'+str(i+1)],gradients['db'+str(i+1)],gradients['da'+str(i)] = LayerBackward(gradients['dh'+str(i+1)], parameters['W'+str(i+1)], parameters['b'+str(i+1)], layer_wise_outputs['a'+str(i)])
        gradients['dh'+str(i)] = ActivationBackward(gradients['da'+str(i)], layer_wise_outputs['h'+ str(i)] , activation_func[i-1])
        
    gradients['dW'+str(1)],gradients['db'+str(1)],gradients['da'+str(0)] = LayerBackward(gradients['dh'+str(1)], parameters['W'+str(1)], parameters['b'+str(1)], X)    
    
    return gradients


"""###**Optimisers**

I created a base class for optimizers that includes two attributes: `learning rate` and `weight decay`. Any optimizer can inherit from this class. The child classes differ in their `update` methods, and the update equations for the optimizers implemented are listed below


**References** : [CS6910 Lecture 5](https://iitm-pod.slides.com/arunprakash_ai/cs6910-lecture-5),  [Optimiser Algorithms](https://cs229.stanford.edu/proj2015/054_report.pdf)

To add a new optimiser, we need to write a new iherited class and write the update method accordingly.

###**Regularisation**
I assigned `weight decay` $λ$ as a common attribute to all the optimiser classes. I changed gradients at every update step accordingly.

"""

class Optimiser:
    def __init__(self, lr, weight_decay):
        self.lr = lr
        self.wd = weight_decay

    def update(self, parameters, gradients):
        raise NotImplementedError

class SGD(Optimiser):
    def __init__(self, lr, weight_decay):
        super().__init__(lr,weight_decay)

    def update(self, parameters, gradients):
        L = len(parameters) // 2 
        for l in range(1, L + 1):
          parameters["W" + str(l)] = (1-self.wd*self.lr)*parameters["W" + str(l)] - self.lr * gradients["dW" + str(l)]
          parameters["b" + str(l)] = (1-self.wd*self.lr)*parameters["b" + str(l)] - self.lr * gradients["db" + str(l)]
        return parameters

class Momentum(Optimiser):
    def __init__(self, lr, beta, weight_decay):
        super().__init__(lr,weight_decay)
        self.beta = beta
        self.v = {}
  

    def update(self, parameters, gradients):
        L = len(parameters)//2
        if self.v == {}:
          for l in range(1, L + 1):
            self.v["W"+str(l)] = 0
            self.v["b"+str(l)] = 0
        for l in range(1, L + 1):
          gradients["dW" + str(l)]= gradients["dW" + str(l)]+self.wd*parameters["W" + str(l)]
          gradients["db" + str(l)] = gradients["db" + str(l)]+self.wd*parameters["b" + str(l)]
          
          self.v["W"+str(l)] = self.beta * self.v["W"+str(l)] + (1-self.beta) * gradients["dW" + str(l)]
          parameters["W" + str(l)] = parameters["W" + str(l)] - self.lr * self.v["W"+str(l)]
          self.v["b"+str(l)] = self.beta * self.v["b"+str(l)] + (1-self.beta) * (gradients["db" + str(l)])
          parameters["b" + str(l)] = parameters["b" + str(l)] - self.lr * self.v["b"+str(l)]
        return parameters

class Nesterov(Optimiser):
    def __init__(self, lr, gamma,weight_decay):
        super().__init__(lr,weight_decay)
        self.gamma = gamma
        self.look_ahead = {}
        self.v = {}
        

    def update(self, parameters, gradients):
        L = len(parameters)//2
        if self.v == {}:
          for l in range(1, L + 1):
            self.v["W"+str(l)] = 0
            self.v["b"+str(l)] = 0
        if self.look_ahead == {}:
          for l in range(1, L + 1):
            self.look_ahead["W"+str(l)] = 0
            self.look_ahead["b"+str(l)] = 0

        for l in range(1, L + 1):
          gradients["dW" + str(l)]= gradients["dW" + str(l)]+self.wd*parameters["W" + str(l)]
          gradients["db" + str(l)] = gradients["db" + str(l)]+self.wd*parameters["b" + str(l)]
            
          self.look_ahead["W"+str(l)] = parameters["W" + str(l)]-self.gamma*self.v["W" + str(l)]
          parameters["W" + str(l)] = self.look_ahead["W"+str(l)] - self.lr * gradients["dW" + str(l)]
          self.v["W"+str(l)] = self.gamma * self.v["W"+str(l)] + self.lr * gradients["dW" + str(l)]

          self.look_ahead["b"+str(l)] = parameters["b" + str(l)]-self.gamma*self.v["b" + str(l)]
          parameters["b" + str(l)] = self.look_ahead["b"+str(l)] - self.lr * gradients["db" + str(l)]
          self.v["b"+str(l)] = self.gamma * self.v["b"+str(l)] + self.lr * gradients["db" + str(l)]

          
        return parameters

class RMSprop(Optimiser):
    def __init__(self, lr, decay_rate, eps,weight_decay):
        super().__init__(lr,weight_decay)
        self.decay_rate = decay_rate
        self.eps = eps
        self.s = {}


    def update(self, parameters, gradients):
        
        L = len(parameters)//2
        if self.s == {}:
          for l in range(1, L + 1):
            self.s["W"+str(l)] = 0
            self.s["b"+str(l)] = 0
        for l in range(1, L + 1):
          gradients["dW" + str(l)]= gradients["dW" + str(l)]+self.wd*parameters["W" + str(l)]
          gradients["db" + str(l)] = gradients["db" + str(l)]+self.wd*parameters["b" + str(l)]
            
          self.s["W"+str(l)] = self.decay_rate * self.s["W"+str(l)] + (1 - self.decay_rate) * (gradients["dW" + str(l)]**2)
          parameters["W" + str(l)] = parameters["W" + str(l)] - self.lr * (gradients["dW" + str(l)]) / (np.sqrt(self.s["W"+str(l)]) + self.eps)

          self.s["b"+str(l)] = self.decay_rate * self.s["b"+str(l)] + (1 - self.decay_rate) * (gradients["db" + str(l)]**2)
          parameters["b" + str(l)] = (1-self.wd)*parameters["b" + str(l)] - self.lr * (gradients["db" + str(l)]) / (np.sqrt(self.s["b"+str(l)]) + self.eps)
        
        return parameters
        

class Adam(Optimiser):
    def __init__(self, lr, beta1, beta2, eps,weight_decay):
        super().__init__(lr,weight_decay)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = {}
        self.v = {}
        self.t = 0

    def update(self, parameters, gradients):
        L = len(parameters)//2
        if self.m == {}:
          for l in range(1, L + 1):
            self.m["W"+str(l)] = 0
            self.m["b"+str(l)] = 0
        if self.v == {}:
          for l in range(1, L + 1):
            self.v["W"+str(l)] = 0
            self.v["b"+str(l)] = 0
        self.t += 1

        for l in range(1, L + 1):
          gradients["dW" + str(l)]= gradients["dW" + str(l)]+self.wd*parameters["W" + str(l)]
          gradients["db" + str(l)] = gradients["db" + str(l)]+self.wd*parameters["b" + str(l)]
            
          self.m["W" + str(l)] = self.beta1 * self.m["W" + str(l)] + (1 - self.beta1) * (gradients["dW" + str(l)])
          self.v["W" + str(l)] = self.beta2 * self.v["W" + str(l)] + (1 - self.beta2) * ((gradients["dW" + str(l)])**2)
          m_hat = self.m["W" + str(l)] / (1 - self.beta1 ** self.t)
          v_hat = self.v["W" + str(l)] / (1 - self.beta2 ** self.t)
          parameters["W" + str(l)] = parameters["W" + str(l)] - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

          self.m["b" + str(l)] = self.beta1 * self.m["b" + str(l)] + (1 - self.beta1) * (gradients["db" + str(l)])
          self.v["b" + str(l)] = self.beta2 * self.v["b" + str(l)] + (1 - self.beta2) * ((gradients["db" + str(l)])**2)
          m_hat = self.m["b" + str(l)] / (1 - self.beta1 ** self.t)
          v_hat = self.v["b" + str(l)] / (1 - self.beta2 ** self.t)
          parameters["b" + str(l)] = parameters["b" + str(l)] - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
        
        
        return parameters

 
class Nadam(Optimiser):
    def __init__(self, lr, beta1, beta2, eps,weight_decay):
        super().__init__(lr,weight_decay)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = {}
        self.v = {}
        self.t = 0

    def update(self, parameters, gradients):
        L = len(parameters)//2
        if self.m == {}:
          for l in range(1, L + 1):
            self.m["W"+str(l)] = 0
            self.m["b"+str(l)] = 0
        if self.v == {}:
          for l in range(1, L + 1):
            self.v["W"+str(l)] = 0
            self.v["b"+str(l)] = 0
        self.t += 1

        for l in range(1, L + 1):
          gradients["dW" + str(l)]= gradients["dW" + str(l)]+self.wd*parameters["W" + str(l)]
          gradients["db" + str(l)] = gradients["db" + str(l)]+self.wd*parameters["b" + str(l)]
            
          self.m["W" + str(l)] = self.beta1 * self.m["W" + str(l)] + (1 - self.beta1) * (gradients["dW" + str(l)]-self.wd*parameters["W" + str(l)])
          self.v["W" + str(l)] = self.beta2 * self.v["W" + str(l)] + (1 - self.beta2) * ((gradients["dW" + str(l)]-self.wd*parameters["W" + str(l)])**2)
          m_hat = self.m["W" + str(l)] / (1 - self.beta1 ** self.t)
          v_hat = self.v["W" + str(l)] / (1 - self.beta2 ** self.t)
          m_hat_fin = (self.beta1*m_hat)+(1-self.beta1)*gradients["dW" + str(l)]/(1-(self.beta1)**self.t)
          parameters["W" + str(l)] = (1-self.wd)*parameters["W" + str(l)] - self.lr * (m_hat_fin) / (np.sqrt(v_hat) + self.eps)

          self.m["b" + str(l)] = self.beta1 * self.m["b" + str(l)] + (1 - self.beta1) * (gradients["db" + str(l)]-self.wd*parameters["b" + str(l)])
          self.v["b" + str(l)] = self.beta2 * self.v["b" + str(l)] + (1 - self.beta2) * ((gradients["db" + str(l)]-self.wd*parameters["b" + str(l)])**2)
          m_hat = self.m["b" + str(l)] / (1 - self.beta1 ** self.t)
          v_hat = self.v["b" + str(l)] / (1 - self.beta2 ** self.t)
          m_hat_fin = (self.beta1*m_hat)+(1-self.beta1)*gradients["db" + str(l)]/(1-(self.beta1)**self.t)
          parameters["b" + str(l)] = (1-self.wd)*parameters["b" + str(l)] - self.lr * (m_hat_fin) / (np.sqrt(v_hat) + self.eps)
        
        
        return parameters


"""###**Neural Network Class**

Implentation of the class using all the functions generated above. This class has 2 methods. 
####Class Attributes

While initisaling the network, this is the convention I followed. Network Class takes a dictionary N as input which has information of the input size, size of hidden layers, output size, activation function at each layer, weight initialisation.


```
> N = {'n' : [784,[64,64,64],10],
          'activation_func' : ['sigmoid','sigmoid','sigmoid','softmax'],
          'initialisation' : 'Random'
      }
```

####Methods of the class


1.  **Train:**
 
 Fits the model to the given dataset. Takes Train Dataset, Validation Data set, Batch size, Optimiser, Number of epochs as it's input arguments and fits model's parameters to the train data.

 
2.   **Test:**

        Tests the model on Test dataset given as input attributes to this method with the existing parameters.


"""

class Network():
  def __init__(self, N, log):
    self.n = N['n']
    self.parameters = initialize_parameters(self.n[0],self.n[1],self.n[2], N['initialisation'])
    self.activation_func = N['activation_func']
    self.l = len(self.parameters)//2
    self.log = log

  def train(self,X,y,X_val,y_val,loss, batch_size, optimiser, epochs):
    m = X.shape[1]
    count = 0
    while(count < epochs):
      count = count+1
      training_loss = 0
      training_score = 0
      for i in np.arange(start=0, stop=X.shape[1], step=batch_size):
        batch_count = batch_size
        if i + batch_size > X.shape[1]:
          batch_count = X.shape[1] - i + 1
        layer_wise_outputs = ForwardPropagation(X[:,i:i+batch_size], self.parameters, self.activation_func)  
        gradients = BackPropagate(self.parameters, layer_wise_outputs, X[:,i:i+batch_size], y[:,i:i+batch_size], self.activation_func, loss)
        self.parameters = optimiser.update(self.parameters, gradients)
        training_loss = training_loss + cost(y[:,i:i+batch_size], layer_wise_outputs['a'+str(self.l)],loss)
        training_score = training_score+ accuracy_score(softmax_to_label(y[:,i:i+batch_size]),  layer_wise_outputs['a'+str(self.l)])
        
      training_loss_fin = training_loss/np.ceil(m/batch_size)
      print("Epoch:"+str(count))
      print("Training Loss after "+ str(count) +"th epoch =" +str(training_loss_fin))

      training_score_fin = training_score/np.ceil(m/batch_size)
      print("Training score after "+ str(count) +"th epoch =" + str(100*training_score_fin))

      validation_outputs = ForwardPropagation(X_val, self.parameters, self.activation_func)
      validation_loss = cost(y_val, validation_outputs['a'+str(self.l)],loss)
      print("Validation Loss after "+ str(count) +"th epoch =" + str(validation_loss))
      validation_score = accuracy_score(softmax_to_label(y_val), validation_outputs['a'+str(self.l)])
      print("Validation Score after "+ str(count) +"th epoch =" + str(100*validation_score))
      if(self.log == True):
        metrics = {"train loss": training_loss_fin, "train score": training_score_fin , "val loss": validation_loss, "accuracy": validation_score}
        wandb.log(metrics)

    
  def test(self,X,Y):
    test_outputs = ForwardPropagation(X, self.parameters, self.activation_func)
    print(f"Test Accuracy = {100*accuracy_score(Y, test_outputs['a'+str(self.l)])} %")
    if(self.log == True):
      wandb.log({'Test Accuracy' : accuracy_score(Y, test_outputs['a'+str(self.l)])})


  """###**Helper Functions -2** """

def build_optimiser(optimiser_str,learning_rate,weight_decay):
  if optimiser_str == 'sgd':
    optimiser = SGD(lr = learning_rate, weight_decay=weight_decay)
  elif optimiser_str == 'momentum':
    optimiser = Momentum(lr = learning_rate, beta = 0.9, weight_decay=weight_decay)
  elif optimiser_str == 'nesterov':
    optimiser = Nesterov(lr = learning_rate, gamma = 0.9, weight_decay=weight_decay)
  elif optimiser_str == 'rmsprop':
    optimiser = RMSprop(lr = learning_rate, decay_rate = 0.1,eps = 1e-6, weight_decay=weight_decay)
  elif optimiser_str == 'adam':
    optimiser = Adam(lr = learning_rate, beta1 = 0.9, beta2 = 0.99 ,eps = 1e-6, weight_decay=weight_decay)
  elif optimiser_str == 'nadam':
    optimiser = Nadam(lr = learning_rate, beta1 = 0.9, beta2 = 0.99 ,eps = 1e-6, weight_decay=weight_decay)
  return optimiser





