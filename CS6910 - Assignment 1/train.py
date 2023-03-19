import numpy as np
from keras.datasets import fashion_mnist, mnist
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import wandb
import a1
import sys, getopt

def build_optimiser_train(optimiser_str,learning_rate,weight_decay,momentum,beta,beta1,beta2,epsilon):
  if optimiser_str == 'sgd':
    optimiser = a1.SGD(lr = learning_rate, weight_decay=weight_decay)
  elif optimiser_str == 'momentum':
    optimiser = a1.Momentum(lr = learning_rate, beta = momentum, weight_decay=weight_decay)
  elif optimiser_str == 'nag':
    optimiser = a1.Nesterov(lr = learning_rate, gamma = momentum, weight_decay=weight_decay)
  elif optimiser_str == 'rmsprop':
    optimiser = a1.RMSprop(lr = learning_rate, decay_rate = beta,eps = epsilon, weight_decay=weight_decay)
  elif optimiser_str == 'adam':
    optimiser = a1.Adam(lr = learning_rate, beta1 = beta1, beta2 = beta2 ,eps = epsilon, weight_decay=weight_decay)
  elif optimiser_str == 'nadam':
    optimiser = a1.Nadam(lr = learning_rate, beta1 = beta1, beta2 = beta2 ,eps = epsilon, weight_decay=weight_decay)
  return optimiser

def main(argv):
    myprojectname = 'A1'
    myname = 'Aditya Nanda Kishore'
    epochs = 10
    batch_size = 32
    loss = 'CE'

    num_layers = 3
    hidden_size = 64
    activation_func = 'leakyRelu'
    optimiser_str = 'nadam'
    learning_rate = 1e-3
    momentum = 0.9
    beta = 0.9
    beta1 = 0.9
    beta2 = 0.99
    epsilon = 1e-6

    dataset = 'fashion_mnist'

    weight_decay = 0
    weight_initialisation = 'Xavier'

    argv = sys.argv[1:]
    opts, args = getopt.getopt(argv, "h:f:g:d:e:b:l:o:r:m:i:j:k:p:y:z:n:s:a:" ,
                               ["wandb_project=","wandb_entity=","dataset=","epochs=","batch_size=",
                                "loss=","optimiser=","learning_rate=", "momentum=", "beta=", "beta1=", "beta2=", "epsilon=",
                                "weight_decay=","weight_init=", "num_layers=", "hidden_size=", "activation=" ])

    for opt, arg in opts:
        if opt == '-h':
            print ('train.py --wandb_project <myprojectname> --wandb_entity <myname> -d <dataset> -e <epochs> -b <batch_size> -l <loss> -o <optimiser> -r <learning rate> -m <momentum> -i <beta> -j <beta1> -k <beta2> -p <epsilon> -y <weight_decay> -z <weight_init> -n <num_layers> -s <hidden_size> -a <activation>')
            sys.exit()
        elif opt in ("-f", "--wandb_project"):
            myprojectname = arg
        elif opt in ("-g", "--wandb_entity"):
            myname = arg
        elif opt in ("-d", "--dataset"):
            dataset = arg
        elif opt in ("-e", "--epochs"):
            epochs = int(arg)
        elif opt in ("-b", "--batch_size"):
            batch_size = int(arg)
        elif opt in ("-l", "--loss"):
            loss = arg
        elif opt in ("-o", "--optimiser"):
            optimiser_str = arg   
        elif opt in ("-r", "--learning_rate"):
            learning_rate = float(arg)
        elif opt in ("-m", "--momentum"):
            momentum = float(arg) 
        elif opt in ("-i", "--beta"):
            beta = float(arg)
        elif opt in ("-j", "--beta1"):
            beta1 = float(arg)
        elif opt in ("-k", "--beta2"):
            beta2 = float(arg)
        elif opt in ("-p", "--epsilon"):
            beta2 = float(arg)
        elif opt in ("-y", "--weight_decay"):
            weight_decay = float(arg)
        elif opt in ("-z", "--weight_init"):
            weight_initialisation = arg
        elif opt in ("-n", "--num_layers"):
            num_layers = int(arg)
        elif opt in ("-s", "--hidden_size"):
            hidden_size = int(arg)
        elif opt in ("-a", "--activation"):
            activation_func = arg

    wandb.login()
    wandb.init(project = myprojectname+ '_'+ myname)
    

    if (dataset == 'fashion_mnist'):
        (X, y), (X_test, y_test) = fashion_mnist.load_data()
    elif (dataset == 'mnist'):
        (X, y), (X_test, y_test) = mnist.load_data()

    X = X.reshape(X.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)


    X = X/255
    X_test = X_test/255
    y = a1.one_hot_encode(y)


    X = X.T
    X_test = X_test.T
    y = y.T
    X_train, X_val, y_train,y_val  = train_test_split(X.T, y.T, test_size=0.1)
    X_train, X_val, y_train,y_val  = X_train.T, X_val.T, y_train.T,y_val.T 

    N = {'n' : [784,[hidden_size]*num_layers,10],
         'activation_func' : [activation_func]*num_layers+['softmax'],
         'initialisation' : weight_initialisation
        }


    if (loss == "mean_squared_error"):
        loss = 'MSE'
    elif (loss == "cross_entropy"):
        loss = 'CE'
    
    optimiser = build_optimiser_train(optimiser_str=optimiser_str,learning_rate=learning_rate,weight_decay=weight_decay,momentum=momentum,beta=beta,beta1=beta1,beta2=beta2,epsilon=epsilon)

    network = a1.Network(N,True)
    network.train(X_train , y_train, X_val, y_val, loss, batch_size, optimiser, epochs)
    network.test(X_test, y_test)

if __name__ == "__main__":
   main(sys.argv[1:])