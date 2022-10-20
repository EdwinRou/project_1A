import numpy as np
import random
from tqdm import tqdm #sert à visualiser l'avancement des calculs lorsqu'on lance le programme
import matplotlib.pyplot as plt


# Quelques fonctions utiles pour la suite :
def vectorized_result(j):
    """Return a 10-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere.  This is used to convert a digit
    (0...9) into a corresponding desired output from the neural
    network."""
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e

def sigmoid(z):
  return 1/(1+np.exp(-z)) # ATTENTION peut valoir 0 ou 1 dans python

def sigmoid_prime(z): #dérivé de la fonction sigmoid
  return sigmoid(z)*(1-sigmoid(z))


# Classe pour les fonctions coûts :
class QuadraticCost(object):
    @staticmethod
    def cost(a, y):
      if type(y) == np.int64 :
        y = vectorized_result(y)
      return 0.5*np.linalg.norm(a-y)**2
    @staticmethod
    def delta(z, a, y):
        return (a-y) * sigmoid_prime(z)


class CrossEntropyCost(object):

    @staticmethod
    def cost(a, y):
      if type(y) == np.int64 :
        y = vectorized_result(y)
      return np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))

    @staticmethod
    def delta(z, a, y):
        return a - y


# Creation du Réseau de neuronne avec une classe :
class Network(object):

    def __init__(self, sizes,  Cost = CrossEntropyCost): #sizes = [784,_,10] = [taille_data, taille_layer_1, output_layer]
        self.num_layers = len(sizes)-1
        self.sizes = sizes
        self.cost = Cost
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)/np.sqrt(x) for x, y in zip(sizes[:-1], sizes[1:])]
        self.train_cost = []
        self.test_cost = []
        self.acc = []
        #initialize biases and weights randomly


    def feedforward(self, value):
        for i in range(self.num_layers):
           value = sigmoid(np.dot(self.weights[i],value) + self.biases[i])
        return value

    # Test unitaire :

    def test_feedforward(self): #vérifie que feedforward() renvoie bien le type de valeur souhaité
      data = np.ones((784,1))
      result = self.feedforward(data)
      return np.shape(result) == (10,1) and np.max(result) <= 1 and np.min(result)>=0


    def accuracy(self, test_data):
      total = 10000
      sucess = 0
      cost = 0
      for (value,number) in test_data :
        a = self.feedforward(value)
        if np.argmax(a) == number :
          sucess +=1
        cost += (self.cost).cost(a,number)
      return sucess/total, cost/total


    def SGD(self,training_data, nb_training, batch_size, eta, test_data = None):
      training_data = list(training_data)
      n = len(training_data)
      self.test_cost.append(batch_size) # store value of batch_size for progression()
      self.test_cost.append(n) # store value of len(training_data) for progression()
      for j in tqdm(range(nb_training)):
          random.shuffle(training_data)
          if n%batch_size == 0 :
              batches = [training_data[i:i + batch_size] for i in range(0,n-batch_size+1, batch_size)]
          else :
              batches = [training_data[i:i + batch_size] for i in range(0,n-batch_size+1, batch_size)] + [training_data[n-n%batch_size:]]
          for batch in (batches):
              self.update_batch(batch, eta)
          if test_data :
              test_data = list(test_data)
              acc , mean_cost = self.accuracy(test_data)
              self.test_cost.append(mean_cost)
              self.acc.append(acc)
              print("Accuracy de {}% après {} entraînement(s)".format(100*acc, j+1))

    def update_batch(self, batch, eta):

        m = len(batch)
        sum_dw = [np.zeros((y, x)) for x, y in zip(self.sizes[:-1], self.sizes[1:])]
        sum_db = [np.zeros((y, 1)) for y in self.sizes[1:]]
        batch_cost = 0
        for value,number in batch :
            list_dw, list_db, cost = self.backprop(value, number)
            sum_dw = [w + dw for w,dw in zip(sum_dw,list_dw)]
            sum_db = [b + db for b,db in zip(sum_db,list_db)]
            batch_cost += cost

        self.weights = [w-(eta/m)*dw for w,dw in zip(self.weights,sum_dw)]
        self.biases = [b-(eta/m)*db for b,db in zip(self.biases,sum_db)]
        self.train_cost.append(batch_cost/m)

    def backprop(self, value, number):

      z = [np.zeros((y, 1)) for y in self.sizes[1:]]
      db = [np.zeros((y, 1)) for y in self.sizes[1:]]
      dw = [np.zeros((y, x)) for x, y in zip(self.sizes[:-1], self.sizes[1:])]
      output = value
      a = [value]

      for i in range(self.num_layers):
        z[i] = np.dot(self.weights[i],output) + self.biases[i]
        output = sigmoid(z[i])
        a.append(output)
      cost = (self.cost).cost(output, number)
      delta = (self.cost).delta(z[-1], output, number)
      db[-1] = delta
      dw[-1] = np.outer(delta,a[-2])

      for j in range(2,self.num_layers+1):
        delta = np.dot(np.transpose(self.weights[-j+1]),delta) * sigmoid_prime(z[-j])
        db[-j] = delta
        dw[-j] = np.outer(delta,a[-j-1])
      return (dw,db,cost)

    def progression(self):
        batch_size, nb_data = self.test_cost.pop(0), self.test_cost.pop(0)
        train_cost_by_training = self.train_cost[::nb_data//batch_size]
        plt.subplot(1, 2, 1)
        plt.xlabel("Nb of trainings")
        plt.plot(self.test_cost, label = 'test_cost')
        plt.plot(train_cost_by_training, label = 'train_cost')
        plt.legend()
        plt.title("Costs over trainings")
        plt.subplot(1, 2, 2)
        plt.xlabel("Nb of trainings")
        plt.plot(self.acc)
        plt.title("Accuracy")
        plt.suptitle("Performance")
        plt.show()
