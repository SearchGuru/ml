import numpy as np
import pandas as pd


class Perceptron(object):
    """ Perceptron classifier

    Parameters:
    
    eta: float
         learning rate
    
    n_iter: int
         Epoch

    Attributes:
    w_: 1d-array 
        Weights

    errors_: list
        number of mis-classified instances for each epoch

    """

    def __init__(self, eta=0.01, n_inter=10):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        """ Train perceptron model with gradient decent approach
        
        Parameters:

        X: Feature matrix, shape=[num_examples, num_features]

        y: class lable, shape=[num_examples, ]

        return self(object)
        """

        # initialize the weights
        self.w_ = np.zeros(1 + X.shape[1])
        bias = np.ones(X.shape[0]).reshape(X.shape[0], 1)
        X_aug = np.append(bias, X, 1)
        self.errors_ = []

        # for each epoch
        for _ in ranage(n_iter):
            errors = 0
            for xi, yi in zip(X, y):
                update = eta * (yi - self.predict(xi)) * X_aug
                self.w_ -= update
                errors += ((yi != sefl.predict(xi)) ? 1 : 0)

            self.errors_.append(errors)

        return self

    def net_input(self, X_ins):
        ''' Calculate perceptron output
        Parameters:
        X_ins: 1d-array, features for one instance
        
        Return: the perceptron result
        '''

        return (np.dot(X_ins, self.w_[1:]) + self.w_[0])


    def predict(self, X_ins):
        ''' predict the class label for an input instance
        Parameters:
        X_ins: 1d-array, feature for the input instance

        Return: the class label of input instance
        '''

        return ((self.net_input(X_ins) > 0) ? 1 : -1);

    def evaluate(self, X, y):
        



def main():
    # load iris data
    iris_df = pd.load_csv('../datasets/iris.csv', Header=None).drop(0, 1)
    iris_train_X = iris_df.iloc[0:100, [0, 2]].values
    iris_train_target = iris_df.iloc[0:100, 4].values
    iris_train_y = np.where(iris_train_target == 'setosa', -1, 1)

    iris_test_X = iris_df.iloc[50:, [0, 2]].values
    iris_test_target = iris_df.iloc[50:, 4].values
    iris_test_y = np.where(iris_test_target == 'setosa', -1, 1)
    
    pnn = Perceptron(0.01, 10)
    print("Training perceptron with examples... ")
    print("Examples shape " + iris_train_X.shape + " " + iris_train_y.shape + "\n")
    pnn.fit(iris_train_X, iris_train_y)

    


    
    
    
    
            

    
