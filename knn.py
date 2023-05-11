import numpy as np


class KNN(object):

    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors
        self.X_train = None
        self.Y_train = None

    def fit(self, X, Y):
        self.X_train = X
        self.Y_train = Y

    def euclidean_distance(self, point):
        return np.sqrt(np.sum((point - self.X_train)**2, axis=1))

    def bigger_neighbors_class(self, neighbors):
        legitimo = 0
        # print("neighbors type",type(neighbors))
        for indice, fila in neighbors.items():
            if self.Y_train[indice] == 1:
                legitimo += 1

        phishing = self.n_neighbors - legitimo

        if legitimo > phishing:
            return 1
        else:
            return -1

    def predict(self, X):
        prediction = []
        # print("X type",type(X))

        for indice, fila in X.iterrows():
            a = self.euclidean_distance(fila)
            vecinos = a.nsmallest(self.n_neighbors, keep='first')
            prediction.append(self.bigger_neighbors_class(vecinos))

        return prediction

    def accuracy_score(self, y_test, predictions):
        return np.sum(y_test == predictions) / len(y_test)
