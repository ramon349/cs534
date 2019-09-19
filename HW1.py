# Please do not use other libraries except for numpy
import numpy as np
class Ridge:

    def __init__(self):
        self.intercept = 0
        self.coef = None

    def fit(self, X, y, coef_prior=None, lmbd=1.0):

        n, m = X.shape
        self.coef = np.zeros(m)
        if coef_prior:
            print(f"coef_prior was evaluated as falsy")
            coef_prior = np.zeros(m)

        # a) normalize X
        x_mu =  np.mean(X,axis=0) #reminder that axis 0 is column wise mean 
        x_sigma = np.std(X,axis=0) 
        X = (X-x_mu)/x_sigma 
        print(f"Column wise mean: {np.mean(X,axis=0 )} ") 
        print(f"Column wise std: {np.std(X,axis=0)}")

        # b) adjust coef_prior according to the normalization parameters
        coef_prior = None

        # c) get coefficients
        ...
        self.intercept = None 
        self.coef = None

        # d) adjust coefficients for de-normalized X
        self.intercept = None
        self.coef = None

        return 0

    def get_coef(self):
        return self.intercept, self.coef

class ForwardStagewise:

    def __init__(self):
        self.intercept = 0
        self.path = []

    def fit(self, X, y, cannot_link=[], epsilon=1e-5, max_iter=1000):

        # a) normalize X

        # b-1) implement incremental forwward-stagewise
        # b-2) implement cannot-link constraints

        # c) adjust coefficients for de-normalized X

        # d) construct the "path" numpy array
        #     path: l-by-m array,
        #               where l is the total number of iterations
        #               m is the number of features in X.
        #               The first row, path[0,:], should be all zeros.

        return 0

    def get_coef_path(self):
        return self.intercept, self.path