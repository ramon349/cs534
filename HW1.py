# Please do not use other libraries except for numpy
import numpy as np
class Ridge:

    def __init__(self):
        self.intercept = 0
        self.coef = None
    def fit(self, X, y, coef_prior=None, lmbd=1.0):
        n, m = X.shape
        if coef_prior == None :
            coef_prior = np.zeros((m,))
        # a) normalize X
        x_mu =  np.mean(X,axis=0) #reminder that axis 0 is column wise mean 
        x_sigma = np.std(X,axis=0)
        X = (X-x_mu)/x_sigma 
        #normalize y   
        y_mu = np.mean(y) 
        y_sigma = np.std(y) 
        y = (y - y_mu) / y_sigma
        self.coef = np.zeros(m) 
        # b) adjust coef_prior according to the normalization parameters
        coef_prior = np.multiply(np.true_divide(x_sigma,y_sigma), coef_prior)
        # c) get coefficients
        A = np.linalg.inv(np.dot(X.T,X) - lmbd)
        B = np.dot(X.T,y) - lmbd*coef_prior
        out = np.dot(A,B)
        # d) adjust coefficients for de-normalized X
        self.intercept = y_mu
        self.coef =   np.multiply(np.true_divide(y_sigma,x_sigma),out)
    def get_coef(self): 
        print(self.intercept)
        return (self.intercept, self.coef)

class ForwardStagewise:

    def __init__(self):
        self.intercept = 0
        self.path = []
        self.excluded = set()  
        self.cannot_link = []
    def fit(self, X, y, cannot_link=[], epsilon=1e-5, max_iter=1000):
        self.cannot_link= cannot_link
        self.excluded = set()
        # a) normalize X
        x_mu  = np.mean(X,axis=0)
        x_sig =  np.std(X,axis=0) 
        X =  (X - x_mu)/x_sig
        y =  y- np.mean(y) 
        n,m = X.shape
        # b-1) implement incremental forwward-stagewise
        nsteps = 100000
        delta = .0001
        beta = np.zeros(m)
        r = y
        for s in range(nsteps):
            corr_best,j_best, gamma_best =  0,0, 0
            for j in range(m):
                xj_norm = np.linalg.norm(X[:,j],2) 
                r_norm = np.linalg.norm(r,2) 
                corr_j = np.dot(X[:,j], r)/(xj_norm*r_norm) 
                if corr_best< corr_j  and not self.is_excluded(j):
                   j_best,corr_best = j,corr_j 
                   self.update_exclusion(j)
            beta[j_best] += delta*np.sign(np.dot(X[:,j],r)) 
            r -= beta[j_best]*X[:,j_best]
        print(beta)
        # b-2) implement cannot-link constraints

        # c) adjust coefficients for de-normalized X

        # d) construct the "path" numpy array
        #     path: l-by-m array,
        #               where l is the total number of iterations
        #               m is the number of features in X.
        #               The first row, path[0,:], should be all zeros.

        return 0 
    def update_exclusion(self,selected_var):
        if self.cannot_link: 
            for link_group in self.cannot_link:
                if selected_var in link_group: 
                    set_diff = set(link_group) -set([selected_var]) 
                    [self.excluded.add(e) for e in set_diff]
                    print(self.excluded)

    def is_excluded(self,idx): 
        return idx in self.excluded
    def get_coef_path(self):
        return self.intercept, self.path