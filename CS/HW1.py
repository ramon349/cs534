# Please do not use other libraries except for numpy
import numpy as np
import matplotlib.pyplot as plt
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
        X = np.true_divide( X-x_mu,x_sigma )
        #normalize y   
        y_mu = np.mean(y) 
        y_sigma = np.std(y) 
        y = np.true_divide(y - y_mu,y_sigma)
        self.coef = np.zeros(m) 
        # b) adjust coef_prior according to the normalization parameters
        coef_prior = np.multiply(np.true_divide(x_sigma,y_sigma), coef_prior)
        # c) get coefficients
        A = np.linalg.inv(np.dot(X.T,X)  + lmbd*np.identity(m) )
        B = np.dot(X.T,y) + lmbd*coef_prior
        out = np.dot(A,B)
        # d) adjust coefficients for de-normalized X 
        self.intercept = y_mu
        denormalize_factor = np.true_divide(y_sigma,x_sigma)
        self.coef =   np.multiply(denormalize_factor,out) 
        """"
        This implementation does not penalize the intercept. 
        Since data is centered we calculate coefficients using centered data 
        Then calculate the intercept using the mean y value 
        Reference:  Hastie,Trevorl et al; Elements of Statistical Learning; Page 64 

        """
    def get_coef(self): 
        print(self.intercept)
        return (self.intercept, self.coef)

class ForwardStagewise:

    def __init__(self):
        self.intercept = 0
        self.path = []
        self.excluded = set()  
        self.cannot_link = []
    def fit(self, X, y, cannot_link=[], epsilon=1e-5, max_iter=1000,alg=1):
        self.cannot_link= cannot_link
        self.excluded = set()
        # a) normalize X 
        x_mu  = np.mean(X,axis=0)
        x_sig =  np.std(X,axis=0) 
        y_mu = np.mean(y)
        y_sig = np.std(y)
        if alg ==1:
            X =  (X - x_mu )/x_sig
            y = y - y_mu
        else:
            X = X - x_mu 
            x_lnorm = np.zeros(x_sig.shape)
            for i in range (0,X.shape[1]): 
                x_lnorm[i] = np.linalg.norm(X[:,i])
                X[:,i] =  X[:,i] /x_lnorm[i]
            y = y - y_mu 
            y_sig =1 
            x_sig =1
        
        # b-1) implement incremental foryward-stagewise
        # b-2) implement cannot-link constraints
        # d) construct the "path" numpy array 
        if alg ==1 :
            (beta,self.path) =  self.ForwardStagewise(X,y,max_iter,epsilon)
        else: 
            (beta,self.path) =  self.incremental_ForwardStagewise(X,y,max_iter,epsilon)
        std_factor = np.true_divide(y_sig,x_sig) 
        # c) adjust coefficients for de-normalized X
        for i in range(self.path.shape[0]):
            self.path[i,:] = np.multiply(self.path[i,:],std_factor)
            #rows are taken back to original space
        self.intercept = y_mu
        self.coef =   np.multiply(std_factor,beta)

    def ForwardStagewise(self,X,y,nsteps,delta):
        n,m = X.shape 
        path = np.zeros((nsteps+1,m))
        beta = np.zeros(m)
        for s in range(1,nsteps+1):
            r = y - np.dot(X, beta)
            mse_min, j_best, gamma_best = np.inf, 0, 0
            for j in range(m):
                gamma_j = np.dot(X[:,j], r)/np.dot(X[:,j], X[:,j])
                mse = np.mean(np.square(r - gamma_j * X[:,j]))
                if mse < mse_min and not self.is_excluded(j):
                    mse_min, j_best, gamma_best = mse, j, gamma_j
            if np.abs(gamma_best) > 0:
                beta[j_best] += gamma_best * delta
                self.update_exclusion(j_best)
            path[s,:] =  beta
        return (beta,path)
    def incremental_ForwardStagewise(self,X,y,nsteps,epsilon):
        """ 
        Implementation of incremental forward stagewise regression 
        Code structure taken from lecture sldies and class book. 
        Reference:  Hastie,Trevorl et al; Elements of Statistical Learning; Page 64 
        """
        r = y 
        n,m =  X.shape
        beta = np.zeros(m)
        path =  np.zeros((nsteps+1,m))
        for s in range(1,nsteps+1):
            corr_best,j_best =  -1,0
            for j in range(m):
                xj_norm = np.linalg.norm(X[:,j]) 
                r_norm = np.linalg.norm(r) 
                corr_j = np.dot(X[:,j], r)/(xj_norm*r_norm) #calculating correlation of j
                if corr_j > corr_best  and not self.is_excluded(j):
                   j_best,corr_best= j,corr_j
            delta_j = epsilon*np.sign(np.dot(X[:,j_best],r)) 
            beta[j_best] +=  delta_j
            path[s,:] =  beta
            r  = r - delta_j*X[:,j_best]
            self.update_exclusion(j_best) 
        return (beta,path)

    def update_exclusion(self,selected_var):
        """ check cannot link set to update excluded variable set
       input: 
            selected_var: Variable index found to be most useful 
        
        """
        sel_var = set([selected_var]) #selected variable is turned into a single
        for link_group in self.cannot_link:
            if selected_var in link_group: 
                set_diff = set(link_group) - sel_var #variables to exclude using set diff 
                [self.excluded.add(e) for e in set_diff] #add each variable to exclusion list

    def is_excluded(self,idx):
        """ Check set of exlcuded variables for membership
        input: 
            idx: index of variable we are interested in adding
        """
        return idx in self.excluded

    def get_coef_path(self):
        return self.intercept, self.path