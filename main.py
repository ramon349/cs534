import numpy as np 
from HW1 import Ridge
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from matplotlib import pyplot 

# Load the diabetes dataset
diabetes = datasets.load_diabetes()



tester = Ridge()
tester.fit(diabetes.data,diabetes.target,coef_prior=None )
other =linear_model.Ridge(fit_intercept= True, alpha=1,normalize=False)
X = (diabetes.data - np.mean(diabetes.data,axis=0) )/np.std(diabetes.data,axis=0) 
other.fit(X,diabetes.target)
(inter,coeff) = tester.get_coef() 
print("----Sklearn Ridge coefficients -----")
print( other.coef_) 
print("-----Ramon's Coefficients----")
print(coeff)

pyplot.plot(other.coef_) 

pyplot.plot(coeff)
pyplot.legend(["SklearnCoeff","RamonCoeff"])
pyplot.show()