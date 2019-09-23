import numpy as np 
from HW1 import Ridge
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from matplotlib import pyplot 
# Load the diabetes dataset
diabetes = datasets.load_diabetes()
##my impleentation 
tester = Ridge() 
tester.fit(np.copy(diabetes.data),np.copy(diabetes.target),lmbd =1,coef_prior=None )
other =linear_model.Ridge(fit_intercept=True, alpha=.001,normalize=False) 
other.fit(diabetes.data,np.copy(diabetes.target))
(inter,coeff) = tester.get_coef() 
print("----Sklearn Ridge coefficients -----")
print( other.coef_)  
print(other.intercept_)
print("-----Ramon's Coefficients----")
print(coeff)
print(inter)
pyplot.plot(other.coef_,marker='*') 

pyplot.plot(coeff,marker='*')
pyplot.legend(["SklearnCoeff","RamonCoeff"])
pyplot.show(block=True)