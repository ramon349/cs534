import numpy as np 
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from matplotlib import pyplot 

# Load the diabetes dataset
diabetes = datasets.load_diabetes()



##my impleentation 
#jtester = Ridge() 
ridge_model =linear_model.Ridge(fit_intercept= True, alpha=0,normalize=False)
ridge_model.fit(np.copy(diabetes.data ),np.copy(diabetes.target))
reg_coef,reg_inter = ridge_model.coef_ ,ridge_model.intercept_
X = np.copy(diabetes.data)
X_mu = np.mean(X,axis=0)
X_std = np.std(X,axis=0)
y_std = np.std(diabetes.target) 
y_mu = np.mean(diabetes.target)
y = (diabetes.target - y_mu)/y_std
X =  (X - X_mu )/X_std
ridge_model =linear_model.Ridge(fit_intercept= True, alpha=0,normalize=False)
ridge_model.fit(X,y)
(std_coef,std_inter) = (ridge_model.coef_,ridge_model.intercept_)
print("----Normal Ridge coefficients -----")
print( reg_coef)  
print("----- Standardzied Coefficients----")
print(std_coef)
print("----- Conversion of Coefficients ---")
print( np.multiply(np.true_divide(X_std,y_std ),reg_coef) )
