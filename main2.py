import numpy as np 
from HW1 import ForwardStagewise 
from sklearn import datasets,linear_model 
import matplotlib.pyplot as plt 


diabetes = datasets.load_diabetes() 
tester = ForwardStagewise()
tester.fit(np.copy(diabetes.data),np.copy(diabetes.target),cannot_link=[[2,3]],epsilon=.001,max_iter=200)
(intercept,path) = tester.get_coef_path() 
print(path.shape)
print(path[-1,:])
print(tester.coef)
for i in range(path.shape[1]):
    plt.plot(path[:,i])
plt.show()