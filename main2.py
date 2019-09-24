import numpy as np 
from HW1 import ForwardStagewise 
from sklearn import datasets,linear_model 
import matplotlib.pyplot as plt 


diabetes = datasets.load_diabetes() 
tester = ForwardStagewise()
#tester.fit(np.copy(diabetes.data),np.copy(diabetes.target),cannot_link=[],epsilon=.01,max_iter=2000,alg=1)
tester.fit(np.copy(diabetes.data),np.copy(diabetes.target),cannot_link=[],epsilon=.1,max_iter=20000,alg=2)
(intercept,path) = tester.get_coef_path() 
for i in range(path.shape[1]): 
    plt.plot(path[:,i])
plt.show()