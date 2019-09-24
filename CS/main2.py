import numpy as np 
from HW1 import ForwardStagewise 
from sklearn import datasets,linear_model 
import matplotlib.pyplot as plt 

""""
Note: for thsi problem i wasn't sure if we had to follow the code in the lecture sldies or the 
pseudocode described in the book. I therefore incldued a slight modification where you can run
the lecturle slide version refered to  as alg=1 (default value ) or you can run the incremental 
stagewise found in the book refered to as alg=2  
"""

diabetes = datasets.load_boston() 
tester = ForwardStagewise()
tester.fit(np.copy(diabetes.data),np.copy(diabetes.target),cannot_link=[],epsilon=.01,max_iter=2000)
#tester.fit(np.copy(diabetes.data),np.copy(diabetes.target),cannot_link=[],epsilon=.1,max_iter=20000,alg=2)
(intercept,path) = tester.get_coef_path() 
for i in range(path.shape[1]): 
    plt.plot(path[:,i])
plt.show()