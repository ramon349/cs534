import numpy as np 
from HW1 import ForwardStagewise 
from sklearn import datasets,linear_model 

diabetes = datasets.load_diabetes() 
tester = ForwardStagewise()
tester.fit(np.copy(diabetes.data),np.copy(diabetes.target),cannot_link=[[2,3]])