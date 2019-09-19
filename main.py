import numpy as np 
from HW1 import Ridge
import csv 


data = np.loadtxt(open("Boston.csv", "r"), delimiter=",", skiprows=1,usecols=range(1,13))

tester = Ridge()
tester.fit(data,np.zeros((data.shape[0],1)),coef_prior=np.array([]) )