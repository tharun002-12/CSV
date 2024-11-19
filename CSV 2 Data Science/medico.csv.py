import numpy as np 
import pandas as pd 
from urllib.request import urlopen 
from pgmpy.models import BayesianModel 
from pgmpy.estimators import MaximumLikelihoodEstimator 
from pgmpy.inference import VariableElimination 
Cleveland_data_URL = "heart.csv" 
names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak','slope', 'ca', 
'thal', 'heartdisease'] 
heartDisease = pd.read_csv(Cleveland_data_URL, names=names)
heartDisease = heartDisease.replace('?', np.nan) 
# Display the data 
print("Few examples from the dataset are given below") 
print(heartDisease.head()) 
model = BayesianModel([('age', 'trestbps'), ('age', 'fbs'), ('sex', 'trestbps'), ('sex', 'trestbps'), 
('exang', 'trestbps'), ('trestbps', 'heartdisease'), ('fbs', 'heartdisease'), ('heartdisease', 'restecg'), 
('heartdisease', 'thalach'), ('heartdisease', 'chol')]) 
# Learning CPDs using Maximum Likelihood Estimators 
model.fit(heartDisease, estimator=MaximumLikelihoodEstimator) 
HeartDisease_infer = VariableElimination(model) 
# Computing the probability of heartdisease given age=28 
q = HeartDisease_infer.query(variables=['heartdisease'], evidence={'age': 28}) 
print(q['heartdisease']) 
