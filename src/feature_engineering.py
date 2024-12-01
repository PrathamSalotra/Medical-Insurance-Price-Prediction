import pandas as pd
from sklearn.preprocessing import StandardScaler

def feature_engineering(data):
    scaler = StandardScaler()
    numerical_features = ['age', 'bmi', 'children']
    data[numerical_features] = scaler.fit_transform(data[numerical_features])
    return data
