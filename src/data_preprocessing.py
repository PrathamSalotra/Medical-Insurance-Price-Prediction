import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

def preprocess_data(data):

    data = data.dropna()

    data = pd.get_dummies(data, columns=['sex', 'smoker', 'region'], drop_first=True)
    
    return data

def split_data(data):
    X = data.drop('charges', axis=1)
    y = data['charges']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test
