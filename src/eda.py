import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def eda(data):

    print(data.describe())


    sns.pairplot(data)
    plt.show()


    data_encoded = pd.get_dummies(data, drop_first=True)


    plt.figure(figsize=(10, 6))
    sns.heatmap(data_encoded.corr(), annot=True)
    plt.show()
