import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display


class Athelete:
        
    df = pd.read_csv('/home/pop/Desktop/python/MachineLearning/LinearRegression/MedalPrediction/atheletes.csv')
    display(df)

    @classmethod
    def dataSplit(cls):
        X = cls.df.iloc[:, [4,9]] #Feature
        Y = cls.df.iloc[:, 8] #Target
        return X,Y

    
    def preDataVisualization(X, Y):
        display(X)
        # X1 = X.iloc[:, 1] #previous medal
        # display(X1)
        # X2 = X.iloc[:, 2] #
        # display(X2)
        # display(Y)
        # plt.scatter(X, Y, color="red")
        # plt.show()
    
    
        