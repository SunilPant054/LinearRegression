
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display


class Athelete:
        
    df = pd.read_csv('/home/pop/Desktop/python/MachineLearning/LinearRegression/MedalPrediction/atheletes.csv')
    display(df)

    @classmethod
    def dataSplit(cls):
        # X = cls.df.iloc[:, [4,9]] #Feature
        X1 = cls.df.iloc[:, 4]
        X2 = cls.df.iloc[:, 9]
        Y = cls.df.iloc[:, 8] #Target
        return X1,X2,Y

    
    def preDataVisualization(X1, X2, Y):
        # display(X1)
        # display(X2)
        # X1 = X.iloc[:, 1] #previous medal
        # display(X1)
        # X2 = X.iloc[:, 2] #
        # display(X2)
        # display(Y)
        plt.scatter(X1, Y, color="red")
        plt.xlabel("Atheletes")
        plt.ylabel("Medals")
        plt.plot(X1,Y)
        plt.show()
        plt.scatter(X2, Y, color="green")
        plt.xlabel("Previous Medals")
        plt.ylabel("Medals")
        plt.show()
    
        