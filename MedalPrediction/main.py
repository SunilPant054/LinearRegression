import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class PlayerMain:

    df = pd.read_csv('/home/pop/Desktop/python/MachineLearning/LinearRegression/MedalPrediction/atheletes.csv')
    print(df)

    @classmethod
    def preDataVisualization(cls):
        plt.scatter(cls.atheletes, cls.medals, color="red")
        plt.show()