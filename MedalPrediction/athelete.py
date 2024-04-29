
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


class Athelete:
        
    df = pd.read_csv('/home/pop/Desktop/python/MachineLearning/LinearRegression/MedalPrediction/atheletes.csv')
    df = df.dropna() #removes rows with missing values in them 
    display(df)

    @classmethod
    def dataSplit(cls):
        # X = cls.df.iloc[:, [4,9]] #Feature
        X1 = cls.df.iloc[:, 4]
        X2 = cls.df.iloc[:, 9]
        Y = cls.df.iloc[:, 8] #Target
        return X1,X2,Y

    
    def preDataVisualization(X1, X2, Y):
        #X1 = athelete and X2= previous medals
        plt.scatter(X1, Y, color="red")
        plt.xlabel("Atheletes")
        plt.ylabel("Medals")
        plt.show()

        plt.scatter(X2, Y, color="green")
        plt.xlabel("Previous Medals")
        plt.ylabel("Medals")
        plt.show()

    @classmethod
    def dataSplitTrainTest(cls):
        predictors = ['athletes', 'prev_medals']
        X = cls.df[predictors]
        Y = cls.df.iloc[:, 8]
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
        return X_train, X_test, y_train, y_test
    
    @classmethod
    def trainModel(cls, X_train, X_test, y_train, y_test):
        model = LinearRegression()
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        #Conert -ve values to 0 and rounding off decimal values 
        predictions = np.where(predictions > 0, np.round(predictions), 0)
        print(predictions)
            
        # display(cls.df)
        # print(predictions)
        
        plt.scatter(y_test, predictions, color="blue")
        plt.xlabel("Actual medals")
        plt.ylabel("Predicted medals")
        plt.show()

        plt.scatter(athelete_feature, y_test, color="black")
        plt.plot(athelete_feature, predictions, color="blue", linewidth=3)
        plt.show()
        # plt.xticks(())
        # plt.yticks(())
        # predictions1 = model.predict(X_train)
        # predictions1 = np.where(predictions1 > 0, np.round(predictions1), 0)
        # predictions1.shape

        # plt.scatter(y_train, model.predict(X_train), color="red")
        # plt.plot()
        # plt.show()
        
        
        
        
    
        