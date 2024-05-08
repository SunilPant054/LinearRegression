
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
        y_prediction_train = model.predict(X_train)
        #Conert -ve values to 0 and rounding off decimal values 
        y_prediction_train = np.where(y_prediction_train > 0, np.round(y_prediction_train), 0)
      
        

        #Visualization for training set
        plt.scatter(y_train, y_prediction_train)
        plt.xlabel('Actual Medal')
        plt.ylabel('Predicted Medal')
        plt.show()


        #For test data
        y_prediction_test = model.predict(X_test)
        y_prediction_test = np.where(y_prediction_test > 0, np.round(y_prediction_test), 0)
        

        #Visualization for test data
        plt.scatter(y_test, y_prediction_test)
        plt.xlabel('Actual Medal')
        plt.ylabel('Predicted Medal')
        plt.show()
      
        
        
        
        
    
        