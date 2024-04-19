import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import seaborn as sns

df = pd.read_csv('/home/pop/Desktop/python/MachineLearning/LinearRegression/SalaryPrediction/Salary.csv')
df.head()


#splitting dataset 
X = df.iloc[:, :-1].values  #Features => Years of Experience => inependent variable
y = df.iloc[:, -1].values  #target => salary => dependent variable


#divide the dataset in some amount of training and testing data

# random_state => seed value used by random number generator
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

#Implement classifier based on simple linear regression 
model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)
print(predictions)

sns.histplot(predictions-y_test)

plt.scatter(X_train, y_train, color='red')
plt.plot(X_train, model.predict(X_train))
plt.show()