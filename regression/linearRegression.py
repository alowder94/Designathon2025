from pathlib import Path

import pandas as pd
import numpy as np

#Used for graphing / creating visualizations
import matplotlib.pyplot as plt
import seaborn as sns

# Used to create your model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Used to calculate metrics/evaluate your model
from sklearn import metrics

#Set local directory so I can run from a terminal anywhere -- this has nothing to do with the model
baseDir = Path(__file__).resolve().parent

#Using Kaggle Data
df = pd.read_csv(f"{baseDir}/DataFiles/USA_Housing.csv")

#Good practice to look at these two statistics when getting started - basically just gives you a bit of insight into what you're workig with
print(df.info()) # Columns/data types, num of rows
print(df.describe()) # Stats - mean, std, max, min

# Data Analysis -- trying to understand the data and relationships within
# sns.displot(df['Price']) # This will show distribution of data
# sns.heatmap(df.drop("Address", axis=1).corr(), annot=True) # Shows heatmap of correlations within our dataset
# plt.show()

# Next clean data - in this case remove address column - and split columns into the target variable (what you are trying to predict) and the features you are using to predict
X = df[df.drop(["Address", "Price"], axis=1).columns] #Data Frame of all columns you are using for your prediction
Y = df['Price'] # DataFrame of values you are trying to predict -- target values

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.4, train_size=0.6, random_state=101) # Random state basically is just a way of ensuring you always see the same random split. it is optional
 
#Instantiate and train the linear regression model
lm = LinearRegression()
lm.fit(x_train, y_train)

#Looking at different attributes of the model - showing intercept and coefficients here
print(lm.intercept_) # Shows correlation?
print(lm.coef_) # Shows correlation per attribute

#Create DF out of coefficients for analysis - basically trying to see what the cause/effect ratio is for each datapoint. Attempt to see what is important in the data and what is not
cdf = pd.DataFrame(lm.coef_, index=X.columns, columns=['Coeff']) 
print("Coefficient Analysis: {}".format(cdf)) #So based off of this coefficient analysis - you can say that using this Linear Regression model, a 1 unit increase to Area Income in associated with a 21.5 unit increase in Price

# Testing and gettig predictions
predictions = lm.predict(x_test)
print(predictions)

# Show a distribution histogram of actual test values vs predicted values -- showing a visualization of how my model performed
sns.displot(y_test - predictions) # This is going to show residuals - which is basically real values - predicted values....this is a measure of accuracy
# plt.scatter(y_test, predictions) #This would show a scatterplot of the same data, basically showing test against predictions
plt.show()

# Now that we have our model, you can calculate Mean Absolute Error, Mean Squared Error, and Root Mean Squared Error
print("Mean Absolute Error: {}".format(metrics.mean_absolute_error(y_test, predictions)))
print("Mean Squared Error: {}".format(metrics.mean_squared_error(y_test, predictions)))
print("Root Mean Squared Error: {}".format(np.sqrt(metrics.mean_squared_error(y_test, predictions)))) # There is no root mean squared error method, so you just calculate this yourself
