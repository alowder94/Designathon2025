import pandas as pd

# Data Visualization - used here to show the losses after testing the model
import matplotlib.pyplot as plt

# using dataset directly form sklearn
from sklearn.datasets import load_breast_cancer

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix

#Using Keras here -- keras is a user friendly API built on top of Tensorflow. It provides users with a way to build a model using features of TF, without having to fully implement the backwards passes, loss functions, etc. Also offers native support for things like Early Stopping to avoid overtraining
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping


#Importing this dataset and reading in as dataframe directly from sklearn
dataset = load_breast_cancer()

# Scikit learn exposes this dataset in a format that isn't necessarilly compatible with pandas/keras/tensorflow...so pulling the data and formatting here
df = pd.DataFrame(dataset['data'], columns=dataset['feature_names'])

# Print stats about the data...things like datatypes, if it is required/non-null, etc
df.info()
# Going to skip data analysis here - the main point of this file is to teach a classification model using TF as well as show early stop callbacks during training and Drop Layers - both to avoid overtraining to the training data

# Splitting target from the rest of the dataset - target will be used for training and testing but will not be present in "real life" uses of this model
X = df.values
y = dataset['target']

# Splitting into train and testing datasets, with corresponding target sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=101)

# Initializing Scaler - used to scale all of the data to roughly the same range - helps with accuracy of the data
scaler = MinMaxScaler()

# "Train" scaler and scale our training data -- then use that same scaler to transform our testing data. It is important that the same scaler is used for both, and that the second time around you DO NOT retrain, as it is a different dataset and will result in a differently configured scaler
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize model - here we are using a sequential model...meaning that inputs are put through our model layers in a sequential fashion. Think from layer a
model = Sequential()
# https://www.geeksforgeeks.org/activation-functions-neural-networks
model.add(Dense(30, activation="relu"))
model.add(Dropout(rate=0.3))
model.add(Dense(20, activation="relu"))
model.add(Dropout(rate=0.2))
model.add(Dense(10, activation="relu"))
model.add(Dropout(rate=0.1))
model.add(Dense(1, activation="sigmoid")) #Output layer - using sigmoid function to return probability of positive classification -- this is becuase this is a binary classification -- meaning we are essentially looking for a 1/0, or true/false output from this model

# Loss function - difference between the output of the model and the expected result || optimizer is how the model will adjust the parameters (train itsself) each iteration
model.compile(loss="binary_crossentropy", optimizer="adam")

epochCount = 3000
patience = 25

#Loss == observed loss on training data || Val_Loss == observed loss on testing data (data not used to train the model / data the model has never "seen" )
# Defining our early stop - we set a monitor, mode, and patience (how many "failing" epocs before triggering early stop) -- setting verbose here to basically show each epoch...
early_stop = EarlyStopping(monitor="val_loss", mode="min", verbose=1, patience=patience)

# Training the model, presenting training data and corresponding target values. This also auto-validates for us, so including the test dataset as "validation_data". Callback to early stop function to allow our model to stop training if it hits our early stop parameters
model.fit(x=X_train, y=y_train, epochs=epochCount, validation_data=(X_test, y_test), callbacks=[early_stop]) #This epoch number is overkill for this size of dataset - however this will be used to show the usefulness of early stopping callbacks as well as Dropout Layers to avoid overtraining

# Pulling losses from our model's traiing history - only used to show what how the loss values improve over the training process
losses = pd.DataFrame(model.history.history)
losses.plot()
plt.title("Losses Over Time")
plt.show()

# Performing our own test here...not needed but we are going to use these values for out classification report and out confusion matrix
predictions = (model.predict(X_test) >= 0.5).astype("int32")
print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))
