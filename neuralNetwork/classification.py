import pandas as pd

# Data Visualization - used here to show the losses after testing the model
import matplotlib.pyplot as plt

# using dataset directly form sklearn
from sklearn.datasets import load_iris

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report

#Using Keras here -- keras is a user friendly API built on top of Tensorflow. It provides users with a way to build a model using features of TF, without having to fully implement the backwards passes, loss functions, etc. Also offers native support for things like Early Stopping to avoid overtraining
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical


#Importing this dataset and reading in as dataframe directly from sklearn
dataset = load_iris()

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

# Used to convert to "one hot encoding" of labels ie [0, 1, 0] would indicate the second option is true
y_train = to_categorical(y_train, num_classes=3)
y_test = to_categorical(y_test, num_classes=3)

# Initializing Scaler - used to scale all of the data to roughly the same range - helps with accuracy of the data
scaler = MinMaxScaler()

# "Train" scaler and scale our training data -- then use that same scaler to transform our testing data. It is important that the same scaler is used for both, and that the second time around you DO NOT retrain, as it is a different dataset and will result in a differently configured scaler
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize model - here we are using a sequential model...meaning that inputs are put through our model layers in a sequential fashion. Think from layer a
model = Sequential()
# https://www.geeksforgeeks.org/activation-functions-neural-networks
model.add(Dense(40, activation="relu"))
model.add(Dropout(rate=0.4))
model.add(Dense(30, activation="relu"))
model.add(Dropout(rate=0.3))
model.add(Dense(20, activation="relu"))
model.add(Dropout(rate=0.2))
model.add(Dense(10, activation="relu"))
model.add(Dropout(rate=0.1))
model.add(Dense(3, activation="softmax")) #Output layer - using softmax function to return probability of positive classification over a set of multple possible values (in this case 3 possible values)

# Loss function - difference between the output of the model and the expected result || optimizer is how the model will adjust the parameters (train itsself) each iteration
model.compile(loss="categorical_crossentropy", optimizer="adam")

epochCount = 3000 # number of times it is going to iterate over the traning process
patience = 25 # Setting for how long it is going to "tolerate" no improvement (or minimal improvement) before triggering the early stop

# Defining our early stop - we set a monitor, mode, and patience (how many "failing" epocs before triggering early stop) -- setting verbose here to basically show each epoch...
early_stop = EarlyStopping(monitor="val_loss", mode="min", verbose=1, patience=patience)

# Training the model, presenting training data and corresponding target values. This also auto-validates for us, so including the test dataset as "validation_data". Callback to early stop function to allow our model to stop training if it hits our early stop parameters
model.fit(x=X_train, y=y_train, epochs=epochCount, validation_data=(X_test, y_test), callbacks=[early_stop]) #This epoch number is overkill for this size of dataset - however this will be used to show the usefulness of early stopping callbacks as well as Dropout Layers to avoid overtraining

# Pulling losses from our model's traiing history - only used to show what how the loss values improve over the training process
losses = pd.DataFrame(model.history.history)
losses.plot()
plt.title("Losses Over Time")
#Loss == observed loss on training data || Val_Loss == observed loss on testing data (data not used to train the model / data the model has never "seen" )
# Show the graph we have created over the past couple lines of code
plt.show()

# Performing our own test here...not needed but we are going to use these values for our classification report
predictions = (model.predict(X_test) >= 0.5).astype("int32")
# Printing our classificaion report -- read more here: https://medium.com/@kohlishivam5522/understanding-a-classification-report-for-your-machine-learning-model-88815e2ce397
print(classification_report(y_test, predictions))
