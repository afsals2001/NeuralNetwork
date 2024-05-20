import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score

# Load the dataset
heart_data = pd.read_csv("heart.csv")

# Display basic information about the dataset
print(heart_data.head())
print(heart_data.tail())
print(heart_data.shape)
print(heart_data.info())
print(heart_data.isnull().sum())
print(heart_data.describe())
print(heart_data["target"].value_counts())

# Split the data into features and target
x = heart_data.drop('target', axis=1)
y = heart_data['target']

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=2)
print(x.shape, x_train.shape, x_test.shape)

# Initialize the Gradient Boosting model
model = GradientBoostingClassifier()

# Train the model
model.fit(x_train, y_train)

# Accuracy on training data
x_train_prediction = model.predict(x_train)
training_data_accuracy = accuracy_score(x_train_prediction, y_train)
print("Accuracy Prediction for Training Data: ", training_data_accuracy * 100)

# Accuracy on testing data
x_test_prediction = model.predict(x_test)
test_data_accuracy = accuracy_score(x_test_prediction, y_test)
print("Accuracy Prediction for Testing Data: ", test_data_accuracy * 100)

# Making a prediction on new data
input_data = (34, 0, 1, 118, 210, 0, 1, 192, 0, 0.7, 2, 0, 2)

# Convert the input data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)

# Reshape the numpy array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

# Make a prediction
prediction = model.predict(input_data_reshaped)
print(prediction)

# Interpret the prediction
if prediction[0] == 0:
    print("The person does not have heart disease")
else:
    print("The person has heart disease")
