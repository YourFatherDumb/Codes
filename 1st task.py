import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Load the data from the URL
url = "http://bit.ly/w-data"
data = pd.read_csv(url)

# Extract the study hours and percentage columns
study_hours = data["Hours"]
percentage = data["Scores"]

# Reshape the input data to fit the expected shape by scikit-learn
study_hours = np.array(study_hours).reshape(-1, 1)

# Create a linear regression model and fit the data
model = LinearRegression()
model.fit(study_hours, percentage)

# Predict the percentage for a given study hour
study_hour = 9.25
predicted_percentage = model.predict([[study_hour]])

print(f"Predicted percentage for {study_hour} study hours: {predicted_percentage[0]}")
