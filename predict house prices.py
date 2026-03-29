Google collab link: https://colab.research.google.com/drive/14_P01bq_44D-kyk9VLuvmztFz56fqfCs?usp=sharing
#DL LAB QUES 4
#Design a neural network for predicting house prices using Boston Housing Price Dataset
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Added import for pandas
# from sklearn.datasets import load_boston # Removed deprecated import
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# 1. Load the Boston Housing dataset
# Note: The Boston Housing dataset is deprecated in scikit-learn.
# For educational purposes, we will proceed with it, but for new projects,
# consider using alternative datasets like California Housing.
# Replaced load_boston() with direct download and processing
data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
X = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
y = raw_df.values[1::2, 2]

print()
print(f"Original shape of features (X): {X.shape}")
print(f"Original shape of targets (y): {y.shape}")

# 2. Preprocess Data
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features (important for neural networks)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print()
print(f"Shape of scaled training features: {X_train_scaled.shape}")
print(f"Shape of scaled testing features: {X_test_scaled.shape}")

# 3. Define the Neural Network Model for Regression
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)))
model.add(Dense(64, activation='relu'))
model.add(Dense(1))
# Output layer with 1 unit for regression (no activation for linear output)

model.summary()

# 4. Compile and Train Model
# Use Mean Squared Error (MSE) as the loss function for regression
model.compile(optimizer='Adam', loss='mse', metrics=['mae'])

# Mean Absolute Error (MAE) for interpretability
print()
print("Model compiled successfully. Starting training...")

history = model.fit(
                     X_train_scaled, y_train, epochs=100, # Increased epochs for better learning
                     batch_size=32,
                     validation_split=0.2 # Use a portion of training data for validation
                     )

print()
print("Model training complete. History stored in 'history' variable.")

# 5. Evaluate Model Performance
test_loss, test_mae = model.evaluate(X_test_scaled, y_test)
print()
print(f"\nTest Loss (MSE): {test_loss:.4f}")
print(f"Test MAE: {test_mae:.4f}")

# 6. Visualize Training History
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss (MSE)')
plt.plot(history.history['val_loss'], label='Validation Loss (MSE)')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label='Training MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.title('Model Mean Absolute Error')
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.legend()

plt.tight_layout()
plt.show()
