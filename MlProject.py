# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

# Load the data
data = pd.read_csv('hyperspectral_data.csv')  # Replace with the actual file name

# Inspect the first few rows of the dataset
print("First few rows of the dataset:")
print(data.head())

# Check for missing values
print("\nMissing values in the dataset:")
print(data.isnull().sum())

# Handle missing values (if any)
data = data.fillna(data.mean())  # Replace missing values with the mean of each column

# Check again after filling missing values
print("\nMissing values after filling with mean:")
print(data.isnull().sum())

# Preprocessing: Scaling the data (if necessary)
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Visualize the first few columns of the dataset
plt.figure(figsize=(10, 6))
sns.pairplot(pd.DataFrame(data_scaled, columns=data.columns[:5]))  # Adjust column selection as needed
plt.show()

# PCA (for visualization)
pca = PCA(n_components=2)  # Reducing to 2 components for 2D visualization
data_pca = pca.fit_transform(data_scaled)

# Plot the PCA result
plt.figure(figsize=(10, 6))
plt.scatter(data_pca[:, 0], data_pca[:, 1], c='blue', edgecolors='k', alpha=0.7)
plt.title('PCA - 2D Projection of the Data')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()

# t-SNE (for non-linear dimensionality reduction)
tsne = TSNE(n_components=2, random_state=42)
data_tsne = tsne.fit_transform(data_scaled)

# Plot the t-SNE result
plt.figure(figsize=(10, 6))
plt.scatter(data_tsne[:, 0], data_tsne[:, 1], c='green', edgecolors='k', alpha=0.7)
plt.title('t-SNE - 2D Projection of the Data')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.show()

# Model Training (Linear Regression)
target_column = 'target'  # Replace 'target' with the actual column name

# Separate features (X) and target (y)
X = data.drop(columns=[target_column])
y = data[target_column]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

# Initialize and train Random Forest model
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions and evaluate
rf_pred = rf_model.predict(X_test)
rf_mse = mean_squared_error(y_test, rf_pred)
rf_r2 = r2_score(y_test, rf_pred)

print(f'Random Forest Mean Squared Error: {rf_mse}')
print(f'Random Forest R-squared: {rf_r2}')


# Perform cross-validation
cv_scores = cross_val_score(model, X, y, cv=5)  # 5-fold cross-validation
print(f'Cross-validation scores: {cv_scores}')
print(f'Mean cross-validation score: {cv_scores.mean()}')