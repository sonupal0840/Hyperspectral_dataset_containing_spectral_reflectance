import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os

# Load data
def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        print("Data loaded successfully.")
        print("Columns in dataset:", df.columns)
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

# Preprocess data
def preprocess_data(df, target_column):
    if df is None:
        raise ValueError("Dataframe is None. Ensure the file is loaded correctly.")
    
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset!")

    if 'hsi_id' in df.columns:
        df = df.drop(columns=['hsi_id'])  # Remove non-numeric column if present
    
    df_numeric = df.select_dtypes(include=['number']).copy()
    df_numeric.fillna(df_numeric.mean(), inplace=True)  # Fill missing values
    
    # Outlier Removal using IQR
    Q1 = df_numeric.quantile(0.25)
    Q3 = df_numeric.quantile(0.75)
    IQR = Q3 - Q1
    df_filtered = df_numeric[~((df_numeric < (Q1 - 1.5 * IQR)) | (df_numeric > (Q3 + 1.5 * IQR))).any(axis=1)].copy()
    
    df_filtered = df_filtered.reset_index(drop=True)  # Fix indexing issue

    # Normalize only the remaining data
    scaler = MinMaxScaler()
    df_filtered[df_filtered.columns] = scaler.fit_transform(df_filtered)
    
    print("Data preprocessing completed.")
    return df_filtered

# Visualize spectral bands
def visualize_spectral_bands(df):
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Correlation Matrix of Spectral Bands')
    plt.show()

# Perform PCA
def apply_pca(df, n_components=2):
    try:
        pca = PCA(n_components=n_components)
        principal_components = pca.fit_transform(df)
        
        # Plot Explained Variance
        plt.figure(figsize=(10, 6))
        plt.plot(np.cumsum(pca.explained_variance_ratio_))
        plt.xlabel('Number of Components')
        plt.ylabel('Explained Variance')
        plt.title('PCA Explained Variance')
        plt.show()
        
        print("PCA applied successfully.")
        return principal_components, pca.explained_variance_ratio_
    except Exception as e:
        print(f"Error in PCA: {e}")
        return None, None

# Perform t-SNE
def apply_tsne(df, n_components=2, perplexity=30, random_state=42):
    try:
        perplexity = min(perplexity, len(df) - 1)
        tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=random_state)
        tsne_results = tsne.fit_transform(df)
        print("t-SNE applied successfully.")
        return tsne_results
    except Exception as e:
        print(f"Error in t-SNE: {e}")
        return None

# Train regression model with checkpointing
def train_regression_model(X, y, checkpoint_path="best_model.pkl"):
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        param_grid = {
            'hidden_layer_sizes': [(128, 64), (256, 128)],
            'activation': ['relu'],
            'solver': ['adam'],
            'max_iter': [1000],
            'early_stopping': [True]
        }
        
        if os.path.exists(checkpoint_path):
            print("Loading saved model...")
            best_model = joblib.load(checkpoint_path)
        else:
            model = MLPRegressor()
            grid_search = GridSearchCV(model, param_grid, cv=3, scoring='r2', n_jobs=-1)
            grid_search.fit(X_train, y_train)
            
            best_model = grid_search.best_estimator_
            joblib.dump(best_model, checkpoint_path)
        
        y_pred = best_model.predict(X_test)
        
        mae = mean_absolute_error(y_test, y_pred)
        rmse = mean_squared_error(y_test, y_pred) ** 0.5
        r2 = r2_score(y_test, y_pred)
        
        print("Model training completed.")
        print(f"Best Model Performance: MAE={mae:.4f}, RMSE={rmse:.4f}, R2={r2:.4f}")
        
        # Scatter plot for actual vs predicted
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred)
        plt.xlabel("Actual Values")
        plt.ylabel("Predicted Values")
        plt.title("Actual vs Predicted DON Concentration")
        plt.show()
        
        return best_model, mae, rmse, r2
    except Exception as e:
        print(f"Error in model training: {e}")
        return None, None, None, None

# Main execution
def main(file_path, target_column):
    df = load_data(file_path)
    df = preprocess_data(df, target_column)
    
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Data visualization of spectral bands
    visualize_spectral_bands(df)
    
    # Perform PCA and t-SNE for dimensionality reduction
    pca_results, _ = apply_pca(X)
    tsne_results = apply_tsne(X)
    
    # Train regression model and evaluate performance
    model, mae, rmse, r2 = train_regression_model(X, y)
    
    print("Pipeline execution completed.")

if __name__ == "__main__":
    file_path = "hyperspectral_data.csv"  # Provide correct path to the dataset
    target_column = "vomitoxin_ppb"  # Update based on your target column name
    main(file_path, target_column)
