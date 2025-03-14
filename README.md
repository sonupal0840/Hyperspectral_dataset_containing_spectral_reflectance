# Hyperspectral Data Analysis Project

## Project Overview
This project focuses on analyzing hyperspectral reflectance data from corn samples to predict **vomitoxin_ppb** levels using machine learning. The pipeline includes:

- **Data Preprocessing** (handling missing values, outlier removal, normalization)
- **Exploratory Data Analysis (EDA)** (visualizing spectral correlations)
- **Dimensionality Reduction** (PCA, t-SNE)
- **Model Training & Evaluation** (MLPRegressor with hyperparameter tuning)

## Project Structure
```
├── hyperspectral_data.csv  # Dataset file
├── MlProject.py            # Main script
├── best_model.pkl          # Saved model checkpoint
├── README.md               # Project documentation
└── requirements.txt        # Dependencies
```

## Installation & Setup
### 1. Create a virtual environment (optional but recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

## Running the Project
```bash
python MlProject.py
```
Ensure that `hyperspectral_data.csv` is in the same directory or provide the correct path in `file_path` inside `MlProject.py`.

## Model Performance
- **Mean Absolute Error (MAE):** *TBD*
- **Root Mean Squared Error (RMSE):** *TBD*
- **R-squared Score (R²):** *TBD*

## Results & Insights
- **Data Preprocessing:** Removed outliers using IQR and normalized features.
- **Visualization:** Heatmaps reveal strong correlations between spectral bands.
- **Dimensionality Reduction:** PCA helps retain key variance, while t-SNE aids visualization.
- **Model Selection:** The tuned MLPRegressor provides robust predictions.

## Future Improvements
- Try additional models (Random Forest, XGBoost) for comparison.
- Optimize feature selection to improve model accuracy.
- Test deep learning models (CNNs) for hyperspectral data processing.

---
For any queries, contact **Sonu Pal** at **sonupal0840@gmail.com** 📧

