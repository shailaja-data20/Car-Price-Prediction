# 🚗 Car Price Prediction

A complete, end-to-end regression project that predicts the **selling price of a used car** based on its features (brand, year, mileage, fuel type, transmission, etc.).

Built in Google Colab / Jupyter using `pandas`, `scikit-learn`, `matplotlib`, and `seaborn`.

---

## 📌 Project Overview

**Goal:** Build a regression model that predicts `Selling_Price` for a used car given its specifications.

**Workflow:**
1. Import libraries
2. Load the dataset
3. Understand the data (shape, types, missing values, duplicates)
4. Clean the data (remove duplicates, normalize text columns)
5. Exploratory Data Analysis (EDA) — distributions, scatter plots, box plots, correlation heatmap
6. Feature engineering — `Car_Age`, `Km_per_Year`
7. Define features (`X`) and target (`y`)
8. Train-test split (80/20)
9. Preprocess numerical + categorical features (imputation, scaling, one-hot encoding)
10. Train multiple regression models
11. Evaluate and compare models (MAE, RMSE, R²)
12. Select the best-performing model
13. Predict the price of a new car
14. Interactive prediction function for custom inputs
15. Save the trained model with `joblib`

---

## 📂 Dataset

The notebook expects a CSV file named `Car_Price_Prediction_Dataset.csv` with the following columns:

| Column | Description |
|---|---|
| `Brand` | Car manufacturer/brand |
| `Year` | Manufacturing year |
| `Km_Driven` | Total kilometers driven |
| `Fuel_Type` | Petrol / Diesel / CNG / Electric, etc. |
| `Transmission` | Manual / Automatic |
| `Owner` | Ownership history (First, Second, etc.) |
| `Location` | City/region of sale |
| `Engine_CC` | Engine displacement (cc) |
| `Mileage` | Fuel efficiency |
| `Seats` | Number of seats |
| `Insurance` | Insurance status |
| `Selling_Price` | **Target variable** — price in lakhs |

> ⚠️ The notebook currently loads the dataset from a local Windows path (`C:\Users\Yerra\...`). Update the `pd.read_csv(...)` path in the "Load the dataset" cell to point to your own copy of the CSV, or upload it directly if running in Google Colab.

---

## 🛠️ Requirements

```bash
pip install numpy pandas matplotlib seaborn scikit-learn joblib
```

Or, if running in Google Colab, all required libraries are pre-installed.

---

## ▶️ How to Run

1. Open `Car_Price_Prediction_ML_Google_Colab.ipynb` in Jupyter or Google Colab.
2. Update the dataset path in the data-loading cell to match your file location.
3. Run all cells in order (top to bottom).
4. Review the EDA plots and model comparison table.
5. Use the `predict_car_price(...)` function near the end of the notebook to get a price estimate for any custom car by changing the input arguments.
6. The best-performing model is automatically saved as `car_price_prediction_model.pkl`.

---

## 🤖 Models Trained

| Model | Notes |
|---|---|
| Linear Regression | Baseline linear model |
| Ridge Regression | Regularized linear model (α = 1.0) |
| Random Forest Regressor | 300 trees, ensemble of decision trees |
| Gradient Boosting Regressor | 200 estimators, boosted trees |

All models are wrapped in a single `scikit-learn` `Pipeline` with a shared preprocessing step:
- **Numerical features:** median imputation → standard scaling
- **Categorical features:** most-frequent imputation → one-hot encoding

This prevents data leakage since preprocessing is fit only on the training data.

---

## 📊 Evaluation Metrics

Models are compared using:
- **MAE** (Mean Absolute Error) — average absolute prediction error (lower is better)
- **RMSE** (Root Mean Squared Error) — penalizes larger errors more (lower is better)
- **R² Score** — proportion of variance explained by the model (higher is better, 1.0 = perfect)

The model with the highest R² score on the test set is automatically selected as the "best model" for prediction and saving.

---

## 🔮 Making Predictions

Example usage of the prediction helper function defined in the notebook:

```python
price = predict_car_price(
    brand='Toyota', year=2022, km_driven=30000,
    fuel_type='Petrol', transmission='Automatic', owner='First',
    location='Hyderabad', engine_cc=1800, mileage=18.0,
    seats=5, insurance='Yes'
)
print(f'Estimated price: ₹{price:.2f} lakh')
```

---

## 💾 Output

- `car_price_prediction_model.pkl` — the trained, serialized best-performing pipeline (preprocessing + model), ready to be loaded with `joblib.load(...)` for future predictions without retraining.

---

## 📁 Suggested Project Structure

```
├── Car_Price_Prediction_ML_Google_Colab.ipynb   # Main notebook
├── Car_Price_Prediction_Dataset.csv              # Input dataset (not included)
├── car_price_prediction_model.pkl                # Saved model (generated after running)
└── README.md                                      # This file
```

---

## 📝 Notes

- Feature engineering adds `Car_Age` (current year − manufacturing year) and `Km_per_Year` (kilometers driven per year of age) to improve model performance.
- The current notebook hardcodes `2026` as the reference year for age calculation — update this if running in a different year for accuracy.
- Random seeds (`random_state=42`) are fixed throughout for reproducibility.
