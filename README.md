# California_House_Prices
# 🏠 California Housing Price Prediction

This project predicts **California housing prices** using the famous housing dataset.  
We applied data preprocessing (scaling & encoding), trained multiple machine learning models, and evaluated their performance.

---

## 📊 Models Tested
We trained and evaluated the following models:
- **Linear Regression**
- **Decision Tree Regressor**
- **Random Forest Regressor**
- **K-Nearest Neighbors (KNN)**

---

## ⚙️ Preprocessing
- **Numerical features** → Standardized using `StandardScaler`.
- **Categorical feature (`ocean_proximity`)** → Encoded using `OneHotEncoder`.
- Combined using a `ColumnTransformer` inside a `Pipeline`.

---

## 🧪 Results

| Model              | MAE       | RMSE      | R²    |
|--------------------|-----------|-----------|-------|
| Linear Regression  | 50,702    | 70,031    | 0.626 |
| Decision Tree      | 43,628    | 69,254    | 0.634 |
| Random Forest      | **31,500**| **48,774**| **0.818** |
| KNN                | 40,879    | 61,512    | 0.711 |

✅ **Best Model:** Random Forest Regressor  

---

## 💾 Save & Load Best Model
We saved the **Random Forest pipeline** (preprocessing + model) using `joblib`.

```python
import joblib

# Save model
joblib.dump(rf_pipeline, "random_forest_pipeline.pkl")

# Load model
loaded_rf = joblib.load("random_forest_pipeline.pkl")
y_pred = loaded_rf.predict(X_test[:5])
