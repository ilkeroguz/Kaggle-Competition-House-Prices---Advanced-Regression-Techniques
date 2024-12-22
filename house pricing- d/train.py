from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pandas as pd
from utils import preprocess_data, evaluate_model, save_model
import joblib

# Load dataset
train_set = pd.read_csv("data/train.csv")

# Separate target variable
y = train_set["SalePrice"]
X = train_set.drop(columns=["SalePrice"])

# Preprocess data and save feature columns
X, feature_columns = preprocess_data(X)
joblib.dump(feature_columns, "feature_columns.pkl")

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=43, test_size=0.5)

# Train model
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
mse, rmse, r2 = evaluate_model(y_test, y_pred)
print(f"MSE: {mse}")
print(f"r^2: {r2}")
print(f"Root Mean Squared Error (RMSE): {rmse}")

# Save model
save_model(model, "model.pkl")
