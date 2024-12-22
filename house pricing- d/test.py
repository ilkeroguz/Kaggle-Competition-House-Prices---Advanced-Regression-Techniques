import pandas as pd
from utils import preprocess_data, load_model
import joblib

# Load dataset
test_set = pd.read_csv("data/test.csv")

# Load feature columns
feature_columns = joblib.load("feature_columns.pkl")

# Preprocess test data with the same feature columns
X_test, _ = preprocess_data(test_set, feature_columns=feature_columns)

# Load model
model = load_model("model.pkl")

# Make predictions
y_pred = model.predict(X_test)

# Output predictions
output = pd.DataFrame({"Id": test_set["Id"], "SalePrice": y_pred})
output.to_csv("submission.csv", index=False)
print("Predictions saved to submission.csv")