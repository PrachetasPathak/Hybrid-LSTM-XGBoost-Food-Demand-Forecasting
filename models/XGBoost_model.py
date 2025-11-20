import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from xgboost import XGBRegressor

# Load datasets
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
fulfilment_center_info = pd.read_csv('fulfilment_center_info.csv')
meal_info = pd.read_csv('meal_info.csv')

# Merge datasets
train = train.merge(fulfilment_center_info, on='center_id', how='left').merge(meal_info, on='meal_id', how='left')
test = test.merge(fulfilment_center_info, on='center_id', how='left').merge(meal_info, on='meal_id', how='left')

# Impute missing values for numerical columns
numerical_features = train.select_dtypes(include=[np.number]).columns.tolist()
numerical_features.remove('num_orders')  # Exclude target variable
imputer = SimpleImputer(strategy='mean')
train[numerical_features] = imputer.fit_transform(train[numerical_features])
test[numerical_features] = imputer.transform(test[numerical_features])

# Label encode categorical columns
categorical_features = train.select_dtypes(include=['object']).columns
label_encoder = LabelEncoder()

for col in categorical_features:
    train[col] = label_encoder.fit_transform(train[col].astype(str))
    test[col] = label_encoder.transform(test[col].astype(str))

# Separate features and target
X = train.drop(columns=['num_orders'])
y = train['num_orders']
X_test = test.drop(columns=['num_orders'], errors='ignore')

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

# Train-validation split
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


# ----- XGBoost Model -----
xgb_model = XGBRegressor(random_state=42)
xgb_model.fit(X_train, y_train)

# XGBoost Predictions
xgb_val_preds = xgb_model.predict(X_val)

# Evaluate XGBoost
xgb_r2 = r2_score(y_val, xgb_val_preds)
xgb_mse = mean_squared_error(y_val, xgb_val_preds)
print(f'XGBoost R2 Score: {xgb_r2}')
print(f'XGBoost Mean Squared Error: {xgb_mse}')

# Plot XGBoost Results
plt.figure(figsize=(12, 6))
plt.plot(y_val.values, label='True Values', color='blue')
plt.plot(xgb_val_preds, label='XGBoost Predictions', color='green')
plt.title('XGBoost: True vs Predicted')
plt.xlabel('Samples')
plt.ylabel('Num Orders')
plt.legend()
plt.show()
