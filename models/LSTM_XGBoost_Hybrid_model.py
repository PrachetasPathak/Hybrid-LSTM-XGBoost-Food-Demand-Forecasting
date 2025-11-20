import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler, PowerTransformer
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, BatchNormalization
from keras.optimizers import AdamW
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from xgboost import XGBRegressor

# Load datasets
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
fulfilment_center_info = pd.read_csv('fulfilment_center_info.csv')
meal_info = pd.read_csv('meal_info.csv')

# Merge datasets
train = train.merge(fulfilment_center_info, on='center_id', how='left').merge(meal_info, on='meal_id', how='left')
test = test.merge(fulfilment_center_info, on='center_id', how='left').merge(meal_info, on='meal_id', how='left')

# Impute missing values
numerical_features = train.select_dtypes(include=[np.number]).columns.tolist()
numerical_features.remove('num_orders')
imputer = SimpleImputer(strategy='median')
train[numerical_features] = imputer.fit_transform(train[numerical_features])
test[numerical_features] = imputer.transform(test[numerical_features])

# Encode categorical variables
categorical_features = train.select_dtypes(include=['object']).columns
label_encoders = {}
for col in categorical_features:
    le = LabelEncoder()
    train[col] = le.fit_transform(train[col].astype(str))
    test[col] = le.transform(test[col].astype(str))
    label_encoders[col] = le

# Feature Scaling & Transformation
scaler = StandardScaler()
power_transformer = PowerTransformer()
train[numerical_features] = power_transformer.fit_transform(train[numerical_features])
test[numerical_features] = power_transformer.transform(test[numerical_features])
train[numerical_features] = scaler.fit_transform(train[numerical_features])
test[numerical_features] = scaler.transform(test[numerical_features])

# Separate features & target
X = train.drop(columns=['num_orders'])
y = np.log1p(train['num_orders'])  # Log transform target for stability
X_test = test.drop(columns=['num_orders'], errors='ignore')

# Train-validation split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Reshape data for LSTM
X_train_lstm = np.reshape(X_train.values, (X_train.shape[0], X_train.shape[1], 1))
X_val_lstm = np.reshape(X_val.values, (X_val.shape[0], X_val.shape[1], 1))

# Define Optimized LSTM Model
lstm_model = Sequential([
    LSTM(units=512, activation='tanh', return_sequences=True, input_shape=(X_train_lstm.shape[1], 1)),
    Dropout(0.3),  # Reduced dropout for better learning
    BatchNormalization(),
    
    LSTM(units=256, activation='tanh', return_sequences=True),
    Dropout(0.3),
    BatchNormalization(),

    LSTM(units=128, activation='tanh', return_sequences=False),
    Dropout(0.3),
    
    Dense(units=64, activation='relu'),
    Dense(units=1)
])

# Compile Model with AdamW Optimizer
lstm_model.compile(optimizer=AdamW(learning_rate=0.0005, weight_decay=0.01), loss='mean_squared_error')

# Callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)
lr_reduction = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-5)

# Initialize Optimized XGBoost Model
xgb_model = XGBRegressor(
    random_state=42,
    n_estimators=800,  # Increased estimators for better learning
    max_depth=12,      # More depth for complex patterns
    learning_rate=0.02, # Reduced learning rate for stability
    subsample=0.9,      # Slightly reduced subsampling to avoid overfitting
    colsample_bytree=0.8,
    gamma=0.1,
    reg_lambda=3,
    reg_alpha=1
)

# Train Models Iteratively (Interleaved Training)
n_iterations = 10
chunk_size = X_train.shape[0] // n_iterations
stacked_preds = []

for i in range(n_iterations):
    start_idx = i * chunk_size
    end_idx = (i + 1) * chunk_size if i != n_iterations - 1 else X_train.shape[0]
    X_train_chunk = X_train.iloc[start_idx:end_idx]
    y_train_chunk = y_train.iloc[start_idx:end_idx]
    X_train_chunk_lstm = X_train_lstm[start_idx:end_idx]
    
    # Train LSTM on mini-batches
    lstm_model.fit(
        X_train_chunk_lstm, y_train_chunk,
        epochs=20, batch_size=64, verbose=1, 
        validation_data=(X_val_lstm, y_val),
        callbacks=[early_stop, lr_reduction]
    )

    # Extract LSTM features for stacking
    lstm_features_chunk = lstm_model.predict(X_train_chunk_lstm).flatten()
    X_train_chunk_xgb = np.column_stack((X_train_chunk, lstm_features_chunk))

    # Train XGBoost on LSTM-enhanced features
    xgb_model.fit(X_train_chunk_xgb, y_train_chunk)

    # Predict on this chunk & store results
    preds_chunk = xgb_model.predict(X_train_chunk_xgb)
    stacked_preds.extend(preds_chunk)

# Evaluate Optimized Model
stacked_preds = np.expm1(stacked_preds)  # Convert back from log scale
actual_values = np.expm1(y_train[:len(stacked_preds)])
stacked_r2 = r2_score(actual_values, stacked_preds)
stacked_mse = mean_squared_error(actual_values, stacked_preds)
stacked_rmse = np.sqrt(stacked_mse)
stacked_mape = mean_absolute_percentage_error(actual_values, stacked_preds)

print(f'✅ Optimized Model R2 Score: {stacked_r2}')
print(f'✅ Optimized Model MSE: {stacked_mse}')
print(f'✅ Optimized Model RMSE: {stacked_rmse}')
print(f'✅ Optimized Model MAPE: {stacked_mape}')

# Plot Results
plt.figure(figsize=(15, 6))
plt.subplot(1, 2, 1)
plt.scatter(actual_values, stacked_preds, color='blue', alpha=0.6)
plt.plot([actual_values.min(), actual_values.max()], [actual_values.min(), actual_values.max()], 'r--')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Predicted vs Actual Values')

plt.subplot(1, 2, 2)
plt.plot(actual_values[:100], label='Actual', color='blue')
plt.plot(stacked_preds[:100], label='Predicted', color='red')
plt.xlabel('Sample Index')
plt.ylabel('Number of Orders')
plt.title('Actual vs Predicted Trend')
plt.legend()
plt.show()
