import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, explained_variance_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Data Loading and Preprocessing
def load_and_preprocess_data(train_path, test_path, fulfillment_path, meal_info_path):
    # Load datasets
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    fulfillment_center_info = pd.read_csv(fulfillment_path)
    meal_info = pd.read_csv(meal_info_path)

    # Merge datasets
    train = train.merge(fulfillment_center_info, on='center_id', how='left').merge(meal_info, on='meal_id', how='left')
    test = test.merge(fulfillment_center_info, on='center_id', how='left').merge(meal_info, on='meal_id', how='left')

    # Debugging: Print dataset shapes
    print(f"Train dataset shape: {train.shape}, Test dataset shape: {test.shape}")

    # Impute missing values for numerical columns
    numerical_features = train.select_dtypes(include=[np.number]).columns.tolist()
    numerical_features.remove('num_orders')
    imputer = SimpleImputer(strategy='median')
    train[numerical_features] = imputer.fit_transform(train[numerical_features])
    test[numerical_features] = imputer.transform(test[numerical_features])

    # Label encode categorical columns
    categorical_features = train.select_dtypes(include=['object']).columns
    label_encoder = LabelEncoder()

    for col in categorical_features:
        train[col] = label_encoder.fit_transform(train[col].astype(str))
        test[col] = label_encoder.transform(test[col].astype(str))

    # Feature engineering
    train['center_meal_interaction'] = train['center_id'] * train['meal_id']
    test['center_meal_interaction'] = test['center_id'] * test['meal_id']

    # Separate features and target
    X_train = train.drop(columns=['num_orders'])
    y_train = train['num_orders']
    X_test = test.drop(columns=['num_orders'], errors='ignore')

    return X_train, y_train, X_test

# Scaling and Transformation
def scale_and_transform(X_train, y_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    y_train_log = np.log1p(y_train)

    return X_train_scaled, y_train_log, X_test_scaled

# Model Building
def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=16, activation='tanh', input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Cross-Validation Evaluation
def evaluate_model_with_cv(X, y, num_splits=2):
    kf = KFold(n_splits=num_splits, shuffle=True, random_state=42)

    metrics = {
        "R2": [],
        "MSE": [],
        "RMSE": [],
        "MAE": [],
        "MAPE": [],
        "Explained Variance": [],
        "RRMSE": []
    }

    all_y_true = []
    all_y_pred = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Reshape for LSTM
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

        model = build_lstm_model((X_train.shape[1], 1))

        early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
        lr_reduction = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6)

        print(f"Training on fold with {len(train_index)} samples")
        model.fit(X_train, y_train, epochs=30, batch_size=8, validation_split=0.2, 
                  callbacks=[early_stopping, lr_reduction], verbose=1)

        y_pred = model.predict(X_test).flatten()

        # Collect true and predicted values for plotting
        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)

        # Metrics Calculation
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        r2 = r2_score(y_test, y_pred)
        evs = explained_variance_score(y_test, y_pred)
        rrmse = rmse / np.mean(y_test)

        metrics["R2"].append(r2)
        metrics["MSE"].append(mse)
        metrics["RMSE"].append(rmse)
        metrics["MAE"].append(mae)
        metrics["MAPE"].append(mape)
        metrics["Explained Variance"].append(evs)
        metrics["RRMSE"].append(rrmse)

    return {k: np.mean(v) for k, v in metrics.items()}, all_y_true, all_y_pred

# Main Script
train_path = 'train.csv'  # Replace with actual path
test_path = 'test.csv'  # Replace with actual path
fulfillment_path = 'fulfilment_center_info.csv'  # Replace with actual path
meal_info_path = 'meal_info.csv'  # Replace with actual path

X_train, y_train, X_test = load_and_preprocess_data(train_path, test_path, fulfillment_path, meal_info_path)
X_train_scaled, y_train_log, X_test_scaled = scale_and_transform(X_train, y_train, X_test)

# Convert to numpy arrays for KFold
X = np.array(X_train_scaled)
y = np.array(y_train_log)

# Evaluate model and get results
results, all_y_true, all_y_pred = evaluate_model_with_cv(X, y)

# Display metrics
print("LSTM Performance Metrics:")
for metric, value in results.items():
    print(f"{metric}: {value:.4f}")

# Plot Results in a Detailed and Styled Format
plt.figure(figsize=(20, 10))

# Scatter Plot: Predicted vs Actual
# plt.subplot(2, 2, 1)
plt.scatter(all_y_true, all_y_pred, color='blue', alpha=0.6, label='Predicted vs Actual')
plt.plot([min(all_y_true), max(all_y_true)], [min(all_y_true), max(all_y_true)], color='red', linestyle='--', label='Ideal Fit')
plt.title('Predicted vs Actual Values (num_orders)', fontsize=14)
plt.xlabel('Actual num_orders', fontsize=12)
plt.ylabel('Predicted num_orders', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True)

# # Histogram: Distribution of Errors
# errors = np.array(all_y_true) - np.array(all_y_pred)
# plt.subplot(2, 2, 2)
# plt.hist(errors, bins=30, color='purple', alpha=0.7, edgecolor='black')
# plt.title('Distribution of Prediction Errors', fontsize=14)
# plt.xlabel('Error (Actual - Predicted)', fontsize=12)
# plt.ylabel('Frequency', fontsize=12)
# plt.grid(True)

# # Line Plot: Actual vs Predicted Over Sample Indices
# plt.subplot(2, 1, 2)
# plt.plot(all_y_true[:100], label='Actual num_orders', color='green', marker='o', linestyle='dashed', markersize=4)
# plt.plot(all_y_pred[:100], label='Predicted num_orders', color='orange', marker='x', linestyle='dashed', markersize=4)
# plt.title('Actual vs Predicted num_orders Over Samples', fontsize=14)
# plt.xlabel('Sample Index', fontsize=12)
# plt.ylabel('num_orders', fontsize=12)
# plt.legend(fontsize=10)
# plt.grid(True)

plt.tight_layout()
plt.show()
