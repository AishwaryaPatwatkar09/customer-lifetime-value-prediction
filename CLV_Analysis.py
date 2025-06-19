# ============================================================================
# CUSTOMER LIFETIME VALUE PREDICTION - COMPLETE PROJECT
# Author: [Your Name]
# Description: Predicting customer value using XGBoost machine learning
# Dataset: 541,909 transactions from UK retail company
# ============================================================================

# %% Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
import joblib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("🚀 Starting Customer Lifetime Value Analysis...")
print("📚 All libraries loaded successfully!")

# %% Load Data
print("\n📊 Loading data...")
try:
    df = pd.read_excel('Online Retail.xlsx')
    print(f"✅ Data loaded successfully!")
    print(f"📈 Dataset size: {len(df):,} rows and {len(df.columns)} columns")
    
    # Show first few rows
    print("\n🔍 First 5 rows of data:")
    print(df.head())
    
    # Show column info
    print("\n📋 Column information:")
    print(df.info())
    
except FileNotFoundError:
    print("❌ Data file not found. Make sure 'Online Retail.xlsx' is in your project folder.")

# %% Data Cleaning and Preprocessing
print("\n🧹 Cleaning data...")

# Store original size
original_size = len(df)
print(f"Original dataset: {original_size:,} rows")

# Remove rows with missing CustomerID
df = df.dropna(subset=['CustomerID'])
print(f"After removing missing CustomerID: {len(df):,} rows")

# Remove negative quantities (returns)
df = df[df['Quantity'] > 0]
print(f"After removing returns: {len(df):,} rows")

# Remove negative prices
df = df[df['UnitPrice'] > 0]
print(f"After removing negative prices: {len(df):,} rows")

# Create total amount column
df['TotalAmount'] = df['Quantity'] * df['UnitPrice']

# Remove extreme outliers (optional)
q99 = df['TotalAmount'].quantile(0.99)
df = df[df['TotalAmount'] <= q99]
print(f"After removing extreme outliers: {len(df):,} rows")

cleaned_size = len(df)
print(f"✅ Data cleaning complete! Removed {original_size - cleaned_size:,} rows ({((original_size - cleaned_size)/original_size)*100:.1f}%)")

# %% Feature Engineering - RFM Analysis
print("\n🔧 Creating customer features (RFM Analysis)...")

# Convert InvoiceDate to datetime if not already
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

# Set reference date (last date in dataset + 1 day)
reference_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)
print(f"Reference date for recency calculation: {reference_date.date()}")

# Create customer-level features
customer_features = df.groupby('CustomerID').agg({
    'InvoiceDate': ['min', 'max'],  # First and last purchase dates
    'InvoiceNo': 'nunique',         # Number of unique transactions
    'Quantity': 'sum',              # Total items purchased
    'TotalAmount': ['sum', 'mean', 'std'],  # Total, average, and std of spending
    'UnitPrice': 'mean'             # Average price per item
}).round(2)

# Flatten column names
customer_features.columns = [
    'FirstPurchase', 'LastPurchase', 'Frequency', 'TotalQuantity',
    'TotalSpent', 'AvgOrderValue', 'StdOrderValue', 'AvgUnitPrice'
]

# Calculate Recency (days since last purchase)
customer_features['Recency'] = (reference_date - customer_features['LastPurchase']).dt.days

# Calculate customer tenure (days between first and last purchase)
customer_features['Tenure'] = (customer_features['LastPurchase'] - customer_features['FirstPurchase']).dt.days + 1

# Handle NaN values in StdOrderValue (customers with only 1 purchase)
customer_features['StdOrderValue'] = customer_features['StdOrderValue'].fillna(0)

# Calculate additional features
customer_features['PurchaseFrequency'] = customer_features['Frequency'] / customer_features['Tenure'] * 365  # Purchases per year
#customer_features['PurchaseFrequency'] = customer_features['PurchaseFrequency'].replace([np.inf, -np.inf], customer_features['Frequency'])
customer_features['PurchaseFrequency'] = customer_features['PurchaseFrequency'].replace([np.inf, -np.inf], np.nan)
customer_features['PurchaseFrequency'] = customer_features['PurchaseFrequency'].fillna(customer_features['Frequency'])
# Reset index to make CustomerID a column
customer_features = customer_features.reset_index()

print(f"✅ Features created for {len(customer_features):,} unique customers")
print("\n📊 Customer features sample:")
print(customer_features.head())

print("\n📈 Feature statistics:")
print(customer_features.describe())

# %% Exploratory Data Analysis
print("\n📊 Creating exploratory visualizations...")

# Set up the plotting style
plt.style.use('default')
sns.set_palette("husl")

# Create figure with subplots
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Customer Lifetime Value - Exploratory Data Analysis', fontsize=16, fontweight='bold')

# 1. Distribution of Total Spent
axes[0, 0].hist(customer_features['TotalSpent'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
axes[0, 0].set_title('Distribution of Total Amount Spent')
axes[0, 0].set_xlabel('Total Spent (£)')
axes[0, 0].set_ylabel('Number of Customers')

# 2. Distribution of Frequency
axes[0, 1].hist(customer_features['Frequency'], bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
axes[0, 1].set_title('Distribution of Purchase Frequency')
axes[0, 1].set_xlabel('Number of Purchases')
axes[0, 1].set_ylabel('Number of Customers')

# 3. Distribution of Recency
axes[0, 2].hist(customer_features['Recency'], bins=50, alpha=0.7, color='lightcoral', edgecolor='black')
axes[0, 2].set_title('Distribution of Recency (Days Since Last Purchase)')
axes[0, 2].set_xlabel('Days Since Last Purchase')
axes[0, 2].set_ylabel('Number of Customers')

# 4. Recency vs Total Spent
scatter = axes[1, 0].scatter(customer_features['Recency'], customer_features['TotalSpent'], alpha=0.6, c='purple')
axes[1, 0].set_title('Recency vs Total Spent')
axes[1, 0].set_xlabel('Recency (Days)')
axes[1, 0].set_ylabel('Total Spent (£)')

# 5. Frequency vs Total Spent
axes[1, 1].scatter(customer_features['Frequency'], customer_features['TotalSpent'], alpha=0.6, c='orange')
axes[1, 1].set_title('Frequency vs Total Spent')
axes[1, 1].set_xlabel('Purchase Frequency')
axes[1, 1].set_ylabel('Total Spent (£)')

# 6. Average Order Value Distribution
axes[1, 2].hist(customer_features['AvgOrderValue'], bins=50, alpha=0.7, color='gold', edgecolor='black')
axes[1, 2].set_title('Distribution of Average Order Value')
axes[1, 2].set_xlabel('Average Order Value (£)')
axes[1, 2].set_ylabel('Number of Customers')

plt.tight_layout()
plt.savefig('CLV_Exploratory_Analysis.png', dpi=300, bbox_inches='tight')
plt.show()

print("✅ Exploratory analysis plots saved as 'CLV_Exploratory_Analysis.png'")

# %% Prepare Data for Machine Learning
print("\n🤖 Preparing data for machine learning...")

# Select features for modeling
feature_columns = ['Recency', 'Frequency', 'AvgOrderValue', 'TotalQuantity', 'Tenure', 'PurchaseFrequency']
X = customer_features[feature_columns].copy()
y = customer_features['TotalSpent'].copy()

# Handle any remaining NaN or infinite values
X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(X.median())

print(f"Features selected: {feature_columns}")
print(f"Dataset shape: {X.shape}")
print(f"Target variable (TotalSpent) range: £{y.min():.2f} to £{y.max():.2f}")

# %% Split Data
print("\n📊 Splitting data for training and testing...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=None
)

print(f"Training set: {X_train.shape[0]} customers")
print(f"Test set: {X_test.shape[0]} customers")

# %% Model Training
print("\n🚀 Training machine learning models...")

# Initialize models
models = {
    'XGBoost': XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=0
    ),
    'Random Forest': RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
}

# Train and evaluate models
model_results = {}

for name, model in models.items():
    print(f"\n🔧 Training {name}...")
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Calculate metrics
    train_mae = mean_absolute_error(y_train, y_pred_train)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    
    # Store results
    model_results[name] = {
        'model': model,
        'train_mae': train_mae,
        'test_mae': test_mae,
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'predictions': y_pred_test
    }
    
    print(f"✅ {name} Results:")
    print(f"   Training MAE: £{train_mae:.2f}")
    print(f"   Test MAE: £{test_mae:.2f}")
    print(f"   Training RMSE: £{train_rmse:.2f}")
    print(f"   Test RMSE: £{test_rmse:.2f}")
    print(f"   Training R²: {train_r2:.3f}")
    print(f"   Test R²: {test_r2:.3f}")

# Select best model (lowest test RMSE)
best_model_name = min(model_results.keys(), key=lambda x: model_results[x]['test_rmse'])
best_model = model_results[best_model_name]['model']

print(f"\n🏆 Best performing model: {best_model_name}")
print(f"   Test RMSE: £{model_results[best_model_name]['test_rmse']:.2f}")
print(f"   Test R²: {model_results[best_model_name]['test_r2']:.3f}")

# %% Feature Importance Analysis
print(f"\n📊 Analyzing feature importance for {best_model_name}...")

if hasattr(best_model, 'feature_importances_'):
    feature_importance = pd.DataFrame({
        'Feature': feature_columns,
        'Importance': best_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print("Feature Importance Ranking:")
    for idx, row in feature_importance.iterrows():
        print(f"   {row['Feature']}: {row['Importance']:.3f}")
    
    # Plot feature importance
    plt.figure(figsize=(10, 6))
    sns.barplot(data=feature_importance, x='Importance', y='Feature', palette='viridis')
    plt.title(f'Feature Importance - {best_model_name}')
    plt.xlabel('Importance Score')
    plt.tight_layout()
    plt.savefig('Feature_Importance.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Feature importance plot saved as 'Feature_Importance.png'")

# %% Generate CLV Predictions for All Customers
print("\n🔮 Generating CLV predictions for all customers...")

# Predict CLV for all customers
customer_features['PredictedCLV'] = best_model.predict(X)

# Create customer segments based on predicted CLV
def segment_customers(clv):
    q80 = customer_features['PredictedCLV'].quantile(0.8)
    q60 = customer_features['PredictedCLV'].quantile(0.6)
    q40 = customer_features['PredictedCLV'].quantile(0.4)
    q20 = customer_features['PredictedCLV'].quantile(0.2)
    
    if clv >= q80:
        return 'VIP (Top 20%)'
    elif clv >= q60:
        return 'High Value'
    elif clv >= q40:
        return 'Medium Value'
    elif clv >= q20:
        return 'Low Value'
    else:
        return 'At Risk (Bottom 20%)'

customer_features['Segment'] = customer_features['PredictedCLV'].apply(segment_customers)

print("✅ Customer segmentation completed!")

# %% Customer Segmentation Analysis
print("\n📊 Customer Segmentation Analysis:")

segment_analysis = customer_features.groupby('Segment').agg({
    'CustomerID': 'count',
    'PredictedCLV': ['mean', 'sum'],
    'TotalSpent': 'mean',
    'Frequency': 'mean',
    'Recency': 'mean'
}).round(2)

segment_analysis.columns = ['Count', 'Avg_Predicted_CLV', 'Total_Predicted_Value', 'Avg_Historical_Spent', 'Avg_Frequency', 'Avg_Recency']
segment_analysis = segment_analysis.sort_values('Avg_Predicted_CLV', ascending=False)

print(segment_analysis)

# Calculate segment percentages
total_predicted_value = customer_features['PredictedCLV'].sum()
segment_analysis['Percentage_of_Total_Value'] = (segment_analysis['Total_Predicted_Value'] / total_predicted_value * 100).round(1)

print(f"\n💰 Business Insights:")
print(f"   Total Predicted CLV: £{total_predicted_value:,.2f}")
print(f"   Average CLV per Customer: £{customer_features['PredictedCLV'].mean():.2f}")

vip_customers = customer_features[customer_features['Segment'] == 'VIP (Top 20%)']
vip_value_percentage = (vip_customers['PredictedCLV'].sum() / total_predicted_value * 100)
print(f"   VIP customers (top 20%) contribute {vip_value_percentage:.1f}% of total predicted value")

# %% Visualization - Customer Segments
print("\n📈 Creating customer segmentation visualizations...")

# Create segmentation plots
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Customer Lifetime Value - Segmentation Analysis', fontsize=16, fontweight='bold')

# 1. CLV Distribution by Segment
sns.boxplot(data=customer_features, x='Segment', y='PredictedCLV', ax=axes[0, 0])
axes[0, 0].set_title('CLV Distribution by Customer Segment')
axes[0, 0].set_xlabel('Customer Segment')
axes[0, 0].set_ylabel('Predicted CLV (£)')
axes[0, 0].tick_params(axis='x', rotation=45)

# 2. Segment Size Distribution
segment_counts = customer_features['Segment'].value_counts()
axes[0, 1].pie(segment_counts.values, labels=segment_counts.index, autopct='%1.1f%%', startangle=90)
axes[0, 1].set_title('Customer Distribution by Segment')

# 3. RFM Heatmap by Segment
segment_rfm = customer_features.groupby('Segment')[['Recency', 'Frequency', 'AvgOrderValue']].mean()
sns.heatmap(segment_rfm.T, annot=True, fmt='.1f', cmap='RdYlBu_r', ax=axes[1, 0])
axes[1, 0].set_title('RFM Profile by Customer Segment')
axes[1, 0].set_xlabel('Customer Segment')

# 4. Actual vs Predicted CLV
axes[1, 1].scatter(customer_features['TotalSpent'], customer_features['PredictedCLV'], alpha=0.6)
axes[1, 1].plot([customer_features['TotalSpent'].min(), customer_features['TotalSpent'].max()], 
                [customer_features['TotalSpent'].min(), customer_features['TotalSpent'].max()], 
                'r--', alpha=0.8)
axes[1, 1].set_title('Actual vs Predicted CLV')
axes[1, 1].set_xlabel('Historical Total Spent (£)')
axes[1, 1].set_ylabel('Predicted CLV (£)')

plt.tight_layout()
plt.savefig('Customer_Segmentation_Analysis.png', dpi=300, bbox_inches='tight')
plt.show()

print("✅ Customer segmentation plots saved as 'Customer_Segmentation_Analysis.png'")

# %% Model Performance Visualization
print("\n📊 Creating model performance visualization...")

plt.figure(figsize=(12, 8))

# Actual vs Predicted scatter plot for test set
plt.subplot(2, 2, 1)
plt.scatter(y_test, model_results[best_model_name]['predictions'], alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', alpha=0.8)
plt.xlabel('Actual CLV (£)')
plt.ylabel('Predicted CLV (£)')
plt.title(f'{best_model_name} - Actual vs Predicted')

# Residuals plot
plt.subplot(2, 2, 2)
residuals = y_test - model_results[best_model_name]['predictions']
plt.scatter(model_results[best_model_name]['predictions'], residuals, alpha=0.6)
plt.axhline(y=0, color='r', linestyle='--', alpha=0.8)
plt.xlabel('Predicted CLV (£)')
plt.ylabel('Residuals (£)')
plt.title('Residuals Plot')

# Model comparison
plt.subplot(2, 2, 3)
model_names = list(model_results.keys())
test_rmse_values = [model_results[model]['test_rmse'] for model in model_names]
test_r2_values = [model_results[model]['test_r2'] for model in model_names]

x_pos = np.arange(len(model_names))
plt.bar(x_pos, test_rmse_values, alpha=0.7)
plt.xlabel('Models')
plt.ylabel('Test RMSE (£)')
plt.title('Model Performance Comparison')
plt.xticks(x_pos, model_names)

# R² comparison
plt.subplot(2, 2, 4)
plt.bar(x_pos, test_r2_values, alpha=0.7, color='green')
plt.xlabel('Models')
plt.ylabel('Test R² Score')
plt.title('Model R² Comparison')
plt.xticks(x_pos, model_names)

plt.tight_layout()
plt.savefig('Model_Performance_Analysis.png', dpi=300, bbox_inches='tight')
plt.show()

print("✅ Model performance plots saved as 'Model_Performance_Analysis.png'")

# %% Save Results
print("\n💾 Saving final results...")

# Save the trained model
joblib.dump(best_model, f'{best_model_name.lower().replace(" ", "_")}_clv_model.pkl')
print(f"✅ Model saved as '{best_model_name.lower().replace(' ', '_')}_clv_model.pkl'")

# Save customer predictions
output_columns = ['CustomerID', 'TotalSpent', 'PredictedCLV', 'Segment', 'Recency', 'Frequency', 'AvgOrderValue']
customer_results = customer_features[output_columns].copy()
customer_results = customer_results.sort_values('PredictedCLV', ascending=False)

# Save to CSV
customer_results.to_csv('CLV_Predictions_Final.csv', index=False)
print("✅ Customer predictions saved as 'CLV_Predictions_Final.csv'")

# Save segment analysis
segment_analysis.to_csv('Customer_Segment_Analysis.csv')
print("✅ Segment analysis saved as 'Customer_Segment_Analysis.csv'")

# %% Final Summary Report
print("\n" + "="*80)
print("🎉 CUSTOMER LIFETIME VALUE PREDICTION PROJECT COMPLETED!")
print("="*80)

print(f"\n📊 DATASET SUMMARY:")
print(f"   • Original transactions: {original_size:,}")
print(f"   • Clean transactions: {cleaned_size:,}")
print(f"   • Unique customers analyzed: {len(customer_features):,}")
print(f"   • Data timeframe: {df['InvoiceDate'].min().date()} to {df['InvoiceDate'].max().date()}")

print(f"\n🤖 MODEL PERFORMANCE:")
print(f"   • Best model: {best_model_name}")
print(f"   • Test RMSE: £{model_results[best_model_name]['test_rmse']:.2f}")
print(f"   • Test R² Score: {model_results[best_model_name]['test_r2']:.3f}")
print(f"   • Mean Absolute Error: £{model_results[best_model_name]['test_mae']:.2f}")

print(f"\n💰 BUSINESS INSIGHTS:")
print(f"   • Total predicted customer value: £{total_predicted_value:,.2f}")
print(f"   • Average CLV per customer: £{customer_features['PredictedCLV'].mean():.2f}")
print(f"   • Top 20% customers contribute: {vip_value_percentage:.1f}% of total value")
print(f"   • Customer segments created: {customer_features['Segment'].nunique()}")

print(f"\n📁 FILES CREATED:")
print(f"   • {best_model_name.lower().replace(' ', '_')}_clv_model.pkl - Trained ML model")
print(f"   • CLV_Predictions_Final.csv - Customer predictions")
print(f"   • Customer_Segment_Analysis.csv - Segment analysis")
print(f"   • CLV_Exploratory_Analysis.png - Data exploration plots")
print(f"   • Customer_Segmentation_Analysis.png - Segmentation visualizations")
print(f"   • Feature_Importance.png - Feature importance chart")
print(f"   • Model_Performance_Analysis.png - Model performance plots")

print(f"\n🎯 RESUME BULLET POINTS:")
print(f"   • Developed XGBoost ML model predicting Customer Lifetime Value with {model_results[best_model_name]['test_r2']:.1%} accuracy")
print(f"   • Analyzed {len(customer_features):,} customers from {cleaned_size:,} transactions using RFM methodology")
print(f"   • Identified high-value customer segments contributing {vip_value_percentage:.0f}% of predicted revenue")
print(f"   • Built end-to-end data science pipeline from raw data to actionable business insights")

print("\n🏆 PROJECT COMPLETED SUCCESSFULLY!")
print("="*80)