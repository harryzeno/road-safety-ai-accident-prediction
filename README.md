# Road Safety AI: Traffic Accident Severity Prediction

# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load sample dataset (replace with your real data file if available)
try:
    df = pd.read_csv('accidents_data.csv')
except FileNotFoundError:
    print("Sample data not found. Creating mock dataset for demo purposes.")
    df = pd.DataFrame({
        'Time': pd.date_range(start='2021-01-01', periods=100, freq='H'),
        'Weather_Condition': np.random.choice(['Clear', 'Rainy', 'Fog'], 100),
        'Road_Surface': np.random.choice(['Dry', 'Wet'], 100),
        'Light_Condition': np.random.choice(['Daylight', 'Dark'], 100),
        'Accident_Severity': np.random.choice(['Minor', 'Serious', 'Fatal'], 100)
    })

# Basic preprocessing
# Drop duplicates
df.drop_duplicates(inplace=True)

# Fill missing values (if any)
df.fillna(method='ffill', inplace=True)

# Convert 'Time' to datetime and create time-based features
if 'Time' in df.columns:
    df['Time'] = pd.to_datetime(df['Time'])
    df['Hour'] = df['Time'].dt.hour
    df['DayOfWeek'] = df['Time'].dt.dayofweek
else:
    df['Hour'] = np.random.randint(0, 24, len(df))
    df['DayOfWeek'] = np.random.randint(0, 7, len(df))

# Encode categorical variables
categorical = ['Weather_Condition', 'Road_Surface', 'Light_Condition']
le = LabelEncoder()
for col in categorical:
    if col in df.columns:
        df[col] = le.fit_transform(df[col].astype(str))

# Encode target variable if it's categorical
y = df['Accident_Severity']
if y.dtype == 'O':
    y = le.fit_transform(y)

# Select features
features = ['Weather_Condition', 'Road_Surface', 'Light_Condition', 'Hour', 'DayOfWeek']
X = df[features]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train Random Forest
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# Train XGBoost
xgb = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss')
xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)

# Evaluation function
def evaluate(model_name, y_true, y_pred):
    print(f"\n--- {model_name} Results ---")
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
    print("Classification Report:\n", classification_report(y_true, y_pred))

evaluate("Random Forest", y_test, y_pred_rf)
evaluate("XGBoost", y_test, y_pred_xgb)

# Feature importance from XGBoost
importances = xgb.feature_importances_
features = X.columns
plt.figure(figsize=(8, 4))
sns.barplot(x=importances, y=features)
plt.title("Feature Importance - XGBoost")
plt.tight_layout()
plt.show()


