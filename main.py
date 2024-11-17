import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb

# Configuration dictionary for different use cases
DATASET_CONFIG = {
    'bank': {
        'file': 'bank_churn.csv',
        'title': 'Bank Customer Churn Prediction',
        'target': 'Exited',
        'drop_columns': ['RowNumber', 'CustomerId', 'Surname'],
        'preprocessing': None
    },
    'telco': {
        'file': 'WA_Fn-UseC_-Telco-Customer-Churn.csv',
        'title': 'Telco Customer Churn Prediction',
        'target': 'Churn',
        'drop_columns': ['customerID'],
        'preprocessing': lambda df: df.assign(Churn=df['Churn'].map({'No': 0, 'Yes': 1}))
    },
    'ecommerce': {
        'file': 'Ecomm.csv',
        'title': 'E-Commerce Customer Churn Prediction',
        'target': 'Churn',
        'drop_columns': ['CustomerID', 'PreferredLoginDevice', 'CouponUsed'],
        'preprocessing': None
    }
}

# Let user select the dataset
dataset_type = st.sidebar.selectbox(
    'Select Dataset',
    list(DATASET_CONFIG.keys()),
    format_func=lambda x: DATASET_CONFIG[x]['title']
)

# Load configuration
config = DATASET_CONFIG[dataset_type]

# Set title
st.title(config['title'])

# Load the data
@st.cache_data(show_spinner=False)
def load_data(file_path, _preprocessing, drop_columns):
    df_churn = pd.read_csv(file_path)
    if _preprocessing:
        df_churn = _preprocessing(df_churn)
    df_churn = df_churn.drop(drop_columns, axis=1)
    return df_churn

# Load the selected dataset
df_churn = load_data(config['file'], config['preprocessing'], config['drop_columns'])

# Display data overview
st.header('Churn Data Overview')
st.write('Data Dimension: ' + str(df_churn.shape[0]) + ' rows and ' + str(df_churn.shape[1]) + ' columns.')
st.dataframe(df_churn)

# Prepare features and target
if config['target'] not in df_churn.columns:
    st.error(f"Target column '{config['target']}' not found in dataset!")
    st.stop()

X = df_churn.drop(config['target'], axis=1)
y = df_churn[config['target']]

# Preprocess features
num_cols = X.select_dtypes(include=['int64', 'float64']).columns
cat_cols = X.select_dtypes(include=['object']).columns

ordinal_encoder = OrdinalEncoder()
X[cat_cols] = ordinal_encoder.fit_transform(X[cat_cols])

transformer = RobustScaler()
X[num_cols] = transformer.fit_transform(X[num_cols])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model selection
classifier_name = st.sidebar.selectbox(
    'Select a Classifier',
    ('Random Forest', 'Decision Tree', 'XGBoost')
)

def get_classifier(clf_name):
    if clf_name == 'Random Forest':
        clf = RandomForestClassifier()
    elif clf_name == 'Decision Tree':
        clf = DecisionTreeClassifier()
    else:
        clf = xgb.XGBClassifier()
    return clf

clf = get_classifier(classifier_name)

# Fit model
clf.fit(X_train, y_train)

# User input for prediction
st.sidebar.header('User Input for Prediction')
user_input = {}
for col in X.columns:
    if col in num_cols:
        user_input[col] = st.sidebar.number_input(f'Enter {col}', value=float(df_churn[col].mean()))
    else:
        user_input[col] = st.sidebar.selectbox(f'Select {col}', options=df_churn[col].unique())

# Convert user input to DataFrame
user_input_df = pd.DataFrame([user_input])

# Preprocess user input for prediction
user_input_df[cat_cols] = ordinal_encoder.transform(user_input_df[cat_cols])
user_input_df[num_cols] = transformer.transform(user_input_df[num_cols])

# Make prediction on user input
if st.sidebar.button('Predict Churn'):
    user_pred = clf.predict(user_input_df)
    user_pred_proba = clf.predict_proba(user_input_df)[0][user_pred[0]]
    
    if user_pred[0] == 1:
        st.write(f"Predicted Churn: Yes (Probability: {user_pred_proba:.2f})")
        st.write("The customer is likely to leave the company's services.")
    else:
        st.write(f"Predicted Churn: No (Probability: {user_pred_proba:.2f})")
        st.write("The customer will likely not leave the company's services.")