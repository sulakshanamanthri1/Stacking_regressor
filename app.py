import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load cleaned data
df = pd.read_csv("cleaned_data.csv")

st.title("💎 Diamond Price Prediction with Stacking Regressor")
st.write("This app predicts diamond prices using a stacking regressor.")

# Features & target
X = df.drop("price", axis=1)
y = df["price"]

# Convert categorical to numeric
X = pd.get_dummies(X, drop_first=True)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Base models
estimators = [
    ('rf', RandomForestRegressor(n_estimators=100, random_state=42)),
    ('gb', GradientBoostingRegressor(n_estimators=100, random_state=42))
]

# Stacking Regressor
stack_model = StackingRegressor(
    estimators=estimators,
    final_estimator=LinearRegression()
)

stack_model.fit(X_train, y_train)
y_pred = stack_model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)

st.subheader("📊 Model Performance")
st.write(f"Mean Squared Error: {mse:.2f}")

# User input
st.sidebar.header("Input Features")
carat = st.sidebar.slider("Carat", float(df.carat.min()), float(df.carat.max()), 0.5)
depth = st.sidebar.slider("Depth", float(df.depth.min()), float(df.depth.max()), 61.0)
table = st.sidebar.slider("Table", float(df.table.min()), float(df.table.max()), 57.0)

# Example prediction (simplified)
sample = pd.DataFrame([[carat, depth, table]], columns=["carat","depth","table"])
sample = pd.get_dummies(sample).reindex(columns=X.columns, fill_value=0)

prediction = stack_model.predict(sample)[0]
st.subheader("💰 Predicted Price")
st.write(f"${prediction:.2f}")
