import streamlit as st
import pandas as pd

from src.regression_utils import (
    load_data,
    train_model
)

st.set_page_config(
    page_title="Diamond Price Prediction",
    layout="wide"
)

st.title("💎 Diamond Price Prediction using Stacking Regressor")

st.write(
    "Enter diamond specifications to predict the price."
)

df = load_data()

@st.cache_resource
def get_model():
    return train_model()

model = get_model()

st.subheader("Diamond Specifications")

col1, col2 = st.columns(2)

with col1:

    carat = st.number_input(
        "Carat",
        min_value=0.1,
        max_value=5.0,
        value=1.0
    )

    cut = st.selectbox(
        "Cut",
        sorted(df["cut"].unique())
    )

    color = st.selectbox(
        "Color",
        sorted(df["color"].unique())
    )

    clarity = st.selectbox(
        "Clarity",
        sorted(df["clarity"].unique())
    )

with col2:

    depth = st.number_input(
        "Depth",
        min_value=40.0,
        max_value=80.0,
        value=61.5
    )

    table = st.number_input(
        "Table",
        min_value=40.0,
        max_value=100.0,
        value=57.0
    )

    x = st.number_input(
        "Length (x)",
        min_value=0.0,
        max_value=15.0,
        value=5.5
    )

    y = st.number_input(
        "Width (y)",
        min_value=0.0,
        max_value=15.0,
        value=5.5
    )

    z = st.number_input(
        "Depth (z)",
        min_value=0.0,
        max_value=15.0,
        value=3.5
    )

if st.button("Predict Price"):

    input_df = pd.DataFrame({

        "carat": [carat],
        "cut": [cut],
        "color": [color],
        "clarity": [clarity],
        "depth": [depth],
        "table": [table],
        "x": [x],
        "y": [y],
        "z": [z]
    })

    prediction = model.predict(
        input_df
    )[0]

    st.success(
        f"Predicted Diamond Price: ${prediction:,.2f}"
    )

st.markdown("---")

st.subheader("Dataset Preview")

st.dataframe(
    df.head()
)