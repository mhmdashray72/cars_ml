import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Cars Analytics App",
    page_icon="🚗",
    layout="wide"
)

# =========================
# LOAD DATA
# =========================
@st.cache_data

def load_data():
    df = pd.read_csv("cars_dataset.csv")

    # Drop unnecessary columns
    drop_cols = ["Unnamed: 0", "vin", "lot"]

    for col in drop_cols:
        if col in df.columns:
            df.drop(col, axis=1, inplace=True)

    # Handle missing values
    df["mileage"] = df["mileage"].fillna(df["mileage"].median())

    # Remove duplicates
    df = df.drop_duplicates()

    return df


df = load_data()

# =========================
# MODEL TRAINING
# =========================

X = df.drop("price", axis=1)
y = df["price"]

categorical_cols = X.select_dtypes(include="object").columns.tolist()
numerical_cols = X.select_dtypes(exclude="object").columns.tolist()

numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
        st.balloons()
