
import os

import streamlit as st
import pandas as pd
import requests

# Base URL of the Flask backend
BACKEND_URL = os.getenv("BACKEND_URL", "http://backend:7860").rstrip("/")


def post_to_backend(endpoint, **kwargs):
    try:
        return requests.post(
            f"{BACKEND_URL}{endpoint}",
            timeout=60,
            **kwargs
        )
    except requests.RequestException as error:
        st.error(f"Could not reach the prediction API at {BACKEND_URL}: {error}")
        return None

# Page title
st.title("SuperKart Sales Forecast")
st.write(
    "Enter the product details below to predict sales."
)

# Collect product details
Product_Weight = st.number_input(
    "Product Weight",
    min_value=0.0,
    value=20.0,
    step=0.1
)

Product_Sugar_Content = st.selectbox(
    "Product Sugar Content",
    ["Low Sugar", "No Sugar", "Regular"]
)

Product_Allocated_Area = st.number_input(
    "Product Allocated Area",
    min_value=0.0,
    value=0.027,
    step=0.001,
    format="%.3f"
)

Product_MRP = st.number_input(
    "Product MRP",
    min_value=0.0,
    value=117.08,
    step=1.0
)

Store_Size = st.selectbox(
    "Store Size",
    ["Small", "Medium", "High"]
)

Store_Location_City_Type = st.selectbox(
    "Store Location City Type",
    ["Tier 1", "Tier 2", "Tier 3"]
)

Store_Type = st.selectbox(
    "Store Type",
    ["Supermarket Type2", "Departmental Store", "Food Mart", "Supermarket Type1", "Supermarket Type3"]
)

Product_Id_char = st.selectbox(
    "Product Id Character",
    ["FD", "DR", "NC"]
)

Store_Age_Years = st.number_input(
    "Store Age in Years",
    min_value=0,
    value=12,
    step=1
)

Product_Type_Category= st.selectbox(
    "Product Type Category",
    ["Perishable", "Non Perishables"]
)


# Create JSON payload
product_data = {
    "Product_Weight": Product_Weight,
    "Product_Sugar_Content": Product_Sugar_Content,
    "Product_Allocated_Area": Product_Allocated_Area,
    "Product_MRP": Product_MRP,
    "Store_Size": Store_Size,
    "Store_Location_City_Type": Store_Location_City_Type,
    "Store_Type": Store_Type,
    "Product_Id_char": Product_Id_char,
    "Store_Age_Years": Store_Age_Years,
    "Product_Type_Category": Product_Type_Category
}

# Single Prediction
if st.button("Predict", type="primary"):

    response = post_to_backend("/v1/predict", json=product_data)

    if response is not None:
        if response.ok:
            result = response.json()

            st.success(f"Predicted Sales: {result['Prediction']:.2f}")
        else:
            st.error(
                f"Prediction API returned HTTP {response.status_code}: "
                f"{response.text[:300]}"
            )

# Batch Prediction
st.subheader("Batch Prediction")

uploaded_file = st.file_uploader(
    "Upload a CSV file",
    type=["csv"]
)

if uploaded_file is not None:

    if st.button("Predict for Batch", type="primary"):

        response = post_to_backend(
            "/v1/predictbatch",
            files={"file": uploaded_file}
        )

        if response is not None:
            if response.ok:
                results = response.json()

                st.success("Predictions completed successfully!")

                try:
                    if isinstance(results, list):
                        df = pd.DataFrame(results)
                    elif isinstance(results, dict):
                        # Check if all values are scalars
                        if all(not isinstance(v, (list, dict)) for v in results.values()):
                            df = pd.DataFrame([results])
                        else:
                            df = pd.DataFrame(results)
                    else:
                        df = pd.DataFrame({"Result": [results]})

                    st.dataframe(df, use_container_width=True)

                except Exception as e:
                    st.error(f"Unable to display results as a table: {e}")
                    st.json(results)
            else:
                st.error(
                    f"Prediction API returned HTTP {response.status_code}: "
                    f"{response.text[:300]}"
                )
