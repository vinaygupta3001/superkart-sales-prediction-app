import joblib
import pandas as pd
from flask import Flask, request, jsonify
from pathlib import Path

# Initialize Flask app with a name
app = Flask("SuperKart Sales Prediction API")

# Load the trained sales prediction model
model = joblib.load(Path(__file__).with_name("superkart_model.joblib"))

MODEL_FEATURES = [
    'Product_Weight',
    'Product_Sugar_Content',
    'Product_Allocated_Area',
    'Product_MRP',
    'Store_Size',
    'Store_Location_City_Type',
    'Store_Type',
    'Product_Id_Char',
    'Store_Age_Years',
    'Product_Type_Category'
]

# Define a route for the home page
@app.get('/')
def home():
    return "Welcome to the SuperKart Sales Prediction API"

# Define an endpoint to predict sales for a single store
@app.post('/v1/predict')
@app.post('/v1/product')
@app.post('/v1/sales')
def predict_sales():
    # Get JSON data from the request
    product_data = request.get_json()

    # Extract relevant product features from the input data
    sample = {
        'Product_Weight': product_data['Product_Weight'],
        'Product_Sugar_Content': product_data['Product_Sugar_Content'],
        'Product_Allocated_Area': product_data['Product_Allocated_Area'],
        'Product_MRP': product_data['Product_MRP'],
        'Store_Size': product_data['Store_Size'],
        'Store_Location_City_Type': product_data['Store_Location_City_Type'],
        'Store_Type': product_data['Store_Type'],
        'Product_Id_Char': product_data['Product_Id_char'],
        'Store_Age_Years': product_data['Store_Age_Years'],
        'Product_Type_Category': product_data['Product_Type_Category']
    }

    # Convert the extracted data into a DataFrame
    input_data = pd.DataFrame([sample])

    # Make a sales prediction using the trained model
    prediction = model.predict(input_data).tolist()[0]


    # Return the prediction as a JSON response
    return jsonify({'Prediction': round(prediction, 2)})

# Define an endpoint to predict sales for a batch of products
@app.post('/v1/predictbatch')
@app.post('/v1/productbatch')
def predict_sales_batch():
    # Get the uploaded CSV file from the request
    file = request.files.get('file')
    if file is None:
        return jsonify({'error': 'Upload a CSV file using the "file" field.'}), 400

    # Read the file into a DataFrame
    input_data = pd.read_csv(file)

    if 'Product_Id_Char' not in input_data.columns and 'Product_Id_char' in input_data.columns:
        input_data = input_data.rename(columns={'Product_Id_char': 'Product_Id_Char'})

    missing_columns = [column for column in MODEL_FEATURES if column not in input_data.columns]
    if missing_columns:
        return jsonify({
            'error': f"CSV is missing required columns: {', '.join(missing_columns)}"
        }), 400

    # Make predictions for the batch data and convert raw predictions into a readable format
    predictions = model.predict(input_data[MODEL_FEATURES])

    return jsonify({
        str(index): [round(float(value), 2)]
        for index, value in enumerate(predictions)
    })


# Run the Flask app in debug mode
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7860, debug=True)
