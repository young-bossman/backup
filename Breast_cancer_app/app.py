from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import pickle
import os

app = Flask(__name__)

# Path to the trained machine learning model
MODEL_PATH = 'model/model_saved.pkl'

# Load the trained model from the specified path
with open(MODEL_PATH, 'rb') as file:
    model = pickle.load(file)

# Route to render the main page with options
@app.route('/')
def index():
    return render_template('index.html')

# Route to render the manual input form page
@app.route('/manual_input')
def manual_input():
    return render_template('manual_input.html')

# Route to render the file upload page
@app.route('/file_upload')
def file_upload():
    return render_template('file_upload.html')

# Route to handle file upload and predict results
@app.route('/predict', methods=['POST'])
def predict():
    # Check if the file part is in the request
    if 'file' not in request.files:
        return "No file part"
    
    file = request.files['file']
    
    # Check if a file is selected
    if file.filename == '':
        return "No selected file"
    
    if file:
        # Save the uploaded file
        file_path = os.path.join('uploads', file.filename)
        file.save(file_path)
        
        # Load the data from the file into a DataFrame
        data = pd.read_excel(file_path)
        
        # Predict using the loaded model
        predictions = model.predict(data)
        
        # Append predictions to the data
        data['Prediction'] = predictions
        
        # Prepare the result text
        results = []
        for index, prediction in enumerate(predictions):
            if prediction == 0:
                results.append(f'Row {index + 1}: The tissue is Malignant (Cancerous)')
            else:
                results.append(f'Row {index + 1}: The tissue is Benign (Non-cancerous)')
        
        # Render the results page with the prediction details
        result_text = '\n'.join(results)
        return render_template('result.html', result=result_text)

# Route to handle manual input prediction
@app.route('/single_predict', methods=['POST'])
def single_predict():
    # Retrieve input data from the form
    input_data = [
        request.form.get('mean_radius'), request.form.get('mean_texture'), request.form.get('mean_perimeter'),
        request.form.get('mean_area'), request.form.get('mean_smoothness'), request.form.get('mean_compactness'),
        request.form.get('mean_concavity'), request.form.get('mean_concave_points'), request.form.get('mean_symmetry'),
        request.form.get('mean_fractal_dimension'), request.form.get('radius_error'), request.form.get('texture_error'),
        request.form.get('perimeter_error'), request.form.get('area_error'), request.form.get('smoothness_error'),
        request.form.get('compactness_error'), request.form.get('concavity_error'), request.form.get('concave_points_error'),
        request.form.get('symmetry_error'), request.form.get('fractal_dimension_error'), request.form.get('worst_radius'),
        request.form.get('worst_texture'), request.form.get('worst_perimeter'), request.form.get('worst_area'),
        request.form.get('worst_smoothness'), request.form.get('worst_compactness'), request.form.get('worst_concavity'),
        request.form.get('worst_concave_points'), request.form.get('worst_symmetry'), request.form.get('worst_fractal_dimension')
    ]
    
    # Convert the input data to a numpy array
    input_data_as_numpy_array = np.asarray(input_data, dtype=float)
    
    # Reshape the array for a single prediction
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    
    # Make prediction using the model
    prediction = model.predict(input_data_reshaped)
    
    # Determine the prediction result
    result = 'The tissue is Malignant (Cancerous)' if prediction[0] == 0 else 'The tissue is Benign (Non-cancerous)'
    
    # Render the result page
    return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
