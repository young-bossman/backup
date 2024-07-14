from flask import Flask, request, render_template, send_file
import pandas as pd
import numpy as np
import pickle
import os

app = Flask(__name__)
MODEL_PATH = 'model/model_saved.pkl'

# Load the trained model
with open(MODEL_PATH, 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/manual_input')
def manual_input():
    return render_template('manual_input.html')

@app.route('/file_upload')
def file_upload():
    return render_template('file_upload.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file part"
    file = request.files['file']
    if file.filename == '':
        return "No selected file"
    if file:
        file_path = os.path.join('uploads', file.filename)
        file.save(file_path)
        data = pd.read_excel(file_path)
        predictions = model.predict(data)
        data['Prediction'] = predictions
        results = []
        for index, prediction in enumerate(predictions):
            if prediction == 0:
                results.append(f'prediction {index + 1}: The tissue is Malignant')
            else:
                results.append(f'Row {index + 1}: Benign')
        return render_template('file_upload.html', results=results)

@app.route('/single_predict', methods=['POST'])
def single_predict():
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
    input_data_as_numpy_array = np.asarray(input_data, dtype=float)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    prediction = model.predict(input_data_reshaped)
    
    result = 'The tissue is Malignant' if prediction[0] == 0 else 'Benign'
    return render_template('manual_input.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
