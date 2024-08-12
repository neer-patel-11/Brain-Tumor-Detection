from flask import Flask, render_template, request, redirect, url_for
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import os

app = Flask(__name__)

# Load your trained model
model = load_model('brain_tumor_model.h5')

# Define the path for uploaded images
UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the file from post request
        file = request.files['file']

        # Save the file to the upload folder
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        # Load and preprocess the image
        image = load_img(filepath, target_size=(224, 224))  # Change the target size to match your model's input size
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)

        # Make prediction
        prediction = model.predict(image)

        # Process the prediction result (e.g., round or use threshold to determine output)
        result = "Tumor Detected" if prediction[0][0] > 0.5 else "No Tumor Detected"

        return render_template('result.html', prediction=result, image_path=filepath)

if __name__ == "__main__":
    app.run(debug=True)
