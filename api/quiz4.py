import matplotlib.pyplot as plt
import sys
# Import datasets, classifiers and performance metrics
from sklearn import datasets, metrics, svm
from sklearn.model_selection import train_test_split
import pdb
from joblib import dump,load
import numpy as np
# import skimage
# from skimage.transform import resize
import pandas as pd
from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer

app = Flask(__name__)

# Load SVM model
svm_model = load('./models/M23CSA011_svm_(0.001, 1).joblib')

# Load Decision Tree model
tree_model = load('./models/M23CSA011_tree_(10,).joblib')

# Load Logistic Regression model
lr_model = load("./models/M23CSA011_lr_('lbfgs',).joblib")

# Define a dictionary to map model names to their loaded models
models = {'svm': svm_model, 'tree': tree_model, 'lr': lr_model}

# Function to load models based on model_name
def load_model(model_name):
    return models.get(model_name) 

@app.route('/predict/<string:model_name>', methods=['POST'])
def compare_digits(model_name):
    try:
        model = load_model(model_name)
        # Get the two image files from the request
        data = request.get_json()  # Parse JSON data from the request body
        image1 = data.get('image', [])
        # image2 = data.get('image2', [])

        # Preprocess the images and make predictions
        digit1 = predict_digit(image1,model)
        # digit2 = predict_digit(image2)

        # Compare the predicted digits and return the result
        # result = digit1 == digit2

        return jsonify({"digit":digit1})
    except Exception as e:
        return jsonify({'error': str(e)})
    
def predict_digit(image,model):
    try:
        # Convert the input list to a numpy array and preprocess for prediction
        img_array = np.array(image, dtype=np.float32).reshape(1,-1) / 255.0
    
        prediction = model.predict(img_array)
        digit = int(np.argmax(prediction))

        return digit
    except Exception as e:
        return str(e)


if __name__ == '__main__':
    app.run()