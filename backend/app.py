import logging
from flask import Flask, request, jsonify, render_template
import os
import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LeakyReLU, Conv2D, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Initialize Flask app
app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.DEBUG)

# Function to build the Discriminator Model
def build_discriminator(input_shape=(128, 128, 3)):
    model = Sequential([
        Conv2D(64, kernel_size=3, strides=2, padding='same', input_shape=input_shape),
        LeakyReLU(alpha=0.2),
        Conv2D(128, kernel_size=3, strides=2, padding='same'),
        LeakyReLU(alpha=0.2),
        Dropout(0.3),  # Added Dropout to prevent overfitting
        Flatten(),
        Dense(1, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0002, beta_1=0.5), metrics=['accuracy'])
    return model

# Load pre-trained model weights
discriminator = build_discriminator()
weights_path = os.path.join(os.getcwd(), 'discriminator_weights_499.weights.h5')
if os.path.exists(weights_path):
    discriminator.load_weights(weights_path)
else:
    logging.error(f"Error: Weights file not found at {weights_path}")

# Function to preprocess and augment the image before prediction
def preprocess_image(image):
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    image = cv2.resize(image, (128, 128))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype('float32') / 255.0
    image = np.expand_dims(image, axis=0)
    
    augmented_image = next(datagen.flow(image, batch_size=1))
    return augmented_image

# Define the prediction function
def predict_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return "Error: Invalid image file"

    preprocessed_image = preprocess_image(image)
    prediction = discriminator.predict(preprocessed_image)
    confidence = prediction[0][0]
    
    threshold = 0.45
    if confidence > threshold:
        result = f"Real (Confidence: {confidence:.2f})"
    else:
        result = f"Fake (Confidence: {confidence:.2f})"
    
    logging.info(f"Prediction result: {result} for image {image_path}")
    return result

# Define the home route for uploading the image
@app.route('/')
def home():
    return render_template('index.html')  # Render the upload form

# Route to handle the file upload and prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if not os.path.exists('uploads'):
        os.makedirs('uploads')

    filename = os.path.join('uploads', file.filename)
    file.save(filename)

    # Predict whether the uploaded image is real or fake
    result = predict_image(filename)
    
    # Return the result as JSON response
    return jsonify({'result': result, 'image_url': filename})

if __name__ == '__main__':
    # Run the app
    app.run(debug=True)
