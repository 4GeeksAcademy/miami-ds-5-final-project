from flask import Flask, request, render_template, url_for
from pickle import load
from PIL import Image
import numpy as np
import joblib
import io
import os

app = Flask(__name__)

# Set up paths
UPLOAD_FOLDER = 'static/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load KMeans clustering model
kmeans_model = load(open("./models/kmodel.dat", "rb"))

# Load your trained models
model_dict = {i: joblib.load(f'./.venv/{i}Model95acc.joblib') for i in range(4)}

# Dictionary to map class indices to class names and descriptions
class_dict = {
    0: ('adipose', 'Adipose tissue, which consists of fat cells.'),
    1: ('background', 'Background, which represents non-relevant areas.'),
    2: ('debris', 'Debris, including particles or waste materials.'),
    3: ('lymphocytes', 'Lymphocytes, a type of white blood cell.'),
    4: ('mucus', 'Mucus, a slippery secretion produced by mucous membranes.'),
    5: ('smooth muscle', 'Smooth muscle tissue, found in various organs.'),
    6: ('normal colon mucosa', 'Normal colon mucosa, the inner lining of the colon.'),
    7: ('cancer-associated stroma', 'Cancer-associated stroma, the supportive tissue around cancer cells.'),
    8: ('colorectal adenocarcinoma epithelium', 'Colorectal adenocarcinoma epithelium, a type of cancerous tissue.')
}

# Function to preprocess the image
def preprocess_image(image):
    image = image.resize((28, 28))
    image_array = np.array(image) / 255.0  # Normalize pixel values to [0, 1]
    return image_array

# Function to extract RGB statistics from an image
def extract_rgb_statistics(image):
    image_array = np.array(image)
    r_channel = image_array[:, :, 0].flatten()
    g_channel = image_array[:, :, 1].flatten()
    b_channel = image_array[:, :, 2].flatten()
    r_stats = [np.mean(r_channel), np.std(r_channel), np.min(r_channel), np.max(r_channel)]
    g_stats = [np.mean(g_channel), np.std(g_channel), np.min(g_channel), np.max(g_channel)]
    b_stats = [np.mean(b_channel), np.std(b_channel), np.min(b_channel), np.max(b_channel)]
    features = np.array(r_stats + g_stats + b_stats)
    return features

@app.route("/", methods=["GET", "POST"])
def index():
    class_prediction = None
    description = None
    error_message = None

    if request.method == "POST":
        try:
            image_file = request.files.get('fileInput')

            if image_file:
                # Save the uploaded image
                image_path = os.path.join(UPLOAD_FOLDER, 'uploaded_image.jpg')
                image = Image.open(io.BytesIO(image_file.read())).convert('RGB')
                image.save(image_path)

                # Preprocess and predict
                image_array = preprocess_image(image)
                image_array = image_array.reshape(1, 28, 28, 3)
                rgb_features = extract_rgb_statistics(image)
                cluster_label = kmeans_model.predict([rgb_features])[0]
                model = model_dict.get(cluster_label)
                if model is None:
                    class_prediction = "Model not found for the predicted cluster."
                else:
                    class_probs = model.predict(image_array)
                    predicted_class_index = np.argmax(class_probs)
                    class_name, class_description = class_dict.get(predicted_class_index, ("Unknown class", "No description available."))
                    class_prediction = class_name
                    description = class_description
                    print(f"Class Probabilities: {class_probs}")
                    print(f"Predicted Class Index: {predicted_class_index}")
                    print(f"Class Prediction: {class_name}")

        except Exception as e:
            error_message = str(e)
            print(f"Error: {error_message}")

    return render_template("index.html", prediction=class_prediction, description=description, error=error_message)

if __name__ == "__main__":
    app.run(debug=True)
