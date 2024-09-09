from flask import Flask, request, render_template
from pickle import load
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
import joblib
import io

app = Flask(__name__)

# Load KMeans clustering model
kmeans_model = load(open("./models/kmodel.dat", "rb"))

# Load your trained models
model_dict = {i: joblib.load(f'./.venv/{i}Model95acc.joblib') for i in range(4)}

class_dict = {
    0: 'adipose', 
    1: 'background', 
    2: 'debris', 
    3: 'lymphocytes', 
    4: 'mucus', 
    5: 'smooth muscle', 
    6: 'normal colon mucosa', 
    7: 'cancer-associated stroma', 
    8: 'colorectal adenocarcinoma epithelium'
}

# Function to preprocess the image
def preprocess_image(image):
    # Resize image to 28x28
    image = image.resize((28, 28))
    
    # Convert image to array and normalize
    image_array = np.array(image) / 255.0  # Normalize pixel values to [0, 1]
    
    return image_array

# Function to extract RGB statistics from an image
def extract_rgb_statistics(image):
    # Convert the image into an array (height, width, 3)
    image_array = np.array(image)

    # Separate into R, G, B channels
    r_channel = image_array[:, :, 0].flatten()
    g_channel = image_array[:, :, 1].flatten()
    b_channel = image_array[:, :, 2].flatten()

    # Compute statistics (mean, std, min, max) for each channel
    r_stats = [np.mean(r_channel), np.std(r_channel), np.min(r_channel), np.max(r_channel)]
    g_stats = [np.mean(g_channel), np.std(g_channel), np.min(g_channel), np.max(g_channel)]
    b_stats = [np.mean(b_channel), np.std(b_channel), np.min(b_channel), np.max(b_channel)]

    # Concatenate all statistics into a single feature vector
    features = np.array(r_stats + g_stats + b_stats)
    return features

@app.route("/", methods=["GET", "POST"])
def index():
    cluster_description = None
    error_message = None

    if request.method == "POST":
        try:
            image_file = request.files.get('fileInput')

            if image_file:
                # Load the image directly from memory in RGB format
                image = Image.open(io.BytesIO(image_file.read())).convert('RGB')

                # Step 1: Preprocess the image
                image_array = preprocess_image(image)
                image_array = image_array.reshape(1, 28, 28, 3)  # Reshape to (1, 28, 28, 3) for prediction

                # Step 2: Extract RGB statistics
                rgb_features = extract_rgb_statistics(image)

                # Step 3: Predict the group/cluster using KMeans
                cluster_label = kmeans_model.predict([rgb_features])[0]

                # Convert cluster index to a descriptive name if needed
                cluster_description = f"Cluster {cluster_label}"

        except Exception as e:
            error_message = str(e)
            print(f"Error: {error_message}")

    return render_template("index.html", cluster=cluster_description, error=error_message)

if __name__ == "__main__":
    app.run(debug=True)
