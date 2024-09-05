from flask import Flask, request, render_template
from pickle import load
import torchvision.transforms as transforms
import torch
from PIL import Image
import io

app = Flask(__name__)
model = load(open("../models/kmodel.dat", "rb"))
class_dict = {
    "0": "adipose",
    "1": "Ibackground",
    "2": "debris",
    "3": "lymphocytes",
    "4": "mucus",
    "5": "smooth muscle",
    "6": "normal colon mucosa",
    "7": "cancer-associated stroma",
    "8": "colorectal adenocarcinoma epithelium"
}

# Define the preprocessing transformation
data_transform = transforms.Compose([
    transforms.Resize((28, 28)),       # Ensure the image is 28x28
    transforms.ToTensor(),             # Convert image to tensor
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Same normalization as used during training
])

@app.route("/", methods=["GET", "POST"])
def index():
    pred_class = None

    if request.method == "POST":
        image_file = request.files['fileInput']
        
        if image_file:
            # Load the image directly from memory
            image = image.open(io.BytesIO(image_file.read())).convert('RGB')  # Convert to RGB to match model input

            # TODO: Apply the preprocessing transformations & add batch dimension
            image_tensor = ()

            # Check the shape of the processed image tensor
            print(image_tensor.shape)  # Should be (1, 3, 28, 28)

            # TODO: Use the model to make a prediction

    return render_template("index.html", prediction=pred_class)

if __name__ == "__main__":
    app.run(debug=True)