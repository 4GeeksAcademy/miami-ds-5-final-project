from flask import Flask, request, render_template, send_from_directory
import torchvision.transforms as transforms
import torch
from pickle import load
from PIL import Image
import io
import os

# Import your model definition
from model_module import SimpleCNN

app = Flask(__name__)

# Ensure the 'static/uploads' directory exists
UPLOAD_FOLDER = 'static/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load your trained model
model = load(open("trained_cnn.pkl", "rb"))

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

# Define the preprocessing transformation
data_transform = transforms.Compose([
    transforms.Resize((28, 28)),       # Ensure the image is 28x28
    transforms.ToTensor(),             # Convert image to tensor
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Same normalization as used during training
])

@app.route("/", methods=["GET", "POST"])
def index():
    pred_class = None
    uploaded_image_path = None

    if request.method == "POST":
        image_file = request.files['fileInput']
        
        if image_file:
            # Save the uploaded image to the static/uploads directory
            filename = image_file.filename
            image_path = os.path.join(UPLOAD_FOLDER, filename)
            image_file.save(image_path)
            uploaded_image_path = image_path  # Save the path to use in HTML

            # Load the image for processing
            image = Image.open(image_path).convert('RGB')  # Convert to RGB to match model input

            # Apply the preprocessing transformations
            image_tensor = data_transform(image).unsqueeze(0)  # Add batch dimension

            # Use the model to make a prediction
            model.eval()  # Set model to evaluation mode
            with torch.no_grad():
                output = model(image_tensor)
                _, predicted = torch.max(output, 1)
                pred_label = predicted.item()
                pred_class = class_dict[pred_label]

    return render_template("index.html", prediction=pred_class, image_path=uploaded_image_path)

if __name__ == "__main__":
    app.run(debug=True)





