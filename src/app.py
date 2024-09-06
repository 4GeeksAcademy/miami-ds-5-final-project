from flask import Flask, request, render_template
import torchvision.transforms as transforms
import torch
from pickle import load
from PIL import Image
import io

# Import your model definition
from model_module import SimpleCNN

app = Flask(__name__)

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

    if request.method == "POST":
        image_file = request.files['fileInput']
        
        if image_file:
            # Load the image directly from memory
            image = Image.open(io.BytesIO(image_file.read())).convert('RGB')  # Convert to RGB to match model input

            # Apply the preprocessing transformations
            image_tensor = data_transform(image).unsqueeze(0)  # Add batch dimension

            # Check the shape of the processed image tensor
            print(image_tensor.shape)  # Should be (1, 3, 28, 28)

            # Use the model to make a prediction
            model.eval()  # Set model to evaluation mode
            with torch.no_grad():
                output = model(image_tensor)
                _, predicted = torch.max(output, 1)
                pred_label = predicted.item()
                pred_class = class_dict[pred_label]

    return render_template("index.html", prediction=pred_class)

if __name__ == "__main__":
    app.run(debug=True)




