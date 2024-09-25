from flask import Flask, request, render_template, url_for
import pandas as pd
from pickle import load
from PIL import Image
import numpy as np
import io
import os
import time
import torch
import psutil
import multiprocessing
from torchvision import transforms
from new_model_training import Encoder, ClassificationHead, GeminiContrast


app = Flask(__name__)
print('app registered')
pid = os.getpid()
process = psutil.Process(pid)

def monitor_memory():
    pid = os.getpid()
    process = psutil.Process(pid)
    
    while True:
        memory_info = process.memory_info()
        memory_usage_mb = memory_info.rss / (1024 ** 2)
        
        system_memory = psutil.virtual_memory()
        free_memory_mb = system_memory.available / (1024 ** 2)
        
        print(f"Process memory usage: {memory_usage_mb:.2f} MB")
        print(f"System free memory: {free_memory_mb:.2f} MB")
        time.sleep(5)


for i in ['cell', 'cancer']:
    webp_image = Image.open(f'static/uploads/{i}.webp')
    rgba_image = webp_image.convert("RGBA")
    data = rgba_image.getdata()
    new_data = []
    for item in data:
        if item[0] > 100 and item[1] > 100 and item[2] > 100:
            new_data.append((255, 255, 255, 0))
        else:
            new_data.append(item)
    rgba_image.putdata(new_data)
    rgba_image.save(f'static/uploads/{i}.png', 'PNG')

    print("Conversion complete: WEBP to PNG with transparent background.")

# Set up paths
UPLOAD_FOLDER = 'static/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Set up Gemini Contrastive Model
def load_models():
    Castor = Encoder()
    Pollux = Encoder()
    Castor.load_state_dict(torch.load('models/Best-Castor-89-6.pth', map_location=torch.device('cpu')))
    Pollux.load_state_dict(torch.load('models/Best-Pollux-89-6.pth', map_location=torch.device('cpu')))
    class_head = ClassificationHead(256, 9)
    class_head.load_state_dict(torch.load('models/Best_Diviner-89-6.pth', map_location=torch.device('cpu')))
    gemini = GeminiContrast(Castor, Pollux, class_head)
    print('models loaded')
    return gemini


# Dictionary to map class indices to class names and descriptions
class_dict = {
    0: {'name': 'Adipose', 'description': 'Adipose tissue, commonly known as body fat, is a connective tissue that stores energy in the form of fat.'},
    1: {'name': 'Background', 'description': 'Background refers to the non-specific or irrelevant areas in the image that do not correspond to any class of interest.'},
    2: {'name': 'Debris', 'description': 'Debris includes small fragments or remnants that are often considered as waste material in the histology slide.'},
    3: {'name': 'Lymphocytes', 'description': 'Lymphocytes are a type of white blood cell crucial for the immune system, involved in the body’s defense against pathogens.'},
    4: {'name': 'Mucus', 'description': 'Mucus is a viscous fluid secreted by mucous membranes that serves to protect and lubricate tissues.'},
    5: {'name': 'Smooth Muscle', 'description': 'Smooth muscle tissue is a type of involuntary muscle found in various organs and structures, responsible for involuntary movements.'},
    6: {'name': 'Normal Colon Mucosa', 'description': 'Normal colon mucosa refers to the healthy lining of the colon, which is important for proper digestive function.'},
    7: {'name': 'Cancer-Associated Stroma', 'description': 'Cancer-associated stroma is the supportive tissue surrounding cancer cells that can influence tumor growth and progression.'},
    8: {'name': 'Colorectal Adenocarcinoma Epithelium', 'description': 'Colorectal adenocarcinoma epithelium refers to the cancerous epithelial cells found in colorectal cancer.'}
}

# Function to preprocess the image
def preprocess_image(image):
    supervised_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])
])
    image = image.resize((28, 28))
    image = supervised_transforms(image)
    print('preprocessed')
    return image

# Function to extract RGB statistics from an image
def extract_rgb_statistics(image):
    img = np.array(image)
    red, green, blue = img[:, :, 0].flatten().tolist(), img[:, :, 1].flatten().tolist(), img[:, :,
                                                                                            2].flatten().tolist()
    colors = {'red': red, 'green': green, 'blue': blue}
    funcs = {'_avg': np.mean, '_std': np.std, '_max': np.max, '_min': np.min}
    results = {}
    for _name, func in funcs.items():
        for name, color in colors.items():
            results[name + _name] = func(color)
    print('rgb done')
    return results

@app.route("/", methods=["GET", "POST"])
def index():
    print('page loaded')
    class_prediction = None
    description = None
    rgb_features = None
    text_rgb = None
    alt_rgb_1 = None
    image_url = None
    error_message = None

    if request.method == "POST":
        print('post triggered')
        try:
            image_file = request.files.get('fileInput')

            if image_file:
                print('image found')
                # memory_process = multiprocessing.Process(target=monitor_memory)
                # memory_process.start()
                # Save the uploaded image
                image_path = os.path.join(UPLOAD_FOLDER, 'uploaded_image.jpg')
                image = Image.open(io.BytesIO(image_file.read())).convert('RGB')
                image.save(image_path)
                image_url = url_for('static', filename=f'uploads/uploaded_image.jpg')

                # Preprocess and predict
                print('preprocessing')
                image_array = preprocess_image(image)
                print('rgb-ing')
                rgb_features = extract_rgb_statistics(image)
                text_rgb = [rgb_features[f'{i}_avg'] - (rgb_features[f'{i}_std'] * 2) if (rgb_features[f'{i}_avg'] - (rgb_features[f'{i}_std'] * 2)) >= 0 else 0 for i in ['red', 'green', 'blue']]
                alt_rgb_1 = [rgb_features[f'{i}_avg'] + (rgb_features[f'{i}_std'] * 2) if (rgb_features[f'{i}_avg'] + (rgb_features[f'{i}_std'] * 2)) <= 255 else 255 for i in ['red', 'green', 'blue']]
                rgb_features = [v for k, v in rgb_features.items() if '_avg' in k]
                gemini = load_models()
                print('model warming up')
                class_probs = gemini(image_array.unsqueeze(0))
                print('model done running')
                predicted_class_index = class_probs.max(1)
                print('model returned a prediction')
                class_info = class_dict.get(predicted_class_index.indices.item(), {'name': 'Unknown', 'description': 'No description available'})
                class_prediction = class_info['name']
                description = class_info['description']
                # memory_process.terminate()
        except Exception as e:
            error_message = str(e)
            print(f"Error: {error_message}")
    return render_template("index.html", prediction=class_prediction, description=description, error=error_message, rgb_features=rgb_features, text_rgb = text_rgb, alt_rgb_1=alt_rgb_1, image=image_url)

if __name__ == "__main__":
    app.run()
