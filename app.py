from flask import Flask, request, render_template, url_for
import os
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import requests

app = Flask(__name__)

# Check if CUDA is available and set device accordingly
device = torch.device('cpu')

# Load the pre-trained ResNet model with the new 'weights' argument
model = models.resnet50(weights='DEFAULT')  # Updated to use 'weights'
model.to(device)
model.eval()

# Define the preprocessing transform
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load class names
LABELS_URL = 'https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json'
labels = requests.get(LABELS_URL).json()

def predict_image(img):
    img_tensor = preprocess(img)
    img_tensor = img_tensor.unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(img_tensor)
        _, predicted = torch.max(output, 1)
    
    return labels[predicted.item()]

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file part'
        
        file = request.files['file']
        
        if file.filename == '':
            return 'No selected file'
        
        if file:
            filename = file.filename
            file_path = os.path.join('static', filename)
            file.save(file_path)
            img = Image.open(file_path).convert('RGB')  # Ensure image is in RGB format
            prediction = predict_image(img)
            img_url = url_for('static', filename=filename)
            return render_template('index.html', prediction=prediction, image_url=img_url)
    
    return render_template('index.html')

if __name__ == '__main__':
    if not os.path.exists('static'):
        os.makedirs('static')
    app.run(debug=True)
