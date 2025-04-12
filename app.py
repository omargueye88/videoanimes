from flask import Flask, send_file, render_template
import cv2
import torch
import torchvision.transforms as T
import numpy as np
from PIL import Image
from torch import nn
from torchvision import models
from waitress import serve
from multiprocessing import cpu_count
import os

app = Flask(__name__)

# Nombre de cœurs CPU
workers = cpu_count()

# Charger le modèle VGG19 une fois
from torchvision.models import vgg19, VGG19_Weights
weights = VGG19_Weights.DEFAULT
model = vgg19(weights=weights).features.eval()

# Transformation de l'image
def transform_image(image):
    transform = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

# Appliquer le style
def apply_style(model, image):
    with torch.no_grad():
        transformed_image = transform_image(image)
        styled_image = model(transformed_image)
        return styled_image

# Traiter une vidéo
def process_video(input_video, output_video, model):
    cap = cv2.VideoCapture(input_video)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video, fourcc, 30.0, (640, 480))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        styled_frame = apply_style(model, pil_image)
        styled_frame = styled_frame.squeeze(0).permute(1, 2, 0).cpu().numpy()
        styled_frame = np.clip(styled_frame, 0, 1)
        styled_frame = (styled_frame * 255).astype(np.uint8)
        styled_frame = cv2.cvtColor(styled_frame, cv2.COLOR_RGB2BGR)
        out.write(styled_frame)

    cap.release()
    out.release()

# Routes Flask
@app.route('/')
def index():
    return render_template("index.html")

@app.route('/video')
def serve_video():
    input_video = "input_video.mp4"
    output_video = "output_styled_video.avi"
    process_video(input_video, output_video, model)
    return send_file(output_video, mimetype='video/avi')

# Lancer le serveur avec Waitress
if __name__ == "_main_":
    print("Serveur en cours de démarrage...")
    serve(app, host='0.0.0.0', port=5000, threads=4, channel_timeout=60, _quiet=True)