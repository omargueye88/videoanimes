from flask import Flask, send_file
import cv2
import torch
import torchvision.transforms as T
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torch import nn
from torchvision import models
from waitress import serve
from flask import Flask, send_file, render_template

app = Flask(__name__)
from multiprocessing import cpu_count

# Obtenir le nombre de cœurs CPU disponibles
workers = cpu_count()

# Charger le modèle une seule fois
from torchvision.models import vgg19, VGG19_Weights

weights = VGG19_Weights.DEFAULT
model = vgg19(weights=weights).features.eval()

def transform_image(image):
    transform = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

def apply_style(model, image):
    with torch.no_grad():
        transformed_image = transform_image(image)
        styled_image = model(transformed_image)
        return styled_image

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

@app.route('/')
def index():
    return render_template("index.html")
@app.route('/video')
def serve_video():
    input_video = "input_video.mp4"
    output_video = "output_styled_video.avi"
    process_video(input_video, output_video, model)
    return send_file(output_video, mimetype='video/avi')

if __name__ == "__main__":
    from waitress import serve
    import os
    print("serveur en cours de demarrage...")
    port = int(os.environ.get("port", 5000))
    serve(app, host='0.0.0.0', port=port, threads=4 ,channel_timeout=60, )
    
     # Utilise soit Waitress, soit app.run()