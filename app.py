from flask import Flask, request, jsonify, render_template, send_from_directory
import torch
from PIL import Image
from torchvision import transforms
from transformers import SwinForImageClassification
import os

app = Flask(__name__, template_folder="templates", static_folder="static")

# Define folders
UPLOAD_FOLDER = "static/uploads"
CONVERTED_FOLDER = "static/converted"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(CONVERTED_FOLDER, exist_ok=True)

# Load the trained model
MODEL_PATH = "C:/Users/jatin/LymphoScan_Project/models/swin_model.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SwinForImageClassification.from_pretrained(
    "microsoft/swin-tiny-patch4-window7-224",
    num_labels=10,
    ignore_mismatched_sizes=True
).to(device)

model.load_state_dict(torch.load(MODEL_PATH, map_location=device), strict=False)
model.eval()

# Preprocessing pipeline
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0).to(device)

# Home route to load the main web page
@app.route('/')
def home():
    return render_template('index.html')

# Optional route to convert TIFF images to PNG
@app.route("/convert_tiff", methods=["POST"])
def convert_tiff():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    filename = file.filename
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    new_filename = os.path.splitext(filename)[0] + ".png"
    new_filepath = os.path.join(CONVERTED_FOLDER, new_filename)

    try:
        img = Image.open(filepath)
        img.convert("RGB").save(new_filepath, format="PNG")

        if os.path.exists(new_filepath):
            return jsonify({"converted_image_url": f"/converted/{new_filename}"})
        else:
            return jsonify({"error": "Converted file not found"}), 500

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Route to serve converted files
@app.route("/converted/<filename>")
def serve_converted_image(filename):
    try:
        return send_from_directory(CONVERTED_FOLDER, filename)
    except FileNotFoundError:
        return "File not found", 404

# Prediction API
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    image = Image.open(file.stream).convert("RGB")
    image = preprocess_image(image)

    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs.logits, 1)

    class_names = [
        'Benign',
        'Blood Cancer',
        'Chronic Lymphocytic Leukemia (CLL)',
        'Early Stage Lymphoma',
        'Follicular Lymphoma (FL)',
        'Mantle Cell Lymphoma (MCL)',
        'Pre-Leukemia Condition',
        'Pro-Lymphocytic Leukemia',
        'Non-Cancerous Abnormality',
        'Unknown / Unclassified'
    ]

    predicted_class = class_names[predicted.item()]
    return jsonify({"prediction": predicted_class})

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)