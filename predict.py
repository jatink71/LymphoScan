import sys
import torch
from PIL import Image
from torchvision import transforms
from transformers import SwinForImageClassification

if len(sys.argv) < 2:
    print("Usage: python predict.py <image_path>")
    sys.exit(1)

IMAGE_PATH = sys.argv[1]
MODEL_PATH = "C:/Users/jatin/LymphoScan_Project/models/swin_model.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SwinForImageClassification.from_pretrained(
    "microsoft/swin-tiny-patch4-window7-224",
    num_labels=10,
    ignore_mismatched_sizes=True
).to(device)

model.load_state_dict(torch.load(MODEL_PATH, map_location=device), strict=False)
model.eval()

image = Image.open(IMAGE_PATH).convert("RGB")
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
image = transform(image).unsqueeze(0).to(device)

with torch.no_grad():
    output = model(image)
    _, predicted = torch.max(output.logits, 1)

class_names = ['Benign', 'Blood_Cancer', 'CLL', 'Early', 'FL', 'MCL', 'Pre', 'Pro', 'non_cancer', 'unknown']
print(f"Prediction: {class_names[predicted.item()]}")