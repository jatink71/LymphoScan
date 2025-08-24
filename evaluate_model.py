import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from transformers import SwinForImageClassification
from sklearn.metrics import classification_report, confusion_matrix

VAL_DIR = "C:/Users/jatin/LymphoScan_Project/data/validation"
MODEL_PATH = "C:/Users/jatin/LymphoScan_Project/models/swin_model.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_dataset = datasets.ImageFolder(VAL_DIR, transform=transform)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

class_names = val_dataset.classes
model = SwinForImageClassification.from_pretrained(
    "microsoft/swin-tiny-patch4-window7-224",
    num_labels=len(class_names),
    ignore_mismatched_sizes=True
).to(device)

model.load_state_dict(torch.load(MODEL_PATH, map_location=device), strict=False)
model.eval()

true_labels = []
pred_labels = []

with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images).logits
        _, preds = torch.max(outputs, 1)
        true_labels.extend(labels.cpu().numpy())
        pred_labels.extend(preds.cpu().numpy())

print(classification_report(true_labels, pred_labels, target_names=class_names, digits=4))
print(confusion_matrix(true_labels, pred_labels))