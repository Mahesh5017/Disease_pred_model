import torch
from torchvision import transforms
from PIL import Image

def predict_image(img_path):
    checkpoint = torch.load("D:/DISEASE_PRED/model/crop_disease_model.pt", map_location=torch.device("cpu"))
    class_names = checkpoint["class_names"]
    model = torch.hub.load("pytorch/vision", "resnet18", weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, len(class_names))
    model.load_state_dict(checkpoint["model"])
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    image = Image.open(img_path).convert("RGB")
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        label = class_names[predicted.item()]

    if "___" in label:
        plant, disease = label.split("___", 1)
    elif "_" in label:
        parts = label.split("_")
        plant = parts[0]
        disease = "_".join(parts[1:])
    else:
        plant, disease = label, "Unknown"

    return {"plant": plant, "disease": disease}