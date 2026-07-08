import io
import json
import base64
from io import BytesIO

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)

model.load_state_dict(
    torch.load('pneumonia_classifier.pth', map_location='cpu', weights_only=False)
)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class_names = ['NORMAL', 'PNEUMONIA']

def validate_input(job_input):
    if job_input is None:
        return None, "Please provide an image"
    if isinstance(job_input, str):
        try:
            job_input = json.loads(job_input)
        except json.JSONDecodeError:
            return None, "Invalid JSON input"
    image_data = job_input.get('image')

    if image_data is None:
        return None, "Please provide an image"
    if not isinstance(image_data, str):
        return None, "Image data must be a base64 encoded string"
    
    return {'image': image_data}, None

def handler(job):
    job_input = job['input']

    validated_data, error_message = validate_input(job_input)
    if error_message:
        return {'error': error_message}

    image_base64 = validated_data['image']

    try:
        image_bytes = base64.b64decode(image_base64)
        image = Image.open(BytesIO(image_bytes)).convert('RGB')
        image = transform(image)
        image = image.unsqueeze(0)

        with torch.no_grad():
            outputs = model(image)
            _, predictions = torch.max(outputs, 1)
            
        predicted_class = class_names[predictions.item()]

        return {'prediction': predicted_class}
    
    except base64.binascii.Error:
        return {'error': 'Invalid base64 encoded image data'}
    except IOError:
        return {'error': 'Invalid image data'}
    except Exception as e:
        return {'error': f'An unexpected error occurred: {e}'}


if __name__ == "__main__":
    import sys

    image_path = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "chest_xray/test/PNEUMONIA/person100_bacteria_475.jpeg"
    )

    with open(image_path, "rb") as image_file:
        image_base64 = base64.b64encode(image_file.read()).decode("utf-8")

    job = {"input": {"image": image_base64}}
    result = handler(job)

    print(f"Image: {image_path}")
    print(f"Result: {result}")
