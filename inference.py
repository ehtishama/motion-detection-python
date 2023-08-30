import torch
import torchvision
from motion_detect import detect_motion
import time
import numpy as np
from PIL import Image


# model setup
print('Setting up Classification model...\n')

classes = ['Cheetah', 'Deer', 'Duck', 'Eagle', 'Elephant', 'Fox',
           'Giraffe', 'Goose', 'Horse', 'Leopard', 'Lion', 'Monkey',
           'Ostrich', 'Owl', 'Parrot', 'Rhinoceros', 'Snake', 'Sparrow',
           'Tiger', 'Zebra']

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

TRAINED_MODEL_PATH = './models/mobilenet_openimages_subset_acc_82'
trained_model = torchvision.models.mobilenet_v2()

in_features = trained_model.classifier[1].in_features
out_features = 20

trained_model.classifier[1] = torch.nn.Linear(in_features=in_features, out_features=out_features)
trained_model_weights = torch.load(TRAINED_MODEL_PATH, map_location=device)
trained_model.load_state_dict(trained_model_weights)

# set the model to inference mode
trained_model.eval()

# transforms for the input
transforms = torchvision.transforms.Compose([
    # torchvision.transforms.PILToTensor(),
    torchvision.transforms.Resize(256),
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


#define callback
def on_motion_detected(motion_frame):
    
    print(f'Received motion frame')
    print(f'Classifying...')
    
    inference(motion_frame)
    
# 
def inference(image):
    # converts cv representation to PIL
    image = Image.fromarray(image)
    
    transformed_image = transforms(image)
    transformed_image = torch.unsqueeze(transformed_image, dim=0)

    output = trained_model(transformed_image)
    
    probs = torch.nn.functional.softmax(output, dim=1)

    # class_idx = torch.argmax(output, dim=1)
    class_score, class_idx = torch.max(probs, dim=1)
    
    class_label = classes[class_idx]
    
    print(f'Class: {class_label}, Confidence: {class_score.item()}. \n')
    
    
detect_motion(on_motion_detected)