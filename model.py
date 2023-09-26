import torch
import torchvision
from PIL import Image

"""Loads trained calssification model and  defines classify funtion."""

TRAINED_MODEL_PATH = './models/mobilenet_openimages_subset_acc_82'
TARGET_LABELS = ['Cheetah', 'Deer', 'Duck', 'Eagle', 'Elephant', 'Fox',
           'Giraffe', 'Goose', 'Horse', 'Leopard', 'Lion', 'Monkey',
           'Ostrich', 'Owl', 'Parrot', 'Rhinoceros', 'Snake', 'Sparrow',
           'Tiger', 'Zebra']
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Import mobilenet from torchvision.
trained_model = torchvision.models.mobilenet_v2()

# Modify the output layer of model to accomodate only 20 target classes.
in_features = trained_model.classifier[1].in_features
out_features = 20
trained_model.classifier[1] = torch.nn.Linear(in_features=in_features, out_features=out_features)

# Load pre-trained weights.
trained_model_weights = torch.load(TRAINED_MODEL_PATH, map_location=DEVICE)
trained_model.load_state_dict(trained_model_weights)
trained_model.eval()

# Setup transformation pipeline for inputs. 
transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize(256),
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def classify(image):
    """Classify the input image.

    Args:
        image (np.Array): numpy representation of image.
    Returns:
        tuple(label, score) 
    """
    
    image = Image.fromarray(image)
    transformed_image = transforms(image)
    transformed_image = torch.unsqueeze(transformed_image, dim=0) # Creates a batch of single image i.e. (1x3x224x224).

    output = trained_model(transformed_image)
    
    # Get prob distribution by applying softmax funciton to the model output. 
    probs = torch.nn.functional.softmax(output, dim=1)

    #
    max_score, max_score_idx = torch.max(probs, dim=1)
    predicted_label = TARGET_LABELS[max_score_idx]
    
    #
    print(f'Predicted Class: {predicted_label}, Confidence Score: {max_score.item()}. \n')
    
    return (predicted_label, max_score.item())
    
  
