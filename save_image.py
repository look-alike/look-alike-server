# pickle 파일에 이미지를 저장하기 위해 만들었습니다.
import os
import glob
import cv2
import torch
import pickle
import numpy as np
from arcface_model import CustomArcFaceModel
from albumentations import Compose, Resize
from albumentations.pytorch import ToTensorV2

def load_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def preprocess_image(image):
    if isinstance(image, np.ndarray):
        preprocess = Compose([
            Resize(224, 224),
            ToTensorV2()
        ])
        image = preprocess(image=image)['image']
        image = image.float() / 255.0
        return image.unsqueeze(0)
    elif isinstance(image, torch.Tensor):
        return image.float() / 255.0
    else:
        raise ValueError("Unsupported image format.")


num_classes = 11  # 분류할 클래스의 수
device = torch.device('cpu')

model = CustomArcFaceModel(num_classes)
model.load_state_dict(torch.load('arcface.pth', map_location=torch.device('cpu')))
model.eval()

celebrity_image_dict = {}

celebrity_initial_list = ['shg', 'idh', 'she', 'ijh', 'cde', 'chj', 'har', 'jjj', 'jsi', 'ojy', 'smo']

embeddings_dict = {}

for celebrity_initial in celebrity_initial_list:
    
  image_folder = f'/Users/jang-youngjoon/dev-projects/youtuber-look-alike/pre-processed-image/{celebrity_initial}/'
  image_files = glob.glob(os.path.join(image_folder, '*.jpg'))
  embeddings = []

  for image_file in image_files:
        image = load_image(image_file)
        preprocessed_image = preprocess_image(image).to(device)

        with torch.no_grad():
            embedding = model(preprocessed_image)
            embeddings.append(embedding.squeeze().cpu().numpy())

  embeddings_dict[celebrity_initial] = embeddings

  with open('trained_celebrity_embeddings.pkl', 'wb') as f:
      pickle.dump(embeddings_dict, f)
