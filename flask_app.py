from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import base64
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2
from arcface_model import CustomArcFaceModel
import pickle
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

# Load trained celebrity embeddings
with open('./trained_celebrity_embeddings.pkl', 'rb') as f:
    trained_embeddings_dict = pickle.load(f)


trained_embeddings = []
for celebrity in trained_embeddings_dict:
    for embedding in trained_embeddings_dict[celebrity]:
        trained_embeddings.append(embedding)

trained_embeddings = torch.tensor(trained_embeddings)

# threshold 설정
sum = 0;
threshold = {}
celebrity_initial_list = ['shg', 'idh', 'she', 'ijh', 'cde', 'chj', 'har', 'jjj', 'jsi', 'ojy', 'smo']
for celebrity_initial in celebrity_initial_list:
    file_length = len(trained_embeddings_dict[celebrity_initial])
    sum += file_length
    # print(f'{celebrity_initial}: {len(trained_embeddings_dict[celebrity_initial])}')
    threshold[celebrity_initial] = sum

def get_initial(number):
    if number is None:  # number가 None인 경우 체크
        return None
    else:
        sorted_initials = sorted(threshold.items(), key=lambda x: x[1])

        for i in range(len(sorted_initials) - 1):
            if sorted_initials[i][1] < number <= sorted_initials[i + 1][1]:
                return sorted_initials[i + 1][0]

    return None

# 모델 불러오기
num_classes = 11  # 분류할 클래스의 수
device = torch.device('cpu')

model = CustomArcFaceModel(num_classes)
model.load_state_dict(torch.load('./arcface.pth', map_location=torch.device('cpu')))

model.eval()

# 이미지 전처리
preprocess = A.Compose([
    A.Resize(224, 224),
    ToTensorV2()
])

def crop_face(image):
    face_cascade = cv2.CascadeClassifier("./haarcascade_frontalface_default.xml")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.2, 5)

    for (x, y, w, h) in faces:
        cropped_image = image[y: y + h, x: x + w]
        resized_image = cv2.resize(cropped_image, (224, 224))
        if resized_image.shape[0] > 0 and resized_image.shape[1] > 0: # 이미지가 존재하는지 확인
            return resized_image
    return None  # 404 대신 None 반환

def predict_celebrity(image):
    with torch.no_grad():
        cropped_image = crop_face(image)
        if cropped_image is None:  # None인 경우 체크
            return [None, 0]  # celebrity_initial 및 정확도를 None, 0으로 설정
        else:
            image = preprocess(image=cropped_image)['image']
            image = image.float() / 255.0
            image = image.unsqueeze(0).to(device)
            user_face_embedding = model(image).squeeze()

            closest_celebrity, max_similarity = model.find_most_similar_celebrity(user_face_embedding, trained_embeddings)
            return [closest_celebrity, max_similarity.item()]
    
@app.route('/')
@cross_origin
def index():
    return 'success'

@app.route('/predict', methods=['POST'])
@cross_origin()
def predict():
    try:
        image_data = request.form.get('image')
        image_decoded = base64.b64decode(image_data)
        nparr = np.frombuffer(image_decoded, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        prediction = predict_celebrity(image)
        celebrity_initial = get_initial(prediction[0])
        print('이니셜:', celebrity_initial, '정확도:', prediction[1])
        return jsonify({'celebrity_initial': celebrity_initial, 'accuracy': prediction[1]})
    except Exception as e:
        print(e)
        return jsonify({'error': 'Error occurred during prediction'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=4000)

