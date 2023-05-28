import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import glob     
import json
import os

# 1. 모델 불러오기
model = tf.keras.models.load_model('my_model_noodle_v2.h5', custom_objects={'KerasLayer': hub.KerasLayer})


# 경로설정
image_directory = r'C:\Users\sj990\Desktop\ex3'

# jpg, png, jpeg 형식 (리스트형식으로 저장된다, 경로에는 파일한장만 존재하게)
image_files = glob.glob(image_directory + '\*.png') + glob.glob(image_directory + '\*.jpg') + glob.glob(image_directory + '\*.jpeg')

# 리스트중 첫번째것만 사용
image_path = image_files[0]


# 이미지전처리, 사이즈조정만
img = tf.keras.preprocessing.image.load_img(image_path, target_size=(256, 256))
img = tf.keras.preprocessing.image.img_to_array(img)
img = img.astype('float32') / 255.0
img = np.expand_dims(img, axis=0)

# 이미지 예측
preds = model.predict(img)

class_labels = []

# 3. txt파일 불러오기
with open('class_labels_noodle.txt', 'r') as file:
    for line in file:
        class_labels.append(line.strip())

# 결과값 상위 1개
top_preds_idx = preds[0].argsort()[ : : -1][ : 1]
top_preds_labels = [class_labels[idx] for idx in top_preds_idx]
top_preds_probs = preds[0][top_preds_idx]

results = []

#json형식 results
for label, prob in zip(top_preds_labels, top_preds_probs):
    result = {
        '상품': label,
        '정확도': round(float(prob), 2)
    }
    results.append(result)
        
        
'''
# json파일 저장필요할시 주석 제거후 사용
with open('results.json', 'w', encoding='utf-8') as file:
    json.dump(results, file, ensure_ascii=False)
'''


#출력예시
print(results)

'''
# 사용한 이미지 파일 삭제 필요할시 주석 제거후 사용
if results != None :
    os.remove(image_path)
'''

