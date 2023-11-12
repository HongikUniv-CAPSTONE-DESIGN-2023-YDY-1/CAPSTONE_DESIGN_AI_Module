#!/usr/bin/env python
# coding: utf-8

# In[14]:


import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import glob     
import json
import os
import shutil
import subprocess

#########################################################################
#########################################################################
# 수정해야되는 초기경로설정들 
# 1. 어플에서 올라온 이미지폴더 경로, 폴더안에 이미지는 항상 한장만 존재해야한다. -> 작업이 끝난후 이미지는 삭제
application_image = r'C:\Users\sj990\Desktop\작업폴더예시\img_folder\img_folder'

# 2. yolov5모델 경로 
yolo_model = r'C:\Users\sj990\Desktop\작업폴더예시\model_label\best_9.21.pt'

# 3. 모델, 라벨경로
objects_model_path = r'C:\Users\sj990\Desktop\작업폴더예시\model_label\my_model_all_v2.h5'
objects_label_path = r'C:\Users\sj990\Desktop\작업폴더예시\model_label\my_model_all_label.txt'

#########################################################################
########################################################################

image_directory = application_image
# jpg, png, jpeg 형식 (리스트형식으로 저장된다, 경로에는 파일한장만 존재하게)
image_files = glob.glob(image_directory + '/*.png') + glob.glob(image_directory + '/*.jpg') + glob.glob(image_directory + '/*.jpeg')
# 리스트중 첫번째것만 사용
first_image = image_files[0]

# 객체검출 실행                                                conf조절하면서 객체검출
#!python yolov5/detect.py --weights "{yolo_model}" --img 640 --conf 0.5 --source "{first_image}" --project ../yolo --save-crop
command = f'python yolov5/detect.py --weights "{yolo_model}" --img 640 --conf 0.4 --source "{first_image}" --project ./ --save-crop'
subprocess.run(command, shell=True)



# 검출경로
crops_path = os.path.join('exp', 'crops')

# 객체 폴더 있는지 탐지
objects_path = os.path.join(crops_path, 'objects')


if os.path.exists(objects_path):
    object_files = os.listdir(objects_path)
    
    # crop이미지가 한장일때만 crop이미지로 사용
    if len(object_files) > 1:
        yolo_img = application_image
        
    else:
        yolo_img = os.path.join(crops_path, 'objects')
        
else:
    yolo_img = application_image

###############################################################
    
yolo_image_directory = yolo_img
# jpg, png, jpeg 형식 (리스트형식으로 저장된다, 경로에는 파일한장만 존재하게)
yolo_image_files = glob.glob(yolo_image_directory + '/*.png') + glob.glob(yolo_image_directory + '/*.jpg') + glob.glob(yolo_image_directory + '/*.jpeg')
# 리스트중 첫번째것만 사용
yolo_image_path = yolo_image_files[0]

#####################모델정의#####################
class Model:
    # 초기화
    def __init__(self, image_path, model_path, class_labels_path):
        self.image_path = image_path
        self.model_path = model_path
        self.class_labels_path = class_labels_path
        
    #예측 (전처리된이미지와 클래스라벨을 삽입)
    def preds_model(self, img, class_labels):
        model = tf.keras.models.load_model(self.model_path,  custom_objects={'KerasLayer': hub.KerasLayer})
        preds = model.predict(img) 
        
        # 결과값 상위 1개
        top_preds_idx = preds[0].argsort()[ : : -1][ : 1]
        top_preds_labels = [class_labels[idx] for idx in top_preds_idx]
        top_preds_probs = preds[0][top_preds_idx]
        return top_preds_labels, top_preds_probs
    
    #이미지전처리_1 (기본)
    def img_pre_default(self):  
        img = tf.keras.preprocessing.image.load_img(self.image_path, target_size=(256, 256))
        img = tf.keras.preprocessing.image.img_to_array(img)
        img = img.astype('float32') / 255.0
        img = np.expand_dims(img, axis=0)
        return img

    #라벨불러오기
    def class_label(self):
        class_labels = []
        with open(self.class_labels_path, 'r') as file:
            for line in file:
                class_labels.append(line.strip())
        return class_labels
    
    #결과출력
    def print_result(self, top_preds_labels, top_preds_probs):
        results = []
        #json형식 results
        for label, prob in zip(top_preds_labels, top_preds_probs):
            result = {
                '상품': label,
                '정확도': round(float(prob), 2)
            }
            results.append(result)
        print(results)
        return results
#####################모델정의#####################

###########특징분류함수###########
def save_result(image_path, model_path, label_path):
    model = Model(image_path, model_path, label_path)
    
    #예측
    result_labels, result_probs = model.preds_model(model.img_pre_default(), model.class_label())
    
    #결과출력
    result = model.print_result(result_labels, result_probs)
    
    # json파일 저장
    with open('results.json', 'w', encoding='utf-8') as file:
        json.dump(result, file, ensure_ascii=False)

        
############특징분류############
image_path = yolo_image_path #yolo탐지된건 crop된이미지가, 아닐시에는 원본이미지가 들어간다.
save_result(image_path, objects_model_path, objects_label_path)
###############################      




# 작업끝난후 yolo 이미지 폴더 전체 삭제
shutil.rmtree("exp")

# 작업끝난후 어플리케이션에서 올라온 이미지만 삭제
for file in os.scandir(application_image):
    os.remove(file.path)


