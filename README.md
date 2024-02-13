# 편킹 - AI모듈
> 사용자가 어플리케이션을 통해 촬영한 상품 이미지의 상품명을 분류하는 모델

## 시스템 구조
<img src = https://github.com/HongikUniv-CAPSTONE-DESIGN-2023-YDY-1/CAPSTONE_DESIGN_AI_Module/assets/117420071/7c123b7d-ccb0-4712-b089-f5fca59da3cb width = "600" height = "400">

 
## 주요 담당 기능 
### 상품 이미지 분류 
> 학습된 인공지능 모델에 사용자가 촬영한 이미지를 입력하고 모델은 입력된 이미지에 대한 상품명을 분류한다.
> 
> 개발 완료된 본 모델을 서버에 적재하여 어플리케이션에서 전송된 상품 이미지에 대한 결과를 json형식으로 반환한다.
> 
> 서버에서는 이 결과값을 이용하여 편의점 할인상품 정보와 결합하고 최종적으로 상품에 대한 할인정보를 제공 가능하게 된다.
>
> <img src = https://github.com/HongikUniv-CAPSTONE-DESIGN-2023-YDY-1/CAPSTONE_DESIGN_AI_Module/assets/117420071/7847c493-1b11-4083-bea8-95e1e6fcaf44>

> 
<a href="https://robinjoon.notion.site/c234ada4cf0748768a6836648de5b31c?pvs=4"><img src="https://img.shields.io/badge/상품 인식 모듈 성능 개선기 -black?style=for-the-badge&logo=Notion&logoColor=00000">

## 기술 스택

<img src="https://img.shields.io/badge/Tensorflow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white">

<img src="https://img.shields.io/badge/pytorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white">

<img src="https://img.shields.io/badge/YOLO-00FFFF?style=for-the-badge&logo=yolo&logoColor=white">

<br>
<br>


## 학습과정
학습에 사용한 패키지들은 다음과같다.

- tensorflow : 딥러닝 모델을 구축 및 학습
- tensorflow-hub : 전이학습 모델 불러오기
- tensorflow.keras : 딥러닝 api로 신경망 모델을 빌드 및 학습
- numpy : 다차원 행렬과 연산 및 수치계산

<br>
<br>

#### a) 전이학습

상품의 할인 정보를 제공하기 위해선 사용자가 어떤 상품의 할인 정보를 요구하는지 시스템이 정확히 인식해야 한다. 따라서 이 모듈이 상품의 이미지를 정확히 인식하는 것이 인식 속도 등 모듈의 다른 성능 지표보다 더 중요하다. 따라서, AI모델의 정확도를 최대한 높이기 위하여 전이학습 기법을 사용했다.

학습에 사용할 이미지에는10280종의 상품 이미지가 존재하지만 각 상품당 이미지는 114장으로 비교적 적다. 따라서, 높은 정확도를 얻기 위하여 요구하는 데이터셋의 양이 비교적 적게 필요한 EfficientNet-B0모델을 사용하여 학습을 진행했다. 

- VGG16	- 수백개 이상
- ResNet	- 수천개 이상
- EfficientNet	- 수십개 이상

미리 훈련된EfficientNet-B0모델을 tensorflow-hub에서 불러오고 마지막 출력층을 동결시킨후 512, 256, 128, 64개의 뉴런을 가진 완전 연결 레이어를 추가하여 특징을 추출한 뒤 분류를 수행하였다. DropOut, L2정규화를 사용해 과적합을 방지했다.

<br>
<br>

#### b) 이미지 전처리

기존의 EfficientNet의 이미지 crop 비율이 우리가 수집한 이미지 데이터와 맞지 않아 상품 이미지의 일부만 인식하는 문제가 발생했다. 이를 해결하기 위해 crop비율을 (0.08, 1.0)에서 (0.6, 1.0)로 조정했다. 

 ![d1d1](https://github.com/HongikUniv-CAPSTONE-DESIGN-2023-YDY-1/CAPSTONE_DESIGN_AI_Module/assets/117420071/64f0c614-8411-4cda-9cae-84a9f22ceaad)
 <br>
           
기존의 EfficientNet-B0모델이 학습한 ImageNet이미지는 우리가 모델의 학습을 위해 수집한 상품 이미지와 비율이 크게 달랐다. 이미지 전체 중 객체의 비율이ImageNet이미지보다 크기 때문에 256x256사이즈로 resize 시켰다.

상품당 114장의 한정된 이미지로는 상품분류를 위한 학습이 이루어지기에는 부족하기 때문에 고강도의 데이터 증강기법이 필요하다. 사용한 데이터 증강기법은 아래와 같다.
기존 수량	변경 수량	증강 과정	증강 기법
- 20도	이미지 회전 (-20° ~ +20°)
- 이미지 좌우 이동 (-30% ~ +30%)
- 이미지 상하 이동 (-30% ~ +30%)
- 이미지 변형 (-30% ~ +30%)
- 이미지 확대, 축소(0.8 ~ 1.2)
- 이미지 수평 뒤집기
- 이미지 잘라내기

<br>
<br>

#### c) 학습 정확도

샘플 데이터를 이용하여 프로토타입을 제작하고 파인 튜닝 과정을 통해 하이퍼 파라미터 값을 조정하여 최적의 학습 조건을 도출했다. 하이퍼 파라미터 최적화 작업은 총 52회에 걸쳐 수행되었으며, 이 중 가장 높은 정확도를 달성한 모델의 하이퍼 파라미터 값과 정확도는 다음과 같다.

- optimizer = RMSprop(lr=0.007)
- kernel_regularizer = l2(0.005)
- batch_size : 64
- epoch : 50
- 완전연결 레이어층의 활성화함수 : ReLu 함수 사용

train accuracy : 97%     val accuracy : 96%    test accuracy : 93%

<br>
 ![image](https://github.com/HongikUniv-CAPSTONE-DESIGN-2023-YDY-1/CAPSTONE_DESIGN_AI_Module/assets/117420071/57d4a03e-407d-44b4-ae20-246bd4cbe74b)

<br>
<br>

#### d) 이외 사용기법
- MirroredStrategy를 사용한 GPU분산학습.
- 검증데이터셋의 정확도가 5회이상 개선되지 않을 시 학습 중지(early stopping 기법 사용)

  
### YOLOv5
사용자가 촬영한 이미지를 분류할 때 이미지에 상품이 아닌 배경이 많이 포함될 경우 정확도가 급격히 떨어지는 문제가 발생했다.


 ![image](https://github.com/HongikUniv-CAPSTONE-DESIGN-2023-YDY-1/CAPSTONE_DESIGN_AI_Module/assets/117420071/aeed5a4e-ea24-4bd0-8034-9acf7e7e0505)
 <br>

이를 해결하기 위해선 상품이 아닌 배경을 제거해야 할 필요가 있었다. 이 작업을 수행하기 위해 객체 탐지 모델로 유명한 YOLOv5를 사용했다. 상품 객체 탐지 모델을 만들기 위하여 EfficientNet-B0모델의 전이학습을 위해 수집한 상품 이미지의 일부를 사용해 YOLOv5를 학습시켰다. 라벨링 과정에서 상품의 전면부 외에 다양한 각도에서의 이미지와 상품 묶음 이미지를 사용했다.

 ![image](https://github.com/HongikUniv-CAPSTONE-DESIGN-2023-YDY-1/CAPSTONE_DESIGN_AI_Module/assets/117420071/19ec6410-6c81-4eb5-9308-71195a31565b)
 <br>

 
아래는 모델의 동작 예시다.

![D2D2D2](https://github.com/HongikUniv-CAPSTONE-DESIGN-2023-YDY-1/CAPSTONE_DESIGN_AI_Module/assets/117420071/b07259ab-7c7a-4a60-b24e-ba42e2b8dc6d)
<br>

모델의 정확도는 다음과 같다.
- precision : 0.991
- recall :  0.956
- mAP50 : 0.985
- mAP50-95 : 0.831

<br>
<br>


### 학습 결과
개발 완료된 모델들을 실행시키면 다음과 같은 JSON 파일이 생성된다.

 ![d3w](https://github.com/HongikUniv-CAPSTONE-DESIGN-2023-YDY-1/CAPSTONE_DESIGN_AI_Module/assets/117420071/cded5313-7958-43ca-a306-57b9418dadeb)
 <br>

#### a) 검증
아래는 기존의 EfficientNet-B0만으로 학습한 모델, 전이학습 기법을 사용한 모델, 그리고 YOLO를 이용한 이미지 crop을 진행한 모델의 정확도 비교다.

- EfficientNet-B0 :	76.38%%	
- Custom EfficientNet-B0 : 87.34%
- YOLOv5 + Custom EfficientNet-B0	: 95.14%

#### b) 배포
서버에서 모듈이 구동되기 위해 필요한 파이썬 버전과 패키지 버전은 다음과 같다.
- python == 3.9.16
- tensorflow == 2.13.0
- tensorflow_hub == 0.14.0
- Pillow == 10.0.0
- keras == 2.13.1
- 
서버에 제공할 모듈의 전체구조는 다음과 같다.

 ![image](https://github.com/HongikUniv-CAPSTONE-DESIGN-2023-YDY-1/CAPSTONE_DESIGN_AI_Module/assets/117420071/befd4464-3f93-4ff4-9440-4d749e58649e)
<br>
- Img_folder : 어플리케이션에서 전송되는 이미지를 임시 저장하는 폴더
- model_label : 전이학습이 진행된 분류모델인 .h5파일, YOLOv5로 학습시킨 객체탐지 모델인 .pt파일, 라벨 파일이 포함된 폴더.
- yolov5 : YOLO를 이용하기 위한 파일들이 포함된 폴더


img_folder내에 임시 저장한 이미지를 객체 탐지 모델을 사용해 crop을 진행한 뒤, crop된 이미지를 분류 모델을 사용해 상품 이름과 부가 정보가 포함된 라벨로 매핑해 JSON을 생성한다.
 이 모듈이 비즈니스 로직 서버와 통신하기 위해 HTTP를 사용해 통신할 수 있도록 개조해야 하는데, 이는 비즈니스 로직 서버를 개발하며 함께 진행했다.

