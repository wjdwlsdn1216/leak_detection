# 상하수도 데이터 기반 누수감지 모델 개발

-------------------------------------------------------------------------------------------------------------------------

## 개요
> ### 누수감지 모델 개발 목적
> 1. 상수관로 스마트 누수 감지 자동화 서비스를 위한 모델 개발
> 2. 전국적으로 발생하는 누수를 빠르게 알아내고 해결하기 위한 목적 

## 개발환경
### pip list
```
python                        3.7.13
ipykernel                     6.13.0
jupyter                       1.0.0
Keras-Preprocessing           1.1.2
lxml                          4.6.2
Markdown                      3.3.7
matplotlib                    3.5.2
numpy                         1.19.5
pandas                        1.3.5
pip                           21.2.4
pydot                         1.4.2
scikit-learn                  1.0.2
scipy                         1.7.3
sklearn                       0.0
tensorboard                   2.9.0
tensorboard-data-server       0.6.1
tensorboard-plugin-wit        1.8.1
tensorboardX                  2.5
torch                         1.8.2+cu111
torchaudio                    0.8.2
torchvision                   0.9.2+cu111
```

## 데이터
> AIHUB 상수관로 데이터
>   [AIHUB 링크](https://aihub.or.kr/aidata/27709)
> 
> 데이터 레이블 및 제공 데이터량
> <img src="/README_img/data.PNG" width="80%" height="80%" title="data" alt="data"></img>

## 알고리즘
> Pytorch
> Neural Network(신경망)
> linear regression(선형회귀)
