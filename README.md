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

## 설치

> 1. clone repo
> 2. install requirements.txt 
> 3. main.py 실행

## 활용 알고리즘

> - Neural Network(신경망)
> - linear regression(선형회귀)

-------------------------------------------------------------------------------------------------------------------------
## 데이터 분석
[data_analysis.ipynb](https://github.com/wjdwlsdn1216/leak_detection/blob/main/raw_data_handling/data_analysis.ipynb)
> ### 1. 독립변수 분석
> + 주파수 대역별 진동수치
>   + 100HZ에서 500HZ 사이가 가장 많은 변동이 있는걸 확인할 수 있다.
> <img src="/README_img/output.png" width="80%" height="80%" title="output" alt="output"></img>
> + 최대 주파수 및 최대 누수 평균 수치
>   + x축의 MAX짝수는 최대 주파수이고, MAX홀수는 최대 누수 수치다.
>   + y축은 평균값을 나타낸 수치이다.
>   + 최대 주파수 수치는 noise -> normal -> other -> out -> in 순서로 주파수 대역이 높아진다.
>   + 최대 누수 수치는 normal -> in -> other -> out -> noise 순서로 누수크기가 높아진다.
> <img src="/README_img/output1.png" width="80%" height="80%" title="output1" alt="output1"></img>
> ### 2. 변수 간의 상관관계
> + lrate(누수확률), llevel(누수레벨), ldate(누수날짜), leaktype(레이블)간의 상관관계
>   + 분석결과 llevel과 lrate이 상관관계가 높다는걸
heatmap을 통해 알 수 있다.
> <img src="/README_img/output2.png" width="80%" height="80%" title="output2" alt="output2"></img>
> ### 3. 누수확률별 누수크기 비교
> + 평균 누수확률 평균 누수크기 수치
>   + 정상음(normal)을 제외하고 나머지 leaktype은 데이터상 전부 90퍼센트 이상 누수확율 발생
>   + 정상음도 70퍼센트 이상 확률로 누수확율이 발생 할수도 있다는걸 알 수 있다.
>   + 이상치 수치가 옥내누수(in), 정상음(normal)에서 많이 보이는걸 알 수 있다.
> <img src="/README_img/output3.png" width="80%" height="80%" title="output3" alt="output3"></img>
> ### 4. 분석 결과
> + 데이터 독립 변수와 종속 변수 분석 결과 lrate(누수확율)은 normal(정상음)을 제외한 나머지 소리는 전부 90프로 이상 누수 발생
> + 평균적으로 140hz에서 530hz 사이의 주파수에서 누수 진동 크기가 가장 많이 발생한다는 것을 알 수 있다.
> + 최대 감지 주파수는 350hz~600hz 사이가 평균이고, 최대 누수 크기는 normal(정상음)을 제외하고 500~600 사이로 비슷하다.
> + 평균적으로 out(옥외누수), in(옥내누수), other(환경음)은 감지되는 최대 주파수와 최대 누수크기 수치의 범위가 가깝다.
> + 결과적으로 sid, site 변수는 감지 기계와 사이트 번호라는 의미없는 데이터라 삭제하고
ldate 변수는 다른 변수와의 상관관계가 매우 낮아서 학습시 제거한다.

## 데이터 전처리

> ### 필요없는 컬럼을 제거한후 모델 input에 적합한 구조로 변환 과정을 거쳤다.
> - 우선 pandas concat 함수로 train, test 데이터를 각각 모두 합쳐줬다.
```python
# train test 데이터 모두 병합
train_data = pd.concat([train_leak_out, train_leak_in, train_leak_noise, train_leak_other, train_leak_normal])
test_data = pd.concat([test_leak_out, test_leak_in, test_leak_noise, test_leak_other, test_leak_normal])

train_data.to_csv('./data/Training/train_data.csv', index=False)
test_data.to_csv('./data/Validation/test_data.csv', index=False)
```
> - 데이터 전처리 핵심 코드 [data_preprocessing.py](https://github.com/wjdwlsdn1216/leak_detection/blob/main/data_preprocessing.py)
```python
def preprocessing(train_data, test_data, le):

    #필요없는 데이터 컬럼 삭제
    train_data = train_data.drop(['site','sid','ldate'], axis = 1)
    test_data = test_data.drop(['site','sid','ldate','leaktype'], axis = 1)

    # label로 지정할 컬럼 실수화
    train_data['leaktype'] = le.fit_transform(train_data['leaktype'])

    # 데이터 넘파이 배열로 변환
    train_data = np.array(train_data)
    test_data = np.array(test_data)

    # 배열 데이터 텐서로 변환 X축은 float32 y축은 int64
    X_train = torch.FloatTensor(train_data[:,1:])
    y_train = torch.LongTensor(train_data[:,0])
    X_test = torch.FloatTensor(test_data[:,:])

    return X_train, y_train, X_test
```

## 데이터 모델링

> ### Pytorch를 활용하여 커스텀 모델을 구축
> - 모델 = linear, 활성화함수 = relu를 활용한 커스텀 모델 구축 [model.py](https://github.com/wjdwlsdn1216/leak_detection/blob/main/model.py)
```python
# 커스텀 모델 지정
class CustomModel(nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()
        
        # 모델 = 선형회귀, 활성화함수 = relu, 과적합 막기위한 각 레이어마다 dropout 지정
        self.sequential = nn.Sequential(
            nn.Linear(533,1024),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(1024,1024),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(1024,512),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(512,512),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(512, 5)
        )
    # 순전파 함수
    def forward(self, x):
        logits = self.sequential(x)
        return logits
```
> - layer간의 대칭적인 가중치 값을 우려하여 가중치 초기화하는 함수를 추가
```python
# 가중치 초기화 함수
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)
```

## 모델 학습 결과 분석

> - tensorboard 라이브러리를 활용한 손실 그래프
> <img src="/README_img/training_loss.PNG" width="80%" height="80%" title="training_loss" alt="training_loss"></img>
> - Confusion Matrix를 활용한 결과 도출
>   - 5개 레이블의 TP, TN, FP, FN 시각화 [Confusion Matrix 설명 링크](https://nittaku.tistory.com/295)
> <img src="/README_img/cf_matrix.png" width="80%" height="80%" title="cf_matrix" alt="cf_matrix"></img>
> - 분류평가지표 (정밀도, 재현율, F1스코어, 정확도)
> <img src="/README_img/accuracy.PNG" width="80%" height="80%" title="accuracy" alt="accuracy"></img>

## 결론

> - 
> - 
> - 

## 참고문서

* Pytorch : https://pytorch.org/tutorials/beginner/examples_nn/polynomial_module.html
* kaggle : https://www.kaggle.com/c/2021sejongai-tp-17011815
* blog : https://nittaku.tistory.com/295
* blog : https://gggggeun.tistory.com/17
