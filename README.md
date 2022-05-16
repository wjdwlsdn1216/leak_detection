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
> - Pytorch
> - Neural Network(신경망)
> - linear regression(선형회귀)

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
> - 데이터 전처리 핵심 코드
```python
def preprocessing(train_data, test_data, le):

    #필요없는 데이터 컬럼 삭제
    train_data = train_data.drop(['site','sid','ldate','lrate','llevel'], axis = 1)
    test_data = test_data.drop(['site','sid','ldate','lrate','llevel','leaktype'], axis = 1)

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

## 결과
