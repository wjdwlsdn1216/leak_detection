import pandas as pd
from sklearn.preprocessing import LabelEncoder

import torch
import torch.nn as nn
import random
random.seed(777)
torch.manual_seed(777)
device = torch.device("cpu")
from sklearn import metrics
import seaborn as sns
import matplotlib.pyplot as plt
# if torch.cuda.is_available():
#     device = torch.device('cuda')

from data_preprocessing import preprocessing, data_loader # 데이터 전처리 함수 로드
from model import CustomModel, init_weights # 모델, 가중치초기화 함수 로드

# 학습
def train(model, train_loader):

    loss = nn.CrossEntropyLoss().to(device) # 다중분류 대표적인 손실함수
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.002) #adam 옵티마이저 선택, 학습률 = 0.001
    epoch = 3

    model.train()
    total_batch = len(train_loader)

    for i in range(epoch+1):
        avg_cost = 0
        
        for X, y in train_loader:
            X = X.to(device)
            y = y.to(device)
            
            # 선형 함수 계산
            H = model(X_train)
            # 손실 계산
            cost = loss(H, y_train)

            # 손실값으로 선형모델 개선
            optimizer.zero_grad()
            cost.backward() # 역전파 진행
            optimizer.step()

            avg_cost += cost / total_batch # 평균손실값 계산

        print("Epoch :", i, "Cost :", format(avg_cost))

    return model

# 예측 함수
def pred(model, X_test):

    model.eval()
    H = model(X_test) # 모델에 x테스트 input
    predict = torch.argmax(H, dim = 1) # 차원 축소

    predict = le.inverse_transform(predict.cpu()) #int > object 변환
    pred = pd.DataFrame(columns=['leaktype']) # 예측값 대입하기 위한 데이터프레임
    pred['leaktype'] = predict

    return pred

if __name__ == "__main__":

    le = LabelEncoder()
    # 데이터 로드
    train_data = pd.read_csv('./data/Training/train_data.csv')
    test_data = pd.read_csv('./data/Validation/test_data.csv')

    X_train, y_train, X_test = preprocessing(train_data, test_data, le) # 데이터 전처리
    train_loader = data_loader(X_train, y_train) # 데이터 로더에 저장
    
    # 모델 학습
    # model = CustomModel().to(device) # 선형모델 지정
    # model = model.apply(init_weights) # 가중치 초기화 적용
    # print(model) # 모델 구조
    # model = train(model, train_loader) # 모델 학습
    
    # 모델 저장
    # torch.save(model.state_dict(),'./model/model.pth')

    # 모델 로드
    model = CustomModel().to(device)
    model.load_state_dict(torch.load('./model/model.pth'))

    # 예측
    predict = pred(model, X_test) # leaktype 예측 결과 도출
    
    y_pred = list(predict['leaktype']) # 예측결과 리스트 변환
    y_test = list(test_data['leaktype']) # 테스트 데이터 리스트 변환

    labels = ['out','in','noise','other','normal']
    cf_matrix = metrics.confusion_matrix(y_test, y_pred, labels=labels) # 혼동행열 함수

    fig, ax = plt.subplots()
    sns.heatmap(cf_matrix, annot=True, fmt='g', xticklabels=labels, yticklabels=labels) # 히트맵으로 시각화
    ax.invert_yaxis()

    plt.title("Confusion matrix of the leaktype")
    plt.xlabel("Predict labels")
    plt.ylabel("True labels")
    
    plt.savefig('./README_img/cf_matrix.png')
    plt.show()
    
    print(metrics.classification_report(y_test, y_pred, labels=labels))
    print("Accuracy : ",metrics.accuracy_score(y_test, y_pred))


