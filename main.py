import pandas as pd
from sklearn.preprocessing import LabelEncoder

import torch
import torch.nn as nn
import random
random.seed(777)
torch.manual_seed(777)
device = torch.device("cpu")
# if torch.cuda.is_available():
#     device = torch.device('cuda')

from data_preprocessing import preprocessing, data_loader # 데이터 전처리 함수 로드
from model import CustomModel, init_weights # 모델, 가중치초기화 함수 로드

# 학습
def train(model, train_loader):

    loss = nn.CrossEntropyLoss().to(device) # 다중분류 대표적인 손실함수
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.002) #adam 옵티마이저 선택, 학습률 = 0.001
    epoch = 5

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
    H = model(X_test)
    predict = torch.argmax(H, dim = 1)

    predict = le.inverse_transform(predict.cpu()) #int > object 변환
    pred = pd.DataFrame(columns=['leaktype']) # 예측값 대입하기 위한 데이터프레임
    pred['leaktype'] = predict

    return pred

if __name__ == "__main__":

    le = LabelEncoder()
    # 데이터 로드
    train_data = pd.read_csv('./data/Training/train_data.csv')
    test_data = pd.read_csv('./data/Validation/test_data.csv')

    X_train, y_train, X_test = preprocessing(train_data, test_data, le)
    train_loader = data_loader(X_train, y_train)
    
    # 모델 학습
    model = CustomModel().to(device)
    model = model.apply(init_weights)
    print(model) # 모델 구조
    model = train(model, train_loader)
    
    # 모델 저장
    torch.save(model.state_dict(),'./model/model.pth')

    # 모델 로드
    # model = CustomModel().to(device)
    # model.load_state_dict(torch.load('./model/model.pth'))

    # 예측 및 csv로 저장
    predict = pred(model, X_test)
    predict.to_csv("./predict.csv", index = False)

    # 테스트 데이터의 leaktype과 예측한 결과 비교
    compare = predict == test_data[['leaktype']]
    count = compare.value_counts()
    Prediction_result_accuracy = "Accuracy : {:.3f}".format(count[1]/count.sum()*100)
    
    print(Prediction_result_accuracy) # 정확도


