import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

# 데이터 전처리
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

# 데이터 로더 지정
def data_loader(X_train, y_train):
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=64) #배치사이즈 64 지정

    return train_loader