import torch.nn as nn

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

# 가중치 초기화 함수
# 레이어간 대칭적인 가중치를 갖지 않게 하기위함
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform(m.weight) # 균등 분포 사용
        m.bias.data.fill_(0.01)