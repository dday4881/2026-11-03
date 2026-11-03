# import torch
# from torch import nn
# torch.set_num_threads(1)  # 작은 CPU 실습의 실행 조건을 고정한다.
# torch.manual_seed(42)

# x = torch.randn(4,6,3)
# rnn = nn.RNN(3,8,batch_first=True)
# output, h_n = rnn(x)  # 모든 시점 출력과 마지막 은닉 상태를 따로 받는다.
# # TODO B02: 은닉 특성 8개를 두 클래스의 점수로 바꾸는 층을 만든다.
# # 필수 위치 인자: in_features=8, out_features=2; bias=True는 기본 선택값.
# # BEGIN B02
# head = nn.Linear(8, 1)
# # END B02
# logits = head(h_n[-1])  # 마지막 층의 상태: [4,8]
# print("입력:", tuple(x.shape))
# print("output / h_n:", tuple(output.shape), tuple(h_n.shape))
# print("logits:", tuple(logits.shape))
# print("마지막 시점 일치:", torch.allclose(output[:,-1],h_n[-1]))

import torch
from torch import nn
torch.set_num_threads(1)  # 작은 CPU 실습의 실행 조건을 고정한다.
torch.manual_seed(42)

x = torch.randn(4,6,3)
# RNN(Recurrent Neural Network, 순환 신경망) 계층을 생성합니다.
# 3                : input_size, 각 시점(Time Step)에 입력되는 특성(Feature)의 개수
# 8                : hidden_size, 은닉 상태(Hidden State)의 특성 개수
# batch_first=True : 입력 및 출력 텐서의 첫 번째 차원을 배치 크기로 설정
#                    입력 Shape: (Batch, Seq_len, Features)
# num_layers       : 기본값 1, RNN 계층의 개수
# bidirectional    : 기본값 False, 단방향 RNN 사용
#
# 예시 입력 Shape  : (4, 6, 3) → 배치 4개, 시점 6개, 입력 특성 3개
# output Shape     : (4, 6, 8) → 각 시점에서 계산된 은닉 상태 8개
# h_n Shape        : (1, 4, 8) → 마지막 계층의 최종 시점 은닉 상태
rnn = nn.RNN(3, 8, batch_first=True)
output, h_n = rnn(x)  # 모든 시점 출력과 마지막 은닉 상태를 따로 받는다.
# 해설 B02: 은닉 특성 8개를 두 클래스의 점수로 바꾸는 층을 만든다.
# 필수 위치 인자: in_features=8, out_features=2; bias=True는 기본 선택값.
# BEGIN B02
head = nn.Linear(8, 2)
# END B02
logits = head(h_n[-1])  # 마지막 층의 상태: [4,8],  -1이라는 뜻은 마지막 층의 은닉 상태를 의미한다.
print("입력:", tuple(x.shape))
print("output / h_n:", tuple(output.shape), tuple(h_n.shape))
print("logits:", tuple(logits.shape))
print("마지막 시점 일치:", torch.allclose(output[:,-1],h_n[-1]))
