import torch
from torch import nn

# 작은 CPU 실습의 실행 조건을 고정한다.
# 멀티스레딩 연산을 제한하고 무작위 시드를 고정하여, 언제 실행해도 똑같은 예측 값이 나오도록 만듭니다.
torch.set_num_threads(1)  
torch.manual_seed(42)

from pathlib import Path
import pandas as pd

# 파일 경로 및 시계열(CSV) 데이터 읽기
ROOT = Path(__file__).resolve().parent
df = pd.read_csv(ROOT / "series.csv")

# 시간 구간을 먼저 분리한다. test 구간은 모델에 넣지 않는다.
# pandas 데이터프레임에서 'split' 열이 'train'과 'valid'인 행의 'value'값만 쏙 골라내 파이토치 텐서로 변환합니다.
train = torch.tensor(df.loc[df["split"]=="train","value"].values, dtype=torch.float32)
valid = torch.tensor(df.loc[df["split"]=="valid","value"].values, dtype=torch.float32)

# unfold(0, 13, 1): 과거 12개 + 다음 정답 1개인 묶음을 만든다.
# 시계열 데이터를 한 칸씩(1) 밀어가면서 13개짜리 연속된 윈도우 묶음으로 쪼갭니다.
window = train.unfold(0, 13, 1)

# x: 입력 데이터 (13개 중 앞의 12개 덤프). 
# LSTM에 넣으려면 (배치, 시점수, 특징수)의 3차원 형태가 필요하므로 .unsqueeze(-1)로 맨 뒤에 '특징수 1개' 차원을 늘려줍니다.
x = window[:, :12].unsqueeze(-1)

# y: 실제 정답 데이터 (13개 중 맨 마지막 13번째 값) -> (배치 크기, 1) 형태
y = window[:, 12:13]

# LSTM 레이어 정의: 입력 특징 1개 -> 은닉 상태(기억 소자) 8개로 확장
# batch_first=True로 설정하여 데이터의 첫 번째 차원이 '배치 크기(48)'가 되도록 지징합니다.
lstm = nn.LSTM(1, 8, batch_first=True)

# 예측을 위한 최종 선형 층(Head): LSTM이 요약한 8개의 특징을 받아 최종 다음 값 '1개'를 예측합니다.
head = nn.Linear(8, 1)

# LSTM에 데이터를 통과시켜 결과를 뽑아냅니다.
# - output: 모든 시점(1~12번째)의 은닉 출력들의 모음 -> 크기: (배치 48, 시점 12, 특징 8)
# - h_n: 맨 마지막 12번째 시점의 최종 은닉 상태 단 한 장 -> 크기: (레이어수 1, 배치 48, 특징 8)
# - c_n: 맨 마지막 12번째 시점의 세포 상태(장기 기억) 단 한 장 -> 크기: (레이어수 1, 배치 48, 특징 8)
output, (h_n, c_n) = lstm(x)

# TODO B03: 12개 관측의 마지막 은닉 출력으로 다음 값 1개를 예측한다.
# [:, -1, :]는 마지막 시점이다. 이 예제의 Window 길이는 모두 12이다.
# BEGIN B03

# [실험 복습]
# - output[:, 0, :]을 넣으면: 1번째 날(과거) 기억만 보고 예측하므로 오차가 큽니다. (첫 예측: -0.106506 / MSE: 0.484213)
# - output[:, -1, :]을 넣으면: 12번째 날(현재 직전)까지의 흐름이 누적된 최종 요약을 보고 예측하므로 훨씬 정확합니다.
# (Tip: head(h_n[0])을 해도 output[:, -1, :]과 완벽히 일치하는 정상 값이 나옵니다.)
prediction = head(output[:, -1, :])

# END B03

# 전체 데이터 크기 구조를 출력하여 매칭 상태를 확인합니다.
print("입력 / 정답:", tuple(x.shape), tuple(y.shape)) # 출력: (48, 12, 1) (48, 1)
print("h_n / c_n:", tuple(h_n.shape), tuple(c_n.shape)) # 최종 시점 요약본의 크기 (1, 48, 8)

# 예측값 구조와 실제 수치, 그리고 정답 대비 오차를 출력합니다.
print("예측:", tuple(prediction.shape)) # 출력: (48, 1)
print("첫 예측:", round(prediction[0].item(), 6)) # 정상 정상값: 0.014645