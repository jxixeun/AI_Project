import sys, os
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import numpy as np
import knn
import csv
from csv import reader
from sklearn.model_selection import train_test_split
import random

pitch_list =[] # 피치값을 담을 리스트
label_list = [] # 레이블을 담을 리스트

# spice model로 구해진 피치값들이 담겨져 있는 csv파일
# csv의 각 행에 있는 데이터의 수, 즉 row의 길이가 다르기 때문에, 같은 값으로 맞춰줘야 한다.
# 50으로 맞추기 위해서 길이가 50 이하인 row는 pitch_list에 저장하지 않는다. 가장 마지막 데이터는 label이기 때문에, 레이블을 포함해 길이가 51 이상인 row만 pitch_list에 담는다.
with open('/Users/jieun/Downloads/file-remove-uncertainty.csv', 'r', newline='', encoding='utf-8') as f:
    reader = csv.reader(f)
    for row in reader:
        row = list(map(float, row))
        if len(row) >= 51:
            pitch_list.append(row)
            label_list.append(row[-1]) # 가장 마지막 값 = label 값을 따로 리스트에 저장한다.

# 리스트의 길이를 50으로 맞춰주기 위해서 슬라이싱한다.
pitch_data = [ pitch_list[i][:50] for i in range(len(pitch_list))]
# 넘파이 어레이로 만들어 준다.
pitch_data = np.array(pitch_data)
label_data = np.array(label_list)


"""

##### knn을 100번 실행할 경우 아래의 주석처리되지 않은 코드를 지우고 사용

print("-- K-Nearest Neighbor use weighted majority Vote --")

sum = 0 # knn을 100번 실행하고 정확도의 평균을 구할때 사용

for i in range(100):
    # 전체 데이터에서 테스트 데이터와 학습 데이터를 나누기 위해서 train_test_split을 사용한다.
    # random_state를 i값으로 사용한다. 0~99까지의 값이 random_state값으로 들어가게 된다.
    # test_size는 0.01로 약 400개의 값이다.
    train_pitch, test_pitch, train_label, test_label = train_test_split(pitch_data, label_data, test_size=0.01, random_state=i)
    size = len(test_pitch) # 사용할 테스트 데이터의 개수
    # print(len(train_pitch))
    # print(len(test_pitch))
    knn3 = knn.KNNalgorithm(train_pitch, train_label, size, 3) # 학습할 데이터, 레이블, 테스트할 데이터의 갯수, k를 인자로 넣는다.
    knn3_result = knn3.weighted_majority_vote(test_pitch) # 테스트 데이터를 넣어 호출한다.
    cnt = 0 # 정확도를 체크하기 위한 cnt 변수이다.
    for j in range(size):
        # 예측한 값이 담겨있는 knn_result와 실제 레이블인 test_label을 비교해서 맞을 경우 cnt값을 1씩 증가해준다.
        if knn3_result[j] == test_label[j]:
            cnt += 1
        # 예측 레이블과 실제 레이블 결과 출력
        # print("%d th data result %d   label %d" % (j, knn_result[j] , test_label[j]))
    # 정확도를 구해준다.
    print("K = 3 accuracy:%d%%" % (cnt / size * 100))
    # 100개의 평균을 내기 위해서 sum에 구해진 정확도를 더해준다.
    sum += (cnt / size * 100)

# 총 100번 진행했으므로 100으로 나눠서 정확도의 평균을 구한다.
print(sum/100)

"""


##### k=1,3,5,7번로 설정해 값을 구할 때

# test_size=0.1이다. 약 400여개의 값을 테스트 데이터로 사용한다. random_state는 아무런 값을 넣어 사용한다.
train_pitch, test_pitch, train_label, test_label = train_test_split(pitch_data, label_data, test_size=0.1, random_state=0)

size = len(test_pitch) # 사용할 테스트 데이터의 개수

print("-- K-Nearest Neighbor use weighted majority Vote --")

# K=1 일때
knn1 = knn.KNNalgorithm(train_pitch, train_label, size, 1)
knn1_result = knn1.weighted_majority_vote(test_pitch)
cnt = 0  # 정확도를 측정하기위해 사용한 변수. 결과가 일치할 경우 1씩 값을 증가시킨다.
for i in range(size):
    if knn1_result[i] == test_label[i]:
        cnt += 1
    # print("%d th data result %d   label %d" % (i, knn_result[i] , test_label[i]))
print("K = 1 accuracy:%d%%" % (cnt/size*100))

# K=3 일때
knn3 = knn.KNNalgorithm(train_pitch, train_label, size, 3)
knn3_result = knn3.weighted_majority_vote(test_pitch)
cnt = 0
for i in range(size):
    if knn3_result[i] == test_label[i]:
        cnt += 1
    # print("%d th data result %d   label %d" % (i, knn_result[i] , test_label[i]))
print("K = 3 accuracy:%d%%" % (cnt/size*100))

# K=5 일때
knn5 = knn.KNNalgorithm(train_pitch, train_label, size, 5)
knn5_result = knn5.weighted_majority_vote(test_pitch)
cnt = 0
for i in range(size):
    if knn5_result[i] == test_label[i]:
        cnt += 1
    # print("%d th data result %d   label %d" % (i, knn_result[i] , test_label[i]))
print("K = 5 accuracy:%d%%" % (cnt/size*100))

# K=7 일때
knn7 = knn.KNNalgorithm(train_pitch, train_label, size, 7)
knn7_result = knn7.weighted_majority_vote(test_pitch)
cnt = 0
for i in range(size):
    if knn7_result[i] == test_label[i]:
        cnt += 1
    # print("%d th data result %d   label %d" % (i, knn_result[i] , test_label[i]))
print("K = 7 accuracy:%d%%" % (cnt/size*100))

