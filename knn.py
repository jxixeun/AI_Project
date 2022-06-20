import numpy as np


class KNNalgorithm:
    def __init__(self, traindata, traintarget, size, k):
        self.data = traindata  # 학습시킬 데이터
        self.target = traintarget  # 학습시킬 데이터의 레이블
        self.k = k  # K의 갯수
        self.result = np.zeros(size)  # 알고리즘을 통해 예측한 결과들을 담을 리스트
        self.weight = np.zeros(k)  # Weighted majority vote에 사용할 가중치를 담을 리스트

    def distance(self, a, b):
        return np.sqrt(np.sum(np.square((a - b))))  # 유클리드 기반 거리 계산 후 결과 리턴

    def kneigbors(self, p):
        dist = np.zeros(len(self.data))  # 계산할 거리를 담을 리스트
        for i in range(len(self.data)):
            dist[i] = self.distance(self.data[i], p)
        a = np.argsort(dist)  # 계산한 거리를 인덱스로 sort한다
        knearest = a[:self.k]  # 가장 가까운 거리의 인덱스 k개를 뽑아 knearest에 저장한다
        output = np.zeros(self.k)  # k개의 최근접 이웃의 레이블을 담을 리스트
        """
        k개의 최근접 이웃의 레이블(숫자 0.,1.,2.)을 순서대로 output 리스트에 저장한다.
        이때 가중치도 같이 저장한다. 가중치는 1/d (d는 계산한 거리)이다.
        저장 후 output을 리턴한다.
        """
        for i in range(self.k):
            output[i] = self.target[knearest[i]]
            min_dist = min(dist[dist>0])
            if dist[knearest[i]] != 0:
                self.weight[i] = 1 / dist[knearest[i]]
            else:
                self.weight[i] = (1 / min_dist) + 1
        return output

    def majority_vote(self, p):
        for i in range(len(p)):
            output = self.kneigbors(p[i])
            numcount = {0.0: 0, 1.0: 0, 2.0: 0, 3.0: 0, 4.0: 0, 5.0: 0, 6.0: 0, 7.0: 0, 8.0: 0, 9.0: 0, 10: 0, 11: 0,
                        12: 0, 13: 0, 14: 0, 15: 0, 16: 0, 17: 0, 18: 0, 19: 0, 20: 0, 21: 0, 22: 0, 23: 0, 24: 0,
                        25: 0, 26: 0, 27: 0, 28: 0, 29: 0, 30: 0, 31: 0, 32: 0, 33: 0, 34: 0, 35: 0, 36: 0, 37: 0,
                        38: 0, 39: 0, 40: 0, 41: 0, 42: 0, 43: 0, 44: 0, 45: 0, 46: 0, 47: 0, 48: 0}
            """
            for문을 돌며 각 레이블의 갯수를 세어준다. 가중치가 없기 때문에 1을 더해준다.
            numcount에는 각 레이블의 갯수가 저장된다.
            """
            for j in output:
                numcount[j] += 1
            maxlabel = max(numcount, key=numcount.get)  # 가장 많은 투표수를 얻은 레이블의 인덱스 = 예측한 레이블 결과
            self.result[i] = maxlabel  # 각 테스트 데이터마다 예측된 레이블 결과값을 result 리스트에 저장한다
        return self.result  # 테스트 데이터에 대한 전체 예측 레이블 결과 리턴

    def weighted_majority_vote(self, p):
        for i in range(len(p)):
            output = self.kneigbors(p[i])
            numcount = {0.0: 0, 1.0: 0, 2.0: 0, 3.0:0, 4.0:0, 5.0:0, 6.0:0, 7.0:0, 8.0:0, 9.0:0, 10:0, 11:0, 12:0, 13:0, 14:0, 15:0,16:0, 17:0, 18:0, 19:0, 20:0, 21:0, 22:0, 23:0, 24:0, 25:0,26:0, 27:0, 28:0, 29:0, 30:0, 31:0, 32:0,33:0,34:0,35:0,36:0,37:0,38:0,39:0,40:0,41:0,42:0,43:0,44:0,45:0,46:0,47:0,48:0}
            """
            for문을 돌며 각 레이블의 1*가중치 값을 더해준다.
            1 * weight = weight 이므로 가중치만 더한다. 
            numcount에는 가중치가 곱해진 레이블의 투표 수가 저장된다.
            ex) 가중치가 0.5이고 레이블이 0인 경우
            numcount[0]+=0.5 -> 0번째 레이블에 0.5만큼 투표수가 더해진다
            """
            for j in range(len(output)):
                numcount[output[j]] += self.weight[j]
            maxlabel = max(numcount, key=numcount.get)  # 가장 많은 투표수를 얻은 레이블의 인덱스 = 예측한 레이블 결과
            self.result[i] = maxlabel  # 각 테스트 데이터마다 예측된 레이블 결과값을 result 리스트에 저장한다
        return self.result  # 테스트 데이터에 대한 전체 예측 레이블 결과 리턴
