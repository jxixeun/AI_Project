import tensorflow as tf
import tensorflow_hub as hub
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import librosa
from librosa import display as librosadisplay
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import logging
import math
import statistics
import sys
from scipy.io import wavfile
from base64 import b64decode
import music21
from pydub import AudioSegment
import csv

# 코드 참조 https://www.tensorflow.org/hub/tutorials/spice?hl=ko

MAX_ABS_INT16 = 32768.0
EXPECTED_SAMPLE_RATE = 8000 # 데이터 세트의 샘플레이트 값이 8000이기 때문에 8000으로 설정한다.

def convert_audio_for_model(user_file, output_file='converted_audio_file.wav'):
  audio = AudioSegment.from_file(user_file)
  audio = audio.set_frame_rate(EXPECTED_SAMPLE_RATE).set_channels(1)
  audio.export(output_file, format="wav")
  return output_file

path = "C:\\Users\\jieun\\Desktop\\MIR-QBSH-corpus\\waveFile\\year2009" # 변환할 wav파일이 담겨져 있는 폴더
file_lst = os.listdir(path) # 현재 디렉토리 내 모든 파일이 리스트에 담겨진다.
# 각 파일명은 '년도-사람-노래 레이블 번호.wav'

for file in file_lst:
    filepath = path + '/' + file # 파일의 경로를 구한다.
    file_name = file.rsplit('.')[0] # 확장자가 없는 파일 이름을 구한다.
    label = int(file_name.split('-')[2]) # 파일명이 '년도-사람-노래 레이블 번호'이기 때문에, 노래 레이블을 얻기 위해서 -로 나누고 마지막 값을 저장한다.
    converted_audio_file = convert_audio_for_model(filepath)
    sample_rate, audio_samples = wavfile.read(converted_audio_file, 'rb')

    duration = len(audio_samples) / sample_rate
    audio_samples = audio_samples / float(MAX_ABS_INT16)

    model = hub.load("https://tfhub.dev/google/spice/2") # 텐서플로우의 spice모델을 이용해 피치값과 신뢰도를 구한다.
    model_output = model.signatures["serving_default"](tf.constant(audio_samples, tf.float32))

    pitch_outputs = model_output["pitch"]
    pitch_outputs = [float(x) for x in pitch_outputs]
    uncertainty_outputs = model_output["uncertainty"]
    confidence_outputs = 1.0 - uncertainty_outputs # 불확도는 신뢰도의 반대이다. 1-불확도를 해서 신뢰도를 구한다.
    confidence_outputs = [float(y) for y in confidence_outputs]
    indices = range(len(pitch_outputs)) # [0,1,2,....,126]
    # 신뢰도(c)가 0.9이상인 pitch값들을 confident_pitch_outputs에 담는다.
    confident_pitch_outputs = [(i, p) for i, p, c in zip(indices, pitch_outputs, confidence_outputs) if c >= 0.9]
    # pitch값은 confident_pitch_outputs_y이다.
    confident_pitch_outputs_x, confident_pitch_outputs_y = zip(*confident_pitch_outputs)
    # 시작 피치값을 1로 통일 시켜주기 위해서 (음역대를 맞추기 위함) 1에서 시작 피치값을 뺀 값 value를 구한다.
    # value를 pitch값 전체에 더해서 시작 pitch값을 1로 맞춰준다.
    value = 1 - confident_pitch_outputs_y[0]
    outputs = [x+value for x in confident_pitch_outputs_y] # [ 1.0, ..., ]
    # pitch값이 담겨있는 output 리스트에 레이블 값을 넣어준다. 한번에 csv파일로 저장하기 위함이다.
    outputs.append(label)
    # test-file.csv에 피치값 리스트+레이블 값이 한 행씩 추가된다. 디렉토리에 있는 모든 파일의 피치값이 추출되면 종료된다.
    outfile = open('test-file.csv', 'a', newline='')
    out = csv.writer(outfile)
    out.writerow(outputs)



### 신뢰도가 낮은 피치값을 제거하지 않고 사용하는 경우
# 위에는 모두 동일한 코드를 사용한다.

"""
for file in file_lst:
    filepath = path + '/' + file  # 파일의 경로를 구한다.
    file_name = file.rsplit('.')[0]  # 확장자가 없는 파일 이름을 구한다.
    label = int(file_name.split('-')[2])  # 파일명이 '년도-사람-노래 레이블 번호'이기 때문에, 노래 레이블을 얻기 위해서 -로 나누고 마지막 값을 저장한다.
    converted_audio_file = convert_audio_for_model(filepath)
    sample_rate, audio_samples = wavfile.read(converted_audio_file, 'rb')

    duration = len(audio_samples) / sample_rate
    audio_samples = audio_samples / float(MAX_ABS_INT16)

    model = hub.load("https://tfhub.dev/google/spice/2")
    model_output = model.signatures["serving_default"](tf.constant(audio_samples, tf.float32))

    pitch_outputs = model_output["pitch"]
    pitch_outputs = [float(x) for x in pitch_outputs] # 126개의 피치값 (신뢰도가 낮은 피치값 전부 포함)
    pitch_outputs.append(label) # pitch값이 담겨있는 output 리스트에 레이블 값을 넣어준다. 한번에 csv파일로 저장하기 위함.
    # test-file.csv에 피치값 리스트+레이블 값이 한 행씩 추가된다. 디렉토리에 있는 모든 파일의 피치값이 추출되면 종료된다.
    outfile = open('test-file.csv', 'a', newline='')
    out = csv.writer(outfile)
    out.writerow(pitch_outputs)
"""

