# load audio file with Librosa
import numpy as np
import librosa, librosa.display 
import matplotlib.pyplot as plt

FIG_SIZE = (15,10)

file = "./G_AcusticPlug26_1.wav"


sig, sr = librosa.load(file, sr=22050)

# STFT -> spectrogram
hop_length = 512  # 전체 frame 수
n_fft = 2048  # frame 하나당 sample 수

# calculate duration hop length and window in seconds
hop_length_duration = float(hop_length)/sr
n_fft_duration = float(n_fft)/sr

# STFT
stft = librosa.stft(sig, n_fft=n_fft, hop_length=hop_length)

# 복소공간 값 절댓값 취하기
magnitude = np.abs(stft)

# magnitude > Decibels 
log_spectrogram = librosa.amplitude_to_db(magnitude)

# display spectrogram
plt.figure(figsize=FIG_SIZE)
librosa.display.specshow(log_spectrogram, sr=sr, hop_length=hop_length)
plt.xlabel("Time")

# y축 눈금과 눈금 라벨 직접 설정하기
y_ticks = np.arange(0, 500, 100)  # 예를 들어, 0부터 Nyquist 주파수까지 2000Hz 간격으로 눈금 설정
y_labels = [f"{int(y)} Hz" for y in y_ticks]  # 눈금 라벨을 kHz 단위로 설정
plt.yticks(y_ticks, y_labels)  # y축 눈금과 라벨을 적용

# plt.ylabel("Frequency")

# # y축 눈금 보여주기 위한 코드 추가
# plt.yticks(np.arange(0, sr/2, step=2000), map(str, np.arange(0, sr/2, step=2000)))

plt.colorbar(format="%+2.0f dB")
plt.title("Spectrogram (dB)")
plt.show()