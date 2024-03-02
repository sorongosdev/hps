# Project name: Polyphonic note detector using Harmonic Product Spectrum
# Date:         2021.05.19
# Author:       João Nuno Carvalho
# Description:  This is my implementation of a polyphonic note detector using
#               the Harmonic Product Spectrum method.
#               The input is a mono WAV file.
#               The output are the corresponding notes in time. 
# License: MIT Open Source License


import numpy as np
import matplotlib.pyplot as plt
import wave
import math

## Configuration

# Path
path     = "./"

# filename = '18474__pitx__c4.wav'
# filename = 'c_major_guitar.wav'
filename = 'G_AcusticPlug26_1.wav'

note_threshold = 5_000.0    # 120   # 50_000.0   #  3_000.0

# Parameters
sample_rate  = 44100                     # Sampling Frequency
fft_len      = 22050   # 2048                      # Length of the FFT window
overlap      = 0.5                       # Hop overlap percentage between windows
hop_length   = int(fft_len*(1-overlap))  # Number of samples between successive frames

# For the calculations of the music scale.
TWELVE_ROOT_OF_2 = math.pow(2, 1.0 / 12)

def read_wav_file(path, filename):
    # Reads the input WAV file from HDD disc.
    # 경로와 파일 이름으로 wav 파일을 읽음
    wav_handler = wave.open(path + filename,'rb') # Read only.
    # 프레임 레이트 확인
    num_frames = wav_handler.getnframes()
    # 샘플링 레이트 확인
    sample_rate = wav_handler.getframerate()
    # wav 파일의 프레임을 읽어옴
    wav_frames = wav_handler.readframes(num_frames)

    # Loads the file into a NumPy contiguous array.

    # Convert Int16 into float64 in the range of [-1, 1].
    # This means that the sound pressure values are mapped to integer values that can range from -2^15 to (2^15)-1.
    # We can convert our sound array to floating point values ranging from -1 to 1 as follows.
    # int16을 float64로 반환
    signal_temp = np.frombuffer(wav_frames, np.int16)
    signal_array = np.zeros( len(signal_temp), float)

    for i in range(0, len(signal_temp)):
        signal_array[i] = signal_temp[i] / (2.0**15)

    # 변환된 신호 배열 출력
    print("file_name: " + str(filename))
    print("sample_rate: " + str(sample_rate))
    print("input_buffer.size: " + str(len(signal_array)))
    print("seconds: " + to_str_f4(len(signal_array)/sample_rate) + " s")
    print("type [-1, 1]: " + str(signal_array.dtype))
    print("min: " + to_str_f4(np.min(signal_array)) + " max: " + to_str_f4(np.max(signal_array))  )

    # 샘플링율과 신호의 numpy 배열 반환
    return sample_rate, signal_array

# 주어진 버퍼를 최대 길이로 나누어 겹치지 않는 청크(작은 조각)로 분할, 분할된 배열 뷰 반환
def divide_buffer_into_non_overlapping_chunks(buffer, max_len): # max_len은 fft_length를 말함
    # 버퍼의 최대 길이 확인
    buffer_len = len(buffer)
    # 최대길이로 나누어질 수 있는 청크의 수 계산
    chunks = int(buffer_len / max_len)
    print("buffers_num: " + str(chunks))
    # 분할 지점을 나타내는 인덱스 리스트 생성
    division_pts_list = []
    for i in range(1, chunks):
        division_pts_list.append(i * max_len) 
    # 버퍼를 분할 지점으로 나누어 배열 뷰를 생성
    splitted_array_view = np.split(buffer, division_pts_list, axis=0)
    # 분할된 뷰 반환
    return splitted_array_view

## 주어진 데이터에 대한 FFT(Fast Fourier Transform)를 계산하여 FFT 주파수와 FFT 결과를 반환하는 함수
def getFFT(data, rate):
    # Returns fft_freq and fft, fft_res_len.
    # 데이터 길이 확인
    len_data = len(data)
    # 해밍 윈도우 함수 적용
    data = data * np.hamming(len_data)
    # fft 계산
    fft = np.fft.rfft(data)
    # fft 결과 절대값
    fft = np.abs(fft)
    # fft 결과 길이 확인
    ret_len_FFT = len(fft)
    # fft 주파수 계산. 주파수 범위 : 0 ~ 샘플링 주파수의 절반까지
    freq = np.fft.rfftfreq(len_data, 1.0 / sample_rate)
    # return ( freq[:int(len(freq) / 2)], fft[:int(ret_len_FFT / 2)], ret_len_FFT )
    # fft 주파수 / fft 결과/ fft 길이 반환
    return ( freq, fft, ret_len_FFT )

def remove_dc_offset(fft_res):
    # Removes the DC offset from the FFT (First bin's)
    # 첫번째, 두번째, 세번째 bin에 해당하는 값을 0으로 만들고 수정된 값 반환
    # fft 결과는 주파수 성분의 크기를 나타내는 복소수 형태로 표현되는데, 첫번째 bin이 dc offset에 해당함
    # 신호의 평균값을 제거하는 과정임.
    # 평균값을 기준으로 좌우 대칭이 되는 효과
    fft_res[0] = 0.0
    fft_res[1] = 0.0
    fft_res[2] = 0.0
    return fft_res

def freq_for_note(base_note, note_index):
    # See Physics of Music - Notes
    #     https://pages.mtu.edu/~suits/NoteFreqCalcs.html
    
    A4 = 440.0

    base_notes_freq = {"A2" : A4 / 4,   # 110.0 Hz
                       "A3" : A4 / 2,   # 220.0 Hz
                       "A4" : A4,       # 440.0 Hz
                       "A5" : A4 * 2,   # 880.0 Hz
                       "A6" : A4 * 4 }  # 1760.0 Hz  

    scale_notes = { "C"  : -9.0,
                    "C#" : -8.0,
                    "D"  : -7.0,
                    "D#" : -6.0,
                    "E"  : -5.0,
                    "F"  : -4.0,
                    "F#" : -3.0,
                    "G"  : -2.0,
                    "G#" : -1.0,
                    "A"  :  1.0,
                    "A#" :  2.0,
                    "B"  :  3.0,
                    "Cn" :  4.0}

    scale_notes_index = list(range(-9, 5)) # Has one more note.
    note_index_value = scale_notes_index[note_index]
    freq_0 = base_notes_freq[base_note]
    freq = freq_0 * math.pow(TWELVE_ROOT_OF_2, note_index_value) 
    return freq

## 음정(C_2) 반환하는 함수
def get_all_notes_freq():
    ordered_note_freq = []
    ordered_notes = ["C",
                     "C#",
                     "D",
                     "D#",
                     "E",
                     "F",
                     "F#",
                     "G",
                     "G#",
                     "A",
                     "A#",
                     "B"]
    # 2옥타브~6옥타브까지 각 음계의 주파수 계산
    for octave_index in range(2, 7):
        base_note  = "A" + str(octave_index)
        # note_index = 0  # C2
        # note_index = 12  # C3
        for note_index in range(0, 12):
            note_freq = freq_for_note(base_note, note_index)
            note_name = ordered_notes[note_index] + "_" + str(octave_index) # 코드_숫자 로 음정 표시
            ordered_note_freq.append((note_name, note_freq))
    return ordered_note_freq

def find_nearest_note(ordered_note_freq, freq):
    final_note_name = 'note_not_found'
    # 노트까지의 최소 거리를 저장하는 함수.
    last_dist = 1_000_000.0
    # ordered_note_freq 리스트를 순회하면서 각 음표의 이름과 주파수를 가져옴
    for note_name, note_freq in ordered_note_freq:
        # 현재 음표와 목표 주파수 간의 거리(curr_dist)를 계산
        curr_dist = abs(note_freq - freq)
        # 현재 음표가 더 가까운 경우에는 last_dist를 curr_dist로 업데이트하고 final_note_name을 현재 음표의 이름으로 업데이트
        if curr_dist < last_dist:
            last_dist = curr_dist
            final_note_name = note_name
        # 더 이상 가까운 음표를 찾을 수 없을 때 순회 종료
        elif curr_dist > last_dist:
            break    
    return final_note_name

# HPS를 계산하여 주파수의 최댓값을 찾는 함수
def PitchSpectralHps(X, freq_buckets, f_s, buffer_rms):
    # X: 스펙트로그램 데이터 (크기: FFTLength x Observations)
    # freq_buckets: 주파수 버킷
    # f_s: 오디오 데이터의 샘플링 주파수
    # buffer_rms: RMS(Root Mean Square) 값을 기반으로 한 노트 임계값
    """
    NOTE: This function is from the book Audio Content Analysis repository
    https://www.audiocontentanalysis.org/code/pitch-tracking/hps-2/
    The license is MIT Open Source License.
    And I have modified it. Go to the link to see the original.

    computes the maximum of the Harmonic Product Spectrum

    Args:
        X: spectrogram (dimension FFTLength X Observations)
        f_s: sample rate of audio data

    Returns:
        f HPS maximum location (in Hz)
    """

    # initialize

    # hps 계산 순서
    iOrder = 4
    # 최소 주파수
    f_min = 65.41   # C2      300
    # f = np.zeros(X.shape[1])
    f = np.zeros(len(X))

    iLen = int((X.shape[0] - 1) / iOrder)
    # hps 결과를 저장, 주어진 스펙트로그램 데이터에서 일부를 추출하여 초기화됨
    afHps = X[np.arange(0, iLen)]
    # 최소 주파수에 해당하는 인덱스를 계산하는데 사용
    # 주어진 샘플링 주파수 f_s와 스펙트로그램 데이터의 크기를 기반으로 계산
    k_min = int(round(f_min / f_s * 2 * (X.shape[0] - 1)))

    # compute the HPS
    # HPS 알고리즘을 사용하여 afHps 배열을 계산
    # 이를 위해 X의 일부를 추출하고, 추출한 배열과 afHps를 곱하여 HPS를 계산
    for j in range(1, iOrder):
        X_d = X[::(j + 1)]
        afHps *= X_d[np.arange(0, iLen)]

    ## Uncomment to show the original algorithm for a single frequency or note. 
    # f = np.argmax(afHps[np.arange(k_min, afHps.shape[0])], axis=0)
    ## find max index and convert to Hz
    # freq_out = (f + k_min) / (X.shape[0] - 1) * f_s / 2

    # note_threshold_scaled_by_RMS 함수를 사용하여 buffer_rms 값을 기반으로 한 노트 임계값을 계산
    note_threshold = note_threshold_scaled_by_RMS(buffer_rms)

    # afHps에서 노트 임계값보다 큰 값의 인덱스를 찾아 k_min 값을 더한 후, 해당 인덱스를 주파수로 변환하여 freqs_out에 저장
    all_freq = np.argwhere(afHps[np.arange(k_min, afHps.shape[0])] > note_threshold)
    # find max index and convert to Hz
    freqs_out = (all_freq + k_min) / (X.shape[0] - 1) * f_s / 2

    # afHps에서 노트 임계값보다 큰 값의 인덱스 / 해당 값들을 각각 freq_indexes_out와 freq_values_out에 저장
    x = afHps[np.arange(k_min, afHps.shape[0])]
    freq_indexes_out = np.where( x > note_threshold)
    freq_values_out = x[freq_indexes_out]

    # print("\n##### x: " + str(x))
    # print("\n##### freq_values_out: " + str(freq_values_out))

    max_value = np.max(afHps[np.arange(k_min, afHps.shape[0])])
    max_index = np.argmax(afHps[np.arange(k_min, afHps.shape[0])])
    
    ## Uncomment to print the values: buffer_RMS, max_value, min_value
    ## and note_threshold.    
    print(" buffer_rms: " + to_str_f4(buffer_rms) )
    print(" max_value : " + to_str_f(max_value) + "  max_index : " + to_str_f(max_index) )
    print(" note_threshold : " + to_str_f(note_threshold) )

    ## Uncomment to show the graph of the result of the 
    ## Harmonic Product Spectrum. 
    # fig, ax = plt.subplots()
    # yr_tmp = afHps[np.arange(k_min, afHps.shape[0])]
    # xr_tmp = (np.arange(k_min, afHps.shape[0]) + k_min) / (X.shape[0] - 1) * f_s / 2
    # ax.plot(xr_tmp, yr_tmp)
    # plt.show()

    # Turns 2 level list into a one level list.
    # freqs_out와 freq_values_out를 묶어서 저장한 freqs_out_tmp를 반환
    freqs_out_tmp = []
    for freq, value  in zip(freqs_out, freq_values_out):
        freqs_out_tmp.append((freq[0], value))
    
    return freqs_out_tmp

def note_threshold_scaled_by_RMS(buffer_rms):
    # note_threshold는 어떤 음향 신호 세기를 나타낼지에 따라 조절하면 됨
    note_threshold = 1000.0 * (4 / 0.090) * buffer_rms
    return note_threshold

def normalize(arr):
    # Note: Do not use.
    # Normalize array between -1 and 1.
    # Only works if the signal is larger then the final signal and if the positive
    # value is grater in absolute value them the negative value.
    ar_res = (arr / (np.max(arr) / 2)) - 1  
    return ar_res

def to_str_f(value):
    # Returns a string with a float without decimals.
    return "{0:.0f}".format(value)

def to_str_f4(value):
    # Returns a string with a float without decimals.
    return "{0:.4f}".format(value)


def main():
    print("\nPolyphonic note detector\n")
    
    # 음정 반환
    ordered_note_freq = get_all_notes_freq()
    # print(ordered_note_freq)

    # wav 파일을 읽고 신호를 버퍼로 반환
    sample_rate_file, input_buffer = read_wav_file(path, filename)
    # 주어진 버퍼를 최대 길이로 나누어 겹치지 않는 청크(작은 조각)로 분할, buffer_chuncks는 분할된 배열 뷰
    buffer_chunks = divide_buffer_into_non_overlapping_chunks(input_buffer, fft_len)
    # The buffer chunk at n seconds:

    count = 0
    
    ## Uncomment to process a single chunk os a limited number os sequential chunks. 
    # for chunk in buffer_chunks[5: 6]:
    for chunk in buffer_chunks[0: 60]:
        print("\n...Chunk: ", str(count))
                
        # fft 결과 반환
        fft_freq, fft_res, fft_res_len = getFFT(chunk, len(chunk))
        # dc 오프셋 제거. 불필요한 값을 제거하고 주요 주파수 성분이 더 잘드러나게 하기 위함.
        fft_res = remove_dc_offset(fft_res)

        # Calculate Root Mean Square of the signal buffer, as a scale factor to the threshold.
        # rms 계산 (root mean square)
        buffer_rms = np.sqrt(np.mean(chunk**2))

        #
        all_freqs = PitchSpectralHps(fft_res, fft_freq, sample_rate_file, buffer_rms)
        # print("all_freqs ")
        # print(all_freqs)

        for freq in all_freqs:
            note_name = find_nearest_note(ordered_note_freq, freq[0])
            print("=> freq: " + to_str_f(freq[0]) + " Hz  value: " + to_str_f(freq[1]) + " note_name: " + note_name )


        ## Uncomment to print the arrays.
        # print("\nfft_freq: ")
        # print(fft_freq)
        # print("\nfft_freq_len: " + str(len(fft_freq)))

        # print("\nfft_res: ")
        # print(fft_res)

        # print("\nfft_res_len: ")
        # print(fft_res_len)


        ## Uncomment to show the graph of the result of the FFT with the
        ## correct frequencies in the legend. 
        N = fft_res_len
        fft_freq_interval = fft_freq[: N // 4]
        fft_res_interval = fft_res[: N // 4]
        fig, ax = plt.subplots()
        ax.plot(fft_freq_interval, 2.0/N * np.abs(fft_res_interval))
        plt.show()

        count += 1

if __name__ == "__main__":
    main()







