#!/usr/bin/env python3

# NOTE: this example requires PyAudio because it uses the Microphone class

import time
import pickle
import speech_recognition as sr
import joblib
import sys

r = sr.Recognizer()
with sr.Microphone() as source:
    print("Say: Hey Google Play Spotify!")
    audio = r.listen(source)
    print(r.recognize_google(audio))

# write audio to a WAV file
with open("raw_audio.wav", "wb") as f:
    f.write(audio.get_wav_data())

# # Real time emotion detection program


import tensorflow.keras
import pandas as pd
import pyaudio
import librosa
from scipy.stats import zscore
import numpy as np
import matplotlib.pyplot as plt

# ## Load our pre-trained emotion recognition model

model = tensorflow.keras.models.load_model('./[CNN-LSTM]M.h5')

# Read audio file
sample_rate = 16000
max_pad_len = 49100


y, sr = librosa.core.load('./raw_audio.wav', sr=sample_rate, offset=0.5)

# Z-normalization
y = zscore(y)

# Padding or truncated signal
if len(y) < max_pad_len:
    y_padded = np.zeros(max_pad_len)
    y_padded[:len(y)] = y
    y = y_padded
elif len(y) > max_pad_len:
    y = np.asarray(y[:max_pad_len])

# Add to signal list
signal = y


# Number of augmented data
nb_augmented = 2

# Function to add noise to a signals with a desired Signal Noise ratio (SNR)
def noisy_signal(signal, snr_low=15, snr_high=30, nb_augmented=2):

    # Signal length
    signal_len = len(signal)

    # Generate White noise
    noise = np.random.normal(size=(nb_augmented, signal_len))

    # Compute signal and noise power
    s_power = np.sum((signal / (2.0 ** 15)) ** 2) / signal_len
    n_power = np.sum((noise / (2.0 ** 15)) ** 2, axis=1) / signal_len

    # Random SNR: Uniform [15, 30]
    snr = np.random.randint(snr_low, snr_high)

    # Compute K coeff for each noise
    K = np.sqrt((s_power / n_power) * 10 ** (- snr / 10))
    K = np.ones((signal_len, nb_augmented)) * K

    # Generate noisy signal
    return signal + K.T * noise


augmented_signal = noisy_signal(signal)



def mel_spectrogram(y, sr=16000, n_fft=512, win_length=256, hop_length=128, window='hamming', n_mels=128, fmax=4000):

    # Compute spectogram
    mel_spect = np.abs(librosa.stft(y, n_fft=n_fft, window=window, win_length=win_length, hop_length=hop_length)) ** 2

    # Compute mel spectrogram
    mel_spect = librosa.feature.melspectrogram(S=mel_spect, sr=sr, n_mels=n_mels, fmax=fmax)

    # Compute log-mel spectrogram
    mel_spect = librosa.power_to_db(mel_spect, ref=np.max)

    return mel_spect


signal = [signal]
augmented_signal = [augmented_signal]

mel_spect = np.asarray(list(map(mel_spectrogram, signal)))
augmented_mel_spect = [np.asarray(list(map(mel_spectrogram, augmented_signal[i]))) for i in range(len(augmented_signal))]


# Time distributed parameters
win_ts = 128
hop_ts = 64

# Split spectrogram into frames
def frame(x, win_step=128, win_size=64):
    nb_frames = 1 + int((x.shape[2] - win_size) / win_step)
    frames = np.zeros((x.shape[0], nb_frames, x.shape[1], win_size)).astype(np.float32)
    for t in range(nb_frames):
        frames[:,t,:,:] = np.copy(x[:,:,(t * win_step):(t * win_step + win_size)]).astype(np.float32)
    return frames
X_test = frame(mel_spect, hop_ts, win_ts)

X_test = X_test.reshape(X_test.shape[0], X_test.shape[1] , X_test.shape[2], X_test.shape[3], 1)


y_pred = model.predict(X_test)


# Positioning of categorical to numerical
# 0 - Anger
# 1 - Disgust
# 2 - Fear
# 3 - Happy
# 4 - Neutral
# 5 - Sad
# 6 - Surprise

import webbrowser
emotion = np.argmax(y_pred)
if emotion == 0:
    print('Angry')
    url = 'https://open.spotify.com/playlist/6wXpdi3rAsNsTCRvI2DIg3?si=d620a38ce9ea4c7c'
    webbrowser.open_new(url)

elif emotion == 1:
    print('Disgust')
    url = 'https://open.spotify.com/track/3jmdY3jGMYrtXpa47TmvIk?si=11e24de8f2e14cfe'
    webbrowser.open_new(url)
elif emotion == 2:
    print('Fear')
elif emotion == 3:
    print('Happy')
    url = 'https://open.spotify.com/playlist/37i9dQZF1EVJSvZp5AOML2?si=052f42dc40aa4b4a'
    webbrowser.open_new(url)
elif emotion == 4:
    print('Neutral')
    url = 'https://open.spotify.com/playlist/37i9dQZEVXbLRQDuF5jeBp?si=8c061a07a6da4fd8'
    webbrowser.open_new(url)
elif emotion == 5:
    print('Sad')
    url = 'https://open.spotify.com/playlist/37i9dQZF1EId4uQEXLu3QA?si=c5df98edb8514e5a'
    webbrowser.open_new(url)
elif emotion == 6:
    print('Surprise')
    url = 'https://open.spotify.com/track/5UiT4e4DHwZrcVIXojU5um?si=a5e1398bfe804096'
    webbrowser.open_new(url)



