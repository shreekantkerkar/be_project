from __future__ import division, print_function
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write
import wavio as wv
import os 
import shutil
import librosa
import tensorflow as tf
speech_model = tf.keras.models.load_model("ml_folder/Speech-Emotion-Recognition-Model.h5")

# try:
#     shutil.rmtree('songs')
# except:
#     print("unable to delete previous audio data or no song folder is present")

# try: 
#     os.mkdir("songs")
# except: 
#     print("directry is already present")

speech_ouput = []
audio_result  = 0
labels = {0:'disgust',1:'happy',2:'sad',3:'neutral',4:'fear',5:'angry'}
SAMPLE_RATE = 22050

def extract_mfcc(filename): #
    # y, sr = librosa.load(filename, duration=3, offset=0.5) #load audio file
    signal, sample_rate = librosa.load(filename, duration=8, sr=SAMPLE_RATE)
    mfcc = librosa.feature.mfcc(signal, sample_rate, n_mfcc=13, n_fft=2048, hop_length=512)
    mfcc = mfcc.T
    # mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0) #extract mfcc features
    return mfcc

def audio_classifier():
                filename = "candidate_answer.wav"
                try:
                    shutil.copyfile("candidate_answer.wav", "./originalCandidateAnswer.wav")
                except:
                    print("unable to copy file")
                rate = 22050           # samples per second
                T = 8                 # sample duration (seconds)
                n = int(rate*T)        # number of samples
                t = np.arange(n)/rate  # grid of time values

                # f = 440.0              # sound frequency (Hz)
                # wavio.write("sine24.wav", x, rate, sampwidth=3)

                freq = 44100  # audio sampling rate
                x = np.sin(2*np.pi * freq * t) #generate sine wave
                wv.write(filename, x, freq, sampwidth=4) #write audio file
                
                feature = extract_mfcc(filename) #extract mfcc features 
                
                feature = np.array([feature]) #reshape to 3d array
                # feature = feature.reshape(1, 40, 13)
                try:
                    audio_result = speech_model.predict(feature) #predicting 
                except Exception as e:
                    print("Expection in speech detection:", e)
                print(audio_result)
                audio_result = np.argmax(audio_result)
                print(audio_result)
                audio_result = labels[audio_result]
                speech_ouput.append(audio_result)
                print(speech_ouput)
                file1 = open("speech_result.txt", "a")
                file1.write(audio_result)
                file1.write("\n")
                file1.close()
                return audio_result

def get_speech_result():
    return speech_ouput