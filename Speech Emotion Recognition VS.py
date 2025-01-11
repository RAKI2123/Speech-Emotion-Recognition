import tkinter as tk
from tkinter import filedialog
import librosa
import pandas as pd
import soundfile
import os
import numpy as np
import glob
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score


import pickle
from flask import Flask, request, jsonify
import pygame
from threading import Thread

# Define emotions
emotions = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}
observed_emotions = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']

def load_audio(file_path):
    audio, sample_rate = librosa.load(file_path)
    return audio, sample_rate

def extract_feature(file_path, mfcc=True, chroma=True, mel=True):
    audio, sample_rate = librosa.load(file_path)
    result = np.array([])
    if mfcc:
        mfccs = np.mean(librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40).T, axis=0)
        result = np.hstack((result, mfccs))
    if chroma:
        chroma_stft = np.mean(librosa.feature.chroma_stft(y=audio, sr=sample_rate).T, axis=0)
        result = np.hstack((result, chroma_stft))
    if mel:
        mel_spectrogram = np.mean(librosa.feature.melspectrogram(y=audio, sr=sample_rate).T, axis=0)
        result = np.hstack((result, mel_spectrogram))
    return result

folder_path = "C:\RAKESH\Mini Project\code\archive\Actor_01"
permissions = 0o755

# Set permissions for each file
for root, dirs, files in os.walk(folder_path):
    for file in files:
        if file.endswith(".wav"):
            file_path = os.path.join(root, file)
            os.chmod(file_path, permissions)

def load_data(test_size=0.2):
    x, y = [], []
    for file in glob.glob(os.path.join(folder_path, "speech-emotion-recognition-ravdess-data", "Actor_*", "*.wav")):
        file_name = os.path.basename(file)
        emotion = emotions.get(file_name.split("-")[2])
        if emotion not in observed_emotions:
            continue
        feature = extract_feature(file, mfcc=True, chroma=True, mel=True)
        x.append(feature)
        y.append(emotion)
    return train_test_split(np.array(x), y, test_size=test_size, random_state=9)

def train_model():
    try:
        x_train, x_test, y_train, y_test = load_data(test_size=0.25)
        model = MLPClassifier(alpha=0.005, batch_size=187, epsilon=1e-08, hidden_layer_sizes=(300,), learning_rate='adaptive', max_iter=500)
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)
        print("Accuracy: {:.2f}%".format(accuracy * 100))
        df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
        print(df.head(20))
        with open('modelForPrediction1.sav', 'wb') as f:
            pickle.dump(model, f)
        print("Model saved as 'modelForPrediction1.sav'")
    except Exception as e:
        print("An error occurred while training the model:", str(e))

def predict_emotion(audio_file):
    with open('modelForPrediction1.sav', 'rb') as f:
        model = pickle.load(f)
    features = extract_feature(audio_file, mfcc=True, chroma=True, mel=True)
    features = features.reshape(1, -1)
    predicted_emotion = model.predict(features)[0]
    return predicted_emotion

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def handle_prediction():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file found'})
    audio_file = request.files['audio']
    temp_file_path = 'temp.wav'
    audio_file.save(temp_file_path)
    predicted_emotion = predict_emotion(temp_file_path)
    os.remove(temp_file_path)
    return jsonify({'emotion': predicted_emotion})

def open_file_dialog():
    file_path = filedialog.askopenfilename()
    if file_path:
        pygame.mixer.init()
        pygame.mixer.music.load(file_path)
        pygame.mixer.music.play()
        audio_player_window = tk.Toplevel(window)
        audio_player_window.title("Audio Player")
        audio_player_window.geometry("300x100")
        audio_player_window.configure(bg='sky blue')
        file_label = tk.Label(audio_player_window, text="Selected Audio: ", font=label_font)
        file_label.pack(pady=10)
        selected_file_label = tk.Label(audio_player_window, text=file_path, font=('Arial', 12), wraplength=250)
        selected_file_label.pack()
        stop_button = tk.Button(audio_player_window, text="Stop", command=stop_audio, width=10, font=label_font)
        stop_button.pack(pady=10)
        predicted_emotion = predict_emotion(file_path)
        result_label.config(text=f"Predicted Emotion: {predicted_emotion}")
        emoji_label.config(text=get_emoji(predicted_emotion))

def stop_audio():
    pygame.mixer.music.stop()

def get_emoji(emotion):
    emoji_dict = {
        'calm': '😌',
        'happy': '😊',
        'fearful': '😨',
        'disgust': '😖',
        'sad': '😢',
        'surprised': '😮',
        'neutral': '😐',
        'angry': '😡',
    }
    return emoji_dict.get(emotion, '')

window = tk.Tk()
window.title("Speech Based Emotion Recognition")
window.attributes('-fullscreen', True)
window.geometry("400x300")
window.configure(bg='sky blue')
label_font = ('Arial', 25)
title_label = tk.Label(window, text="Speech Based Emotion Recognition", font=('Arial', 30))
title_label.pack(pady=20)
train_button = tk.Button(window, text="Train Model", command=train_model, width=9, font=label_font)
train_button.pack(pady=10)
file_button = tk.Button(window, text="Select Audio File", command=open_file_dialog, width=13, font=label_font)
file_button.pack(pady=30)
result_label = tk.Label(window, text="Predicted Emotion: ", font=label_font, width=25)
result_label.pack(pady=20)
emoji_label = tk.Label(window, text="", font=('Arial', 100))
emoji_label.pack(pady=20)

def run_flask_server():
    app.run(host='0.0.0.0', port=5000)

flask_thread = Thread(target=run_flask_server)
flask_thread.daemon = True
flask_thread.start()

window.mainloop()
