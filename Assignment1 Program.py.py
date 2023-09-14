import os
import librosa
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from pydub import AudioSegment
from pydub.playback import play

# Function to extract audio features (MFCCs) from an audio file
def extract_features(audio_file):
    y, sr = librosa.load(audio_file, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return mfccs

# Function to train an emotion detection model
def train_emotion_model(features, labels):
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    return model, accuracy

# Function to predict emotion (happy or sad) for an audio file
def predict_emotion(model, audio_file):
    features = extract_features(audio_file)
    emotion = model.predict([np.mean(features, axis=1)])
    return "happy" if emotion == 1 else "sad"

def speaker_diarization(audio_file):
    return [("Speaker 1", 0, 10), ("Speaker 2", 15, 25)]

# Function to create line chart for emotion prediction
def plot_emotion_prediction(audio_files, predictions):
    plt.figure(figsize=(10, 5))
    plt.plot(audio_files, predictions, marker='o', linestyle='-')
    plt.xlabel('Audio File')
    plt.ylabel('Emotion Prediction')
    plt.title('Emotion Prediction Over Audio Files')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.show()

# Function to create line chart for speaker diarization
def plot_speaker_diarization(speaker_segments):
    plt.figure(figsize=(10, 5))
    for speaker, start, end in speaker_segments:
        plt.plot([start, end], [0, 0], label=speaker)
    plt.xlabel('Time (s)')
    plt.ylabel('Speaker')
    plt.title('Speaker Diarization')
    plt.legend()
    plt.grid(True)
    plt.show()

# Main function
def main():
    audio_files = [] 
    emotions = []  

    audio_directory = r'C:\Users\prabh\Downloads\harvard.wav'
    for filename in os.listdir(audio_directory):
        if filename.endswith('.wav'):
            audio_file = os.path.join(audio_directory, filename)
            audio_files.append(filename)
            emotion = predict_emotion(emotion_model, audio_file)
            emotions.append(emotion)

    emotion_df = pd.DataFrame({'Audio File': audio_files, 'Predicted Emotion': emotions})
    print(emotion_df)

    plot_emotion_prediction(audio_files, emotions)

    speaker_segments = speaker_diarization(audio_file)

    speaker_df = pd.DataFrame(speaker_segments, columns=['Speaker', 'Start Time (s)', 'End Time (s)'])
    print(speaker_df)

    plot_speaker_diarization(speaker_segments)

if __name__ == "__main__":
    emotion_model, _ = train_emotion_model(features, labels)
    main()
