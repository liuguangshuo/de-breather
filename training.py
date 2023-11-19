#! /usr/bin/python3

import os
import librosa
import librosa.util as librosa_util
import numpy as np
import pandas as pd
import random

from pathlib import Path


def convert_time_to_seconds(time_str):
    """
    Converts a timestamp in the format 'minutes:seconds.milliseconds' to seconds.
    Handles empty strings, missing values (NaN), and malformed entries.
    """
    try:
        if pd.isna(time_str) or time_str.strip() == "":
            return np.nan  # Return NaN for missing or empty values

        time_parts = time_str.split(":")
        minutes = int(time_parts[0])
        seconds, milliseconds = map(int, time_parts[1].split("."))
        return minutes * 60 + seconds + milliseconds / 1000
    except Exception as e:
        # Log the error and the value that caused it
        print(f"Error converting time '{time_str}': {e}")
        return np.nan  # Return NaN for errors


def extract_features(y, sr, starts_ends, fixed_length=1.0):
    """
    Extracts magnitude spectrogram features from the audio file given annotation start and end times.

    Parameters:
    - audio_path: Path to the wav file.
    - starts_ends: List of tuples containing start and end times of breaths in seconds.
    - fixed_length: Duration in seconds to which each segment will be padded or truncated.

    Returns:
    Numpy array of extracted features.
    """
    features = []
    # Loop through annotated intervals to extract features
    for start, end in starts_ends:
        # If either start or end is NaN, skip this interval
        if np.isnan(start) or np.isnan(end):
            continue

        # Convert time to samples
        start_sample = librosa.time_to_samples(start, sr=sr)
        end_sample = librosa.time_to_samples(end, sr=sr)

        # Extract the segment and enforce the fixed length by padding or truncating
        y_segment = y[start_sample:end_sample]
        y_segment = librosa_util.fix_length(
            y_segment, size=librosa.time_to_samples(fixed_length, sr=sr)
        )

        # Compute the magnitude spectrogram
        S = np.abs(librosa.stft(y_segment))

        # Compute log-amplitude spectrogram
        log_S = librosa.amplitude_to_db(S, ref=np.max)

        # Average the spectrogram along the time axis to create a feature vector
        # This reduces the time dimension, creating a consistent feature vector length
        feature_vector = np.mean(log_S, axis=1)

        features.append(feature_vector)

    return np.array(features)


def sample_negative_clips(y, sr, breath_starts_ends, num_negative, fixed_length=1.0):
    """
    Sample negative (non-breath) clips from an audio file.

    Parameters:
    - y: Audio time series.
    - sr: Sampling rate of the audio time series.
    - breath_starts_ends: List of tuples containing start and end times of breaths in seconds.
                          This is used to avoid sampling from these regions.
    - num_negative: Number of negative clips to sample.
    - fixed_length: Duration in seconds to which each clip will be padded or truncated.

    Returns:
    List of negative clip features.
    """
    # Convert breath start and end times to sample indices
    breath_samples = []
    for start, end in breath_starts_ends:
        if not np.isnan(start) and not np.isnan(end):
            start_sample = librosa.time_to_samples(start, sr=sr)
            end_sample = librosa.time_to_samples(end, sr=sr)
            breath_samples.append((start_sample, end_sample))

    # Calculate the duration of the audio file in samples
    audio_length_samples = len(y)

    # Fixed number of samples per clip
    clip_length_samples = librosa.time_to_samples(fixed_length, sr=sr)

    # List to store the features of negative clips
    negative_features = []

    # Sample negative clips
    for _ in range(num_negative):
        while True:
            # Randomly pick a starting sample for a potential negative clip
            neg_start_sample = random.randint(
                0, audio_length_samples - clip_length_samples
            )
            neg_end_sample = neg_start_sample + clip_length_samples

            # Check if the clip overlaps with any breath segment
            overlap_with_breath = any(
                neg_start_sample <= end and neg_end_sample >= start
                for start, end in breath_samples
            )

            # If there's no overlap, extract features for this clip
            if not overlap_with_breath:
                y_segment = y[neg_start_sample:neg_end_sample]
                # Compute the magnitude spectrogram
                S = np.abs(librosa.stft(y_segment))
                # Compute log-amplitude spectrogram
                log_S = librosa.amplitude_to_db(S, ref=np.max)
                # Average the spectrogram along the time axis to create a feature vector
                feature_vector = np.mean(log_S, axis=1)
                negative_features.append(feature_vector)
                break

    return np.array(negative_features)


# Process annotation CSV file and assume it's present in the current working directory
annotations_csv = "/Users/guangshuo/Desktop/dataset/Ciel/annotations.csv"
annotations_df = pd.read_csv(annotations_csv)

# Apply the conversion function to all relevant columns with start and end times
time_columns = annotations_df.columns[annotations_df.columns.str.contains("Start|End")]
for col in time_columns:
    annotations_df[col] = annotations_df[col].apply(convert_time_to_seconds)

# Directory where the WAV files are located
wav_files_dir = "/Users/guangshuo/Desktop/dataset/Ciel"

# Modified dataset creation code
all_features = []
all_labels = []
neg_to_pos_ratio = (
    10  # Ratio of negative (non-breath) examples to positive (breath) examples
)

for index, row in annotations_df.iterrows():
    wav_file_path = Path(wav_files_dir) / f"{row['File Name']}.wav"
    if not wav_file_path.exists():
        print(wav_file_path)
        continue

    # Load the audio file
    y, sr = librosa.load(wav_file_path, sr=None)

    # Create a list of start and end times of annotated breaths for this file
    starts_ends = [
        (row[f"Breath {i} Start"], row[f"Breath {i} End"]) for i in range(1, 9)
    ]

    # Clean up NaN values
    starts_ends = [
        se for se in starts_ends if not np.isnan(se[0]) and not np.isnan(se[1])
    ]

    # Extract positive features (breaths)
    positive_features = extract_features(y, sr, starts_ends)

    # Extract negative features (non-breaths), at the specified ratio
    num_negative_clips = len(positive_features) * neg_to_pos_ratio
    negative_features = sample_negative_clips(y, sr, starts_ends, num_negative_clips)

    # If there are no valid intervals, continue to the next row
    if positive_features.size == 0 and negative_features.size == 0:
        continue

    # Add the features and the corresponding labels to the list
    all_features.extend(positive_features)
    all_labels.extend([1] * len(positive_features))

    all_features.extend(negative_features)
    all_labels.extend([0] * len(negative_features))

# The arrays should already be NumPy arrays, but in case they are lists
all_features = np.array(all_features)
all_labels = np.array(all_labels)

# At this point, we can use `all_features` and `all_labels` to train a machine learning model
# Below is an example using a simple Random Forest classifier from scikit-learn

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    all_features, all_labels, test_size=0.2, random_state=42
)

# Initialize the Random Forest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the classifier
clf.fit(X_train, y_train)

# Predict on the test set
y_pred = clf.predict(X_test)

# Print classification report
print(classification_report(y_test, y_pred))
