#! /usr/bin/python3

import os
import librosa
import librosa.util as librosa_util
import numpy as np
import pandas as pd
import pickle
import random
import sys

from datetime import datetime
from pathlib import Path

from training import (
    convert_time_to_seconds,
    extract_features,
    sample_negative_clips,
)

# Process annotation CSV file and assume it's present in the current working directory
annotations_csv = "/Users/guangshuo/Desktop/dataset/Daniel/annotations.csv"
annotations_df = pd.read_csv(annotations_csv)

# Apply the conversion function to all relevant columns with start and end times
time_columns = annotations_df.columns[annotations_df.columns.str.contains("Start|End")]
for col in time_columns:
    annotations_df[col] = annotations_df[col].apply(convert_time_to_seconds)

# Directory where the WAV files are located
wav_files_dir = "/Users/guangshuo/Desktop/dataset/Daniel"

# Modified dataset creation code
all_features = []
all_labels = []
neg_to_pos_ratio = (
    10  # Ratio of negative (non-breath) examples to positive (breath) examples
)

for index, row in annotations_df.iterrows():
    wav_paths = list(Path(wav_files_dir).glob(f"{row['File Name']}*.wav"))
    if not wav_paths:
        print(row["File Name"])
        continue
    wav_file_path = wav_paths[0]

    # Load the audio file
    y, sr = librosa.load(wav_file_path, sr=None)

    # Create a list of start and end times of annotated breaths for this file
    starts_ends = [
        (row[f"Breath {i} Start"], row[f"Breath {i} End"]) for i in range(1, 8)
    ]

    # Clean up NaN values
    starts_ends = [
        se for se in starts_ends if not np.isnan(se[0]) and not np.isnan(se[1]) and se[1] > se[0]
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
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Initialize the Random Forest classifier
# clf = RandomForestClassifier(n_estimators=100, random_state=42)
with open(sys.argv[1], "rb") as f:
    clf = pickle.load(f)

# Predict on the test set
pred = clf.predict(all_features)

# Print classification report
print(classification_report(all_labels, pred))
