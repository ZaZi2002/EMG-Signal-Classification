# EMG Signal Classification

## Overview

This project is for Biomedical Engineering Fundamentals course and focuses on the classification of Electromyography (EMG) signals. EMG data from multiple channels are processed, filtered, normalized, and then divided into windows for feature extraction. The extracted features are used to train classification models, including **K-Nearest Neighbors (KNN)** and **Random Forest (RF)** models, for predicting specific muscle activations based on the input EMG data.

## Project Structure

- **Data Loading**: EMG data from different subjects is loaded for analysis.
- **Signal Filtering**: A series of filters (lowpass, highpass, notch) are applied to the raw EMG signals to reduce noise and unwanted frequencies.
- **Data Normalization**: The EMG data is normalized using Z-score normalization.
- **Windowing**: The EMG data is divided into overlapping windows for feature extraction.
- **Feature Extraction**: Seven key features are extracted from the EMG windows, including:
  - Mean Absolute Value (MAV)
  - Standard Deviation (STD)
  - Variance (VAR)
  - Root Mean Square (RMS)
  - Waveform Length (WL)
  - Zero Crossing (ZC)
  - Integrated Absolute Value (IAV)
- **Model Training**: 
  - **K-Nearest Neighbors (KNN)** and **Random Forest (RF)** classifiers are trained on the extracted features.
  - 70% of the data is used for training, and 30% for testing.
- **Validation**: Confusion matrices are generated to assess model performance, and the accuracy of both models is computed.

## Data

- EMG data from **Subject 1** is used by default. Data from **Subject 2** and **Subject 3** is also available but commented out.
- The data is segmented into windows of size 400, with an overlap of 20 samples, ensuring a balance between rest and activation states.

## Feature Extraction

The following features are extracted from each EMG channel in the windowed data:
1. **Mean Absolute Value (MAV)**
2. **Standard Deviation (STD)**
3. **Variance (VAR)**
4. **Root Mean Square (RMS)**
5. **Waveform Length (WL)**
6. **Zero Crossing (ZC)**
7. **Integrated Absolute Value (IAV)**

## Classification Models

- **KNN Model**: A KNN classifier is trained using 5 neighbors and Euclidean distance as the distance metric.
- **Random Forest Model**: A Random Forest classifier is trained with 100 decision trees for classification.

## Results

The performance of the models is validated using confusion matrices and accuracy metrics. 

- **KNN Model Accuracy**: Displayed after model validation.
- **Random Forest Model Accuracy**: Displayed after model validation.

## Visualization

- Spectral analysis of the EMG signal (before and after filtering) is plotted.
- Confusion matrices for both models are displayed as heatmaps for visual performance comparison.

## How to Run

1. **Load Data**: Make sure you have the dataset in the appropriate folder (e.g., `DB2_s1\S1_E1_A1.mat`).
2. **Run the Script**: Execute the code in MATLAB to load, process, classify, and validate the EMG data.
3. **View Results**: The accuracy of both KNN and RF models will be displayed, along with confusion matrices.
