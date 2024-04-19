#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 13:44:13 2024

@author: alexandergadin
test2
"""

import numpy as np
import pandas as pd
import os
import os.path
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as grd
import tensorflow as tf
import zipfile
import shutil
import seaborn as sns
from tqdm import tqdm
import shap

#download locations
download_location= "/Users/alexandergadin/Downloads/archive"
cleancsv_file= '/Users/alexandergadin/Documents/Python/BME_Fondations_Final_Project/all_clean.csv'



shap.initjs()
shap.explainers._deep.deep_tf.op_handlers["AddV2"] = shap.explainers._deep.deep_tf.passthrough

pd.set_option('display.max_columns', 500)

df = pd.read_csv(cleancsv_file, dtype='string', na_values=['-'])
df=df[df.isna().sum().sort_values().keys()]
df

#target selectuon
organism = 'Staphylococcus aureus'
antimicrobial = 'Ciprofloxacin'


if organism == '': #general
     pa_df = df.dropna(axis=1, how='all')
else: #organism specific
     pa_df = df[df['species'] == organism]
     pa_df = pa_df.dropna(axis=1, how='all')
     
pa_df.replace({antimicrobial:'I'}, {antimicrobial:'R'}, inplace=True)
pa_df[antimicrobial].value_counts(dropna=False)

pa_df = pa_df[(pa_df[antimicrobial] == 'S') | (pa_df[antimicrobial] == 'R')][['code', 'species', antimicrobial, 'year', 'institute']]
def load_spectra(row):
    df = pd.read_csv(f'{download_location}/{row["institute"]}/binned_6000/{row["year"]}/{row["code"]}.txt', sep=" ")
    return df['binned_intensity'].to_numpy()
pa_df['bins'] = pa_df.apply(load_spectra, axis=1)

classes = pa_df[antimicrobial].value_counts().apply(lambda x: 500/x).to_numpy()
class_weight = {}
for idx in range(len(classes)):
    class_weight[idx] = classes[idx]

inputs = np.vstack(pa_df['bins'].to_numpy())
targets = pa_df[antimicrobial].apply(lambda x: x == 'R').to_numpy()

(inputs.shape, targets.shape)

from sklearn.model_selection import train_test_split

# Split data
X_train, X_test, y_train, y_test = train_test_split(inputs, targets, test_size=0.2, random_state=42)

# Show shapes
print("Training set shapes:", X_train.shape, y_train.shape)
print("Testing set shapes:", X_test.shape, y_test.shape)



from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Define the neural network model
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

# compile
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Sumamry 
model.summary()

# Training wheels
history= model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

def predict_resistant_microbes(antimicrobial_name, model, threshold=0.5):
    antimicrobial_label = { antimicrobial_name : 'R'}  # mapping push
    
    #Alexander is smarts
    if antimicrobial_name not in antimicrobial_label:
        print("Antimicrobial not found in the mapping.")
        return None
    
    label = antimicrobial_label[antimicrobial_name]
    probabilities = model.predict(X_test)
    probable_resistant_microbes_indices = np.where(probabilities >= threshold)[0]
    probable_resistant_microbes_data = X_test[probable_resistant_microbes_indices]
    
    return probable_resistant_microbes_data
antimicrobial_name = antimicrobial

#optimize threshold for highest
def find_highest_threshold(antimicrobial_name, model):
    highest_threshold_data = None
    highest_threshold = 0.1
    
    for threshold in range(100, 999):  # Range from 0.1 to 0.99 with steps of 0.001
        threshold /= 1000  # Convert to float
        
        probable_resistant_microbes_data = predict_resistant_microbes(antimicrobial_name, model, threshold)
        
        if len(probable_resistant_microbes_data) > 0:
            highest_threshold_data = probable_resistant_microbes_data
            highest_threshold = threshold
    
    return highest_threshold_data, highest_threshold

lowest_threshold_data, lowest_threshold = find_highest_threshold(antimicrobial_name, model)
print("Lowest non-zero threshold:", lowest_threshold)

# Non-optimize route:
# probable_resistant_microbes_data= predict_resistant_microbes(antimicrobial_name, model, threshold=0.03)

import matplotlib.pyplot as plt

def visualize_spectral_data(spectral_data):

    num_bins = spectral_data.shape[0]
    font_size = 14
    plt.figure(figsize=(12, 8), dpi=100)
    plt.plot(range(num_bins), spectral_data.T, alpha=0.5)
    plt.xlabel('Bin Number', fontsize=font_size)
    plt.ylabel('Intensity', fontsize=font_size)
    
    if organism == '': #general
         plt.title(f'Spectral Data for {antimicrobial_name}', fontsize=font_size)
    else: #organism specific
         plt.title(f'Spectral Data for {antimicrobial_name} given {organism}', fontsize=font_size)
    
    plt.grid(True, linewidth=1)
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)
    plt.show()

visualize_spectral_data(lowest_threshold_data[0])

#ROC curve
from sklearn.metrics import roc_curve, auc

probabilities = model.predict(X_test)
fpr, tpr, thresholds = roc_curve(y_test, probabilities)
roc_auc = auc(fpr, tpr)
font_size = 14
plt.figure(figsize=(10, 8), dpi=100)
plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=font_size)
plt.ylabel('True Positive Rate', fontsize=font_size)
if organism == '': #general
    plt.title(f'Receiver Operating Characteristic (ROC) Curve for {antimicrobial_name}', fontsize=font_size)
else: #organism specific
    plt.title(f'Receiver Operating Characteristic (ROC) Curve for {antimicrobial_name} given {organism}', fontsize=font_size)
plt.legend(loc="lower right", fontsize=font_size)

plt.show()





