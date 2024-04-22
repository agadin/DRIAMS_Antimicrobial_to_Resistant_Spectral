# Overview
* `mater_fil_creator.py`- Creates a master `.csv` file of all institutes and years
* `antimicrobials_to_organism_spectral.py` - Trains a simple Tensorflow model to predict resistant organism spectral data for a given antimicrobial

**Produced Figures:** Resistant organism spectral graph and ROC curve

## Creating the master `.csv file`
> [!CAUTION]
> Required step!

The data source can be easily downloaded from [Kaggle](https://www.kaggle.com/datasets/drscarlat/driams/data) which already has the processed CSVs for each institute and year without the added bloat (and data requirement of 144.84 GB!). The original data source can be found [here](https://datadryad.org/stash/dataset/doi:10.5061/dryad.bzkh1899q)[^1]. This guide assumes that the data source is the trimmed-down Kaggle version but the scripts should work for both data sources (the original data source is untested however because uncompressing 144.84 GB takes too long).
[^1]: Weis, Caroline et al. (2022). DRIAMS: Database of Resistance Information on Antimicrobials and MALDI-TOF Mass Spectra [Dataset]. Dryad. https://doi.org/10.5061/dryad.bzkh1899q

After downloading the data, open `mater_fil_creator.py` with your program of choice and edit `download_location` on line 14 to the folder that contains the 4 subfolders, DRIAMS-A to DRIAMS-D. Next, edit `cleancsv_file`, on line 16, to specify where you would like to save the created master csv.

**Expected Output:**
```
DRIAMS-A in 2015
DRIAMS-A in 2017
DRIAMS-A in 2018
DRIAMS-A in 2016
DRIAMS-B in 2018
DRIAMS-D in 2018
DRIAMS-C in 2018
```
> [!NOTE]
> `mater_fil_creator.py` is a fork of Kaggle user LYSINE's script found in [DRIAMS - Master Metadata File](https://www.kaggle.com/code/hlysine/driams-master-metadata-file).

## Running the model 
Open `antimicrobials_to_organism_spectral.py` with your python program of choice. Edit `download_location` on line 25 to the folder that contains the 4 subfolders, DRIAMS-A to DRIAMS-D (the same location that was specified in the previous section. Next, edit `cleancsv_file`, on line 16, to specify where you saved the master csv created in the step before.
The next step is to edit the `organism` variable on line 40 to specify if you would like the model to only produce resistant spectral data for a given microbe for only a specific organism. Leaving this blank will train the model to produce general resistant spectral data across all microbes in the data set. After that, edit `antimicrobial` on line 41 to the antimicrobial you wish to produce probable resistant organism spectral figures for.
After completing the paragraph above, run the now modified `antimicrobials_to_organism_spectral.py` file. The general elements of the script are as follows:
1. **Formatting and Splitting:** The `all_clean.csv` file created in the previous section is converted into a Pandas database and all unnecessary antimicrobial columns will be removed (if the `organism` variable is NOT blank all irrelevant organism rows will be removed as well. Next, on line 50 all intermediary, `I`, resistances are converted to absolute resistances, `R` to increase the available data pool. Finally, the now filtered data is split into 80% training and 20% test. 
2. **Defining and Running the Model:** Exact model definition can be explored in more detail by looking directly at the python file but the underpinnings of the model used are TensorFlow's Keras sequential and dense approaches. 
3. **Thresholding and Plotting:** When queried the output of the model will most often return multiple possible resistant spectral graphs. To circumvent this a simple iterative loop was defined that tests 1,000 threshold values between 0.1 and 0.999 and selects the highest threshold value that returned a non-zero spectral graph. Next, the highest threshold spectral graph is plotted along with the ROC for the model.

**Example Output:**
```
Training set shapes: (5603, 6000) (5603,)
Testing set shapes: (1401, 6000) (1401,)
Model: "sequential_52"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Layer (type)                    ┃ Output Shape           ┃       Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ dense_170 (Dense)               │ (None, 64)             │       384,064 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_171 (Dense)               │ (None, 32)             │         2,080 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_172 (Dense)               │ (None, 1)              │            33 │
└─────────────────────────────────┴────────────────────────┴───────────────┘
 Total params: 386,177 (1.47 MB)
 Trainable params: 386,177 (1.47 MB)
 Non-trainable params: 0 (0.00 B)
/Users/alexandergadin/anaconda3/lib/python3.11/site-packages/keras/src/layers/core/dense.py:86: UserWarning: Do not pass an `input_shape`/`input_dim` argument to a layer. When using Sequential models, prefer using an `Input(shape)` object as the first layer in the model instead.
  super().__init__(activity_regularizer=activity_regularizer, **kwargs)
Epoch 1/10
141/141 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.8656 - loss: 0.5289 - val_accuracy: 0.8894 - val_loss: 0.3435
Epoch 2/10
141/141 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.8815 - loss: 0.3602 - val_accuracy: 0.8894 - val_loss: 0.3386
Epoch 3/10
141/141 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.8885 - loss: 0.3402 - val_accuracy: 0.8894 - val_loss: 0.3346
Epoch 4/10
141/141 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.8812 - loss: 0.3498 - val_accuracy: 0.8894 - val_loss: 0.3309
Epoch 5/10
141/141 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.8818 - loss: 0.3453 - val_accuracy: 0.8894 - val_loss: 0.3272
Epoch 6/10
141/141 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.8787 - loss: 0.3443 - val_accuracy: 0.8894 - val_loss: 0.3233
Epoch 7/10
141/141 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.8793 - loss: 0.3388 - val_accuracy: 0.8894 - val_loss: 0.3192
Epoch 8/10
141/141 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.8882 - loss: 0.3166 - val_accuracy: 0.8894 - val_loss: 0.3143
Epoch 9/10
141/141 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.8836 - loss: 0.3226 - val_accuracy: 0.8894 - val_loss: 0.3087
Epoch 10/10
141/141 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.8884 - loss: 0.3045 - val_accuracy: 0.8894 - val_loss: 0.3054
Lowest non-zero threshold: 0.427
```
Elapsed time: 43.60 seconds

**Example Output Figures:**

![](https://github.com/agadin/BME_Fondations_Final_Project/blob/main/images/Spectrum_Resistant_to_Ciprofloxacin.png)
![](https://github.com/agadin/DRIAMS_Antimicrobial_to_Resistant_Spectral/blob/main/images/Example_ROC_Curve.png)
