#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Creates the master csv file from the data set
Created on Fri Apr 19 12:20:50 2024

@author: alexandergadin
modified from user hlysine at https://www.kaggle.com/code/hlysine/driams-master-metadata-file
"""
import pandas as pd
import os


download_location= "/Users/alexandergadin/Downloads/archive"

cleancsv_file= '/Users/alexandergadin/Documents/Python/BME_Fondations_Final_Project/all_clean.csv'

pd.set_option('display.max_columns', 500)
master_df= pd.DataFrame()

institutes= [x for x in os.listdir(download_location) if os.path.isdir(f'{download_location}/{x}')]

for institute in institutes:
    years = [x for x in os.listdir(f'{download_location}/{institute}/id/') if os.path.isdir(f'{download_location}/{institute}/id/{x}')]
    for year in years:
        print(f'{institute} in {year}')
        df = pd.read_csv(f'{download_location}/{institute}/id/{year}/{year}_clean.csv', dtype='string')
        df['year'] = year
        df['institute'] = institute
        if master_df.empty:
            master_df = df
        else:
            master_df = pd.concat([master_df, df], ignore_index=True, sort=False)
            
master_df=master_df[df.isna().sum().sort_values().keys()] #sort columns

master_df.to_csv(cleancsv_file, index=False)

