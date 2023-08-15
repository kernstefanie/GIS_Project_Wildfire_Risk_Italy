# -*- coding: utf-8 -*-
"""
Compilation of the machine learning dataset

Created on Sat Jul 15 21:10:39 2023

@authors: Stefanie Kern & Iris Haake

"""

# Import libraries
import xarray as xr
from datetime import datetime
import pandas as pd
import rioxarray as rioxr
import numpy as np
from glob import glob


############################################## Fire status ###################################################

# Add a column to both datasets indicating the fire status (0: no fire, 1: fire)
# This will be the target variable for machine learning

# define path to csv file (fire)                    ###################ANPASSEN#######################
fp = "D:/Stefanie/Dokumente/Studium_Geographie/Master_Marburg/Semester_1/GIS/Projektarbeit/data/lev3/fire/fire_grouped_LUC_LST_NDVI_ProxAgri_ProxUrban_DEM_ERA5.csv"

# read data
fire = pd.read_csv(fp,delimiter=',',parse_dates=['acq_date'])
fire.keys()

# define path to csv file (no-fire)
fp = "D:/Stefanie/Dokumente/Studium_Geographie/Master_Marburg/Semester_1/GIS/Projektarbeit/data/lev3/nofire/nofire_LUC_LST_NDVI_ProxAgriUrban_DEM_ERA5.csv"

# read data
nofire = pd.read_csv(fp,delimiter=',',parse_dates=['timeslot'])
nofire

# Add fire status
fire['fire status'] = 1
nofire["fire status"] = 0

fire
nofire


############################################ Unify some columns ##############################################

fire.keys()
nofire.keys()

# Rename and delete some columns
fire = fire.rename(columns={"acq_date": "timeslot"})
fire = fire.drop(["minlat", "minlon", "maxlat", "maxlon", "latBin", "lonBin", "minDate", "maxDate", "FireRadiativePower"], axis=1)
fire

nofire = nofire.rename(columns={"pixel": "lccs_class"})
nofire = nofire.drop(["latBin2", "lonBin2"], axis=1)
nofire

# Add new columns to the dataset, which are specifying the respective month and year for each row
fire["time"] = fire["timeslot"].dt.strftime("%Y-%m")
fire

nofire["time"] = nofire["timeslot"].dt.strftime("%Y-%m")
nofire

# Merge the datasets
df_fire_nofire = pd.concat([nofire, fire], ignore_index=True)

df_fire_nofire = df_fire_nofire.drop(["timeslot"], axis=1)
df_fire_nofire

# Sort columns
custom_sort = ['time','x', 'y','lccs_class', 
               'u10', 'v10', 'd2m', 't2m', 'pev', 'src',
       'slhf', 'ssr', 'str', 'sp', 'sshf', 'e', 'tp', 'swvl1', 'rel_hum',
       'mixRatio', 'vapPres', 'satVapPres', 'VPD', 'LST', 'NDVI', 'ProxAgri',
       'ProxUrban', 'aspect', 
       'curvature', 'dem', 'slope', 'tpi', 'fire status']
df_fire_nofire = df_fire_nofire[custom_sort]
df_fire_nofire

# Sort by date
df_fire_nofire = df_fire_nofire.sort_values(by=['time'])
df_fire_nofire


############################################### Data cleaning ####################################################

# plot histograms
df_fire_nofire.hist(bins=30, figsize=(20, 20));

# Descriptive statistics for each column
df_fire_nofire.describe()

# Remove rows with -9999 (Aspect, Curvature, DEM, Slope, TPI, DistRoad, DistRails)
df_fire_nofire.keys()

data = df_fire_nofire[(df_fire_nofire.aspect != -9999.0) &
                      (df_fire_nofire.dem != -9999.0) &
                      (df_fire_nofire.curvature != -9999.0) &
                      (df_fire_nofire.slope != -9999.0) &
                      (df_fire_nofire.tpi != -9999.0)].copy(deep=True)
data

# plot histograms again
data.hist(bins=30, figsize=(20, 20));

fpout = "D:/Stefanie/Dokumente/Studium_Geographie/Master_Marburg/Semester_1/GIS/Projektarbeit/data/lev3/fire_nofire/fire_nofire_cleaning.csv"
# write data
data.to_csv(fpout,index=False)

# Since very strong correlations between variables affect the predictive power, 
# these are checked and corresponding variables are removed

pear_corr=data.corr(method='pearson')
pear_corr.style.background_gradient(cmap='Greens')

# remove highly correlated variables
data.drop(["mixRatio",'vapPres', 'satVapPres', "d2m", "e"], axis=1, inplace=True)

################################### Subsample for test dataset ###################################

# Set aside a subsample that can be used as an independent test dataset for the final machine learning algorithms
# Here, the last five years of the dataset are used as a test dataset

data['time'] = pd.to_datetime(data['time'])
split_date = pd.datetime(2018,1,1)

df_train = data.loc[data['time'] <= split_date]
df_test = data.loc[data['time'] > split_date]

df_train

# Save training data
# Define path and file name
fpout = "D:/Stefanie/Dokumente/Studium_Geographie/Master_Marburg/Semester_1/GIS/Projektarbeit/data/lev3/fire_nofire/fire_nofire_trainingdata.csv"

# write data
df_train.to_csv(fpout,index=False)

df_test

# Save test data
# Define path and file name
fpout = "D:/Stefanie/Dokumente/Studium_Geographie/Master_Marburg/Semester_1/GIS/Projektarbeit/data/lev3/fire_nofire/fire_nofire_testdata.csv"

# write data
df_test.to_csv(fpout,index=False)
