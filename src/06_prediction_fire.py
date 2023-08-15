# -*- coding: utf-8 -*-
"""
Model application and prediction of fire probability

Created on Mon Jul 17 14:32:25 2023

@author: Stefanie Kern & Iris Haake
"""
# Import libraries
from datetime import datetime
import pandas as pd
import numpy as np
from rasterio import transform
import rasterio
import xarray as xr
import rioxarray as rioxr
from glob import glob
import os # from osgeo import gdal
import matplotlib.pyplot as plt
import matplotlib.colors

########################## Application of the RF-Model to July 2022 ######################
# Now that the predictors dataset for July 2022 is ready, it can be applied the model. For this, the predictor data set and the model must have the same variables. 
# Then the occurrence of fire/non-fire as well as the probability (with and without threshold) can be predicted.

    # Read prepared predictors dataset for July 2022 ######### ANPASSEN ###########
    data = xr.open_dataset("D:/Stefanie/Dokumente/Studium_Geographie/Master_Marburg/Semester_1/GIS/Projektarbeit/data/lev3/AllData202207.nc", decode_coords="all")
    data
    
    # The model requires tabular data. So the 2D data in our dataset has to be reshaped by converting the `Dataset` into a `pandas dataframe`.
    df_predictors = data.to_dataframe()
    df_predictors
    
    # To use the x and y coordinates as predictors, they are converted from indices to columns.
    df_predictors.reset_index(level=['x','y'],inplace=True)
    df_predictors

    # Read trained RF-model
    rf_model_test = joblib.load('D:/Stefanie/Dokumente/Studium_Geographie/Master_Marburg/Semester_1/GIS/Projektarbeit/data/lev4/model/rf_final.model')
    
    # Display features used to train the model
    rf_model_test.feature_names_in_
    
    # Selection of predictor variables in accordance with the used features
    predictors = rf_model_test.feature_names_in_.tolist()
    predictors
    df_predictors.keys()
    
################################### Prediction of Fire/No-Fire Occurrences ################################################################

    # Apply model
    prediction = rf_model_test.predict(df_predictors.fillna(0)[predictors])
    prediction
    
    # Specify raster dimensions to convert the 1D prediction back to a 2D array. MODIS NDVI file is used. 
    raster = rioxr.open_rasterio('D:/Stefanie/Dokumente/Studium_Geographie/Master_Marburg/Semester_1/GIS/Projektarbeit/data/lev0/ndvi/MOD13A3.061__1_km_monthly_NDVI_doy2001001_aid0001.tif')[0]
    raster
    
    # Reshape prediction array to 2D dimensions of the loaded raster image
    prediction = prediction.reshape(raster.shape)
    prediction
    prediction.shape
    
    # To write the prediction into a Geotiff file, for example, a conversion from the 2D `numpy array` into an `xarray DataArray` is necessary.
    # For this the dimensions of the loaded raster image is used again.
    prediction_da = xr.DataArray(prediction,dims=raster.dims,coords=[raster.y,raster.x]).astype(np.float32)
    prediction_da.plot()
    
    # Set non-relevant land use classes to `no data`
    prediction = np.where(np.isnan(data.lccs_class[0]),np.nan, prediction)
    
    # Convert results into DataArray using the dims/coords information of the loaded raster image
    prediction_da = xr.DataArray(prediction,dims=raster.dims,coords=[raster.y,raster.x]).astype(np.float32)
    prediction_da.plot()
    
    # Write prediction
    prediction_da.rio.to_raster("D:/Stefanie/Dokumente/Studium_Geographie/Master_Marburg/Semester_1/GIS/Projektarbeit/data/lev4/prediction/fire_prediction_202207_RF.tif")
    
######################################### Predict Fire Probability ##################################################

    fire_prob = rf_model_test.predict_proba(df_predictors.fillna(0)[predictors])
    
    # Keep only probability of fires
    fire_prob[:,1]
    
    # Reshape prediction array to 2D dimensions of the loaded raster image
    fire_prob_reshaped = fire_prob[:,1].reshape(raster.shape)
    
    # Set non-relevant land use classes to `no data`
    fire_prob_reshaped = np.where(np.isnan(data.lccs_class[0]),np.nan,fire_prob_reshaped)
    
    # Convert the 2D `numpy array` into an `xarray DataArray`.
    # Convert results into DataArray using the dims/coords information of the label data
    prediction_prob_da = xr.DataArray(fire_prob_reshaped,dims=raster.dims,coords=[raster.y,raster.x]).astype(np.float32)
    
    # predict 
    prediction_prob_da.plot()
    
    def NormalizeData(data):
        return (data - np.min(data)) / (np.max(data) - np.min(data))

    prediction_prob_da = NormalizeData(prediction_prob_da)
    prediction_prob_da.plot(cmap=matplotlib.colors.LinearSegmentedColormap.from_list("", ['burlywood','bisque','mistyrose','lightcoral', "firebrick"]))
    
    # Read prediction of probability
    prediction_prob_da.rio.to_raster("D:/Stefanie/Dokumente/Studium_Geographie/Master_Marburg/Semester_1/GIS/Projektarbeit/data/lev4/prediction/fire_prediction_prob_202207_RF.tif")
