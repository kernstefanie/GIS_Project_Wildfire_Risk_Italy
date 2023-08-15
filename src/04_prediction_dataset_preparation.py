# -*- coding: utf-8 -*-
"""
Preparation of the dataset for prediction

Created on Sat Jul 15 21:37:58 2023

@authors: Stefanie Kern & Iris Haake

"""

# Import libraries
import xarray as xr
import numpy as np
import rioxarray as rioxr
import os 
from glob import glob

###########################################################################################################

# Compile all predictor variables for a month (in this case July 2022) 
# for which the prediction of fire occurences will be made


################################ ERA5 ########################################

# Load data into DataSet
era5 = xr.open_dataset("D:/Stefanie/Dokumente/Studium_Geographie/Master_Marburg/Semester_1/GIS/Projektarbeit/data/lev1/era5/ERA5_ClimateVariables_2001to2022.nc", decode_coords="all")
era5

# Select July 2022
era5_select = era5.sel(time='2022-07')
era5_select

# Adjust ERA5 resolution (originally 9 km) to the resolution of other variables (1 km)

# check if the crs information is available
era5_select.rio.crs

# Read in the reference file MODIS NDVI with the target CRS and resolution
ndvi_tomatch = rioxr.open_rasterio('D:/Stefanie/Dokumente/Studium_Geographie/Master_Marburg/Semester_1/GIS/Projektarbeit/data/lev0/ndvi/MOD13A3.061__1_km_monthly_NDVI_doy2001001_aid0001.tif')
ndvi_tomatch

ndvi_tomatch.rio.crs

# Reproject ERA5 to reference dataset
era5_select_reproj = era5_select.interp(x=xr.DataArray(ndvi_tomatch.x), y=xr.DataArray(ndvi_tomatch.y), method="nearest")
era5_select_reproj

# Have a look at the processed data
era5_select_reproj.t2m.plot()


################################# Land Surface Temperature (LST) ######################################

# Load data
lst = xr.open_dataset("D:/Stefanie/Dokumente/Studium_Geographie/Master_Marburg/Semester_1/GIS/Projektarbeit/data/lev1/lst/LST_2001to2022.nc", decode_coords="all")
lst

# Select July 2022
lst_select = lst.sel(time='2022-07')
lst_select

# Add LST data to ERA5 dataset
era5_select_reproj['LST'] = lst_select.LST
era5_select_reproj


########################################## NDVI ###################################################

# Load data
ndvi = xr.open_dataset("D:/Stefanie/Dokumente/Studium_Geographie/Master_Marburg/Semester_1/GIS/Projektarbeit/data/lev1/ndvi/NDVI_2001to2022.nc", decode_coords="all")
ndvi

# Select July 2022
ndvi_select = ndvi.sel(time='2022-07')
ndvi_select

# Add NDVI to ERA5 dataset
era5_select_reproj['NDVI'] = ndvi_select.NDVI
era5_select_reproj


########################################## Land Cover Data (LUC) ###################################################

# Load data
lccs_class = xr.open_dataset("D:/Stefanie/Dokumente/Studium_Geographie/Master_Marburg/Semester_1/GIS/Projektarbeit/data/lev1/luc/LUC_IT_2001to2020.nc", decode_coords="all")
lccs_class

# The land cover data have an annual resolution
# Therefore,the lccs Dataset is converted into a DataArray before adding it to the ERA5 dataset

# Select land cover data for 2020
lccs_class_select = lccs_class.sel(time='2020-01-01').to_array(dim='lccs_class')
lccs_class_select

# Add Landcover data to ERA5 dataset
era5_select_reproj['lccs_class'] = (["time","y","x"], lccs_class_select.values)
era5_select_reproj


########################################## Distance to urban areas ###################################################

# Load data
fn_tif = "D:/Stefanie/Dokumente/Studium_Geographie/Master_Marburg/Semester_1/GIS/Projektarbeit/data/lev2/proximity_urban/IT_2020_prox_urban.tif"

da = rioxr.open_rasterio(fn_tif)
da

# Add data to ERA5 dataset
era5_select_reproj['ProxUrban'] = (["time","y","x"],da.values)
era5_select_reproj


########################################## Distance to agricultural areas ###################################################

# Load data
fn_tif = "D:/Stefanie/Dokumente/Studium_Geographie/Master_Marburg/Semester_1/GIS/Projektarbeit/data/lev2/proximity_agriculture/IT_2020_prox_agric.tif"
da = rioxr.open_rasterio(fn_tif)
da

# Add data to ERA5 dataset
era5_select_reproj['ProxAgri'] = (["time","y","x"],da.values)
era5_select_reproj

fpout = ('D:/Stefanie/Dokumente/Studium_Geographie/Master_Marburg/Semester_1/GIS/Projektarbeit/data/lev1/dem_var/*.tif')
########################################## DEM and derived variables ###################################################

# Create list of tif files
filelist = glob('D:/Stefanie/Dokumente/Studium_Geographie/Master_Marburg/Semester_1/GIS/Projektarbeit/data/lev1/dem_var/*.tif')
filelist

# loop over all files in the list, extract pixel values and add them to the dataframe
for x in filelist:
    varname = os.path.splitext(os.path.basename(x))[0]
    print(varname)
    rds = rioxr.open_rasterio(x)
    era5_select_reproj[varname] = (["time","y","x"],rds.values)
    
era5_select_reproj


########################################## Additional data filtering ###################################################

# Mask out urban areas and areas outside of Italy
era5_select_reproj = era5_select_reproj.where((era5_select_reproj['lccs_class'] > 30) & (era5_select_reproj['lccs_class'] < 150) & (era5_select_reproj['dem'] != -9999.))

era5_select_reproj.lccs_class.plot()

# Set -9999 to NoData
era5_select_reproj['aspect'] = era5_select_reproj['aspect'].where(era5_select_reproj['aspect'] != -9999.)
era5_select_reproj['curvature'] = era5_select_reproj['curvature'].where(era5_select_reproj['curvature'] != -9999.)
era5_select_reproj['dem'] = era5_select_reproj['dem'].where(era5_select_reproj['dem'] != -9999.)
era5_select_reproj['slope'] = era5_select_reproj['slope'].where(era5_select_reproj['slope'] != -9999.)
era5_select_reproj['tpi'] = era5_select_reproj['tpi'].where(era5_select_reproj['tpi'] != -9999.)

era5_select_reproj.slope.plot()

########################################## Write final dataset to netCDF file ###################################################

outfilename = 'D:/Stefanie/Dokumente/Studium_Geographie/Master_Marburg/Semester_1/GIS/Projektarbeit/data/lev3/AllData202207.nc' 
era5_select_reproj.to_netcdf(path=outfilename)