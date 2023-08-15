# -*- coding: utf-8 -*-
"""
Extraction and preparation of predictor variables for fire and non-fire pixels

Created on Sat Jul 15 15:19:53 2023

@authors: Stefanie Kern & Iris Haake

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


######################### 1. Extract predictor variables of fire pixels #########################

# The pixel values of the predictor variables have to be assembled for the fire pixel coordinates. 
# First, extract the land use information and sort out the fire events in urban areas. 
# Starting from the cleaned dataset, the pixel values of the predictor variables are now extracted and compiled into a final data frame.

######### 1.1 Land Use Classes (LUC) #########

    # First, read the filtered data
        # define path to csv file 
        fp = "D:/Stefanie/Dokumente/Studium_Geographie/Master_Marburg/Semester_1/GIS/Projektarbeit/data/lev1/fire/fire_grouped.csv"
        # read data
        df_fire = pd.read_csv(fp,delimiter=',',parse_dates=['acq_date'])
        df_fire
        
    # Rename lon,lat to x,y
    df_fire.rename(columns={'meanlat': 'y', 'meanlon': 'x'}, inplace=True)
    df_fire
    
    # Extract all the annual time steps (annual resolution)
    time_slots = df_fire.acq_date.dt.year.unique()
    time_slots
    
    # First open LUC data from 2001 to 2020 
    lucfn = 'D:/Stefanie/Dokumente/Studium_Geographie/Master_Marburg/Semester_1/GIS/Projektarbeit/data/lev1/luc/LUC_IT_2001to2020.nc'
    luc = xr.open_dataset(lucfn, decode_coords="all") 
    luc
    
    # Extracting all LUC data for each fire pixel with a loop
    df= pd.DataFrame()

    for time_slot in time_slots:
        print(time_slot)
        
        luc_data = luc.sel(time=str(time_slot), method="nearest") # land use data
        current_data = df_fire[df_fire.acq_date.dt.year == time_slot] # fire data
        
        px_values = luc_data.lccs_class.interp(x=xr.DataArray(current_data.x), y=xr.DataArray(current_data.y), method="nearest").to_dataframe()
        px_values.reset_index(inplace=True)
        px_values = px_values.rename(columns ={'dim_0':'index'})
        
        current_data.reset_index(inplace=True)
        concat_data = pd.concat([current_data,px_values.lccs_class],axis=1, join='inner')
        df = pd.concat([df, concat_data], ignore_index=True)

    df
    df.lccs_class.hist()
    
    # Since wildfires are not studied in urban area, these data should be excluded. (See the full LUC-classification table via https://datastore.copernicus-climate.eu/documents/satellite-land-cover/D5.3.1_PUGS_ICDR_LC_v2.1.x_PRODUCTS_v1.1.pdf)
    df_selectedLUC = df[(df.lccs_class > 20) & (df.lccs_class < 190)]
    df_selectedLUC
    
    # Delete index column and NA
    df_selectedLUC = df_selectedLUC.drop(columns=['index'])
    df_selectedLUC
    
    df_selectedLUC.dropna(inplace=True)
    df_selectedLUC
    
    # Write as csv file
        # define path and file name 
        fpout = "D:/Stefanie/Dokumente/Studium_Geographie/Master_Marburg/Semester_1/GIS/Projektarbeit/data/lev3/fire/fire_grouped_LUC.csv"
        
        # write data
        df_selectedLUC.to_csv(fpout,index=False)

######### 1.2 Monthly LST and NDVI data #########
    
    # Write a loop for extracting pixel values from LST and NDVI
        
        # function to extract pixel values from raster file
        def extractPxValues(time_slots_def,df_def,ds_def):
            # Instantiate empty pandas-DataFrame
            df_loop = pd.DataFrame()
        
            # Iterate over all available time slots
            for time_slot in time_slots_def:
                print(time_slot)
                
                # Clip dataframe to current time slot
                current_data = df_def.loc[df_def['acq_date'].dt.strftime("%Y-%m") == time_slot]
            
                # Load data scene of the current time slot
                ds_data = ds_def.sel(time=str(time_slot), method="nearest")
            
                # Extract pixel values for coordinates
                px_values = ds_data.interp(x=xr.DataArray(current_data.x), y=xr.DataArray(current_data.y), method="nearest").to_dataframe()
                
                # Prepare columns for data merging
                px_values.reset_index(inplace=True)
                px_values = px_values.rename(columns ={'dim_0':'index'})
                px_values.drop(['x','y'], axis=1, inplace=True)
                current_data.reset_index(inplace=True)
            
                # Merge extracted data and fire data into one DataFrame
                concat_data = pd.concat([current_data,px_values],axis=1, join='inner')
            
                # Append the merged data to the final DataFrame
                df_loop = pd.concat([df_loop,concat_data], ignore_index=True)
                
            return df_loop
    
    # Read fire dataset
        # define path to csv file 
        fp = "D:/Stefanie/Dokumente/Studium_Geographie/Master_Marburg/Semester_1/GIS/Projektarbeit/data/lev3/fire/fire_grouped_LUC.csv"
        # read data
        df = pd.read_csv(fp,delimiter=',',parse_dates=['acq_date'])
        df
    
    # LST:
        # open netcdf file
        fn = 'D:/Stefanie/Dokumente/Studium_Geographie/Master_Marburg/Semester_1/GIS/Projektarbeit/data/lev1/lst/LST_2001to2022.nc'
        ds = xr.open_dataset(fn, decode_coords="all")
        ds
        
        # create monthly time slots for the loop
        time_slots = df['acq_date'].dt.strftime("%Y-%m").unique().tolist()
        
        # call the function to extract the pixel values
        df_extract_LST = extractPxValues(time_slots,df,ds)
        df_extract_LST

        # show column names
        df_extract_LST.keys()
        
        # remove some columns
        df_extract_LST.drop(['time','index','spatial_ref'], axis=1, inplace=True)
        df_extract_LST.keys()
        
        # remove NA values
        df_extract_LST.dropna(inplace=True)
        df_extract_LST
        
        # Write csv file
            # define path and file name
            fpout = "D:/Stefanie/Dokumente/Studium_Geographie/Master_Marburg/Semester_1/GIS/Projektarbeit/data/lev3/fire/fire_grouped_LUC_LST.csv"
            # write data
            df_extract_LST.to_csv(fpout,index=False)
    
    # NDVI:
        # Read fire dataset
        fp = "D:/Stefanie/Dokumente/Studium_Geographie/Master_Marburg/Semester_1/GIS/Projektarbeit/data/lev3/fire/fire_grouped_LUC_LST.csv"
        df = pd.read_csv(fp,delimiter=',',parse_dates=['acq_date'])
        
        # Read NDVI file
        fn = 'D:/Stefanie/Dokumente/Studium_Geographie/Master_Marburg/Semester_1/GIS/Projektarbeit/data/lev1/ndvi/NDVI_2001to2022.nc'
        ds = xr.open_dataset(fn, decode_coords="all")
        ds

        # create monthly time slots for the loop
        time_slots = df['acq_date'].dt.strftime("%Y-%m").unique().tolist()
        
        # call the function to extract the pixel values
        df_extract_NDVI = extractPxValues(time_slots,df,ds)
        df_extract_NDVI
        
        # show column names
        df_extract_NDVI.keys()
        
        # remove some columns
        df_extract_NDVI.drop(['time','index','spatial_ref'], axis=1, inplace=True)
        df_extract_NDVI.keys()
        
        # remove NA values
        df_extract_NDVI.dropna(inplace=True)
        df_extract_NDVI
        
        # Write data
        fpout = "D:/Stefanie/Dokumente/Studium_Geographie/Master_Marburg/Semester_1/GIS/Projektarbeit/data/lev3/fire/fire_grouped_LUC_LST_NDVI.csv"
        df_extract_NDVI.to_csv(fpout,index=False)

########## 1.3 Annual distance to agricultural and urban land data #########
    # Proximity of agricultural areas:
        # Read fire dataset
        fp = 'D:/Stefanie/Dokumente/Studium_Geographie/Master_Marburg/Semester_1/GIS/Projektarbeit/data/lev3/fire/fire_grouped_LUC_LST_NDVI.csv'
        df = pd.read_csv(fp,delimiter=',',parse_dates=['acq_date'])
        df
        
        # Extract all the annual time steps (annual resolution)
        time_slots = df.acq_date.dt.year.unique()
        time_slots
        
        # Extraction of all pixel values over all months is done with a loop
            # Instantiate empty pandas-DataFrame
            df_loop = pd.DataFrame()
            
            # Iterate over all available time slots
            for time_slot in time_slots:
                print(time_slot)
                
                # Clip dataframe to current time slot
                current_data = df[df.acq_date.dt.year == time_slot]
                
                # Since land use data are only available through 2020, we use these data for 2021 and 2022.
                if time_slot == 2021 or time_slot == 2022:
                    time_slot = 2020
            
                # Create filename
                fn_tif = pd.to_datetime(str(time_slot)).strftime("D:/Stefanie/Dokumente/Studium_Geographie/Master_Marburg/Semester_1/GIS/Projektarbeit/data/lev2/proximity_agriculture/IT_%Y_prox_agric.tif")
            
                # Load data scene of the current time slot
                da = rioxr.open_rasterio(fn_tif)
                ds = da.to_dataset('band')
                ds = ds.rename({1: 'ProxAgri'})
                
                # Extract pixel values for coordinates
                px_values = ds.interp(x=xr.DataArray(current_data.x), y=xr.DataArray(current_data.y), method="nearest").to_dataframe()
            
                # Prepare columns for data merging
                px_values.reset_index(inplace=True)
                px_values = px_values.rename(columns ={'dim_0':'index'})
                current_data.reset_index(inplace=True)
                
                # Merge extracted data and fire data into one DataFrame
                concat_data = pd.concat([current_data,px_values],axis=1, join='inner')
            
                # Append the merged data to the final DataFrame
                df_loop = pd.concat([df_loop,concat_data], ignore_index=True)
                
            df_loop
            
        # Delete some columns and NA
        df_loop.keys()
        
        df_loop.drop(['index','spatial_ref'], axis=1, inplace=True)
        df_loop.keys()
        
        df_loop.dropna(inplace=True)
        df_loop
        
        # Write data
        fpout = "D:/Stefanie/Dokumente/Studium_Geographie/Master_Marburg/Semester_1/GIS/Projektarbeit/data/lev3/fire/fire_grouped_LUC_LST_NDVI_ProxAgri.csv"
        df_loop.to_csv(fpout,index=False)
        
    # Proximity of urban areas:
        # Read fire data
        fp = "D:/Stefanie/Dokumente/Studium_Geographie/Master_Marburg/Semester_1/GIS/Projektarbeit/data/lev3/fire/fire_grouped_LUC_LST_NDVI_ProxAgri.csv"
        df_fire = pd.read_csv(fp,delimiter=',',parse_dates=['acq_date'])

        # define time slots
        time_slots = df_fire.acq_date.dt.year.unique()

        # Iterate over all available time slots
        df_loop = pd.DataFrame()

        for time_slot in time_slots:
            print(time_slot)
            
            # Clip dataframe to current time slot
            current_data = df_fire[df_fire.acq_date.dt.year == time_slot]
            
            # Since land use data are only available through 2020, we use these data for 2021 and 2022.
            if time_slot == 2021 or time_slot == 2022:
                time_slot = 2020
        
            # Create filename
            fn_tif = pd.to_datetime(str(time_slot)).strftime("D:/Stefanie/Dokumente/Studium_Geographie/Master_Marburg/Semester_1/GIS/Projektarbeit/data/lev2/proximity_urban/IT_%Y_prox_urban.tif")
        
            # Load data scene of the current time slot
            da = rioxr.open_rasterio(fn_tif)
            ds = da.to_dataset('band')
            ds = ds.rename({1: 'ProxUrban'})
            
            # Extract pixel values for coordinates
            px_values = ds.interp(x=xr.DataArray(current_data.x), y=xr.DataArray(current_data.y), method="nearest").to_dataframe()
        
            # Prepare columns for data merging
            px_values.reset_index(inplace=True)
            px_values = px_values.rename(columns ={'dim_0':'index'})
            px_values.drop(['x','y'], axis=1, inplace=True)
            current_data.reset_index(inplace=True)
            
            # Merge extracted data and fire data into one DataFrame
            concat_data = pd.concat([current_data,px_values],axis=1, join='inner')
        
            # Append the merged data to the final DataFrame
            df_loop = pd.concat([df_loop,concat_data], ignore_index=True)
        
        # Delete some columns and NA
        df_loop.drop(['index','spatial_ref'], axis=1, inplace=True)
        df_loop.dropna(inplace=True)
        df_loop
        
        # write csv file
        fpout = "D:/Stefanie/Dokumente/Studium_Geographie/Master_Marburg/Semester_1/GIS/Projektarbeit/data/lev3/fire/fire_grouped_LUC_LST_NDVI_ProxAgri_ProxUrban.csv"
        df_loop.to_csv(fpout,index=False)
        
########## 1.4 Topographic variables: Aspect, Curvature, DEM, Slope and TPI #########

    # Read fire data
    fp = 'D:/Stefanie/Dokumente/Studium_Geographie/Master_Marburg/Semester_1/GIS/Projektarbeit/data/lev3/fire/fire_grouped_LUC_LST_NDVI_ProxAgri_ProxUrban.csv'
    df = pd.read_csv(fp,delimiter=',',parse_dates=['acq_date'])
    
    # Create list of tif files
    filelist = glob('D:/Stefanie/Dokumente/Studium_Geographie/Master_Marburg/Semester_1/GIS/Projektarbeit/data/lev1/dem_var/*.tif')
    filelist

    # loop over all files in the list, extract pixel values and add them to the dataframe
    for x in filelist:
        varname = os.path.splitext(os.path.basename(x))[0]
        print(varname)
        rds = rioxr.open_rasterio(x)
        pixelvalues = rds.interp(x=xr.DataArray(df.x), y=xr.DataArray(df.y), method="nearest").values
        df[varname] = pixelvalues[0][:]
    
    df
    
    # Delete NA
    df.dropna(inplace=True)
    df
    df.sort_values(by="dem", ascending=True)
    
    # write csv file
    fpout = "D:/Stefanie/Dokumente/Studium_Geographie/Master_Marburg/Semester_1/GIS/Projektarbeit/data/lev3/fire/fire_grouped_LUC_LST_NDVI_ProxAgri_ProxUrban_DEM.csv"
    df.to_csv(fpout,index=False)
    
########## 1.5 Monthly ERA5 Climate data #########
    
    # Read fire data
    fp = 'D:/Stefanie/Dokumente/Studium_Geographie/Master_Marburg/Semester_1/GIS/Projektarbeit/data/lev3/fire/fire_grouped_LUC_LST_NDVI_ProxAgri_ProxUrban_DEM.csv'
    df_ERA5 = pd.read_csv(fp,delimiter=',',parse_dates=['acq_date'])
    df_ERA5
    
    # All monthly time steps are extracted since the ERA5 data has a monthly resolution.
    time_slots = df_ERA5['acq_date'].dt.strftime("%Y-%m").unique().tolist()
    time_slots

    # Read ERA5 data 
    fn = 'D:/Stefanie/Dokumente/Studium_Geographie/Master_Marburg/Semester_1/GIS/Projektarbeit/data/lev1/era5/ERA5_ClimateVariables_2001to2022.nc'
    ds = xr.open_dataset(fn, decode_coords="all") 
    ds
    
    # Extract pixel values over all months with a loop
        
        # Instantiate empty pandas-DataFrame
        df = pd.DataFrame()
        
        # Iterate over all available time slots
        for time_slot in time_slots:
            print(time_slot)
            
            # Clip dataframe to current time slot
            current_data = df_ERA5.loc[df_ERA5['acq_date'].dt.strftime("%Y-%m") == time_slot]
            
            # Load data scene of the current time slot
            ds_data = ds.sel(time=str(time_slot), method="nearest")
                
            # Extract pixel values for coordinates in the 1km resolution (with "interp")
            px_values = ds_data.interp(x=xr.DataArray(current_data.x), y=xr.DataArray(current_data.y), method="nearest").to_dataframe()
            
            # Prepare columns for data merging
            px_values.reset_index(inplace=True)
            px_values = px_values.rename(columns ={'dim_0':'index'})
            current_data.reset_index(inplace=True)
            
            # Merge extracted data and fire data into one DataFrame
            concat_data = pd.concat([current_data,px_values],axis=1, join='inner')
            
            # Append the merged data to the final DataFrame
            df = pd.concat([df,concat_data], ignore_index=True)
            
        df
    
    # Delete NA
    df.dropna(inplace=True)
    df
    
    # Delete columns
    df.keys()

    df = df.drop(columns=['x.1', 'y.1', 'index',"count",'time','spatial_ref'])
    df.keys()
    df
    
    # Write csv file
    fpout = "D:/Stefanie/Dokumente/Studium_Geographie/Master_Marburg/Semester_1/GIS/Projektarbeit/data/lev3/fire/fire_grouped_LUC_LST_NDVI_ProxAgri_ProxUrban_DEM_ERA5.csv"
    df.to_csv(fpout,index=False)
    
##################### Final fire dataset with all predictor variables ##############
    
    # Read fire data
    fp = 'D:/Stefanie/Dokumente/Studium_Geographie/Master_Marburg/Semester_1/GIS/Projektarbeit/data/lev3/fire/fire_grouped_LUC_LST_NDVI_ProxAgri_ProxUrban_DEM_ERA5.csv'
    df_fire = pd.read_csv(fp,delimiter=',',parse_dates=['acq_date'])
    df_fire
    df_fire.keys()
    
    # drop columns
    df_fire.drop(['x.1', 'y.1'], axis=1, inplace=True)
    # drop NA
    df_fire.dropna(inplace=True)
    df_fire
    
    # Write csv file
    fpout = "D:/Stefanie/Dokumente/Studium_Geographie/Master_Marburg/Semester_1/GIS/Projektarbeit/data/lev3/fire/fire_grouped_LUC_LST_NDVI_ProxAgri_ProxUrban_DEM_ERA5.csv"
    df_fire.to_csv(fpout,index=False)
    
########################### 2. Extract coordinates of non-fire pixels ###########################
 
# Read fire data to get the time slots
# Define path to csv file                    
fp = 'D:/Stefanie/Dokumente/Studium_Geographie/Master_Marburg/Semester_1/GIS/Projektarbeit/data/lev3/fire/fire_grouped_LUC_LST_NDVI_ProxAgri_ProxUrban_DEM_ERA5.csv'

# Read data
df_fire = pd.read_csv(fp,delimiter=',',parse_dates=['acq_date'])
df_fire
    
# Create spatial buffer around all coordinates to avoid close fire and non-fire pixels
# define a function to create latitude and longitude columns for spatial aggregation
#https://stackoverflow.com/questions/39254704/pandas-group-bins-of-data-per-longitude-latitude
def to_bin(x):
    return np.floor(x / step) * step #numpy.floor: rounding down to integer
    
# Define step size of 0.2 degrees since the ERA5 Land data have a spatial resolution of about 0.1 degrees
step = 0.2
    
# Apply function to latitude and longitude column and create new latitude and longitude columns for spatial buffering
df_fire["latBin2"] = to_bin(df_fire.y)
df_fire["lonBin2"] = to_bin(df_fire.x)
df_fire
    
# Create the time slots for the loop over years
time_slots_year = df_fire.acq_date.dt.year.unique()
time_slots_year
    
# Apply function from StackExchange to extract the coordinates for the selected land use classes
# https://gis.stackexchange.com/questions/427143/extract-selected-pixel-values-with-their-coordinates
def coordinates_and_values(raster, pixel_values):
    """
    this function extracts longitude and latitudes of list of raster pixel values
    raster: load raster with rasterio
    pixel_values: list of pixel values 
    exception: pixel value must be found more than one time in raster else len of float error
    """
    df = pd.DataFrame()                         # create dataframe
    raster_band = raster.read(1)                # read raster band
    for i in pixel_values:                      # iterate between list of pixel values
        cols, rows = np.where(raster_band == i) # extract row and column numbers for each pixel value
        cols, rows = transform.xy(raster.transform, cols, rows) # transform column and row numbers to coordinates
        values = np.array([i] * len(cols))      # create array containing n pixel value of n coordinates
        df_i = pd.DataFrame(zip(cols, rows, values), columns=['lon','lat', 'pixel']) # create dataframe for one pixel value
        df = pd.concat([df,df_i], ignore_index=True) # append to get dataframe of lon and lat of list of pixel values
    return df
    
# Extract the coordinates for relevant land use classes (30 to 150) for the years 2000 to 2020
# Create list with pixel values
pixel_values = list(range(30,160,10))
pixel_values
    
# Loop to extract pixel coordinates per year

# Instantiate empty pandas-DataFrames
df_loop = pd.DataFrame()
df_coorlucval = pd.DataFrame()
    
# Iterate over all available time slots
for time_slot in time_slots_year:
    # Since land use data are only available through 2020, we use these data for 2021 and 2022
    if time_slot == 2021 or time_slot == 2022:
        time_slot_tif = 2020
    else:
        time_slot_tif = time_slot
    print(time_slot)
        
    # Create filename
    fn_tif = pd.to_datetime(str(time_slot_tif)).strftime("D:/Stefanie/Dokumente/Studium_Geographie/Master_Marburg/Semester_1/GIS/Projektarbeit/data/lev0/luc/LUC_IT_%Y_reproj.tif")
    
    # Open luc tif file
    src = rasterio.open(fn_tif)
            
    # Extract coordinates for pixel values
    coorlucval = coordinates_and_values(src,pixel_values)
    
    # apply function to latitude and longitude column and create new latitude and longitude columns for spatial aggregation
    coorlucval["latBin2"] = to_bin(coorlucval.lat)
    coorlucval["lonBin2"] = to_bin(coorlucval.lon)
    
    # set index for combination with df_fire
    coorlucval.set_index(['latBin2','lonBin2'])
        
    # add extracted coordinates to dataframe
    df_coorlucval = pd.concat([df_coorlucval,coorlucval], ignore_index=True)
    
    # Clip fire dataframe to current time slot
    timeselect = df_fire[df_fire['acq_date'].dt.year == time_slot]
        
    # Create a list to loop over the months in a year
    time_slots_months = timeselect['acq_date'].dt.strftime("%Y-%m").unique().tolist()
     
    # Loop to compare the extracted pixel coordinates with fire coordinates for each month
    for time_slot_month in time_slots_months:
        print(time_slot_month)
            
        # Clip dataframe of the current year to the respective months
        current_data = timeselect.loc[timeselect['acq_date'].dt.strftime("%Y-%m") == time_slot_month]
            
        # set index for combination with coordinates dataframe
        current_data.set_index(['latBin2','lonBin2'])
    
        # select only coordinates that are not in the fire df
        nonfirecoordinates = coorlucval[~coorlucval.index.isin(current_data.index)]
            
        # select a random subsample
        samplesize = len(current_data) + 1
        randomSubSample = nonfirecoordinates.sample(n=samplesize, random_state=1)
        
        # add column with timeslot
        randomSubSample['timeslot'] = time_slot_month
    
        # Append the merged data to the final DataFrame
        df_loop = pd.concat([df_loop,randomSubSample], ignore_index=True)
            
# Show results
df_loop
    
# Write final dataframe to csv file
fnout = "D:/Stefanie/Dokumente/Studium_Geographie/Master_Marburg/Semester_1/GIS/Projektarbeit/data/lev3/nofire/nofire_2001_2022.csv"
df_loop.to_csv(fnout)


############################## 3. Extract values of non-fire pixels #############################

########## Land Surface Temperature (LST) ##########

# Define path to the file
fp = "D:/Stefanie/Dokumente/Studium_Geographie/Master_Marburg/Semester_1/GIS/Projektarbeit/data/lev3/nofire/nofire_2001_2022.csv"

# Read data
df = pd.read_csv(fp,delimiter=',',parse_dates=['timeslot'])
df

# Rename the lon, lat columns to x and y to facilitate extraction by coordinates
df.rename(columns={'lat': 'y', 'lon': 'x'}, inplace=True)
df

# Extract all the monthly time steps, because the LST data has a monthly resolution
time_slots = df['timeslot'].dt.strftime("%Y-%m").unique().tolist()
time_slots

# Read in the netcdf file with the LST data
fn = 'D:/Stefanie/Dokumente/Studium_Geographie/Master_Marburg/Semester_1/GIS/Projektarbeit/data/lev1/lst/LST_2001to2022.nc'
ds = xr.open_dataset(fn, decode_coords="all")
ds

# Extraction of the pixel values over all months
# Instantiate empty pandas-DataFrame
df_loop = pd.DataFrame()

# Iterate over all available time slots
for time_slot in time_slots:
    print(time_slot)
    
    # Clip dataframe to current time slot
    current_data = df.loc[df['timeslot'].dt.strftime("%Y-%m") == time_slot]
    
    # Load data scene of the current time slot
    ds_data = ds.sel(time=str(time_slot), method="nearest")
        
    # Extract pixel values for coordinates
    px_values = ds_data.interp(x=xr.DataArray(current_data.x), y=xr.DataArray(current_data.y), method="nearest").to_dataframe()
    
    # Prepare columns for data merging
    px_values.reset_index(inplace=True)
    px_values = px_values.rename(columns ={'dim_0':'index'})
    px_values.drop(['x','y'], axis=1, inplace=True)
    current_data.reset_index(inplace=True)
    
    # Merge extracted data and fire data into one DataFrame
    concat_data = pd.concat([current_data,px_values],axis=1, join='inner')
    
    # Append the merged data to the final DataFrame
    df_loop = pd.concat([df_loop,concat_data], ignore_index=True)

# Show results
df_loop

# Delete the index column and NA records
df_loop.dropna(inplace=True)
df_loop

df_loop.keys()

df_loop.drop(["Unnamed: 0", 'time','index','spatial_ref'], axis=1, inplace=True)
df_loop


########## NDVI ##########

# If we use the existing dataframe we create a copy of it first so that we do not have to change the name of the dataframe in the following.
df = df_loop.copy()
df

time_slots = df['timeslot'].dt.strftime("%Y-%m").unique().tolist()
time_slots

# Open netcdf file
fn = 'D:/Stefanie/Dokumente/Studium_Geographie/Master_Marburg/Semester_1/GIS/Projektarbeit/data/lev1/ndvi/NDVI_2001to2022.nc'
ds = xr.open_dataset(fn, decode_coords="all")
ds

# Function to extract pixel values from raster file
def extractPxValues(time_slots_def,df_def,ds_def):
    # Instantiate empty pandas-DataFrame
    df_loop = pd.DataFrame()

    # Iterate over all available time slots
    for time_slot in time_slots_def:
        print(time_slot)
        
        # Clip dataframe to current time slot
        current_data = df_def.loc[df_def['timeslot'].dt.strftime("%Y-%m") == time_slot]
    
        # Load data scene of the current time slot
        ds_data = ds_def.sel(time=str(time_slot), method="nearest")
    
        # Extract pixel values for coordinates
        px_values = ds_data.interp(x=xr.DataArray(current_data.x), y=xr.DataArray(current_data.y), method="nearest").to_dataframe()
        
        # Prepare columns for data merging
        px_values.reset_index(inplace=True)
        px_values = px_values.rename(columns ={'dim_0':'index'})
        px_values.drop(['x','y'], axis=1, inplace=True)
        current_data.reset_index(inplace=True)
    
        # Merge extracted data and fire data into one DataFrame
        concat_data = pd.concat([current_data,px_values],axis=1, join='inner')
    
        # Append the merged data to the final DataFrame
        df_loop = pd.concat([df_loop,concat_data], ignore_index=True)
        
    return df_loop

# Call the function to extract the pixel values
df_extract_NDVI = extractPxValues(time_slots,df,ds)

# Show results
df_extract_NDVI

# Remove some columns
df_extract_NDVI.drop(['time','index','spatial_ref'], axis=1, inplace=True)
df_extract_NDVI.keys()

# Remove NA values
df_extract_NDVI.dropna(inplace=True)
df_extract_NDVI


########## Distance to agricultural areas ##########

# if we use the existing dataframe we create a copy of it first so that we do not have to change the name of the dataframe in the following.
df = df_extract_NDVI.copy()

# Extract all the annual time steps, because the land cover data has an annual resolution
time_slots = df.timeslot.dt.year.unique()
time_slots

# The extraction of the pixel values over all months is done with a loop
# Instantiate empty pandas-DataFrame
df_loop = pd.DataFrame()

# Iterate over all available time slots
for time_slot in time_slots:
    print(time_slot)
    
    # Clip dataframe to current time slot
    current_data = df[df.timeslot.dt.year == time_slot]
    
    # Since land use data are only available through 2020, we use these data for 2021 and 2022.
    if time_slot == 2021 or time_slot == 2022:
        time_slot = 2020

    # Create filename
    fn_tif = pd.to_datetime(str(time_slot)).strftime("D:/Stefanie/Dokumente/Studium_Geographie/Master_Marburg/Semester_1/GIS/Projektarbeit/data/lev2/proximity_agriculture/IT_%Y_prox_agric.tif")
    
    # Load data scene of the current time slot
    da = rioxr.open_rasterio(fn_tif)
    ds = da.to_dataset('band')
    ds = ds.rename({1: 'ProxAgri'})
    
    # Extract pixel values for coordinates
    px_values = ds.interp(x=xr.DataArray(current_data.x), y=xr.DataArray(current_data.y), method="nearest").to_dataframe()

    # Prepare columns for data merging
    px_values.reset_index(inplace=True)
    px_values = px_values.rename(columns ={'dim_0':'index'})
    px_values.drop(['x','y'], axis=1, inplace=True)
    current_data.reset_index(inplace=True)
    
    # Merge extracted data and fire data into one DataFrame
    concat_data = pd.concat([current_data,px_values],axis=1, join='inner')

    # Append the merged data to the final DataFrame
    df_loop = pd.concat([df_loop,concat_data], ignore_index=True)
    
# Show results
df_loop

# Delete the index column and NA records
df_loop.drop(['index','spatial_ref'], axis=1, inplace=True)
df_loop.keys()

df_loop.dropna(inplace=True)
df_loop


########## Distance to urban areas ##########

# if we use the existing dataframe we create a copy of it first so that we do not have to change the name of the dataframe in the following.
df = df_loop.copy()

# Instantiate empty pandas-DataFrame
df_loop = pd.DataFrame()

# Iterate over all available time slots
for time_slot in time_slots:
    print(time_slot)
    
    # Clip dataframe to current time slot
    current_data = df[df.timeslot.dt.year == time_slot]
    
    # Since land use data are only available through 2020, we use these data for 2021 and 2022.
    if time_slot == 2021 or time_slot == 2022:
        time_slot = 2020

    # Create filename
    fn_tif = pd.to_datetime(str(time_slot)).strftime("D:/Stefanie/Dokumente/Studium_Geographie/Master_Marburg/Semester_1/GIS/Projektarbeit/data/lev2/proximity_urban/IT_%Y_prox_urban.tif")

    # Load data scene of the current time slot
    da = rioxr.open_rasterio(fn_tif)
    ds = da.to_dataset('band')
    ds = ds.rename({1: 'ProxUrban'})
    
    # Extract pixel values for coordinates
    px_values = ds.interp(x=xr.DataArray(current_data.x), y=xr.DataArray(current_data.y), method="nearest").to_dataframe()

    # Prepare columns for data merging
    px_values.reset_index(inplace=True)
    px_values = px_values.rename(columns ={'dim_0':'index'})
    px_values.drop(['x','y'], axis=1, inplace=True)
    current_data.reset_index(inplace=True)
    
    # Merge extracted data and fire data into one DataFrame
    concat_data = pd.concat([current_data,px_values],axis=1, join='inner')

    # Append the merged data to the final DataFrame
    df_loop = pd.concat([df_loop,concat_data], ignore_index=True)
    
# Show results
df_loop

# Delete the index column and NA records
df_loop.drop(['index','spatial_ref'], axis=1, inplace=True)
df_loop.keys()

df_loop.dropna(inplace=True)
df_loop


########## DEM, Aspect, Slope, Curvature and TPI ##########

# if we use the existing dataframe we create a copy of it first so that we do not have to change the name of the dataframe in the following.
df_DEM = df_loop.copy()
df_DEM

# Create list of tif files
filelist = glob('D:/Stefanie/Dokumente/Studium_Geographie/Master_Marburg/Semester_1/GIS/Projektarbeit/data/lev1/dem_var/*.tif')
filelist

# Loop over all files in the list, extract pixel values and add them to the dataframe
for x in filelist:
    varname = os.path.splitext(os.path.basename(x))[0]
    print(varname)
    rds = rioxr.open_rasterio(x)
    pixelvalues = rds.interp(x=xr.DataArray(df_DEM.x), y=xr.DataArray(df_DEM.y), method="nearest").values
    df_DEM[varname] = pixelvalues[0][:]

# Show results
df_DEM

# Remove NA values
df_DEM.dropna(inplace=True)
df_DEM


########## ERA5 Climate Data ##########

# if we use the existing dataframe we create a copy of it first so that we do not have to change the name of the dataframe in the following.
df_ERA5 = df_DEM.copy()
df_ERA5

# Read the netcdf file with the ERA5 Land data
fn = 'D:/Stefanie/Dokumente/Studium_Geographie/Master_Marburg/Semester_1/GIS/Projektarbeit/data/lev1/era5/ERA5_ClimateVariables_2001to2022.nc'
ds = xr.open_dataset(fn, decode_coords="all") 
ds

# Extract all the monthly time steps, because the ERA5 data has a monthly resolution
time_slots = df_ERA5['timeslot'].dt.strftime("%Y-%m").unique().tolist()
time_slots

# The extraction of the pixel values over all months is done with a loop
# function to extract pixel values from raster file
def extractPxValues(time_slots_def,df_def,ds_def):
    # Instantiate empty pandas-DataFrame
    df_loop = pd.DataFrame()

    # Iterate over all available time slots
    for time_slot in time_slots_def:
        print(time_slot)
        
        # Clip dataframe to current time slot
        current_data = df_def.loc[df_def['timeslot'].dt.strftime("%Y-%m") == time_slot]
    
        # Load data scene of the current time slot
        ds_data = ds_def.sel(time=str(time_slot), method="nearest")
    
        # Extract pixel values for coordinates
        px_values = ds_data.interp(x=xr.DataArray(current_data.x), y=xr.DataArray(current_data.y), method="nearest").to_dataframe()
        
        # Prepare columns for data merging
        px_values.reset_index(inplace=True)
        px_values = px_values.rename(columns ={'dim_0':'index'})
        px_values.drop(['x','y'], axis=1, inplace=True)
        current_data.reset_index(inplace=True)
    
        # Merge extracted data and fire data into one DataFrame
        concat_data = pd.concat([current_data,px_values],axis=1, join='inner')
    
        # Append the merged data to the final DataFrame
        df_loop = pd.concat([df_loop,concat_data], ignore_index=True)
        
    return df_loop

# call the function to extract the pixel values
df_extract_ERA5 = extractPxValues(time_slots,df_ERA5,ds)

# Show results
df_extract_ERA5

df_extract_ERA5.keys()

# Remove some columns
df_extract_ERA5.drop(['time','index','spatial_ref'], axis=1, inplace=True)
df_extract_ERA5.keys()

# Remove NA values
df_extract_ERA5.dropna(inplace=True)
df_extract_ERA5


########## Write out final non-fire dataset to csv file ##########

# Define path and file name
fpout = "D:/Stefanie/Dokumente/Studium_Geographie/Master_Marburg/Semester_1/GIS/Projektarbeit/data/lev3/nofire/nofire_LUC_LST_NDVI_ProxAgriUrban_DEM_ERA5.csv"

# Write data
df_extract_ERA5.to_csv(fpout,index=False)
