# -*- coding: utf-8 -*-
"""
Data Processing

Created on Thu Jul 13 10:48:19 2023

@author: Stefanie Kern & Iris Haake

"""
# Import libraries
import pandas as pd
import numpy as np
from datetime import datetime
import xarray as xr
from glob import glob # create filelist with multiple netCDF files
import rioxarray as rioxr # reprojection
import os # directory path

######################################### 1. Fire Data #####################################################

    # define path to the file
    fp = "D:/Stefanie/Dokumente/Studium_Geographie/Master_Marburg/Semester_1/GIS/Projektarbeit/data/lev0/fire/fire_archive_M-C61_366518.csv"
    
    # read data
    fire = pd.read_csv(fp, parse_dates=["acq_date"], index_col="acq_date",sep=",")
    fire
    
    # Plotting values of selected columns of the fire data set, e.g. `Fire Radiative Power`.
    fire.frp.plot()
    
    # Extract and filter data by threshold
    fire_filt = fire.loc[ (fire['confidence'] >= 30) & (fire['scan'] < 1.5)]
    fire_filt
    
    # If the focus is only on presumed vegetation fire `type = 0`, a large loss of data occurs. 
    fire.type.hist()
    
    #So "other static land source" `type = 2` will be also part of the calculation. The column with `type` information is unneeded and can be deleted.
    # Remove column "type"
    fire_filt = fire_filt.drop(["type"], axis = 1)
    fire_filt
    
    # Remove NA
    fire_filt.dropna()
    
    # Write filtered dataset
    # define path and file name
    fpout = "D:/Stefanie/Dokumente/Studium_Geographie/Master_Marburg/Semester_1/GIS/Projektarbeit/data/lev1/fire/fire_filtered.csv"
    
    # write data
    fire_filt.to_csv(fpout)
    
    # Group and aggregate by time, latitude and longitude
    
        # When reading the data, the `acq_date` column is not used as index, because it's part of the aggregation process.
        # define path to csv file
        fp = "D:/Stefanie/Dokumente/Studium_Geographie/Master_Marburg/Semester_1/GIS/Projektarbeit/data/lev1/fire/fire_filtered.csv" 
        # read data 
        filtereddata = pd.read_csv(fp,parse_dates=["acq_date"],sep=",")
        filtereddata
        
        # The spatial and temporal "step size" of 0.02 and 4 Days is applied.
        # define a function to create latitude and longitude columns for spatial aggregation
        def to_bin(x):
            return np.floor(x / step) * step # numpy.floor: rounding down to integer
        
        # Group by space:
            # define step size for spatial aggregation
            step = 0.02
            
            # apply function to latitude and longitude column and create new latitude and longitude columns for spatial aggregation
            filtereddata["latBin"] = to_bin(filtereddata.latitude)
            filtereddata["lonBin"] = to_bin(filtereddata.longitude)
            filtereddata
        
        # Group by time:
            groupedFireData = filtereddata.groupby([pd.Grouper(key='acq_date', freq='4D'),"latBin", "lonBin"]).agg(minDate=('acq_date','min'),
                                                                                                                   maxDate=('acq_date','max'),
                                                                                                                   minlat=('latitude','min'),
                                                                                                                   minlon=('longitude','min'),
                                                                                                                   maxlat=('latitude','max'),
                                                                                                                   maxlon=('longitude','max'),
                                                                                                                   meanlat=('latitude','mean'),
                                                                                                                   meanlon=('longitude','mean'),
                                                                                                                   FireRadiativePower=('frp','mean'),
                                                                                                                   count=('frp','count'))
            groupedFireData
        
    # Write grouped data to csv file
    outfile = 'D:/Stefanie/Dokumente/Studium_Geographie/Master_Marburg/Semester_1/GIS/Projektarbeit/data/lev1/fire/fire_grouped.csv'
    groupedFireData.to_csv(outfile)


######################################### 2. NDVI ####################################################################################################
    # Create list of all the NDVI geotiff files 
    filelist = glob('D:/Stefanie/Dokumente/Studium_Geographie/Master_Marburg/Semester_1/GIS/Projektarbeit/data/lev0/ndvi/*.tif')
    filelist[0:9]
    
    # Separate filename from the complete path information
    geotiff_list = [os.path.basename(x) for x in filelist]
    geotiff_list[0:9]
    
    # Extract datetime from filename and convert it to a datetime object
    timestring = [x[34:41] for x in geotiff_list] #extract the string between position 34 and 41 (adjust if you have another filename).
    timestring[0:9]
    
    timelist = sorted([datetime.strptime(x,'%Y%j') for x in timestring])
    timelist[0:9]
    
    # Create xarray time variable to concatenate the individual NDVI data into one xarray Dataset with x, y coordinates and time.
    time_var = xr.Variable('time',timelist)
    time_var
    
    # Now, a "list comprehension" is applied to loop over all the geotiff files and concat them into an `xarray DataArray` based on the time variable.
    geotiffs_da = xr.concat([rioxr.open_rasterio(i) for i in filelist],dim=time_var)
    geotiffs_da
    
    # Conversion into a xarray dataset
    geotiffs_ds = geotiffs_da.to_dataset('band')
    geotiffs_ds
    
    # Rename variable
    geotiffs_ds = geotiffs_ds.rename({1: 'NDVI'})
    geotiffs_ds
    
    # Apply scaling factor
    geotiffs_ds["NDVI"] = geotiffs_ds.NDVI * 0.0001
    geotiffs_ds
    # Plot some data for visual inspection
    geotiffs_ds.NDVI.sel(time=slice('2020-05', '2020-08')).plot(col="time")
    
    # Check value distribution 
    geotiffs_ds.NDVI.plot.hist()
   
    # Mask values below -0.2
    geotiffs_ds_masked = geotiffs_ds.where(geotiffs_ds >= -0.2, drop=True)
    geotiffs_ds_masked
    geotiffs_ds_masked.NDVI.plot.hist()
    
    geotiffs_ds_masked.NDVI.sel(time=slice('2020-05', '2020-08')).plot(col="time")
    
    # Check crs information
    geotiffs_ds_masked.rio.write_crs(4326, inplace=True)
    geotiffs_ds_masked.rio.crs

    # Reprojection for extent
        # reference NDVI file
        ndvi_tomatch = rioxr.open_rasterio('D:/Stefanie/Dokumente/Studium_Geographie/Master_Marburg/Semester_1/GIS/Projektarbeit/data/lev0/ndvi/MOD13A3.061__1_km_monthly_NDVI_doy2001001_aid0001.tif')
        # Reprojection with NDVI as reference
        geotiffs_ds_masked = geotiffs_ds_masked.rio.reproject_match(ndvi_tomatch)
        geotiffs_ds_masked

    # Write NDVI data
    outfilename = 'D:/Stefanie/Dokumente/Studium_Geographie/Master_Marburg/Semester_1/GIS/Projektarbeit/data/lev1/ndvi/NDVI_2001to2022.nc'
    geotiffs_ds_masked.to_netcdf(path=outfilename)


######################################### 3. Land Surface Temperature (LST) ##############################################################################
    # Create list of all the LST geotiff files 
    filelist = glob('D:/Stefanie/Dokumente/Studium_Geographie/Master_Marburg/Semester_1/GIS/Projektarbeit/data/lev0/lst/*.tif')
    filelist[0:9]
    
    # Separate filename from the complete path information
    geotiff_list = [os.path.basename(x) for x in filelist]
    geotiff_list[0:9]
    
    # Extract datetime from filename and convert it to a datetime object
    timestring = [x[27:34] for x in geotiff_list] #extract the string between position 34 and 41 (adjust if you have another filename).
    timestring[0:9]
    
    timelist = sorted([datetime.strptime(x,'%Y%j') for x in timestring])
    timelist[0:9]
    
    # Create xarray time variable to concatenate the individual LST data into one xarray Dataset with x, y coordinates and time.
    time_var = xr.Variable('time',timelist)
    time_var
    
    # Now, a "list comprehension" is applied to loop over all the geotiff files and concat them into an `xarray DataArray` based on the time variable.
    geotiffs_da = xr.concat([rioxr.open_rasterio(i) for i in filelist],dim=time_var)
    geotiffs_da
    
    # Conversion into a xarray dataset
    geotiffs_ds = geotiffs_da.to_dataset('band')
    geotiffs_ds
    
    # Rename variable
    geotiffs_ds = geotiffs_ds.rename({1: 'LST'})
    geotiffs_ds
    
    # Aggregate 8-day data to monthly averaged data
    ds_mon = geotiffs_ds.resample(time="1MS", skipna=True).mean("time") # MS: month begin
    ds_mon
    
    # Apply scaling factor
    ds_mon["LST"] = ds_mon.LST * 0.02
    
    # Reprojection 
        # reference NDVI file
        ndvi_tomatch = rioxr.open_rasterio('D:/Stefanie/Dokumente/Studium_Geographie/Master_Marburg/Semester_1/GIS/Projektarbeit/data/lev0/ndvi/MOD13A3.061__1_km_monthly_NDVI_doy2001001_aid0001.tif')
        # Reprojection with NDVI as reference
        ds_mon = ds_mon.rio.reproject_match(ndvi_tomatch)
        ds_mon
    
    # Plot some data for visual inspection
    ds_mon.LST.sel(time=slice('2020-05', '2020-08')).plot(col="time")
    
    # Check value distribution 
    ds_mon.LST.plot.hist()
    
    # Mask values below 200 Kelvin
    ds_mon_masked = ds_mon.where(ds_mon >= 200, drop=True)
    ds_mon_masked.LST.plot.hist()
    
    ds_mon_masked.LST.sel(time=slice('2020-05', '2020-08')).plot(col="time")
    
    # Check crs information
    ds_mon_masked.rio.write_crs(4326, inplace=True)
    
    # Reprojection 
        # reference NDVI file
        ndvi_tomatch = rioxr.open_rasterio('D:/Stefanie/Dokumente/Studium_Geographie/Master_Marburg/Semester_1/GIS/Projektarbeit/data/lev0/ndvi/MOD13A3.061__1_km_monthly_NDVI_doy2001001_aid0001.tif')
        # Reprojection with NDVI as reference
        ds_mon_masked = ds_mon_masked.rio.reproject_match(ndvi_tomatch)
        ds_mon_masked
        
    # Write LST data
    outfilename = 'D:/Stefanie/Dokumente/Studium_Geographie/Master_Marburg/Semester_1/GIS/Projektarbeit/data/lev1/lst/LST_2001to2022.nc'
    ds_mon_masked.to_netcdf(path=outfilename)
    

########################################## 4. Land Use Classes (LUC) ##############################################################################################
    # Within a loop, the original global land cover data (LUC time serie) is for each year separately read in, 
    # masked to the study area `Italy`, the class `lccs_class` extracted, reprojected to the `CRS` and `spatial resolution` of the MODIS NDVI reference dataset, 
    # and written out as a new `NC file` per year.
    
    filelist = glob('D:/KFF/Projektarbeit/Daten/Landcover/*.nc')
    filelist
    
    ndvi_tomatch = rioxr.open_rasterio('D:/Stefanie/Dokumente/Studium_Geographie/Master_Marburg/Semester_1/GIS/Projektarbeit/data/lev0/ndvi/MOD13A3.061__1_km_monthly_NDVI_doy2001001_aid0001.tif')
    
    outpath = "D:/Stefanie/Dokumente/Studium_Geographie/Master_Marburg/Semester_1/GIS/Projektarbeit/data/lev0/luc/"
    
    for i in range(len(filelist)):
        ds = xr.open_dataset(filelist[i], decode_coords="all")
        year = str(ds.time.dt.year[0].values)
        print(year)
        ds_sds = ds.lccs_class.sel(lon=slice(6.5,18.6),lat=slice(47.2,35.49)) 
        ds_sds.rio.write_crs(4326,inplace=True)
        brd_luc_repr_match = ds_sds.rio.reproject_match(ndvi_tomatch)
        outfile = outpath + 'LUC_IT_' + year + '_reproj.tif'
        brd_luc_repr_match.rio.to_raster(outfile)
    
    # data preparation for the entire dataset from 2001 to 2022
    filelist = glob('D:/KFF/Projektarbeit/Daten/Landcover/*.nc')
    filelist
    
    # read data
    luc01_20 = xr.open_mfdataset(filelist)
    luc01_20 
    
    # Extract lccs_class and study area Italy
    fr_lccs_2001_2020 = luc01_20.lccs_class.sel(lon=slice(6.5,18.6),lat=slice(47.2,35.49)) 
    fr_lccs_2001_2020
    
    # check crs and reproject with NDVI as reference
    fr_lccs_2001_2020 = fr_lccs_2001_2020.rio.write_crs(4326, inplace=True)
    fr_lccs_2001_2020.rio.crs
    
    # Reprojection with NDVI as reference
    ndvi_tomatch = rioxr.open_rasterio('D:/Stefanie/Dokumente/Studium_Geographie/Master_Marburg/Semester_1/GIS/Projektarbeit/data/lev0/ndvi/MOD13A3.061__1_km_monthly_NDVI_doy2001001_aid0001.tif')
    
    fr_lccs_2001_2020 = fr_lccs_2001_2020.rio.reproject_match(ndvi_tomatch)
    fr_lccs_2001_2020
    
    # Plot some data to get a visual impression
    fr_lccs_2001_2020.sel(time=slice('2017', '2020')).plot(col="time")
    
    # Write prepared dataset (2000-2022) to a netCDF file
    outfileNC_reproj = "D:/Stefanie/Dokumente/Studium_Geographie/Master_Marburg/Semester_1/GIS/Projektarbeit/data/lev1/luc/LUC_IT_2001to2020.nc"
    fr_lccs_2001_2020.to_netcdf(outfileNC_reproj)

########################################## 5. Proximity of urban and agriculture areas ##########################################################
# The proximity is calculated in QGIS. The following steps were performed:
# - Read prepared LUC files (2001-2020).
# - Extract the proximity of each cell from the agricultural areas (10,20,30,40) and the urban areas (190) for each year.
    
    # Let's have a look at the processed data 
        # Proximity of agriculture areas 2020
        filelist = 'D:/Stefanie/Dokumente/Studium_Geographie/Master_Marburg/Semester_1/GIS/Projektarbeit/data/lev2/proximity_agriculture/IT_2020_prox_agric.tif'
        proxagri = rioxr.open_rasterio(filelist)
        proxagri
        
        # check crs
        proxagri.rio.crs
        
        # plot for visual inspection
        proxagri.plot()
        
        # Proximity of urban areas 2020
        filelist = 'D:/Stefanie/Dokumente/Studium_Geographie/Master_Marburg/Semester_1/GIS/Projektarbeit/data/lev2/proximity_urban/IT_2020_prox_urban.tif'
        proxurban = rioxr.open_rasterio(filelist)
        proxurban
        
        # check crs
        proxurban.rio.crs
        
        # plot for visual inspection
        proxurban.plot()

########################################## 6. ERA5 ##############################################################################################
    
    # Set filename
    fp = "D:/Stefanie/Dokumente/Studium_Geographie/Master_Marburg/Semester_1/GIS/Projektarbeit/data/lev0/era5/data.nc"
    
    # Open netcdf file
   # open netcdf file
    ds = xr.open_dataset(fp, decode_coords="all") 
    ds
    
    # Calculate Vapor Pressure Deficit (VPD) -> important variable related to wildfires
    import metpy.calc as mpcalc
        # Calculate relative humidity
        ds['rel_hum'] = mpcalc.relative_humidity_from_dewpoint(ds.t2m,ds.d2m)
        
        # Plot relative humidity
        ds.rel_hum.plot()
        
        # Calculate mixing ratio
        ds['mixRatio'] = mpcalc.mixing_ratio_from_relative_humidity(ds.sp, ds.t2m,ds.rel_hum)
        
        # Plot
        ds.mixRatio.plot()
        
        # Calculate vapor pressure
        ds['vapPres'] = mpcalc.vapor_pressure(ds.sp, ds.mixRatio)
        
        # Plot
        ds.vapPres.plot()
        
        # Calculate saturation vapor pressure
        ds['satVapPres'] = mpcalc.saturation_vapor_pressure(ds.t2m)
        
        # Plot
        ds.satVapPres.plot()
        
        # Calculate vapor pressure deficit (VPD)
        ds['VPD'] = ds.satVapPres - ds.vapPres
        
        # Plot
        ds.VPD.plot()
        
    # Convert units of selected variables
    
        # Convert total precipitation per month from m to mm
        N = ds.time.dt.days_in_month # N = number of days in a month
        ds["tp"] = ds.tp*1000*N 
        ds.tp.plot()
        
        # Convert surface latent heat flux, surface sensible heat flux, surface solar radiation and surface thermal radiation from `J/m²` to `W/m²`
        ds[{"slhf", "sshf", "ssr", "str"}] = ds[{"slhf", "sshf", "ssr", "str"}] / 86400
        ds.sshf.plot()
        ds.str.plot()
        
        # Convert the total and potential evaporation from `m` to `mm`
        ds[{"pev", "e"}] = ds[{"pev", "e"}] * 1000 * ds.time.dt.days_in_month
        ds.e.plot()
        
    # Verify coordinate reference system
    ds.rio.crs
    ds.rio.write_crs(4326,inplace=True)
    ds.rio.crs
    
    # Rename the lon, lat columns to x and y to facilitate extraction by coordinates later
    ds = ds.rename({'longitude': 'x','latitude': 'y'})
    ds
    
    # Have a look at the data
    ds.t2m.sel(time=slice('2012-05', '2012-08')).plot(col="time")
    
    # Write the dataset to a netcdf file
    outfilename = 'D:/Stefanie/Dokumente/Studium_Geographie/Master_Marburg/Semester_1/GIS/Projektarbeit/data/lev1/era5/ERA5_ClimateVariables_2001to2022.nc'
    ds.to_netcdf(path=outfilename)
    
########################################## 7. DEM ###############################################################################################
## Processing Steps in QGIS
# 1) The spatial resolution of all grids was scaled up to 1 km when exporting/saving the data. The number of columns (x = 1428) and rows (y = 1392) was adjusted to that of the reference raster from NDVI.
# 2) Subsequently, reproject all three DEM-TIF files to the coordinate reference system WGS84 (EPSG: 4326) and the extent of reference file NDVI. 
# 3) All three grids were merged together.
# 4) The merged DEM-TIF was cut to the extent of the study area France (Reference: NDVI file`). Maybe you have to set the correct extent again (1).
# 5) The following variables were derived from the DEM: Slope, Aspect, Curvature and TPI.

    # Let's have a look at the processed data 
        # DEM
            filelist = 'D:/Stefanie/Dokumente/Studium_Geographie/Master_Marburg/Semester_1/GIS/Projektarbeit/data/lev1/dem_var/dem.tif'
            dem = rioxr.open_rasterio(filelist)
            dem
            
            # check crs
            dem.rio.crs
            
            # Delete all missing values (-9999) to achieve a better representation of the actual range of values. 
            dem = dem.where(dem != -9999.)  
            
            # Plot for visual impression
            dem.plot()
            
        # Aspect
            filelist = 'D:/Stefanie/Dokumente/Studium_Geographie/Master_Marburg/Semester_1/GIS/Projektarbeit/data/lev1/dem_var/aspect.tif'
            aspect = rioxr.open_rasterio(filelist)
            aspect
            
            # check crs
            aspect.rio.crs
            
            # Delete all missing values (-9999) to achieve a better representation of the actual range of values. 
            aspect = aspect.where(aspect != -9999.)  
            
            # Plot for visual impression
            aspect.plot()
        
        # Slope
            filelist = 'D:/Stefanie/Dokumente/Studium_Geographie/Master_Marburg/Semester_1/GIS/Projektarbeit/data/lev1/dem_var/slope.tif'
            slope = rioxr.open_rasterio(filelist)
            slope
            
            # check crs
            slope.rio.crs
            
            # Delete all missing values (-9999) to achieve a better representation of the actual range of values. 
            slope = slope.where(slope != -9999.)  
            
            # Plot for visual impression
            slope.plot()
        
        # Curvature
            filelist = 'D:/Stefanie/Dokumente/Studium_Geographie/Master_Marburg/Semester_1/GIS/Projektarbeit/data/lev1/dem_var/curvature.tif'
            curvature = rioxr.open_rasterio(filelist)
            curvature
            
            # check crs
            curvature.rio.crs
            
            # Plot for visual impression
            curvature.plot()
        
        # TPI
            filelist = 'D:/Stefanie/Dokumente/Studium_Geographie/Master_Marburg/Semester_1/GIS/Projektarbeit/data/lev1/dem_var/tpi.tif'
            tpi = rioxr.open_rasterio(filelist)
            tpi
            
            # check crs
            tpi.rio.crs
            
            # Delete all missing values (-9999) to achieve a better representation of the actual range of values. 
            tpi = tpi.where(tpi != -9999.)  
            
            # Plot for visual impression
            tpi.plot()
