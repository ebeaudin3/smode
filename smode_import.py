import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import scipy
from glob import glob
import pandas as pd
import gsw
from datetime import datetime, timedelta, UTC
#from parula import parula
from cmocean import cm as cm

def smode_pgon(ax,lw=0.8,c='k'):
    from matplotlib.patches import Polygon
    vertices = np.array([[-126.25, 38.342],[-123.99, 37.707],[-123.354, 37.75], [-122.92, 37.00],
                     [-124.36, 36.337],[-124.16, 36.00],[-125.515, 35.60]])
    polygon = Polygon(vertices, closed=True, edgecolor=c, linewidth=lw, facecolor='none')
    ax.add_patch(polygon)

# Load Pacific Coastline
mat = scipy.io.loadmat('/Users/elise/data/WorldCstLinePacific.mat')['cst']
clat,clon = mat['lat'][0][0], mat['lon'][0][0]

def coastline(ax,lw=1,c='k'):
    ax.plot(clon,clat,'-',color=c,linewidth=lw)
    ax.set_xlim(-128,-121)
    ax.set_ylim(33,39)

def haversine(lat1, lon1, lat2, lon2):
    from scipy.spatial.distance import cdist
    R = 6371  # Earth radius in kilometers
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2) ** 2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distance = R * c
    return distance #in km

def coriolis_parameter(latitude):
    omega = 7.2921e-5  # Angular velocity of the Earth (radians per second)
    phi = np.radians(latitude)  # Convert latitude to radians
    f = 2 * omega * np.sin(phi)  # Calculate the Coriolis parameter
    return f

def interp_ship_on_drifter(underway_data, drifter, dist=1):
    from scipy.interpolate import interp1d
    # drifter data
    time_dr= drifter.time
    lat_dr = drifter.latitude
    lon_dr = drifter.longitude

    def is_night(timestamp,delta=-7): #account for photochemical quenching
        # UTC to PST delta=-7
        return (timestamp.dt.hour < 9-delta) | (timestamp.dt.hour >= 17-delta)
    
    def interpolate_function(time_dr,time_uw,variable):
        num_time_uw      = time_uw.astype('int64') / 1e9 # nanoseconds to seconds
        num_time_drifter = time_dr.astype('int64') / 1e9
        interp_function  = interp1d(num_time_uw, variable, kind='linear',fill_value=np.nan, bounds_error=False)
        interpolated_data= interp_function(num_time_drifter)
        return interpolated_data
        
    # Interpolate ship data onto drifter time
    interp_lon = interpolate_function(time_dr,underway_data.time,underway_data.lon)
    interp_lat = interpolate_function(time_dr,underway_data.time,underway_data.lat)
    interp_chl = interpolate_function(time_dr,underway_data.time,underway_data.chl)
    interp_temp = interpolate_function(time_dr,underway_data.time,underway_data.temp)
    interp_salt = interpolate_function(time_dr,underway_data.time,underway_data.salt)
    interp_rho = interpolate_function(time_dr,underway_data.time,underway_data.rho)
    interp_o2sat = interpolate_function(time_dr,underway_data.time,underway_data.o2sat)
    
    # Save chlorophyll values when dist < 1 km  & nighttime
    distances = haversine(lat_dr, lon_dr, interp_lat, interp_lon) # pairwise distances
    idx = (distances <= dist) & (is_night(time_dr))

    # chlorophyll rate of change
    dChldt_ship = np.empty(len(time_dr[idx]))*np.nan
    dChl = np.diff(interp_chl[idx]) # mg m-3
    dt   = (time_dr[idx].diff(dim='time')*1e-9).astype(float) # seconds
    dChldt_ship[1::] = dChl/dt

    # temperature rate of change
    dSSTdt_ship = np.empty(len(time_dr[idx]))*np.nan
    dSST = np.diff(interp_temp[idx]) # C
    dt   = (time_dr[idx].diff(dim='time')*1e-9).astype(float) # seconds
    dSSTdt_ship[1::] = dSST/dt

    # remove outliers +/3 3e-3 mg m-3 s-1
    condition = np.where(np.abs(dChldt_ship)<3e-3)[0]
    
    dataset = xr.Dataset(
    {
        'lon': ('time', lon_dr[idx].values),
        'lat': ('time', lat_dr[idx].values),
        'chl': ('time', interp_chl[idx]),
        'temp': ('time', interp_temp[idx]),
        'salt': ('time', interp_salt[idx]),
        'rho': ('time', interp_rho[idx]),
        'o2sat': ('time', interp_o2sat[idx]),
        'dChldt': ('time', dChldt_ship),
        'dSSTdt': ('time', dSSTdt_ship),
    },
    coords={'time': time_dr[idx].values},
    attrs={'name': drifter.title[-7::]},
    )
    
    return dataset