import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, glob, json
from scipy.integrate import cumtrapz
from scipy.signal import butter, lfilter

#%% lowpass_filter (butter)
def lowpass_filter(imu, cutoff, fs, order):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    butter_x, butter_y, butter_z = lfilter(b, a, imu[0], axis=0), lfilter(b, a, imu[1], axis=0), lfilter(b, a, imu[2], axis=0)
    filtered_imu = np.array([butter_x, butter_y, butter_z])

    return filtered_imu    

#%% compute_velocity
def compute_velocity_and_displacement(imu, times):
    # sampling rate
    fs = round(len(times)/times[-1],2)  

    # lowpass filter the accelerometer data
    cutoff = 0.8
    order = 4
    filtered_imu = lowpass_filter(imu, cutoff, fs, order)

    # compute velocity
    velocity = np.array([cumtrapz(filtered_imu[0], times, initial=0), 
                        cumtrapz(filtered_imu[1], times, initial=0), 
                        cumtrapz(filtered_imu[2], times, initial=0)])
    
    # compute displacement
    displacement = np.array([cumtrapz(velocity[0], times, initial=0), 
                             cumtrapz(velocity[1], times, initial=0), 
                             cumtrapz(velocity[2], times, initial=0)])
    
    return filtered_imu, velocity, displacement

#%% compute_glasses_traveled
def compute_glasses_traveled(file_lists):
    all_data = pd.DataFrame(columns = ['folder', 'traveled time', 'ramp', 'session', 
                                       'raw acceleration [m/s^2]', 'filtered acceleration [m/s^2]'
                                       'velocity [m/s]', 'displacement [m]'])

    for folder in file_lists:
        # read json file
        with open(os.path.join(folder, 'info.json')) as f:
            info = json.load(f)
            session_id = list(info['template_data']['data'].keys())[0]
            session_info = info['template_data']['data'][session_id][0]
            traveled_time = float(session_info[:session_info.find('sec')])
            ramp = float(session_info[session_info.find('sec')+3:session_info.find('ramp')])
            session = int(session_info[session_info.find('ramp')+4:])

        imu = pd.read_csv(os.path.join(folder, 'imu.csv'))        
        # if IMU data is missing, continue to the next folder
        if len(imu) == 0:
            print(f'Folder "{os.path.basename(folder)}" is empty')
            continue

        # compute velocity 
        raw_imu = imu[['acceleration x [G]', 'acceleration y [G]', 'acceleration z [G]']].values.T * 9.81   # convert unit to m/s^2
        times = imu['timestamp [ns]'].to_numpy()/1e9
        times = times - times[0]
        filtered_imu, velocity, displacement = compute_velocity_and_displacement(imu, times)

        # add as a new row
        new_row = pd.DataFrame({
            'folder': folder,
            'traveled time': traveled_time, 
            'ramp': ramp,
            'session': session,
            'raw acceleration [m/s^2]': [raw_imu], 
            'filtered acceleration [m/s^2]': [filtered_imu],
            'velocity [m/s]': [velocity], 
            'displacement [m]': [displacement],
        })

        # save the new row to all_data
        all_data = pd.concat([all_data, new_row])

    # return all_data
    return all_data



#%% main code 
data_dir = './data'
file_lists = glob.glob(os.path.join(data_dir, '2023*'))
all_data = compute_glasses_traveled(file_lists)

#### now draw all_data
