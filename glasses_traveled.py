import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, glob, json
from scipy.integrate import cumtrapz
from scipy.signal import butter, lfilter
import seaborn as sns

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
    all_data = pd.DataFrame(columns = ['folder', 'traveled time', 'traveled distance',
                                       'speed', 'ramp', 'session', 'time',
                                       'raw acceleration [m/s^2]', 'filtered acceleration [m/s^2]',
                                       'velocity [m/s]', 'displacement [m]'])

    for folder in file_lists:
        # read json file
        with open(os.path.join(folder, 'info.json')) as f:
            info = json.load(f)
            session_id = list(info['template_data']['data'].keys())[0]
            session_info = info['template_data']['data'][session_id][0]
            # if session info is missing, continue to the next folder
            if session_info == '':
                print(f'Folder "{os.path.basename(folder)}" does not have session info')
                continue
            elif session_info == 'stationary':
                traveled_time = 0
                speed = 0
                ramp = 0
                session = 0
            else:
                if (session_info.find('sec') != -1):
                    traveled_time =  float(session_info[:session_info.find('sec')])
                    session_info = session_info[session_info.find('sec')+3:]
                else:
                    traveled_time = 0

                if session_info.find('speed') != -1:
                    speed = float(session_info[:session_info.find('speed')])
                    session_info = session_info[session_info.find('speed')+5:]
                else:
                    speed = 0

                if session_info.find('ramp') != -1:
                    ramp = float(session_info[:session_info.find('ramp')])
                    session_info = session_info[session_info.find('ramp')+4:]
                else:
                    ramp = int(0)

                if session_info.find('inch') == -1:
                    session = int(session_info)
                    traveled_distance = 'max'
                else:
                    traveled_distance = str(session_info[:session_info.find('inch')])
                    session = int(session_info[session_info.find('inch')+4:])

        imu = pd.read_csv(os.path.join(folder, 'imu.csv'))        
        # if IMU data is missing, continue to the next folder
        if len(imu) == 0:
            print(f'Folder "{os.path.basename(folder)}" is empty')
            continue

        # compute velocity 
        raw_imu = imu[['acceleration x [G]', 'acceleration y [G]', 'acceleration z [G]']].values.T * 9.81   # convert unit to m/s^2
        times = imu['timestamp [ns]'].to_numpy()/1e9
        times = times - times[0]
        filtered_imu, velocity, displacement = compute_velocity_and_displacement(raw_imu, times)

        # add as a new row
        new_row = pd.DataFrame({
            'folder': folder,
            'traveled time': traveled_time, 
            'traveled distance': [traveled_distance],
            'speed': speed,
            'ramp': ramp,
            'session': session,
            'time': [times],
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

### plot all_data
colors = sns.color_palette('Set2')
params = all_data[['traveled time', 'speed', 'ramp', 'traveled distance']].drop_duplicates()
fig_num = 0
for id in range(len(params)):
    traveled_time = params.iloc[id]['traveled time']
    speed = params.iloc[id]['speed']
    ramp = params.iloc[id]['ramp']
    traveled_distance = params.iloc[id]['traveled distance']
    data = all_data[(all_data['speed'] == speed) & (all_data['ramp'] == ramp)]

    # create a new figure for each parameter
    fig, ax = plt.subplots(3,3, figsize=(8, 8))    
    ax[0,0].set_title('X-axis')    # first panel shows raw and filtered accelerometer 
    ax[0,1].set_title('Y-axis')             # second panel shows velocity
    ax[0,2].set_title('Z-axis')          # third panel shows displacement    
    for row in range(len(data)):
        time = data['time'].iloc[row]
        for direction in range(3):
            # draw Accelerometer in the first panel
            y = data['raw acceleration [m/s^2]'].iloc[row][direction,:]
            ax[0, direction].plot(time, y, color = colors[direction], alpha = 0.3, linestyle = '-')

            y = data['filtered acceleration [m/s^2]'].iloc[row][direction,:]
            ax[0, direction].plot(time, y, color = colors[direction])

            if row == len(data)-1:
                legend_elements = [plt.Line2D([0], [0], color=colors[direction], alpha = 0.3, linestyle ='-'),                            
                    plt.Line2D([0], [0], color=colors[direction])]
                legend_labels = ['Raw', 'Filtered']
                ax[0, direction].legend(legend_elements, legend_labels, fontsize='x-small')
    
            # draw velocity in the second panel
            y = data['velocity [m/s]'].iloc[row][direction,:]
            ax[1, direction].plot(time, y, color = colors[direction])

            # draw displacement in the last panel
            y = data['displacement [m]'].iloc[row][direction,:]    # change unit to cm
            ax[2, direction].plot(time, y, color = colors[direction])

            if row == len(data)-1:
                ax[0,direction].set_xlabel('time [s]')

    ax[0,0].set_ylabel('Acceleration [m/s$^2$]')
    ax[1,0].set_ylabel('Velocity [m/s]')
    ax[2,0].set_ylabel('Displacement [m]')
    
    suptitle = f'ramp:{ramp}_traveled:{traveled_distance}'
    if traveled_time != 0:
        suptitle = f'traveled time (sec):{traveled_time}_' + suptitle
    if (traveled_time == 0) & (speed == 0) & (ramp == 0):
        suptitle = 'stationary'
        
    fig.suptitle(suptitle)
    
    fig_num += 1
    filename = f'slider_figure_{fig_num}.png'
    fig.savefig(os.path.join('./figures', filename), dpi=150)
        


        


    
