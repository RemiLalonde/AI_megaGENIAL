import numpy as np
import matplotlib.pyplot as plt
import h5py
from scipy import signal
import json

class filterSignal:
    def __init__(self):
        self.threshold = 0

    def open_recording(self, path):
        file = h5py.File(path, 'r')

        print("Keys in the HDF5 file:")
        for key in file.keys():
                print(key)

        dataset_name = 'recordings'
        dataset = file[dataset_name]
        return dataset

    #------------------------------------------------
    #----------------- FILTRAGE ---------------------
    #------------------------------------------------

    def median_filter(self, data):
        window_size = 3
        rows = data.shape[0]  # Number of rows
        cols = data.shape[1]  # Number of columns

        # Initialize a 2D list filled with zeros
        filtered_data = []
        for i in range(rows):
            row = []
            for j in range(cols):
                row.append(0)
            filtered_data.append(row)

        half_window = window_size // 2

        for i in range(half_window, data.shape[0] - half_window):
            for j in range(data.shape[1]):
                window_values = sorted(data[i - half_window:i + half_window + 1, j])
                if window_size % 2 == 1:
                    median = window_values[window_size // 2] 
                else:
                    median = (window_values[window_size // 2] + window_values[window_size // 2 - 1]) / 2.0
                filtered_data[i][j] = median
        # median_timepoint = np.median(data, axis=1)

        # # Step 2: Subtract the median from each channel
        # cmr_matrix = data - median_timepoint[:, np.newaxis]

        return np.array(filtered_data)

    def common_average_reference(self, neuronal_matrix):
        car_matrix = []
        for i in range(neuronal_matrix.shape[0]):
            average_timepoint = np.mean(neuronal_matrix[i, :])
            print(average_timepoint)
            car_matrix.append(neuronal_matrix[i, :] - average_timepoint)

        return car_matrix

    def apply_bandpass_filter(self, data, sampling_rate):
        lowcut = 300  # Lower cutoff frequency in Hz
        highcut = 3000  # Upper cutoff frequency in Hz
        nyquist_freq = 0.5 * sampling_rate
        low = lowcut / nyquist_freq
        high = highcut / nyquist_freq
        order = 4
        b, a = signal.butter(order, [low, high], btype='band')
        filtered_matrix = signal.filtfilt(b, a, data, axis=0)
        return filtered_matrix

    #------------------------------------------------
    #----------------- Détection de potentiel ---------------------
    #------------------------------------------------

    def detect_spikes(self, data, threshold_factor=3.0):
        # spikes_indices = []  # A list to store the indices of spike occurrences
        # median
        # for channel_data in data.T:
        #     spike_index = []  # Iterate over each channel's data
        #     median = np.median(channel_data)
        #     mad = np.median(np.abs(channel_data - median))
        #     self.threshold = mad * threshold_factor

        # return median
        pass

    #------------------------------------------------
    #----------------- Séparation en fenêtre ---------------------
    #------------------------------------------------

    def split_into_windows(self, data, window_size, sampling_rate):
        
        window_length = int(window_size * sampling_rate / 1000)  # Convert window size from ms to number of samples
        num_windows = len(data) // window_length
        windows = np.split(data[:num_windows * window_length], num_windows)
        return windows, window_length


    def filter_windows(self, unfilter_windows, threshold_factor, data):
        min_index_list = []

        median_array = []
        threshold_array = []
        for channel_data in data.T:
            median = np.median(channel_data)
            mad = np.median(np.abs(channel_data - median))
            threshold = mad * threshold_factor
            threshold_array.append(threshold)
            median_array.append(median)

        for window_index, window in enumerate(unfilter_windows):
            min_index = np.unravel_index(np.argmin(window), window.shape)
            if np.abs(window[min_index[0]][min_index[1]] - median_array[min_index[1]]) > threshold_array[min_index[1]]:
                min_index_list.append((min_index[0], window_index))
        return min_index_list

    def reconstruct_windows(self, spike_window_index_list, window_lenght, data):
        align_spike_window = []
        timestamps = {}
        for index, point in enumerate(spike_window_index_list):
            spike_index_in_data = window_lenght * point[1] + point[0]
            start_index = np.maximum(spike_index_in_data - window_lenght/2, 0)
            end_index = spike_index_in_data + window_lenght/2
            new_window = data[int(start_index):int(end_index)]
            align_spike_window.append(new_window)
            timestamps[index] = int(start_index)

        # Get the number of columns
        num_columns = data.shape[1]

        # Plot each column as a separate graph
        for i in range(num_columns):
            plt.plot(align_spike_window[395][:, i], label=f'Column {i + 1}')

        # Add labels and legend
        plt.xlabel('Row Index')
        plt.ylabel('Value')
        plt.legend()

        # Show the plot
        plt.show()

        print(data)
        print(align_spike_window[1])
        align_spike_window.pop(0)
        align_spike_window.pop()
        return align_spike_window, timestamps

    #------------------------------------------------
    #----------------- Plot graphique ---------------------
    #------------------------------------------------

    def show_combined(self, data, denoised_data=None, spikes_indices=None, window_indices=None, window_size=None, sampling_rate=None, windowed_data=None):
        # Plot data and denoised_data
        if denoised_data is not None:
            plt.figure(figsize=(10, 6))

            y_min = np.min(np.min(data))
            y_max = np.max(np.max(data))
            y_range = y_max - y_min

            for i in range(4):
                plt.subplot(4, 2, 2*i+1)
                plt.subplots_adjust(wspace=0.3, hspace=0.3)
                plt.plot(data[:, i])  # Plot one channel's data at a time
                plt.title('Channel {}'.format(i+1))
                plt.ylabel('Signal')
                plt.ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)  # Set y-axis limits
                plt.subplot(4, 2, 2*i+2)
                plt.plot(denoised_data[:, i])  # Plot one channel's denoised data at a time
                plt.title('Denoised Channel {}'.format(i+1))
                plt.ylabel('Signal')
                plt.ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)  # Set y-axis limits

        # Plot data with spike indices
        if spikes_indices is not None:
            plt.figure()

            for i in range(denoised_data.shape[1]):
                plt.subplot(denoised_data.shape[1], 1, i + 1)
                plt.subplots_adjust(wspace=0.3, hspace=0.3)
                plt.plot(denoised_data[:, i]) #label=f'Channel {1 + 1}
                plt.title(f'Channel {i + 1} Signal')
                plt.ylabel('Signal')
                # plt.ylim(-60, 60)
                # plt.xlim(0, 256)
                # Modify the color of spikes in the data
                spike_color = 'r'  # Red color for spikes
                channel_spikes = spikes_indices[i]
                plt.scatter(channel_spikes, denoised_data[channel_spikes, i], c=spike_color, marker='o', label='Spikes')

                plt.legend()

            plt.xlabel('Time')

        # Plot windows in subplots
        if window_indices is not None and window_size is not None and sampling_rate is not None:
            num_windows = len(window_indices)
            print(num_windows)
            num_rows = min(num_windows, 2)
            num_cols = (num_windows - 1) // num_rows + 1

            fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 6 * num_rows))

            for i, window_idx in enumerate(window_indices):
                if num_rows > 1:
                    ax = axes[i // num_cols, i % num_cols]
                else:
                    ax = axes[i % num_cols]

                channel_idx = 0  # You can change this to select a different channel
                ax.plot(windowed_data[i][:, channel_idx]) #, label=f'Channel {channel_idx + 1} Signal'
                ax.set_title(f'Channel {channel_idx + 1}, Window {window_idx + 1}')
                ax.set_ylabel('Signal')
                ax.set_xlabel('Time')
                ax.legend()
                ax.set_ylim(-60, 60)
        plt.tight_layout()
        plt.show()

    def test(): 
        pass


def main():
    window_indices_to_show = [0, 1, 2, 3, 4, 5]
    fs = filterSignal()
    data = fs.open_recording(path='Logiciel/DatasetGenerator/src/data/recordings/recordings_tetrode-mea-s_2min.h5')
    sampling_rate = 32000  # Replace this with the actual sampling rate of your recording data
    # denoised_data = median_filter(data=data, window_size=7)
    denoised_data = fs.apply_bandpass_filter(data=data, sampling_rate=sampling_rate)
    # denoised_data = median_filter(denoised_data)
    # denoised_data = filter(signal=data[:, 1], fs=6400, cutoff=[100, 500], type='band')
    spikes_indices = fs.detect_spikes(data=denoised_data)
    windowed_data, windows_lenght = fs.split_into_windows(denoised_data, window_size=4, sampling_rate=sampling_rate)
    filter_window = fs.filter_windows(windowed_data, 6, denoised_data)
    align_window, timestamps = fs.reconstruct_windows(spike_window_index_list=filter_window, window_lenght=windows_lenght, data=denoised_data)
    #align_window = fs.align_spike(filter_window)
    #print(len(align_window))
    print('FILTER DATA : ')
    print(len(filter_window))
    print('FILTER ALIGN DATA : ')
    print(len(align_window))
    #print(filter_window[0])
    ##print(filter_window)
    print('WINDOWED DATA : ')
    print(len(windowed_data))

    json_file_path = 'Logiciel/DatasetGenerator/src/data/preprocess/recordings_2min.json'
    data_list_serializable = [item.tolist() if isinstance(item, np.ndarray) else item for item in align_window]
    combined_data = {"data": data_list_serializable, "timesteps": timestamps}
    #flatten_data_list_serializable = [[item for sublist in row for item in sublist] for row in data_list_serializable]
    # Save the list to a JSON file
    with open(json_file_path, 'w') as json_file:
        json.dump(combined_data, json_file)
    print(f'Data has been written to {json_file_path}')
   
    #fs.show_combined(data, denoised_data=denoised_data, spikes_indices=spikes_indices, window_indices=window_indices_to_show, window_size=4, sampling_rate=sampling_rate, windowed_data=filter_window)


if __name__ == "__main__":
    main()
