%% Initialize
clc;
clear;
close all;

%% Load Data
% Specify the file name
filename = 'C:\Users\alexl\Desktop\dataProcessing\RAW_EMG_data.h5';  % Update with the correct path

% Get information about the content of the HDF5 file
group_path = '/RAW_EMG';  % Assuming your data is in this group
group_info = h5info(filename, group_path);

%% Extract and Segment Walking Data
% Initialize an empty structure to store segmented data
segmented_data = struct();

% Define segmentation parameters
fs = 1000;  % Sampling frequency, adjust if different
window_size = 10 * fs;  % 10 seconds of data = 10,000 samples
step_size = 5 * fs;     % 5-second overlap = 5,000 samples

% Loop through all walking datasets
for i = 1:length(group_info.Datasets)
    dataset_name = group_info.Datasets(i).Name;
    
    % Check if the dataset name contains '_TW' (indicating treadmill walking)
    if contains(dataset_name, '_TW')
        full_dataset_path = [group_path '/' dataset_name];
        
        % Read the dataset
        data = h5read(filename, full_dataset_path);
        
        % Get the number of samples in the dataset (assuming all muscle signals have the same length)
        num_samples = length(data.time);
        
        % Initialize segment counter
        segment_counter = 1;
        
        % Perform sliding window segmentation
        for start_idx = 1:step_size:(num_samples - window_size + 1)
            end_idx = start_idx + window_size - 1;  % Define the end index of the window
            
            % Extract the segment for each muscle signal
            segment = struct();
            segment.time = data.time(start_idx:end_idx);  % Extract corresponding time segment
            
            % Loop through each muscle signal and extract its segment
            muscle_signals = fieldnames(data);
            muscle_signals = muscle_signals(~strcmp(muscle_signals, 'time'));  % Exclude 'time'
            
            for j = 1:length(muscle_signals)
                muscle = muscle_signals{j};
                segment.(muscle) = data.(muscle)(start_idx:end_idx);  % Extract muscle signal segment
            end
            
            % Store the segmented data with an identifier
            segment_name = [dataset_name '_segment_' num2str(segment_counter)];
            segmented_data.(segment_name) = segment;
            
            % Increment the segment counter
            segment_counter = segment_counter + 1;
        end
        
        % Optionally, display a message indicating the dataset has been segmented
        disp(['Segmented dataset: ' dataset_name]);
    end
end

%% Define Filter Parameters
fs = 1000;  % Sampling frequency (adjust if necessary)

% Design the band-pass Butterworth filter (20 Hz to 499 Hz)
[b_band, a_band] = butter(4, [20 499] / (fs / 2), 'bandpass');

% Design the notch filter at 50 Hz
wo = 50 / (fs / 2);  % Normalized frequency for 50 Hz
bw = wo / 35;  % Bandwidth of the notch filter (adjustable)
[b_notch, a_notch] = designNotchPeakIIR(CenterFrequency=wo, Bandwidth=bw, Response="notch");

%% Apply Filters to Each Segmented Dataset
filtered_data = struct();  % Structure to store filtered data

fields = fieldnames(segmented_data);
for i = 1:numel(fields)
    segment_name = fields{i};
    segment = segmented_data.(segment_name);
    
    % Initialize a structure to hold filtered signals
    filtered_segment = struct();
    
    % Loop through each muscle signal in the segment (excluding 'time')
    muscle_signals = fieldnames(segment);
    muscle_signals = muscle_signals(~strcmp(muscle_signals, 'time'));  % Exclude 'time'
    
    for j = 1:length(muscle_signals)
        muscle = muscle_signals{j};
        
        % Apply the band-pass Butterworth filter
        filtered_signal = filtfilt(b_band, a_band, segment.(muscle));
        
        % Apply the 50 Hz notch filter
        filtered_signal = filtfilt(b_notch, a_notch, filtered_signal);
        
        % Store the filtered signal
        filtered_segment.(muscle) = filtered_signal;
    end
    
    % Keep the 'time' field as it is
    filtered_segment.time = segment.time;
    
    % Store the filtered segment in the filtered_data structure
    filtered_data.(segment_name) = filtered_segment;
    
    % Optionally display a message indicating that the segment has been filtered
    disp(['Filtered segment: ' segment_name]);
end

%% Define CWT Parameters
fs = 1000;  % Sampling frequency (adjust if necessary)
output_filename = 'CWT_EMG_data.h5';  % HDF5 output file

% If the file exists, delete it to avoid appending to old data
if exist(output_filename, 'file') == 2
    delete(output_filename);
end

% Define the scales for CWT (ensure consistent frequency axis)
num_scales = 100;  % Number of scales
min_freq = 20;     % Minimum frequency of interest
max_freq = 500;    % Maximum frequency of interest
scales = (fs ./ linspace(min_freq, max_freq, num_scales));  % Generate scales

%% Apply CWT to Each Filtered Segment and Save to HDF5
fields = fieldnames(filtered_data);
for i = 1:numel(fields)
    segment_name = fields{i};
    segment = filtered_data.(segment_name);
    
    % Loop through each muscle signal in the segment (excluding 'time')
    muscle_signals = fieldnames(segment);
    muscle_signals = muscle_signals(~strcmp(muscle_signals, 'time'));  % Exclude 'time'
    
    for j = 1:length(muscle_signals)
        muscle = muscle_signals{j};
        
        % Apply CWT to the muscle signal
        [coeffs, frequencies] = cwt(segment.(muscle), scales, 'amor', fs);  % 'amor' wavelet used here
        
        % Separate real and imaginary parts
        real_coeffs = real(coeffs);
        imag_coeffs = imag(coeffs);
        
        % Define the dataset paths for real and imaginary parts
        real_cwt_dataset_path = ['/' segment_name '/' muscle '/coeffs_real'];
        imag_cwt_dataset_path = ['/' segment_name '/' muscle '/coeffs_imag'];
        freq_dataset_path = ['/' segment_name '/' muscle '/frequencies'];
        
        % Write the real part of the CWT coefficients to HDF5
        h5create(output_filename, real_cwt_dataset_path, size(real_coeffs));
        h5write(output_filename, real_cwt_dataset_path, real_coeffs);
        
        % Write the imaginary part of the CWT coefficients to HDF5
        h5create(output_filename, imag_cwt_dataset_path, size(imag_coeffs));
        h5write(output_filename, imag_cwt_dataset_path, imag_coeffs);
        
        % Write the frequencies (scales) used for the CWT
        h5create(output_filename, freq_dataset_path, size(frequencies));
        h5write(output_filename, freq_dataset_path, frequencies);
    end
    
    % Optionally display a message indicating the segment has been processed
    disp(['Processed CWT for segment: ' segment_name]);
end

disp(['CWT data saved to: ' output_filename]);