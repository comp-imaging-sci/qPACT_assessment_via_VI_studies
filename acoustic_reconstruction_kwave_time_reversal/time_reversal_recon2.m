% Time-reversal reconstruction script for Study 2 (qPACT).
%
% This script generates 3D time-reversal reconstructions that are used as
% network inputs in Study 2 of the accompanying paper:
%
%   "Application of a Virtual Imaging Framework for Investigating a Deep Learning–Based
%    Reconstruction Method for 3D Quantitative Photoacoustic Computed Tomography (qPACT)"
%
% Description:
%   - Loads simulated photoacoustic sensor data at multiple wavelengths.
%   - Applies optional noise and frequency-domain filtering.
%   - Performs 3D time-reversal reconstruction using the k-Wave toolbox.
%   - Saves reconstructed initial pressure estimates for downstream learning-based analysis.
%
%

clc, clear, close all

addpath(genpath('/home/rcam2/k-wave/'));
addpath(genpath('/home/rcam2/k-wave/binaries/kspaceFirstOrder-CUDA/'));
save_folder = '/shared/anastasio-s2/Phantom/Breast_phantom_UBP/dataset_h_l_time_reversal/';

% Initialize an empty cell array to store image names
image_names = {};

% Define the root directory and subdirectory
root_dir = '/shared/anastasio-s2/Phantom/Breast_phantom_UBP/';
sub_dir = 'dataset_h_l_corrected';

% Combine the root and subdirectory to get the full path
directory = fullfile(root_dir, sub_dir);

% Get the list of files in the directory
file_list = dir(directory);

% Create a set to store unique names
names_set = containers.Map;

% Loop through each file in the directory
for i = 1:length(file_list)
    filename = file_list(i).name;
    % Check if the file has a .mat extension
    if endsWith(filename, '.mat')
        % Extract the part before the first underscore
        name = strsplit(filename, '_');
        name = name{1}; % Get the first part
        % Add to the set (unique names)
        names_set(name) = true; % Using a map to ensure uniqueness
    end
end

% Get the unique names from the set and sort them
image_names = sort(keys(names_set));

% % Display the sorted unique image names
% disp('Sorted unique image names:');
% disp(image_names);

% Define the common paths for natural_shape_A, natural_shape_B, etc.
base_paths = {
    '/shared/anastasio-s2/Phantom/Breast/natural_shape_A/';
    '/shared/anastasio-s2/Phantom/Breast/natural_shape_B/';
    '/shared/anastasio-s2/Phantom/Breast/natural_shape_C/';
    '/shared/anastasio-s2/Phantom/Breast/natural_shape_D/';
};

% Create cell arrays to hold the maximum values
p_w757_max = [];
p_w800_max = [];
p_w850_max = [];


pml_size = [20,20,10];

voxel_size = 0.3e-3;   % [m]
Nx = 600;            % [voxel], 20 voxels padded
Ny = 600;            % [voxel], 20 voxels padded
Nz = 300;
dx = voxel_size;
dy = voxel_size;
dz = voxel_size;
Nt = 3720;
dt = 1e-6/20;


% Define sensor mask, transducer element locations
disp('Setting sensor mask...');
load('/shared/anastasio4/PACT/qPACT/cs2-net/mask_qpact_030.mat', 'mask_qpact_030');
sensor.mask = mask_qpact_030;
clear mask_qpact_030

sound_speed = 1520.6;
rho = 993.36; % [kg/(m^3)]

a = 0.;
alpha_power = 0.;

% Create the computational grid
kgrid = kWaveGrid(Nx, dx, Ny, dy, Nz, dz);
kgrid.setTime(Nt, dt);

% Assign medium
medium.sound_speed  = sound_speed;
medium.density      = rho;
medium.alpha_coeff  = a;
medium.alpha_power  = alpha_power;

input_args = {'PMLSize', pml_size, 'PMLInside', false, ...
    'PlotPML', false, 'Smooth', false, 'DataCast', 'single', 'BinaryPath', '/home/rcam2/k-wave/binaries/kspaceFirstOrder-CUDA/', 'DataPath', '/shared/anastasio-s2/Phantom/Breast_phantom_UBP/tmpdir2'};

load('/shared/anastasio4/PACT/qPACT/cs2-net/closest_sensor_idx_30_25.mat', 'closest_sensor_idx');

% Iterate through the first 100 image names
%for idx = 1:min(100, length(image_names))
for idx = 135:min(200, length(image_names))
    idx
    name = image_names{idx}; % Get the image name
    found = false; % Flag to check if file was found
    
    % Iterate over the base paths to find the correct directory
    for base_path_idx = 1:length(base_paths)
        base_path = base_paths{base_path_idx};
        file_path = fullfile(base_path, name, [name, 'l_p_s1.mat']); % Construct the file path
        
        if isfile(file_path) % Check if the file exists
            fprintf('%d\n', base_path_idx); % Print the shape number
            found = true; % File found, set flag to true
            
            p = load(file_path); % Load the .mat file
            p_w757 = p.p_w757;
            p_w800 = p.p_w800;
            p_w850 = p.p_w850;
            
%             % Store the maximum values in the respective arrays
%             p_w757_max(end + 1) = max(p_w757(:));
%             p_w800_max(end + 1) = max(p_w800(:));
%             p_w850_max(end + 1) = max(p_w850(:));
            break; % Exit the loop as we've found the file
        end
    end
    
    if ~found
        warning('File not found in any known location for name: %s', name);
    end
    
    p_w757_30 = p_w757(int32(squeeze(closest_sensor_idx)),:);
    p_w800_30 = p_w800(int32(squeeze(closest_sensor_idx)),:);    
    p_w850_30 = p_w850(int32(squeeze(closest_sensor_idx)),:);    
    
    clear p_w757 p_w800 p_w850
    
    rng(3*idx+1)
    p_w757_30 = p_w757_30 + normrnd(0., 0.01*0.0015350636, 51080, 3720);
    rng(3*idx*2)
    p_w800_30 = p_w800_30 + normrnd(0., 0.01*0.0015350636, 51080, 3720); 
    rng(3*idx+3)
    p_w850_30 = p_w850_30 + normrnd(0., 0.01*0.0015350636, 51080, 3720);
    
    % Apply FFT along the second dimension
    data_fft_w757 = fft(p_w757_30, [], 2); % FFT along the 2nd dimension
    data_fft_w800 = fft(p_w800_30, [], 2); % FFT along the 2nd dimension    
    data_fft_w850 = fft(p_w850_30, [], 2); % FFT along the 2nd dimension
    
    clear p_w757_30 p_w800_30 p_w850_30

    % Create a mask to zero out frequencies beyond the 700th bin
    % Assuming 3720 frequency bins, we're keeping only the first 700
    frequency_limit = 500; % Set frequency limit
    [n_rows, n_cols] = size(data_fft_w757);

    % Create the mask to retain frequencies below the limit
    frequency_mask = zeros(n_rows, n_cols); % Initialize mask
    frequency_mask(:, 1:frequency_limit) = 1; % Set the frequency range to keep
    frequency_mask(:, n_cols-frequency_limit+1:n_cols) = 1;
    % Apply the mask to the FFT data to zero out unwanted frequencies
    
    filtered_fft_w757 = data_fft_w757 .* frequency_mask;
    filtered_fft_w800 = data_fft_w800 .* frequency_mask;
    filtered_fft_w850 = data_fft_w850 .* frequency_mask;
     
    % Apply the inverse FFT to get back to the time domain
    filtered_data_w757 = ifft(filtered_fft_w757, [], 2, 'symmetric'); % Symmetric ensures real output    
    filtered_data_w800 = ifft(filtered_fft_w800, [], 2, 'symmetric'); % Symmetric ensures real output     
    filtered_data_w850 = ifft(filtered_fft_w850, [], 2, 'symmetric'); % Symmetric ensures real output
    
    clear filtered_fft_w757 filtered_fft_w800 filtered_fft_w850
    
    source.p0 = 0;
    sensor.time_reversal_boundary_data = filtered_data_w757;
    pest_time_reversal_w757 = kspaceFirstOrder3DG(kgrid, medium, source, sensor, input_args{:});
    
    source.p0 = 0;
    sensor.time_reversal_boundary_data = filtered_data_w800;
    pest_time_reversal_w800 = kspaceFirstOrder3DG(kgrid, medium, source, sensor, input_args{:});
    
    source.p0 = 0;
    sensor.time_reversal_boundary_data = filtered_data_w850;
    pest_time_reversal_w850 = kspaceFirstOrder3DG(kgrid, medium, source, sensor, input_args{:});
    
    
    output_filename = [save_folder, name, '_tr_l.mat'];
    % Save the three variables to a .mat file
    save(output_filename, 'pest_time_reversal_w757', 'pest_time_reversal_w800', 'pest_time_reversal_w850');
   
end









