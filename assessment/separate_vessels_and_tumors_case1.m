%%
clc; clear; close all;

addpath(genpath('/home/rcam2/k-wave/'));
addpath(genpath('/home/rcam2/k-wave/binaries/kspaceFirstOrder-CUDA/'));

% Voxel and volume size
VoxelSize = 0.3;           % Voxel size [mm]
Nx = 256;                   % Volume size along x-axis [voxel]
Ny = 512;                   % Volume size along y-axis [voxel]
Nz = 512;                   % Volume size along z-axis [voxel]
VolumeSize = [Nx, Ny, Nz];  % Volume size [voxel]

vesselscaleAll = [VoxelSize:VoxelSize:VoxelSize*5].*1e-3;

% Process 8 cases
for i = 1:8
% for i = 3
% for i = 4
% for i = 7
    % Load the case file
    %load(['/shared/anastasio-s2/PACT/qPACT/learned_qpact/case1/oxy_trial_unhealthy_', int2str(i), '.mat']);
    load(['/home/rcam2/learned_qPACT_results/case1_final/oxy_trial_unhealthy_', int2str(i), '.mat']);
    
    
    % Squeeze predictions_seg and threshold to create a binary segmentation
    predictions_seg = squeeze(predictions_seg);
    predictions_seg(predictions_seg <= 0.7) = 0;
    predictions_seg(predictions_seg > 0.7) = 1;
    
    % Determine number of images (or volumes) in predictions_seg
    numImages = size(predictions_seg, 1);
%     
%     % Create a cell array to hold the final tumor mask for each image
%     finalTumorMasks = cell(numImages, 1);

    % Preallocate a 4D array [numImages x Nx x Ny x Nz]
    finalTumorMasks = false(numImages, Nx, Ny, Nz);

    % Loop over each image/volume in the current case
    for j = 1:numImages
%     for j = 2
%     for j = 5
%     for j = 4
        vesselData = zeros([Nx, Ny, Nz]);
        % Extract the j-th volume
        pred_seg = squeeze(predictions_seg(j,:,:,:));
        
        % Process vessel filtering for different scales
        for k = 1:2  % or use 1:length(vesselscaleAll) if desired
            scale = vesselscaleAll(k);
            vesselDataAll{k} = vesselFilter(pred_seg, [0.3e-3, 0.3e-3, 0.3e-3], [scale, scale, scale]);
            vesselData = vesselData + abs(vesselDataAll{k});
        end
        
        % Normalize the vesselness response to [0,1]
        vesselData = vesselData ./ max(vesselData(:));
        
        % Create a preliminary binary vessel mask from vesselness response
        vesselThreshold = 0.1;  % adjust based on your data
        prelimVesselMask = vesselData > vesselThreshold;
        
        % Perform connected component analysis on the original segmentation
        cc = bwconncomp(pred_seg, 26);
        vesselMaskCC = false(size(pred_seg));
        componentThreshold = 0.6;  % if >40% of a component is vessel, label it as vessel
        
        % Iterate over each connected component of the segmentation
        for comp = 1:cc.NumObjects
            compIndices = cc.PixelIdxList{comp};
            vesselFraction = mean(prelimVesselMask(compIndices));
            if vesselFraction > componentThreshold
                vesselMaskCC(compIndices) = true;
            end
        end
        
        % Define the tumor mask as the segmentation minus the vessel mask
        tumorMask = pred_seg & ~vesselMaskCC;
        
        % Now perform connected component analysis on the tumor mask
        cc_tumor = bwconncomp(tumorMask, 26);
        finalTumorMask = false(size(tumorMask));
        minComponentSize = 1000;  % only keep components larger than 20 voxels
        
        % Iterate over each connected component in the tumor mask
        for comp = 1:cc_tumor.NumObjects
            compIndices = cc_tumor.PixelIdxList{comp};
            if numel(compIndices) > minComponentSize
                finalTumorMask(compIndices) = true;
            end
        end

%         % Create a structuring element for the morphological operations.
%         % For a 3D image, a spherical element is often appropriate.
%         se = strel('sphere', 3);  % Adjust the radius as needed
% 
%         tumorMask_final = imerode(finalTumorMask, se);
% %         tumorMask_final = imerode(tumorMask_final, se);
% %         tumorMask_final = imopen(tumorMask_final, se);
%         tumorMask_final = imdilate(tumorMask_final, se);
%         tumorMask_final = imdilate(tumorMask_final, se);
%         tumorMask_final = imdilate(tumorMask_final, se);
%         tumorMask_final = finalTumorMask.*tumorMask_final;
% 
%         % Convert the mask to a logical (binary) array
%         tumorMask_final = logical(tumorMask_final);

%         % Create a structuring element for the morphological operations.
%         % For a 3D image, a spherical element is often appropriate.
%         se = strel('sphere', 1);  % Adjust the radius as needed
% 
%         tumorMask_final = imerode(finalTumorMask, se);
%         tumorMask_final = imerode(finalTumorMask, se);
%         
% %         tumorMask_final = imerode(tumorMask_final, se);
% %         tumorMask_final = imopen(tumorMask_final, se);
%         tumorMask_final = imdilate(tumorMask_final, se);
%         tumorMask_final = imdilate(tumorMask_final, se);
%         tumorMask_final = imdilate(tumorMask_final, se);
% %         tumorMask_final = imdilate(tumorMask_final, se);
% %         tumorMask_final = imdilate(tumorMask_final, se);
%         tumorMask_final = finalTumorMask.*tumorMask_final;
% 
%         % Convert the mask to a logical (binary) array
%         tumorMask_final = logical(tumorMask_final);

        % Create a structuring element for the morphological operations.
        % For a 3D image, a spherical element is often appropriate.
        se = strel('sphere', 2);  % Adjust the radius as needed

        tumorMask_final = imerode(finalTumorMask, se);
%         tumorMask_final = imerode(finalTumorMask, se);
        
%         tumorMask_final = imerode(tumorMask_final, se);
%         tumorMask_final = imopen(tumorMask_final, se);
        tumorMask_final = imdilate(tumorMask_final, se);
        tumorMask_final = imdilate(tumorMask_final, se);
%         tumorMask_final = imdilate(tumorMask_final, se);
%         tumorMask_final = imdilate(tumorMask_final, se);
%         tumorMask_final = imdilate(tumorMask_final, se);
        tumorMask_final = finalTumorMask.*tumorMask_final;

        % Convert the mask to a logical (binary) array
        tumorMask_final = logical(tumorMask_final);
        
        cc_final = bwconncomp(tumorMask_final, 26);
        tumorMask_final_clean = false(size(tumorMask_final));
        minVoxelThreshold = 1000;
        for comp = 1:cc_final.NumObjects
            compIndices = cc_final.PixelIdxList{comp};
            if numel(compIndices) >= minVoxelThreshold
                tumorMask_final_clean(compIndices) = true;
            end
        end
        
        tumorMask_final = tumorMask_final_clean;
        
        % Store the final tumor mask for the current image
        finalTumorMasks(j, :, :, :) = tumorMask_final;
        
    end
    
    % Save the cell array of tumor masks as a MAT file
%     saveFilename = ['/shared/anastasio-s2/PACT/qPACT/learned_qpact/case1/finalTumorMasks_', num2str(i), '.mat'];
%     saveFilename = ['/home/rcam2/learned_qPACT_results/case1_final/finalTumorMasks_', num2str(i), '.mat'];
    saveFilename = ['/shared/aristotle/PACT/qpact_tumor_vessel_separated/case1/finalTumorMasks_', num2str(i), '.mat'];
    save(saveFilename, 'finalTumorMasks');
end