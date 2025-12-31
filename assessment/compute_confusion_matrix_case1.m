clc; clear; close all;

true_positive = 0;
false_positive = 0;
false_negative = 0;
diceArray = [];  % Array to store Dice coefficients for each true positive match

for i = 1:8
    i
    load(['/home/rcam2/learned_qPACT_results/case1_final/oxy_trial_unhealthy_', int2str(i), '.mat']);
    load(['/shared/anastasio-s3/PACT/aristotle/qpact_tumor_vessel_separated/case1/finalTumorMasks_', int2str(i), '.mat']);
    
    predictions_seg = squeeze(predictions_seg);
    segtumor = squeeze(segtumor);
   
    for j = 1:size(predictions_seg, 1)
        j
        tumor_seg = squeeze(segtumor(j,:,:,:));
              
        % Create ground truth tumor mask: tumor label 2 becomes 1, label 1 becomes 0
        mask_tum = tumor_seg;
        mask_tum(tumor_seg == 1) = 0;
        mask_tum(tumor_seg == 2) = 1;
        
        % Get the segmented tumor mask (original) and convert to double for filtering
        tum_mask_seg = squeeze(finalTumorMasks(j,:,:,:));
        tum_mask_seg = double(tum_mask_seg);
        
        % Filter the masks with a 3D Gaussian filter to ensure connectivity.
        mask_tum_filt = imgaussfilt3(double(mask_tum), 2);
        mask_tum_filt(mask_tum_filt <= 0.1) = 0;
        mask_tum_filt(mask_tum_filt > 0.1) = 1;
        
        tum_mask_seg_filt = imgaussfilt3(double(tum_mask_seg), 2);
        tum_mask_seg_filt(tum_mask_seg_filt <= 0.1) = 0;
        tum_mask_seg_filt(tum_mask_seg_filt > 0.1) = 1;
        
        % --- Connected Component Analysis ---
        % Each connected component in mask_tum_filt is a true tumor.
        cc_true = bwconncomp(mask_tum_filt);
        % Each connected component in tum_mask_seg_filt is a segmented tumor.
        cc_seg = bwconncomp(tum_mask_seg_filt);
        
        % Initialize flags to mark detected components
        seg_detected = zeros(cc_seg.NumObjects,1); % segmented tumor detection flag
        true_detected = zeros(cc_true.NumObjects,1); % true tumor detection flag
        
        % Compare every segmented tumor component with every true tumor component.
        for s = 1:cc_seg.NumObjects
            for t = 1:cc_true.NumObjects
                % Count overlapping voxels between segmented and true tumor components.
                overlap = numel(intersect(cc_seg.PixelIdxList{s}, cc_true.PixelIdxList{t}));
                if overlap > 500
                    seg_detected(s) = 1;
                    true_detected(t) = 1;
                end
            end
        end
        
        % For each true positive segmented tumor, compute the Dice coefficient using original masks.
        for s = 1:cc_seg.NumObjects
            if seg_detected(s) == 1
                % For segmented component s, find the best matching true tumor component by maximum overlap.
                max_overlap = 0;
                best_t = 0;
                for t = 1:cc_true.NumObjects
                    overlap = numel(intersect(cc_seg.PixelIdxList{s}, cc_true.PixelIdxList{t}));
                    if overlap > 500 && overlap > max_overlap
                        max_overlap = overlap;
                        best_t = t;
                    end
                end
                
                if best_t > 0
                    % Compute the centroid of the best-matching true tumor (using the filtered mask).
                    [r, c, z] = ind2sub(size(mask_tum_filt), cc_true.PixelIdxList{best_t});
                    center_true = round([mean(r), mean(c), mean(z)]);
                    
                    % Define a 50x50x50 region centered on the true tumor centroid.
                    win_size = 60;
                    half_win = floor(win_size/2);
                    [dim1, dim2, dim3] = size(mask_tum); % use dimensions of the original mask
                    
                    row_start = max(center_true(1) - half_win + 1, 1);
                    row_end   = min(row_start + win_size - 1, dim1);
                    col_start = max(center_true(2) - half_win + 1, 1);
                    col_end   = min(col_start + win_size - 1, dim2);
                    slice_start = max(center_true(3) - half_win + 1, 1);
                    slice_end   = min(slice_start + win_size - 1, dim3);
                    
                    % Extract the region from the original (unfiltered) true and segmented masks.
                    region_true = mask_tum(row_start:row_end, col_start:col_end, slice_start:slice_end);
                    region_seg  = tum_mask_seg(row_start:row_end, col_start:col_end, slice_start:slice_end);
                    
                    % Compute the Dice coefficient:
                    intersection = sum((region_true & region_seg), 'all');
                    vol_true = sum(region_true, 'all');
                    vol_seg  = sum(region_seg, 'all');
                    if (vol_true + vol_seg) > 0
                        dice = 2 * intersection / (vol_true + vol_seg);
                    else
                        dice = 0;
                    end
                    
                    % Append the computed Dice coefficient to the array.
                    diceArray = [diceArray; dice];
                end
            end
        end
        
        % Update the counts:
        true_positive = true_positive + sum(seg_detected);
        false_positive = false_positive + (cc_seg.NumObjects - sum(seg_detected));
        false_negative = false_negative + (cc_true.NumObjects - sum(true_detected));
    end
end

% Display the final counts
fprintf('True Positive: %d\n', true_positive);
fprintf('False Positive: %d\n', false_positive);
fprintf('False Negative: %d\n', false_negative);

% Display the array of Dice coefficients
disp('Dice Coefficients for True Positives:');
disp(diceArray);

save('results_tumor_metrics_case1.mat', 'true_positive', 'false_positive', 'false_negative', 'diceArray');





% clc; clear; close all;
% 
% true_positive = 0;
% false_positive = 0;
% false_negative = 0;
% diceArray = [];  % Array to store Dice coefficients for each true positive match
% 
% for i = 1:8
%     load(['/home/rcam2/learned_qPACT_results/case1_final/oxy_trial_unhealthy_', int2str(i), '.mat']);
%     load(['/shared/aristotle/PACT/qpact_tumor_vessel_separated/case1/finalTumorMasks_', int2str(i), '.mat']);
%     
%     predictions_seg = squeeze(predictions_seg);
%     segtumor = squeeze(segtumor);
%    
%     for j = 1:size(predictions_seg, 1)
%         tumor_seg = squeeze(segtumor(j,:,:,:));
%               
%         % Create ground truth tumor mask: tumor label 2 becomes 1, label 1 becomes 0
%         mask_tum = tumor_seg;
%         mask_tum(tumor_seg == 1) = 0;
%         mask_tum(tumor_seg == 2) = 1;
%         
%         % Get the segmented tumor mask and convert to double for filtering
%         tum_mask_seg = squeeze(finalTumorMasks(j,:,:,:));
%         tum_mask_seg = double(tum_mask_seg);
%         
%         % Filter the ground truth and segmented masks with a 3D Gaussian filter
%         mask_tum_filt = imgaussfilt3(double(mask_tum), 2);
%         mask_tum_filt(mask_tum_filt <= 0.1) = 0;
%         mask_tum_filt(mask_tum_filt > 0.1) = 1;
%         
%         tum_mask_seg_filt = imgaussfilt3(double(tum_mask_seg), 2);
%         tum_mask_seg_filt(tum_mask_seg_filt <= 0.1) = 0;
%         tum_mask_seg_filt(tum_mask_seg_filt > 0.1) = 1;
%         
%         % --- Connected Component Analysis ---
%         % Each connected component in mask_tum_filt is a true tumor.
%         cc_true = bwconncomp(mask_tum_filt);
%         % Each connected component in tum_mask_seg_filt is a segmented tumor.
%         cc_seg = bwconncomp(tum_mask_seg_filt);
%         
%         % Initialize flags to mark detected components
%         seg_detected = zeros(cc_seg.NumObjects,1); % segmented tumor detection flag
%         true_detected = zeros(cc_true.NumObjects,1); % true tumor detection flag
%         
%         % Compare every segmented tumor component with every true tumor component.
%         for s = 1:cc_seg.NumObjects
%             for t = 1:cc_true.NumObjects
%                 % Count the overlapping voxels between the segmented and true tumor components.
%                 intersection_voxels = numel(intersect(cc_seg.PixelIdxList{s}, cc_true.PixelIdxList{t}));
%                 if intersection_voxels > 200
%                     seg_detected(s) = 1;
%                     true_detected(t) = 1;
%                 end
%             end
%         end
%         
%         
%         
%         
%         % Update the counts:
%         % - A segmented tumor with sufficient overlap counts as a true positive.
%         % - A segmented tumor that does not overlap enough with any true tumor counts as a false positive.
%         % - A true tumor with no matching segmented tumor counts as a false negative.
%         true_positive = true_positive + sum(seg_detected);
%         false_positive = false_positive + (cc_seg.NumObjects - sum(seg_detected));
%         false_negative = false_negative + (cc_true.NumObjects - sum(true_detected));
%     end
% end
% 
% % Display the final counts
% fprintf('True Positive: %d\n', true_positive);
% fprintf('False Positive: %d\n', false_positive);
% fprintf('False Negative: %d\n', false_negative);
% 
