clc, clear, close all

vessel_avg_arr_pred = [];
vessel_avg_arr_true = [];

%minVol = 2000;   % minimum component volume to keep (voxels). Adjust as needed.
minVol = 1000;   % minimum component volume to keep (voxels). Adjust as needed.


for iCase = 1:8
    load(['/home/rcam2/learned_qPACT_results/case1_final/oxy_trial_unhealthy_', int2str(iCase), '_s3.mat']);

    % Squeeze batch dimension-friendly tensors
    predictions_reg = squeeze(predictions_reg);   % [B, Y, X, Z] after squeeze
    labels          = squeeze(labels);
    segmask         = squeeze(segmask);
    segtumor        = squeeze(segtumor);

    % Loop over batch items
    nBatches = size(predictions_reg, 1);
    for j = 1:nBatches
        % Pull volumes for this batch item
        pred_reg = squeeze(predictions_reg(j,:,:,:));
        lab_reg  = squeeze(labels(j,:,:,:));
        seg_all  = squeeze(segmask(j,:,:,:));
        seg_tum  = squeeze(segtumor(j,:,:,:));

        % Ensure logical masks; binarize tumor with threshold (if not already 0/1)
        seg_tum = seg_tum > 0.5;
        seg_all = seg_all > 0.5;

        % Vessel mask = all-vessel-and-tumor mask minus tumor -> vessels only
        seg_vessel = seg_all & ~seg_tum;  % logical 3D

        % If too few vessel voxels, skip safely
        if nnz(seg_vessel) < minVol
            continue
        end

        % Connected components (26-connectivity for 3D)
        CC = bwconncomp(seg_vessel, 26);
        if CC.NumObjects == 0
            continue
        end

        % Label matrix + component volumes
        L = labelmatrix(CC);
        stats = regionprops3(L, 'Volume');

        % Keep only sufficiently large vessels
        keepIDs = find(stats.Volume > minVol);  % indices into component list
        if isempty(keepIDs)
            continue
        end

        maskLarge = ismember(L, keepIDs);

        % Re-index labels to 1..N for kept vessels
%         Lfiltered = L .* uint32(maskLarge);   % keep only large ones
        Lfiltered = L;
        Lfiltered(~maskLarge) = 0;
        
        uniq = unique(Lfiltered);
        uniq(uniq == 0) = [];                 % drop background

        Lreindexed = zeros(size(Lfiltered), 'like', Lfiltered);
        for k = 1:numel(uniq)
            Lreindexed(Lfiltered == uniq(k)) = k;
        end

        % Compute per-vessel averages and append
        vesselIDs = unique(Lreindexed);
        vesselIDs(vesselIDs == 0) = [];
        if isempty(vesselIDs)
            continue
        end

        % Pre-allocate per batch (optional)
        predSO2 = zeros(numel(vesselIDs),1);
        trueSO2 = zeros(numel(vesselIDs),1);

        for v = 1:numel(vesselIDs)
            maskV = (Lreindexed == vesselIDs(v));
            % Using mean(mask) to avoid integer*logical issues
            predSO2(v) = mean(pred_reg(maskV), 'omitnan');
            trueSO2(v) = mean(lab_reg(maskV),  'omitnan');
            % If you'd rather match tumor code exactly:
            % predSO2(v) = sum(double(pred_reg(maskV))) / nnz(maskV);
            % trueSO2(v) = sum(double(lab_reg(maskV)))  / nnz(maskV);
        end

        % Append to global arrays
        vessel_avg_arr_pred = [vessel_avg_arr_pred; predSO2];
        vessel_avg_arr_true = [vessel_avg_arr_true; trueSO2];
    end
end

% ------------------------------------------------------------------------------
% Save arrays as .mat files (edit paths if desired)
% ------------------------------------------------------------------------------
save('/home/rcam2/learned_qPACT_results/case1_final/vessel_avg_err_pred_s3.mat', 'vessel_avg_arr_pred');
save('/home/rcam2/learned_qPACT_results/case1_final/vessel_avg_err_true_s3.mat', 'vessel_avg_arr_true');

%%

load('/home/rcam2/learned_qPACT_results/case4_final/vessel_avg_err_pred_s5.mat', 'vessel_avg_arr_pred');
load('/home/rcam2/learned_qPACT_results/case4_final/vessel_avg_err_true_s5.mat', 'vessel_avg_arr_true');

% 
% ------------------------------------------------------------------------------
% (Optional) Scatter plot of ALL vessels across ALL cases
% ------------------------------------------------------------------------------
figure;
scatter(vessel_avg_arr_true, vessel_avg_arr_pred, 80, 'o', 'MarkerFaceColor', [0.2 0.6 1], 'MarkerEdgeColor', 'k');
xlabel('Ground Truth sO_2');
ylabel('Estimated sO_2');
title('Per-Vessel Average sO_2 (All Cases)');
grid on;
axis equal;
xlim([0.7 1]); ylim([0.7 1]);   % adjust if your sO2 range differs
hold on;
plot([0.7 1], [0.7 1], 'r--', 'LineWidth', 1); % y = x