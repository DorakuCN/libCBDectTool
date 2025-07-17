% MATLAB Debug Version for Detailed Module Comparison
% This version adds comprehensive logging for each processing step

clear all; close all; clc;

% Add paths
addpath('matching');

% Load image
img_path = '../../data/04.png';
fprintf('\n=== MATLAB DEBUG VERSION - DETAILED MODULE ANALYSIS ===\n');
fprintf('Image: %s\n', img_path);

img = imread(img_path);
if size(img, 3) == 3
    img = rgb2gray(img);
end
img = im2double(img);

fprintf('Image size: %d x %d\n', size(img, 2), size(img, 1));
fprintf('Image type: %s, range: [%.3f, %.3f]\n', class(img), min(img(:)), max(img(:)));

%% Step 1: Corner Detection
fprintf('\n=== STEP 1: CORNER DETECTION ===\n');
tic;
corners = findCorners(img);
corner_detection_time = toc;

fprintf('Corner detection completed in %.3f seconds\n', corner_detection_time);
fprintf('Raw corners detected: %d\n', length(corners.p));

% Log corner coordinates and properties
fprintf('\nDetailed corner information:\n');
fprintf('Corners in different regions:\n');

% Define regions for analysis
matlab_region = [42, 350, 423, 562]; % [x_min, y_min, x_max, y_max]
upper_region_count = 0;
middle_region_count = 0;
matlab_region_count = 0;
lower_region_count = 0;

corner_coords = [];
corner_scores = [];

for i = 1:length(corners.p)
    x = corners.p(i, 1);
    y = corners.p(i, 2);
    
    corner_coords = [corner_coords; x, y];
    
    % Get score if available
    if isfield(corners, 'score') && length(corners.score) >= i
        score = corners.score(i);
        corner_scores = [corner_scores; score];
    else
        score = 0.0;
        corner_scores = [corner_scores; score];
    end
    
    % Classify by region
    region_label = '';
    if y < 200
        upper_region_count = upper_region_count + 1;
        region_label = 'UPPER';
    elseif y >= 200 && y < 350
        middle_region_count = middle_region_count + 1;
        region_label = 'MIDDLE';
    elseif x >= matlab_region(1) && x <= matlab_region(3) && y >= matlab_region(2) && y <= matlab_region(4)
        matlab_region_count = matlab_region_count + 1;
        region_label = 'MATLAB';
    else
        lower_region_count = lower_region_count + 1;
        region_label = 'LOWER';
    end
    
    if i <= 20 || strcmp(region_label, 'MATLAB')
        fprintf('  [%2d] (%.1f, %.1f) score=%.3f [%s]\n', i, x, y, score, region_label);
    end
end

fprintf('\nRegion distribution:\n');
fprintf('  Upper region (Y<200):     %d corners (%.1f%%)\n', upper_region_count, 100*upper_region_count/length(corners.p));
fprintf('  Middle region (Y200-350): %d corners (%.1f%%)\n', middle_region_count, 100*middle_region_count/length(corners.p));
fprintf('  MATLAB region:            %d corners (%.1f%%)\n', matlab_region_count, 100*matlab_region_count/length(corners.p));
fprintf('  Lower region (Y>350):     %d corners (%.1f%%)\n', lower_region_count, 100*lower_region_count/length(corners.p));

% Log coordinate bounds
if ~isempty(corner_coords)
    fprintf('\nCoordinate bounds:\n');
    fprintf('  X range: [%.1f, %.1f]\n', min(corner_coords(:,1)), max(corner_coords(:,1)));
    fprintf('  Y range: [%.1f, %.1f]\n', min(corner_coords(:,2)), max(corner_coords(:,2)));
end

% Log score statistics
if ~isempty(corner_scores) && any(corner_scores > 0)
    fprintf('\nScore statistics:\n');
    fprintf('  Score range: [%.3f, %.3f]\n', min(corner_scores), max(corner_scores));
    fprintf('  Average score: %.3f\n', mean(corner_scores));
    fprintf('  High quality corners (>1.0): %d\n', sum(corner_scores > 1.0));
end

%% Step 2: Corner Filtering and Refinement (if applicable)
fprintf('\n=== STEP 2: CORNER PROCESSING ===\n');

% Log corner structure fields
fprintf('Corner structure fields:\n');
field_names = fieldnames(corners);
for i = 1:length(field_names)
    field = field_names{i};
    if strcmp(field, 'p')
        fprintf('  %s: %d x %d (corner coordinates)\n', field, size(corners.(field), 1), size(corners.(field), 2));
    elseif strcmp(field, 'v1') || strcmp(field, 'v2')
        if ~isempty(corners.(field))
            fprintf('  %s: %d x %d (direction vectors)\n', field, size(corners.(field), 1), size(corners.(field), 2));
            % Show first few direction vectors
            for j = 1:min(5, size(corners.(field), 1))
                fprintf('    [%d]: (%.3f, %.3f, %.3f)\n', j, corners.(field)(j,1), corners.(field)(j,2), corners.(field)(j,3));
            end
        else
            fprintf('  %s: empty\n', field);
        end
    elseif strcmp(field, 'score')
        if ~isempty(corners.(field))
            fprintf('  %s: %d elements\n', field, length(corners.(field)));
        else
            fprintf('  %s: empty\n', field);
        end
    else
        fprintf('  %s: %s\n', field, mat2str(size(corners.(field))));
    end
end

%% Step 3: Chessboard Detection
fprintf('\n=== STEP 3: CHESSBOARD DETECTION ===\n');
tic;
boards = chessboardsFromCorners(corners, img);
board_detection_time = toc;

fprintf('Chessboard detection completed in %.3f seconds\n', board_detection_time);
fprintf('Chessboards detected: %d\n', length(boards));

% Detailed board analysis
for i = 1:length(boards)
    fprintf('\nBoard %d details:\n', i);
    
    if isfield(boards{i}, 'energy')
        fprintf('  Energy: %.3f\n', boards{i}.energy);
    end
    
    if isfield(boards{i}, 'corners')
        corner_count = size(boards{i}.corners, 1);
        fprintf('  Corners: %d\n', corner_count);
        
        % Calculate board dimensions
        if corner_count > 0
            x_coords = boards{i}.corners(:, 1);
            y_coords = boards{i}.corners(:, 2);
            fprintf('  Spatial extent: X[%.1f-%.1f] Y[%.1f-%.1f]\n', ...
                min(x_coords), max(x_coords), min(y_coords), max(y_coords));
        end
    end
    
    if isfield(boards{i}, 'idx')
        fprintf('  Corner indices: %s\n', mat2str(size(boards{i}.idx)));
    end
end

%% Step 4: Visualization and Final Results
fprintf('\n=== STEP 4: FINAL RESULTS ===\n');

total_time = corner_detection_time + board_detection_time;
fprintf('Total processing time: %.3f seconds\n', total_time);

fprintf('\nFinal Summary:\n');
fprintf('  Total corners: %d\n', length(corners.p));
fprintf('  MATLAB region corners: %d\n', matlab_region_count);
fprintf('  Detected boards: %d\n', length(boards));

if length(boards) > 0 && isfield(boards{1}, 'energy')
    fprintf('  Best board energy: %.3f\n', boards{1}.energy);
end

% Create detailed visualization
figure('Position', [100, 100, 1200, 800]);

% Original image
subplot(2, 2, 1);
imshow(img);
hold on;
title('Original Image');

% All corners with region coloring
subplot(2, 2, 2);
imshow(img);
hold on;
title(sprintf('All Corners (%d total)', length(corners.p)));

% Draw MATLAB expected region
rectangle('Position', [matlab_region(1), matlab_region(2), ...
    matlab_region(3)-matlab_region(1), matlab_region(4)-matlab_region(2)], ...
    'EdgeColor', 'yellow', 'LineWidth', 2);
text(50, 340, 'MATLAB Region', 'Color', 'yellow', 'FontSize', 12, 'FontWeight', 'bold');

for i = 1:length(corners.p)
    x = corners.p(i, 1);
    y = corners.p(i, 2);
    
    % Color by region
    if y < 200
        color = 'red';      % Upper region
    elseif y >= 200 && y < 350
        color = 'cyan';     % Middle region
    elseif x >= matlab_region(1) && x <= matlab_region(3) && y >= matlab_region(2) && y <= matlab_region(4)
        color = 'green';    % MATLAB region
    else
        color = 'magenta';  % Lower region
    end
    
    plot(x, y, 'o', 'Color', color, 'MarkerSize', 8, 'LineWidth', 2);
    plot(x, y, '.', 'Color', 'white', 'MarkerSize', 4);
end

% MATLAB region only
subplot(2, 2, 3);
imshow(img);
hold on;
title(sprintf('MATLAB Region Corners (%d)', matlab_region_count));

rectangle('Position', [matlab_region(1), matlab_region(2), ...
    matlab_region(3)-matlab_region(1), matlab_region(4)-matlab_region(2)], ...
    'EdgeColor', 'yellow', 'LineWidth', 2);

for i = 1:length(corners.p)
    x = corners.p(i, 1);
    y = corners.p(i, 2);
    
    if x >= matlab_region(1) && x <= matlab_region(3) && y >= matlab_region(2) && y <= matlab_region(4)
        plot(x, y, 'o', 'Color', 'green', 'MarkerSize', 8, 'LineWidth', 2);
        plot(x, y, '.', 'Color', 'white', 'MarkerSize', 4);
        text(x+5, y-5, num2str(i), 'Color', 'white', 'FontSize', 8);
    end
end

% Detected chessboards
subplot(2, 2, 4);
imshow(img);
hold on;
title(sprintf('Detected Chessboards (%d)', length(boards)));

if length(boards) > 0
    plotChessboards(corners, boards, img);
end

% Save results
save_path = 'matlab_debug_results.mat';
save(save_path, 'corners', 'boards', 'corner_coords', 'corner_scores', ...
     'matlab_region_count', 'upper_region_count', 'middle_region_count', 'lower_region_count');

fprintf('\nResults saved to: %s\n', save_path);

%% Export detailed log
log_file = 'matlab_debug_log.txt';
diary(log_file);
fprintf('\n=== MATLAB DETAILED DEBUG LOG ===\n');
fprintf('Timestamp: %s\n', datestr(now));
fprintf('Image: %s\n', img_path);
fprintf('Image size: %d x %d\n', size(img, 2), size(img, 1));
fprintf('Total corners: %d\n', length(corners.p));
fprintf('MATLAB region corners: %d\n', matlab_region_count);
fprintf('Detected boards: %d\n', length(boards));

if ~isempty(boards) && isfield(boards{1}, 'energy')
    fprintf('Best board energy: %.3f\n', boards{1}.energy);
end

fprintf('Corner detection time: %.3f seconds\n', corner_detection_time);
fprintf('Board detection time: %.3f seconds\n', board_detection_time);
fprintf('Total time: %.3f seconds\n', total_time);
diary off;

fprintf('\nDebug log saved to: %s\n', log_file);
fprintf('\n=== MATLAB DEBUG ANALYSIS COMPLETE ===\n'); 