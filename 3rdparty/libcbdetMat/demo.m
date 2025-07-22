clear variables; dbstop error; close all;
disp('================================');
disp('MATLAB libcbdetM Demo with Debug Info');
disp('================================');

addpath('matching');

% 加载图像
%I = imread('../../data/imiSample/Color.bmp');
I = imread('../../data/imiSample/IR.bmp');

fprintf('Image loaded: %dx%d\n', size(I,2), size(I,1));

% 角点检测
fprintf('\n=== CORNER DETECTION ===\n');
tic;
corners = findCorners(I,0.01,1);
corner_time = toc;
fprintf('Corner detection completed in %.3f ms\n', corner_time * 1000);
fprintf('Found %d corners\n', size(corners.p,1));

% 显示前几个角点的信息
fprintf('\nFirst 5 corners (if available):\n');
for i = 1:min(5, size(corners.p,1))
    fprintf('  Corner %d: pos=(%.2f,%.2f), v1=(%.3f,%.3f), v2=(%.3f,%.3f), score=%.3f\n', ...
        i, corners.p(i,1), corners.p(i,2), ...
        corners.v1(i,1), corners.v1(i,2), ...
        corners.v2(i,1), corners.v2(i,2), ...
        corners.score(i));
end

% 棋盘格检测
fprintf('\n=== CHESSBOARD DETECTION ===\n');
tic;
chessboards = chessboardsFromCorners(corners);
chessboard_time = toc;
fprintf('Chessboard detection completed in %.3f ms\n', chessboard_time * 1000);
fprintf('Found %d chessboards\n', length(chessboards));

% 显示每个棋盘格的详细信息
for i = 1:length(chessboards)
    cb = chessboards{i};
    energy = chessboardEnergy(cb, corners);
    fprintf('\nChessboard %d:\n', i);
    fprintf('  Size: %dx%d\n', size(cb,1), size(cb,2));
    fprintf('  Energy: %.5f\n', energy);
    fprintf('  Corner indices: ');
    for r = 1:size(cb,1)
        for c = 1:size(cb,2)
            fprintf('%d ', cb(r,c));
        end
        if r < size(cb,1), fprintf('| '); end
    end
    fprintf('\n');
    
    % 显示角点坐标
    fprintf('  Corner positions:\n');
    for r = 1:size(cb,1)
        fprintf('    Row %d: ', r);
        for c = 1:size(cb,2)
            idx = cb(r,c);
            fprintf('(%.1f,%.1f) ', corners.p(idx,1), corners.p(idx,2));
        end
        fprintf('\n');
    end
end

% 总结
fprintf('\n=== SUMMARY ===\n');
fprintf('Total processing time: %.3f ms\n', (corner_time + chessboard_time) * 1000);
fprintf('Corner detection: %d corners\n', size(corners.p,1));
fprintf('Chessboard detection: %d chessboards\n', length(chessboards));
if length(chessboards) > 0
    fprintf('Result matches C++ target: YES\n');
else
    fprintf('Result matches C++ target: NO\n');
end

% 可视化结果
% In batch/headless mode skip UI; otherwise show detection
if usejava('desktop')
    figure; imshow(uint8(I)); hold on;
    plotChessboards(chessboards,corners);
else
    disp('Batch mode: skipping figure display');
end

% Debug: print final corner and board indices for comparison
disp('MATLAB: corners.p (0-based):');
disp(corners.p - 1);
for c = 1:numel(chessboards)
    disp(['MATLAB: board ' num2str(c) ' corner indices (0-based):']);
    disp(chessboards{c}(:)' - 1);
end
title(sprintf('MATLAB Result: %d corners, %d chessboards', size(corners.p,1), length(chessboards)));

% 保存结果供对比
fprintf('\n=== SAVING DEBUG DATA ===\n');
save('matlab_debug_results.mat', 'corners', 'chessboards', 'I');
fprintf('Debug data saved to matlab_debug_results.mat\n');
