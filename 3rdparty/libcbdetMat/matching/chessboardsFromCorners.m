% Copyright 2012. All rights reserved.
% Author: Andreas Geiger
%         Institute of Measurement and Control Systems (MRT)
%         Karlsruhe Institute of Technology (KIT), Germany

% This is free software; you can redistribute it and/or modify it under the
% terms of the GNU General Public License as published by the Free Software
% Foundation; either version 3 of the License, or any later version.

% This software is distributed in the hope that it will be useful, but WITHOUT ANY
% WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
% PARTICULAR PURPOSE. See the GNU General Public License for more details.

% You should have received a copy of the GNU General Public License along with
% this software; if not, write to the Free Software Foundation, Inc., 51 Franklin
% Street, Fifth Floor, Boston, MA 02110-1301, USA 

function chessboards = chessboardsFromCorners(corners)

fprintf('Structure recovery:\n');

% 添加详细的调试信息
fprintf('=== MATLAB Structure Recovery Debug ===\n');
fprintf('Total corners available: %d\n', size(corners.p,1));
fprintf('Energy thresholds: init=0, final=-10 (MATLAB standard)\n');

% 统计变量
init_failed = 0;
energy_rejected_init = 0;
energy_rejected_final = 0;
success_count = 0;

% intialize chessboards
chessboards = [];

% for all seed corners do
for i=1:size(corners.p,1)
  
  % output progress more frequently for debugging
  if mod(i-1,5)==0 || i <= 8  % 每5个输出进度，前8个全部输出
    fprintf('Processing seed %d/%d\n',i,size(corners.p,1));
  end
  
  % 添加调试：尝试初始化
  fprintf('  Attempting to initialize chessboard for seed %d\n', i);
  
  % init 3x3 chessboard from seed i
  chessboard = initChessboard(corners,i);
  
  % 检查初始化结果
  if isempty(chessboard)
    fprintf('  Seed %d result: FAILED (initialization)\n', i);
    init_failed = init_failed + 1;
    continue;
  else
    fprintf('  Seed %d result: SUCCESS (initialization)\n', i);
  end
  
  % 计算初始能量
  init_energy = chessboardEnergy(chessboard,corners);
  fprintf('  Energy check: seed %d has energy %.5f (threshold: 0)\n', i, init_energy);
  
  % check if this is a useful initial guess
  if init_energy > 0
    fprintf('  REJECTED by energy: %.5f > 0\n', init_energy);
    energy_rejected_init = energy_rejected_init + 1;
    continue;
  end
    
  % try growing chessboard
  fprintf('  Growing chessboard from seed %d...\n', i);
  growth_iterations = 0;
  while 1
    
    % compute current energy
    energy = chessboardEnergy(chessboard,corners);
    
    % compute proposals and energies
    for j=1:4
      proposal{j} = growChessboard(chessboard,corners,j);
      p_energy(j) = chessboardEnergy(proposal{j},corners);
    end
    
    % find best proposal
    [min_val,min_idx] = min(p_energy);
    
    % accept best proposal, if energy is reduced
    if p_energy(min_idx)<energy
      fprintf('    Growth iteration %d: energy %.3f -> %.3f\n', growth_iterations+1, energy, p_energy(min_idx));
      chessboard = proposal{min_idx};
      growth_iterations = growth_iterations + 1;
      
      if 0
        figure, hold on, axis equal;
        chessboards{1} = chessboard;
        plotChessboards(chessboards,corners);
        keyboard;
      end
      
    % otherwise exit loop
    else
      break;
    end
  end
  
  % 计算最终能量
  final_energy = chessboardEnergy(chessboard,corners);
  fprintf('  Final energy check: seed %d has final energy %.5f (threshold: -10)\n', i, final_energy);
    
  % if chessboard has low energy (corresponding to high quality)
  if final_energy < -10
  
    fprintf('  SUCCESS: seed %d -> final_energy %.5f < -10\n', i, final_energy);
    success_count = success_count + 1;
    % check if new chessboard proposal overlaps with existing chessboards
    overlap = zeros(length(chessboards),2);
    for j=1:length(chessboards)
      for k=1:length(chessboards{j}(:))
        if any(chessboards{j}(k)==chessboard(:))
          overlap(j,1) = 1;
          overlap(j,2) = chessboardEnergy(chessboards{j},corners);
          break;
        end
      end
    end

    % add chessboard (and replace overlapping if neccessary)
    if ~any(overlap(:,1))
      chessboards{end+1} = chessboard;
    else
      idx = find(overlap(:,1)==1);
      if ~any(overlap(idx,2)<=chessboardEnergy(chessboard,corners))
        chessboards(idx) = [];
        chessboards{end+1} = chessboard;
      end
    end
  else
    fprintf('  FINAL REJECTED by energy: %.5f > -10\n', final_energy);
    energy_rejected_final = energy_rejected_final + 1;
  end
end

% 添加最终统计
fprintf('\n=== MATLAB Structure Recovery Completed ===\n');
fprintf('Processed: %d/%d corners\n', size(corners.p,1), size(corners.p,1));
fprintf('Init failures: %d\n', init_failed);
fprintf('Energy rejected (init): %d (threshold: 0)\n', energy_rejected_init);
fprintf('Energy rejected (final): %d (threshold: -10)\n', energy_rejected_final);
fprintf('Found: %d chessboards\n', length(chessboards));

fprintf('\n');
