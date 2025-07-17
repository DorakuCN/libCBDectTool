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

function corners = findCorners(img,tau,refine_corners)

% convert to double grayscale image
img = im2double(img);
if length(size(img))==3
  img = rgb2gray(img);
end

% 3 scales
radius(1) = 4;
radius(2) = 8;
radius(3) = 12;

% sobel masks
du = [-1 0 1; -1 0 1; -1 0 1];
dv = du';

% compute image derivatives (for principal axes estimation)
img_du     = conv2(double(img),du,'same');
img_dv     = conv2(double(img),dv,'same');
img_angle  = atan2(img_dv,img_du);
img_weight = sqrt(img_du.^2+img_dv.^2);

% correct angle to lie in between [0,pi]
img_angle(img_angle<0)  = img_angle(img_angle<0)+pi;
img_angle(img_angle>pi) = img_angle(img_angle>pi)-pi;

% scale input image
img     = double(img);
img_min = min(img(:));
img_max = max(img(:));
img     = (img-img_min)/(img_max-img_min);

% template properties
template_props = [0 pi/2 radius(1); pi/4 -pi/4 radius(1); 0 pi/2 radius(2); pi/4 -pi/4 radius(2); 0 pi/2 radius(3); pi/4 -pi/4 radius(3)];

disp('Filtering ...');

% filter image
img_corners = zeros(size(img,1),size(img,2));
for template_class=1:size(template_props,1)
  
  % create correlation template
  template = createCorrelationPatch(template_props(template_class,1),template_props(template_class,2),template_props(template_class,3));
  
  % filter image according with current template
  img_corners_a1 = conv2(img,template.a1,'same');
  img_corners_a2 = conv2(img,template.a2,'same');
  img_corners_b1 = conv2(img,template.b1,'same');
  img_corners_b2 = conv2(img,template.b2,'same');
  
  % compute mean
  img_corners_mu = (img_corners_a1+img_corners_a2+img_corners_b1+img_corners_b2)/4;
  
  % case 1: a=white, b=black
  img_corners_a = min(img_corners_a1-img_corners_mu,img_corners_a2-img_corners_mu);
  img_corners_b = min(img_corners_mu-img_corners_b1,img_corners_mu-img_corners_b2);
  img_corners_1 = min(img_corners_a,img_corners_b);
  
  % case 2: b=white, a=black
  img_corners_a = min(img_corners_mu-img_corners_a1,img_corners_mu-img_corners_a2);
  img_corners_b = min(img_corners_b1-img_corners_mu,img_corners_b2-img_corners_mu);
  img_corners_2 = min(img_corners_a,img_corners_b);
  
  % update corner map
  img_corners = max(img_corners,img_corners_1);
  img_corners = max(img_corners,img_corners_2);
end

% extract corner candidates via non maximum suppression
corners.p = nonMaximumSuppression(img_corners,3,0.025,5);

fprintf('=== MATLAB Corner Detection Debug ===\n');
fprintf('Initial corner candidates from NMS: %d\n', size(corners.p,1));

disp('Refining ...');

% subpixel refinement
if refine_corners
  corners_before_refine = size(corners.p,1);
  corners = refineCorners(img_du,img_dv,img_angle,img_weight,corners,10);
  fprintf('After subpixel refinement: %d corners (removed %d invalid)\n', ...
      size(corners.p,1), corners_before_refine - size(corners.p,1));
end

% remove corners without edges
idx = corners.v1(:,1)==0 & corners.v1(:,2)==0;
corners_before_edge_filter = size(corners.p,1);
corners.p(idx,:)  = [];
corners.v1(idx,:) = [];
corners.v2(idx,:) = [];
fprintf('After edge filter: %d corners (removed %d without edges)\n', ...
    size(corners.p,1), corners_before_edge_filter - size(corners.p,1));

disp('Scoring ...');

% score corners
corners = scoreCorners(img,img_angle,img_weight,corners,radius);
fprintf('Corner scoring completed for %d corners\n', size(corners.p,1));

% 显示评分统计
if size(corners.score,1) > 0
    fprintf('Score statistics: min=%.4f, max=%.4f, mean=%.4f, std=%.4f\n', ...
        min(corners.score), max(corners.score), mean(corners.score), std(corners.score));
    fprintf('Score threshold (tau): %.4f\n', tau);
end

% remove low scoring corners
idx = corners.score<tau;
corners_before_score_filter = size(corners.p,1);
num_below_threshold = sum(idx);
corners.p(idx,:)     = [];
corners.v1(idx,:)    = [];
corners.v2(idx,:)    = [];
corners.score(idx) = [];

fprintf('After score filter (tau=%.4f): %d corners (removed %d below threshold)\n', ...
    tau, size(corners.p,1), num_below_threshold);
fprintf('Final corner detection result: %d corners\n', size(corners.p,1));

% Debug: print corner coordinates for comparison with C++
fprintf('\nMATLAB corners first 10 coordinates (0-based, after conversion):\n');
for i = 1:min(10, size(corners.p,1))
    fprintf('  %d: (%.2f, %.2f)\n', i-1, corners.p(i,1), corners.p(i,2));
end

% make v1(:,1)+v1(:,2) positive (=> comparable to c++ code)
idx = corners.v1(:,1)+corners.v1(:,2)<0;
corners.v1(idx,:) = -corners.v1(idx,:);

% make all coordinate systems right-handed (reduces matching ambiguities from 8 to 4)
corners_n1 = [corners.v1(:,2) -corners.v1(:,1)];
flip       = -sign(corners_n1(:,1).*corners.v2(:,1)+corners_n1(:,2).*corners.v2(:,2));
corners.v2 = corners.v2.*(flip*ones(1,2));

% convert to 0-based index
% convert to 0-based index
corners.p = corners.p - 1;

% Debug: print corners.p for comparison
disp('MATLAB: findCorners output p (0-based coords):');
disp(corners.p);
