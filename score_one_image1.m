function [ f0, feat_max0, f1, feat_max1 ] = score_one_image1( W, W_old, feat, xy, sz, varargin )

if nargin == 5
    level = [1 2 4];
end
v = feat * W;
v_old = feat * W_old;
% v = softmax(v')';

[f0, feat_max0, f1, feat_max1] = lala_spm(feat, v, v_old, xy, sz, level);

end


function [ f0, feat_max0, f1, feat_max1 ] = lala_spm(feat, v, v_old, xy, sz, level)

% max-pooling and spm
feat_dim = size(feat, 2);
v_dim = size(v_old, 2);
n_bins = sum(level.^2);
f0 = zeros( v_dim,  n_bins);
feat_max0 = cell( n_bins, 1 );
f1 = zeros( v_dim,  n_bins);
feat_max1 = cell( n_bins, 1 );
xy(:,1) = xy(:,1) / sz(2);
xy(:,2) = xy(:,2) / sz(1);
findex = 0;

for i = 1:length(level)
    for r = 1:level(i)
        rmin = (r-1)/level(i);
        rmax = (r+0)/level(i);
        for c = 1:level(i)
            cmin = (c-1)/level(i);
            cmax = (c+0)/level(i);  
            
            idx = xy(:,1)>=cmin & xy(:,1)<=cmax & xy(:,2)>=rmin & xy(:,2)<=rmax;
            v_old_ = v_old( idx, : );
            v_ = v( idx, : );
            feat_ = feat( idx, : );
           
            findex = findex + 1;
            
            if isempty(v_old_)
                f0( :, findex ) = single(zeros(v_dim, 1));
                feat_max0{findex} = single(zeros(v_dim, feat_dim));
                f1( :, findex ) = single(zeros(v_dim, 1));
                feat_max1{findex} = single(zeros(v_dim, feat_dim));
            else
                [~, pos_max] = max( v_old_, [], 1 );
                feat_max1{findex} = feat_(pos_max, :);
                % f1(:, findex) = diag(feat_max1{findex} * W);
                f1(:, findex) = v_(pos_max + (0:(length(pos_max)-1))*size(v_, 1));
                [f_tmp, pos_max] = max( v_, [], 1 );
                feat_max0{findex} = feat_(pos_max, :);
                f0(:, findex) = f_tmp;
%                 f( findex, : ) = sum( v_ );
            end                
        end
    end
end

f0 = f0(:);
f1 = f1(:);
feat_max0 = cell2mat(feat_max0);
feat_max1 = cell2mat(feat_max1);

end
