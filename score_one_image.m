function f = score_one_image( W, feat, xy, sz, varargin )

if nargin == 4
    level = [1 2 4];
end
v = feat * W;
% v = softmax(v')';

f = lala_spm(v, xy, sz, level);

f = f / norm(f,2);

end


function f = lala_spm(v, xy, sz, level)

% max-pooling and spm
f = zeros( size(v,2), sum(level.^2) );
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
            v_ = v( idx, : );
           
            findex = findex + 1;
            
            if isempty(v_)
                f( :, findex ) = zeros(size(v, 2), 1);
            else
                f( :, findex ) = max( v_, [], 1 );
%                 f( findex, : ) = sum( v_ );
            end                
        end
    end
end

f = f(:);
% f = [f; pi];
end
