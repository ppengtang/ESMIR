function [CNN_feat, CNN_xy, sz, im] = extr_deep_img_feat(im, para, net)

% resize the image
im = imresize(im, [para.imsz, para.imsz]);
sz = size(im);

patchsize = para.patchsize;
step = para.step;

n_scale = length(patchsize);
rect = cell(n_scale, 1);
for i = 1:n_scale
    xs = 1 : step : sz(2)-patchsize(i)+1;
    ys = 1 : step : sz(1)-patchsize(i)+1;
    [x, y] = meshgrid(xs, ys);
    x = x(:);
    y = y(:);
    rect{i} = [ y, y+patchsize(i)-1, x, x+patchsize(i)-1];
end

CNN_feat = cell(n_scale, 1);
CNN_xy = cell(n_scale, 1);

for i = 1:n_scale
    rect_tmp = rect{i};
    len = size( rect_tmp,1 );
    CNN_feat_tmp = zeros( len, 0 );
    CNN_xy_tmp = zeros( len, 2 );
    for j = 1:len
        r = rect_tmp(j,:);
        f = extr_patch_dfeat_linux(im, r, para, net);
        CNN_feat_tmp(j, 1:length(f)) = f;
        CNN_xy_tmp(j,:) = [ (r(3)+r(4))/2, (r(1)+r(2))/2 ];
    end
    CNN_feat{i} = single(CNN_feat_tmp);
    CNN_xy{i} = single(CNN_xy_tmp);
end

sz = single(sz);

end
