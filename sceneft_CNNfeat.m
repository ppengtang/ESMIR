function sceneft_CNNfeat(para)

database = retr_database_dir(para.path_db, para.fmt);

% Set caffe mode
if para.use_gpu
    caffe.set_mode_gpu();
    caffe.set_device(para.gpu_id);
else
    caffe.set_mode_cpu();
end
phase = 'test'; % run with phase test (so that dropout isn't applied)

% Initialize a network
net = caffe.Net(para.model_def_file, para.model_file, phase);

for i=1:database.imnum
    fprintf('Extracting patch level features: %d of %d.\n', i, database.imnum);
    
    tic
    im = imread( database.path{i} );   
    if size(im, 3) == 1
        im_tmp = zeros([size(im), 3]);
        im_tmp(:, :, 1) = im(:, :);
        im_tmp(:, :, 2) = im(:, :);
        im_tmp(:, :, 3) = im(:, :);
        im = im_tmp;
    end
    [feat, xy, sz] = extr_deep_img_feat( im, para, net);
    toc
        
    [nc, fname] = fileparts( database.path{i} );
    p0 = fullfile( para.path_rgnfeat, database.cname{database.label(i)} );
    if ~exist( p0, 'dir' )
        mkdir(p0);
    end
    
    p = fullfile( para.path_rgnfeat, database.cname{database.label(i)}, fname );
    savefeat( feat, xy, sz, p );
end

end

function savefeat( feat, xy, sz, p )
save( [p, '.mat'], 'feat', 'xy', 'sz' );
end
