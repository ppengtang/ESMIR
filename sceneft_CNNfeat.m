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


tr = zeros( size(database.label) );
% read the train and test image
MITindoor_tr = importdata('TrainImages.txt');
MITindoor_te = importdata('TestImages.txt');

tr_map_cname = containers.Map();
for i = 1:database.nclass
    idx = find( database.label == i );
    tr_map_name = containers.Map();
    for j = 1:length(idx)
        [~, fig_name] = fileparts(database.path{idx(j)});
        tr_map_name(fig_name) = idx(j);
    end
    tr_map_cname(database.cname{i}) = tr_map_name;
end

for i = 1:length(MITindoor_tr)
    [cname, name] = fileparts(MITindoor_tr{i});
    tr_map_name = tr_map_cname(cname);
    tr(tr_map_name(name)) = 1;
end

for i = 1:length(MITindoor_te)
    [cname, name] = fileparts(MITindoor_te{i});
    tr_map_name = tr_map_cname(cname);
    tr(tr_map_name(name)) = 2;
end

% find the position of train and test image
tr_te_position = find(tr ~= 0);

tmp = retr_database_dir(para.path_rgnfeat, '*.mat');
begin = tmp.imnum + 1;
for i=begin:length(tr_te_position)
    position = tr_te_position(i);
    fprintf('Extracting patch level features: %d of %d.\n', position, database.imnum);
    
    tic
    im = imread( database.path{position} );
    [~, im_box] = fileparts(database.path{position});
    box_dir = fullfile(para.box_db, database.cname{database.label(position)}, [im_box, '.mat']);
    
    if size(im, 3) == 1
        im_tmp = zeros([size(im), 3]);
        im_tmp(:, :, 1) = im(:, :);
        im_tmp(:, :, 2) = im(:, :);
        im_tmp(:, :, 3) = im(:, :);
        im = uint8(im_tmp);
    end
    load(box_dir);
    [feat, xy, sz] = extr_img_feat_MITindoor_box( im, inf, para, net, proposals);
    
    toc
    
    feat = single(feat);
    xy = single(xy);
    sz = single(sz);
    
    [nc, fname] = fileparts( database.path{position} );
    p0 = fullfile( para.path_rgnfeat, database.cname{database.label(position)} );
    if ~exist( p0, 'dir' )
        mkdir(p0);
    end
    
    p = fullfile( para.path_rgnfeat, database.cname{database.label(position)}, fname );
    savefeat( feat, xy, sz, p );
end
% matlabpool close;

function savefeat( feat, xy, sz, p )
save( [p, '.mat'], 'feat', 'xy', 'sz' );
