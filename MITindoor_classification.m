function MITindoor_classification()

addpath(fullfile('include', 'liblinear-1.7-single', 'matlab'));
addpath(fullfile('include', 'vlfeat', 'toolbox'));
addpath(fullfile('include', 'utils'));
addpath(fullfile('..', 'caffe', 'matlab'));
vl_setup;


experiment_index = 'release';    %%!!!!

para.nround = 1;   

para.max_imsz = 500;
para.min_imsz = 256;
para.step = 32;                 % step size for computing feature
para.patchsize = [144, 120, 96, 72];  % size of patch for computing feature

para.model_def_file = fullfile('feature_extraction_imagenet',...
    'VGG_CNN_S_deploy_fc7.prototxt');
para.model_file = fullfile('feature_extraction_imagenet',...
    'VGG_CNN_S.caffemodel');
para.use_gpu = 1;
para.gpu_id = 0;

load( fullfile('feature_extraction_imagenet', 'VGG_mean') );
para.IMAGE_MEAN = image_mean;

para.IMAGE_DIM = 256;
para.CROPPED_DIM = 224;

para.poolsize = 2;

para.options = '-s 5 -c 5';
para.options1 = '-s 2 -c 4';               % svm option for image classfication
para.num_cluster = 4;
para.max_iter = 3;
para.max_SGD_iter = 10000;
para.batchsize = 1;
para.lr = 1;
para.gamma = 0.1;
para.lambda_w_init = 0.002;
para.lambda_w = 0.05;
para.lambda_u = 5 / (10 ^ 4);
para.beta = 1;
para.m = 1;
para.pca_dim = 256;

para.path_db = fullfile('data', 'MITindoor_categories');
para.fmt = '*.jpg';
para.path_svm_models = fullfile('data', 'best_4.mat');
para.path_rgnfeat = fullfile('data', sprintf('MITindoorft_VGGCNNfeat_%s', experiment_index ));
para.path_results = fullfile('data', sprintf('MITindoorft_results_%s.mat', experiment_index ));

mkdir(para.path_rgnfeat);

sceneft_CNNfeat(para);

db = retr_database_dir(para.path_db, para.fmt);

tr = zeros( size(db.label) );
% read the train and test image
MITindoor_tr = importdata('TrainImages.txt');
MITindoor_te = importdata('TestImages.txt');

tr_map_cname = containers.Map();
for i = 1:db.nclass
    idx = find( db.label == i );
    tr_map_name = containers.Map();
    for k = 1:length(idx)
        [~, fig_name] = fileparts(db.path{idx(k)});
        tr_map_name(fig_name) = idx(k);
    end
    tr_map_cname(db.cname{i}) = tr_map_name;
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

db.tr = tr;

% pca
if para.pca_dim ~= 0 && ~exist('MITindoor_pca.mat', 'file')
    fprintf('pca.\n');
    tic;
    tr_pos = find( db.tr==1 );
    features = cell(length(tr_pos), 1);
    for i = 1:length(tr_pos)
        imid = tr_pos(i);
        [~, fname] = fileparts( db.path{imid} );
        p = fullfile( para.path_rgnfeat, db.cname{db.label(imid)}, [fname '.mat'] );
        feat = load_feat(p);
        tmp = any(isnan(feat)');
        feat(tmp, :) = [];
        ri = randperm(size(feat, 1));
        ri = ri(1:min(length(ri), 100));
        feat = feat(ri, :);
        features{i} = feat;
    end
    features = cell2mat(features);
    ri = randperm(size(features, 1));
    length(ri)
    ri = ri(1:min(length(ri), 300000));
    features = features(ri, :);
    pca_mean = mean(features, 1);
    [pca_coeff, ~, pca_eig] = pca(features);
    save('MITindoor_pca.mat', 'pca_mean', 'pca_coeff', 'pca_eig');
    clear features;
    toc;
end

load('MITindoor_pca.mat');

load(para.path_svm_models);

% encoding image using attribute classifier
tr = find( db.tr==1 );
te = find( db.tr==2 );

fv_tr = zeros( length(tr), 0 );
fv_te = zeros( length(te), 0 );
for i = 1:length(tr)
    position = tr(i);
    fprintf( 'Scoring %d of %d.\n', position, db.imnum );
    [~, fname] = fileparts( db.path{position} );
    p = [fullfile( para.path_rgnfeat, db.cname{db.label(position)}, fname ), '.mat'];
    load(p, 'feat', 'xy', 'sz');
    f = score_one_image( model_w{ii}, my_pca(feat, pca_mean, pca_coeff, pca_eig, para.pca_dim), xy, sz );
    fv_tr(i, 1:length(f)) = f;
end
for i = 1:length(te)
    position = te(i);
    fprintf( 'Scoring %d of %d.\n', position, db.imnum );
    [~, fname] = fileparts( db.path{position} );
    p = [fullfile( para.path_rgnfeat, db.cname{db.label(position)}, fname ), '.mat'];
    load(p, 'feat', 'xy', 'sz');
    f = score_one_image( model_w{ii}, my_pca(feat, pca_mean, pca_coeff, pca_eig, para.pca_dim), xy, sz );
    fv_te(i, 1:length(f)) = f;
end

label = db.label;

% testing procedure
fv_tr = single(fv_tr);
fv_te = sparse(fv_te);
label_tr = single(label( db.tr==1 ));
label_te = sparse(label( db.tr==2 ));

m2 = train( label_tr, fv_tr, para.options1 );
clear fv_tr;
clear label_tr;

[~, a] = predict( label_te, fv_te, m2 );

acc = a(1);

acc


end


function [feat_new] = my_pca(feat, pca_mean, pca_coeff, pca_eig, pca_dim)

feat_new_tmp = single(bsxfun(@minus, feat, pca_mean) * pca_coeff) * diag(pca_eig .^ (-0.5));
feat_new = single(feat_new_tmp(:, 1:pca_dim));

feat_new = bsxfun(@rdivide, feat_new, norm_matrix(feat_new, 2, 2)+eps);
    
end
