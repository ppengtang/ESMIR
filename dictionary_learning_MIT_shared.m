function dictionary_learning_MIT_shared(para)

db = retr_database_dir(para.path_db, para.fmt);
n = para.nround;
acc = zeros(1,n);
for i = 1:n
    acc(i) = run_one_time(db, para);
end
sacc = std(acc);
macc = mean(acc);
fprintf('Final accuracy: %f +- %f.\n', macc, sacc);
save(para.path_results, 'macc', 'acc', 'sacc');

end

function acc = run_one_time( db, para )

% divide the images into training and testing
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
% training procedure

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

if ~exist('MIT_features.mat', 'file')
n_image = length( find(db.tr==1) );
features_all = cell(n_image, 1);
xy_all = cell(n_image, 1);
sz_all = cell(n_image, 1);
labels_all = single(zeros(n_image, 1));
count = 0;
for i = 1:db.nclass
    %for i = 1:0
    fprintf('loading training features for class %d.\n', i);
    % reduce some of the negative
    tr_pos = find( db.tr==1 & db.label==i );
    
    for j = 1:length(tr_pos)
        count = count + 1;
        
        imid = tr_pos(j);
        
        [~, fname] = fileparts( db.path{imid} );
        p = [fullfile( para.path_rgnfeat, db.cname{db.label(imid)}, fname ), '.mat'];
        load(p);
        
        features_all{count} = my_pca(feat, pca_mean, pca_coeff, pca_eig, para.pca_dim);
        xy_all{count} = xy;
        sz_all{count} = sz;
        labels_all(count) = single(i);
        
    end
    
end

save('MIT_features.mat', 'features_all', 'labels_all', 'xy_all', 'sz_all');

end

load('MIT_features.mat');

fprintf('training models.\n');
[model_w, model_u, fv_tr, label_tr] = shared(features_all, labels_all, xy_all, sz_all, para);

save(para.path_svm_models, 'model_w', 'db', 'model_u');

clear features_all;
clear labels_all;

% testing procedure
label = db.label;

fv_tr = single(fv_tr);
label_tr = single(label_tr);

m2 = train( label_tr, fv_tr, para.options1 );
clear fv_tr;
clear label_tr;

te = find( db.tr==2 );
fv_te = zeros( length(te), 0 );
for i = 1:length(te)
    position = te(i);
    fprintf( 'Scoring %d of %d.\n', position, db.imnum );
    [~, fname] = fileparts( db.path{position} );
    p = [fullfile( para.path_rgnfeat, db.cname{db.label(position)}, fname ), '.mat'];
    load(p, 'feat', 'xy', 'sz');
    f = score_one_image( model_w{end}, my_pca(feat, pca_mean, pca_coeff, pca_eig, para.pca_dim), xy, sz );
    fv_te(i, 1:length(f)) = f;
end

fv_te = sparse(fv_te);
label_te = sparse(label( db.tr==2 ));

[~, a] = predict( label_te, fv_te, m2 );

acc = a(1);

end

function feat = load_feat(p)
f = load(p, 'feat');
feat = f.feat;
end


function [feat_new] = my_pca(feat, pca_mean, pca_coeff, pca_eig, pca_dim)

feat_new_tmp = single(bsxfun(@minus, feat, pca_mean) * pca_coeff) * diag(pca_eig .^ (-0.5));
feat_new = single(feat_new_tmp(:, 1:pca_dim));

feat_new = bsxfun(@rdivide, feat_new, norm_matrix(feat_new, 2, 2)+eps);
    
end
