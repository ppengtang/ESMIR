function [model_w, model_u, fv_tr, labels_tr] = shared(features_all, labels_all, xy_all, sz_all, para)

fprintf('initializing.\n');
tic;

if length(labels_all) ~= length(features_all)
    error('the number of features must equal to the number of labels!');
end
n_images = length(labels_all);
n_classes = length(unique(labels_all));
clusters_center = cell(n_classes, 1);
for i = 1:n_classes
    features_all_tmp = features_all(labels_all == i);
    features_all_tmp = cell2mat(features_all_tmp);
    clusters_center{i} = vl_kmeans(features_all_tmp', para.num_cluster);
end

labels_in = init_label(features_all, labels_all, clusters_center, para.num_cluster);
clear features_all_tmp;

model_w = cell(para.max_iter, 1);
model_u = cell(para.max_iter, 1);
W = single(zeros(size(features_all{1}, 2), para.num_cluster*n_classes));
fprintf('time costs %f min.\n', toc/60);

fprintf('iteration 1.\n');

fprintf('update w.\n');
tic;

count = 1;
lr = para.lr;
for i = 1:5
    for j = 1:para.max_SGD_iter
        if mod(count, n_images) == 1
            sample = randperm(n_images);
        end
        index = sample(mod(count, n_images)+1);
        features_cur = features_all{index};
        W = SGD_W_init(W, features_cur, labels_in{index}, lr, para);
        count = count + 1;
    end
    lr = lr * para.gamma;
end
model_w{1} = W;

fprintf('time costs %f min.\n', toc/60);

fprintf('scoring images.\n');
tic;
fv_tr = zeros(n_images, 0);
for i = 1:n_images
    f = score_one_image( W, features_all{i}, xy_all{i}, sz_all{i} );
    fv_tr(i, 1:length(f)) = f;
end

fv_tr = single(fv_tr);
labels_tr = single(labels_all);
fprintf('time costs %f min.\n', toc/60);

if para.max_iter ~= 1
    fprintf('update u.\n');
    tic;
    U = single(zeros(size(fv_tr, 2), length(unique(labels_all))));
    n_iter = floor(n_images/para.batchsize);
    count = 1;
    lr = para.lr;
    for i = 1:5
        for j = 1:(para.max_SGD_iter/2)
            if mod(count, n_iter) == 1
                sample = randperm(n_images);
            end
            index = sample( (mod(count, n_iter)*para.batchsize+1):(mod(count, n_iter)*para.batchsize+para.batchsize) );
            U = SGD_U(U, fv_tr(index, :), labels_tr(index), lr, para);
            count = count + 1;
        end
        lr = lr * para.gamma;
    end
    model_u{1} = U;

    clear fv_tr;
    clear labels_tr;
    fprintf('time costs %f min.\n', toc/60);
end

for iter = 2:para.max_iter
    fprintf('iteration %d.\n', iter);
    fprintf('update w.\n');
    tic;
    
    U_sub = sub_U(U);
    W_old = W;
    W = single(zeros(size(features_all{1}, 2), para.num_cluster*n_classes));
    count = 1;
    lr = para.lr;
    for i = 1:5
        for j = 1:para.max_SGD_iter
            if mod(count, n_images) == 1
                sample = randperm(n_images);
            end
            index = sample(mod(count, n_images)+1);
            features_cur = features_all{index};
            W = SGD_W(W, W_old, U_sub{labels_all(index)}, features_cur, ...
                xy_all{index}, sz_all{index}, lr, para);
            count = count + 1;
        end
        lr = lr * para.gamma;
    end
    model_w{iter} = W;
    
    fprintf('time costs %f min.\n', toc/60);
    
    fprintf('scoring images.\n');
    tic;
    fv_tr = zeros(n_images, 0);
    for i = 1:n_images
        f = score_one_image( W, features_all{i}, xy_all{i}, sz_all{i} );
        fv_tr(i, 1:length(f)) = f;
    end
    
    fv_tr = single(fv_tr);
    labels_tr = single(labels_all);
    fprintf('time costs %f min.\n', toc/60);
    
    if iter ~= para.max_iter
        fprintf('update u.\n');
        tic;
        U = single(zeros(size(fv_tr, 2), length(unique(labels_all))));
        n_iter = floor(n_images/para.batchsize);
        count = 1;
        lr = para.lr;
        for i = 1:5
            for j = 1:(para.max_SGD_iter/2)
                if mod(count, n_iter) == 1
                    sample = randperm(n_images);
                end
                index = sample( (mod(count, n_iter)*para.batchsize+1):(mod(count, n_iter)*para.batchsize+para.batchsize) );
                U = SGD_U(U, fv_tr(index, :), labels_tr(index), lr, para);
                count = count + 1;
            end
            lr = lr * para.gamma;
        end
        model_u{iter} = U;
        
        clear fv_tr;
        clear labels_tr;
        fprintf('time costs %f min.\n', toc/60);
    end
    
end

end


function [ W_update ] = SGD_W_init(W, features, labels, lr, para)

idx_real = single(labels(:));
score = features * W;

temp = score;
[l,c] = size(temp);
allidx = single((idx_real - 1) * l) + (1:length(idx_real))';
score_real = score(allidx);
temp(allidx) = -inf;
score_tmp = reshape(temp, l, c);

[score_max, idx_max] = max(score_tmp, [], 2);
score_diff = score_real - score_max;

idx = unique([idx_real; idx_max(:)]);
idx1 = (score_diff < para.m);

n = size(features, 1);
W_tmp = para.lambda_w_init * W;
W_tmp(:, idx) = W_tmp(:, idx) + features(idx1, :)' * sign_my(idx_real(idx1), idx_max(idx1), idx) / n;
W_update = W - lr * W_tmp;

end


function [ W_update ] = SGD_W(W, W_old, U_sub, features, xy, sz, lr, para)

W_tmp = para.lambda_w * W;

n_classes = size(U_sub, 2) + 1;
[feat0, feat_max0, feat1, feat_max1] = score_one_image1(W, W_old, features, xy, sz);
feat0 = repmat(feat0(:), 1, n_classes - 1);
feat1 = repmat(feat1(:), 1, n_classes - 1);
feat0(U_sub > 0) = feat1(U_sub > 0);
[feat0, feat0_norm] = my_norm(feat0, 2);

score1 = diag(U_sub' * feat0);
[score_diff1, idx_min] = min(score1);

if score_diff1 < para.m
    feat_max0(U_sub(:, idx_min)>0, :) = feat_max1(U_sub(:, idx_min)>0, :);
    feat_max0 = feat_max0 / feat0_norm(idx_min);
    W_tmp = W_tmp - para.beta ...
        * sum(reshape(bsxfun(@times, U_sub(:, idx_min), feat_max0)', ...
        [para.pca_dim, para.num_cluster*n_classes, 21]), 3);
end

W_update = W - lr * W_tmp;

end


function [ U_update ] = SGD_U(U, features, labels, lr, para)

idx_real = single(labels(:));
score = features * U;

temp = score;
[l,c] = size(temp);
allidx = single((idx_real - 1) * l) + (1:length(idx_real))';
score_real = score(allidx);
temp(allidx) = -inf;
score_tmp = reshape(temp, l, c);

[score_max, idx_max] = max(score_tmp, [], 2);
score_diff = score_real - score_max;

idx = unique([idx_real; idx_max(:)]);
idx1 = (score_diff < para.m);

n = size(features, 1);
U_tmp = para.lambda_u * sign(U);

U_tmp(:, idx) = U_tmp(:, idx) + para.beta * features(idx1, :)' * sign_my(idx_real(idx1), idx_max(idx1), idx) / n;
U_update = U - lr * U_tmp;

end


function [ labels_in ] = init_label(features_all, labels_all, centers, num_cluster)

labels_in = cell(size(features_all));
for i = 1:length(features_all)
    [~, labels] = min( vl_alldist(features_all{i}', centers{labels_all(i)}), [], 2 );
    labels_in{i} = single(labels(:) + (labels_all(i) - 1) * num_cluster);
end

end


function [ result ]  = sign_my( idx_real, idx_max, idx )

n = length(idx);
result = zeros(length(idx_real), length(idx));
result(bsxfun(@minus, repmat(idx_real(:), [1, n]), idx(:)') == 0) = -1;
result(bsxfun(@minus, repmat(idx_max(:), [1, n]), idx(:)') == 0) = 1;
result = single(result);

end


function [ U_sub ] = sub_U(U)

n = size(U, 2);
U_sub = cell(n, 1);
for i = 1:n
    U_sub{i} = bsxfun( @minus, U(:, i), [U(:, 1:i-1), U(:, i+1:end)] );
end

end


function [f, f_norm] = my_norm(feat, n)

f_norm = single(norm_matrix(feat, n, 1)+eps);
f = single(bsxfun(@rdivide, feat, f_norm));

end
