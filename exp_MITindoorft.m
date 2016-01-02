
addpath(fullfile('include', 'liblinear-1.7-single', 'matlab'));
addpath(fullfile('include', 'vlfeat', 'toolbox'));
addpath(fullfile('include', 'utils'));
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
para.options1 = '-s 2 -c 3';               % svm option for image classfication
para.num_cluster = 4;
para.max_iter = 3;
para.max_SGD_iter = 10000;
para.batchsize = 40;
para.lr = 1;
para.gamma = 0.1;
para.lambda_w_init = 0.03;
para.lambda_w = 0.1;
para.lambda_u = 5 / (10 ^ 7);
para.beta = 1;
para.m = 1;
para.pca_dim = 256;

para.path_db = fullfile('data', 'MITindoor_categories');
para.fmt = '*.jpg';
para.path_svm_models = fullfile('data', sprintf('MITindoorft_model_%s.mat', experiment_index ));
para.path_rgnfeat = fullfile('data', sprintf('MITindoorft_VGGCNNfeat_%s', experiment_index ));
para.path_results = fullfile('data', sprintf('MITindoorft_results_%s.mat', experiment_index ));

mkdir(para.path_rgnfeat);

sceneft_CNNfeat(para);

dictionary_learning_MIT_shared(para);
