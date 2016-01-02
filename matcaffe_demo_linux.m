function [scores, maxlabel] = matcaffe_demo_linux( im, net, IMAGE_MEAN, IMAGE_DIM, CROPPED_DIM )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

% Set caffe mode
% if use_gpu
%     caffe.set_mode_gpu();
%     gpu_id = 0;  % we will use the first gpu in this demo
%     caffe.set_device(gpu_id);
% else
%     caffe.set_mode_cpu();
% end
% 
% phase = 'test'; % run with phase test (so that dropout isn't applied)
% 
% % Initialize a network
% net = caffe.Net(model_def_file, model_file, phase);

% prepare oversampled input
% input_data is Height x Width x Channel x Num
% tic;
input_data = {prepare_image(im, IMAGE_MEAN, IMAGE_DIM, CROPPED_DIM)};
% toc;

% do forward pass to get scores
% scores are now Channels x Num, where Channels == 1000
% tic;
% The net forward function. It takes in a cell array of N-D arrays
% (where N == 4 here) containing data of input blob(s) and outputs a cell
% array containing data from output blob(s)
scores = net.forward(input_data);
% toc;

scores = scores{1};
scores = squeeze(scores);
scores = mean(scores,2);

[~, maxlabel] = max(scores);

end


% ------------------------------------------------------------------------
function images = prepare_image(im, IMAGE_MEAN, IMAGE_DIM, CROPPED_DIM)
% ------------------------------------------------------------------------
% d = load('car_image_mean');
% IMAGE_MEAN = d.image_mean;      IMAGE_MEAN=IMAGE_MEAN(:,:,[3,2,1]);
% IMAGE_DIM = 256;
% CROPPED_DIM = 227;

% resize to fixed input size
im = single(im);
%im = imresize(im, [IMAGE_DIM IMAGE_DIM], 'bilinear');
im = imresize(im, [CROPPED_DIM, CROPPED_DIM], 'bilinear');
% permute from RGB to BGR (IMAGE_MEAN is already BGR)
im = im(:, :, [3 2 1]);
im = permute(im, [2 1 3]);
im = im - IMAGE_MEAN;
%im = imresize(im, [CROPPED_DIM, CROPPED_DIM], 'bilinear');
% im = im(:,:,[3 2 1]) - IMAGE_MEAN;

% oversample (4 corners, center, and their x-axis flips)
% images = zeros(CROPPED_DIM, CROPPED_DIM, 3, 10, 'single');
% indices = [0 IMAGE_DIM-CROPPED_DIM] + 1;
% curr = 1;
% for i = indices
%     for j = indices
%         images(:, :, :, curr) = ...
%             permute(im(i:i+CROPPED_DIM-1, j:j+CROPPED_DIM-1, :), [2 1 3]);
%         images(:, :, :, curr+5) = images(end:-1:1, :, :, curr);
%         curr = curr + 1;
%     end
% end
% center = floor(indices(2) / 2)+1;
% images(:,:,:,5) = ...
%     permute(im(center:center+CROPPED_DIM-1,center:center+CROPPED_DIM-1,:), ...
%     [2 1 3]);
% images(:,:,:,10) = images(end:-1:1, :, :, curr);
% 
% for c = 1:10
%     images(:, :, :, c) = images(:, :, :, c) - IMAGE_MEAN(:, :, :);
% end
images = single(im);
end
