function sample = wieght_sample(data, w, k)
%
% Sample randomly, without replacements, from data
%
% Usage:
%   sample = wieght_sample(data, w, k)
%
% Inputs:
%   - data: a data matrix to sample from, treats the columns as vectors and
%                sample from them
%   - w: a wieght per sample
%   - k: how many elements to sample
%
% Output:
%   - sample: the sampled data
% 
%
%   Copyright (c) Bagon Shai
%   Department of Computer Science and Applied Mathmatics
%   Wiezmann Institute of Science
%   http://www.wisdom.weizmann.ac.il/
%  
%   Permission is hereby granted, free of charge, to any person obtaining a copy
%   of this software and associated documentation files (the "Software"), to deal
%   in the Software without restriction, subject to the following conditions:
%  
%   1. The above copyright notice and this permission notice shall be included in 
%       all copies or substantial portions of the Software.
%   2. No commercial use will be done with this software.
%   3. If used in an academic framework - a proper citation must be included.
%  
%   The Software is provided "as is", without warranty of any kind.
%  
%   Jul. 2007
%   

if (length(size(data)) > 2)
    error('weight_sample:data',...
        'unsupported dimensionality of data, currently only 1D or 2D data is supported');
end
if size(data,2) == 1 && size(data,1) > 1
    transpose_flag = true;
    data = data';
else
    transpose_flag = false;
end

% what is the data size
n = size(data,2);

% if no weights are given - sample uniformly
if nargin == 2
    k = w;
    w = [];
end
if isempty(w) 
    w = ones(n,1)./n;
end

if any(w<0)
    error('weight_sample:w','weights must be non negative');
end
% avoid zero division
if all(w==0)
    w = ones(n,1)./n;
end

% convert weights to double
if ~ strcmp(class(w), 'double')
    w = double(w);
end
% make weights a pdf
w = w./sum(w);
si = wieght_sample_mex(n, w, k);

sample = data(:,si);

if transpose_flag
    sample = sample';
end