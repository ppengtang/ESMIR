% function [inds] = resample(w,N)
% 
% Sample N indexes from the weight vector w whose length may be greater
% than N. 
%
% Copyright August, 2007
% Author: Frank Wood fwood@gatsby.ucl.ac.uk
function [inds] = resample(w,N)



M = length(w);

ni = randperm(M); 
w = w(ni);

inds = zeros(1,N);

w = w/sum(w);       % normalize
cdf = cumsum(w);

cdf(end) = 1;

p = linspace(rand*(1/N),1,N);

picked = zeros(1,M);
j=1;
for i=1:N
    while j<M && cdf(j)<p(i)
        j=j+1;
    end
    picked(j) = picked(j)+1;
end

rind=1;
for i=1:M
    if(picked(i)>0)
        for j=1:picked(i)
            inds(rind) = ni(i);
            rind=rind+1;
        end
    end
end



