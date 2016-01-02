function compile_wieght_sample()
% This function compiles the mex library required for weight_sample
% function. 
% Please make sure mex is properly set on your machine:
% to do so simply type
% >> mex -setup
% and follow the instructions.
%

mex -O wieght_sample_mex.cpp