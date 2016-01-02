function [ A_norm ] = norm_matrix( A, n0, n1 )

A_norm = sum(A .^ n0, n1) .^ (1 / n0);

end