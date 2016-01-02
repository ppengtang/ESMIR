function codebook = generate_codebook(all_sift, K)

fprintf('kmeansing...\n');
[dict, assign] = vl_kmeans(all_sift', K, 'verbose', 'Algorithm', 'ELKAN');

for n = 1:size(dict, 2)
    code = dict(:,n);
    ind = assign == n;
    sigma(n) = stdv( all_sift(ind,:), code' );
end

codebook.dict = dict';
codebook.sigma = sigma';
