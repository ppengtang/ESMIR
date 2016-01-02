function all_sift = extr_sift_for_codebook(patchsizes, gridsize, paths, N, M)


all_sift = zeros(N*M, 128, 'single');

rind = randperm( length(paths) );

tic
for n = 1:N
    fea_sift = func_extr_sift(paths{rind(n)}, patchsizes, gridsize); 
    rind2 = randperm( size(fea_sift.data, 1) );   
    all_sift( (n-1)*M+1:n*M, : ) = fea_sift.data( rind2(1:M), : );
    fprintf('extracting sift %d of %d, image id %d\n', n, N, rind(n));
end
toc



function fea_sift = func_extr_sift(im_name, patchsizes, gridsize)
    I = imread(im_name);
	if size(I, 3) > 1
		I = rgb2gray(I);
    end
    
    s = 500 / max( size(I) );
    if s < 1
        I = imresize( I, s );
    end
	
    fea_sift.data = [];
    fea_sift.x = [];
    fea_sift.y = [];
    
    for n = 1:length(patchsizes)
        [sift_arr, grid_x, grid_y] = dense_sift(I, patchsizes(n), gridsize);
        
        fea_sift.data = [fea_sift.data; sift_arr];
        fea_sift.x = [fea_sift.x; grid_x];
        fea_sift.y = [fea_sift.y; grid_y];
    end
    fea_sift.hgt = size(I,1);
    fea_sift.wid = size(I,2);
    