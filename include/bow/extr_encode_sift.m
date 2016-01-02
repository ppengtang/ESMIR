
function code_map = extr_encode_sift(I, cb)
    
	if size(I, 3) > 1
		I = rgb2gray(I);
    end
    
    patchsizes = cb.patchsizes;
    gridsize   = cb.gridsize;
	
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
    
    help_code = coding(cb.dict, fea_sift.data);
    code_map = rmfield(fea_sift, 'data');
    code_map.code = help_code;
    



 
	
	
	
	
	
	