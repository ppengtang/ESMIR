function [FeatureVector] = FeatureExtraction(I, GrayImage, LabImageU8, Rectangle, params)


    Histogram = []; 
    LbpFeature = []; 
    hogFeature = []; 
    
    FeatureVector = [];
    
    % compute LBP feature
    SubGrayImage = GrayImage(Rectangle(1) : Rectangle(2), Rectangle(3) : Rectangle(4));
    resz_SubGrayImage = imresize(SubGrayImage, [params.base_patch, params.base_patch]);
    if params.lbp_on
        [LbpFeature] = lbp_mex(int32(resz_SubGrayImage), 1);
        
        FeatureVector = LbpFeature;
    end
    
    % compute Hog feature
    if params.hog_on
        patch = I(Rectangle(1) : Rectangle(2), Rectangle(3) : Rectangle(4), : );
        patch = imresize(patch, [params.base_patch, params.base_patch]);
        desc = pffhog( double(patch), params.hog_sz );
        hogFeature = desc(:);
%         if norm(hogFeature) <= params.hog_thre
%             hogFeature(:) = 0 / 0;
%         end
        FeatureVector = [ FeatureVector, hogFeature' ];
    end
    
    % compute Lab histogram
    if params.lab_on
        SubColorImage = LabImageU8(Rectangle(1) : Rectangle(2), Rectangle(3) : Rectangle(4), :);
        [Histogram] = LabHistogram(SubColorImage);
        
        FeatureVector = [ FeatureVector, Histogram ];
    end
    
end