function [FeatureVector] = FeatureExtraction_wSift(GrayImage, LabImageU8, Rectangle)
    % compute SIFT feature
%    CenterX = Rectangle(1) + Rectangle(3) / 2;
%    CenterY = Rectangle(2) + Rectangle(4) / 2;
%    [SiftFeature] = SiftFeatureExtraction(SingleGrayImage, CenterX,
%    CenterY);
    
    % compute LBP feature
    SubGrayImage = ExtractSubImage(GrayImage, Rectangle(1), Rectangle(2), Rectangle(3), Rectangle(4));
    resz_SubGrayImage = imresize(SubGrayImage, [64, 64]); 
    
    [LbpFeature] = lbp_mex(int32(resz_SubGrayImage), 1);
%    SubGrayImageR = imresize(SubGrayImage, [64, 64]); 
    
    % hogFeature = HoG(double(resz_SubGrayImage), [8, 12, 2, 0, .2]);
    siftFeature = vl_sift(single(resz_SubGrayImage), 'frames', [33; 33; 16/3; 0], 'FloatDescriptors'); 
    siftFeature = siftFeature / (norm(siftFeature) + 1e-20); 
    % compute Lab histogram
    SubColorImage = ExtractSubImage(LabImageU8, Rectangle(1), Rectangle(2), Rectangle(3), Rectangle(4));
    [Histogram] = LabHistogram(SubColorImage);
    
    % combination
    FeatureVector = [siftFeature(:); LbpFeature'; Histogram'];
    % FeatureVector = FeatureVector';
end