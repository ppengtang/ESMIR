function [LbpFeature] = LbpFeatureExtraction(GrayImage)
    global MAPPING;

    % LBP histogtam calculation
    LbpFeature = lbp(GrayImage, 2, 16, MAPPING, 'nh');
end