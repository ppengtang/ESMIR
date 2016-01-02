function [SiftFeature] = SiftFeatureExtraction(GrayImage, CenterX, CenterY)
    % initialization
    fc = [CenterX; CenterY; 3; 0];
    
    % sift calculation
    [f, SiftFeature] = vl_sift(GrayImage, 'frames', fc);
    
    % display
    % h = vl_plotsiftdescriptor(SiftFeature, f);  
    % set(h, 'color', 'b') ;
    
    % normalization
    L = norm (double(SiftFeature)) + 1e-20;
    SiftFeature = double(SiftFeature) / L;
    SiftFeature = SiftFeature';
end