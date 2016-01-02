function [Histogram] = LabHistogram(LabImage)
    % histograms of three channels
    nBins = 32;
    LHist = imhist(LabImage(:,:,1), nBins); 
    aHist = imhist(LabImage(:,:,2), nBins);
    bHist = imhist(LabImage(:,:,3), nBins);
    
    % normalization
    [m n d] = size(LabImage);
    PixelCount = m * n;
    LHist = LHist / double(PixelCount);
    aHist = aHist / double(PixelCount);
    bHist = bHist / double(PixelCount);
    
    % combination
    Histogram = [LHist' aHist' bHist'];
end