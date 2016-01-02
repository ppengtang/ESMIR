function [SubImage] = ExtractSubImage(OriginalImage, X, Y, W, H)
    % extract
    SubImage = OriginalImage(Y : Y + H - 1, X : X + W - 1, :);
end