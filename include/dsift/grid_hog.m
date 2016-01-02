function sift_arr = grid_hog(I, grid)

if ~exist('grid', 'var')
    grid = [4, 4];
end


I = double(I);
I = mean(I,3);
I = I /max(I(:));


% parameters
num_angles = 8;
alpha = 9; %% parameter for attenuation of angles (must be odd)


sigma_edge = 1;
angle_step = 2 * pi / num_angles;
angles = 0:angle_step:2*pi;
angles(num_angles+1) = []; % bin centers

[hgt wid] = size(I);

[G_X,G_Y]=gen_dgauss(sigma_edge);
I_X = filter2(G_X, I, 'same'); % vertical edges
I_Y = filter2(G_Y, I, 'same'); % horizontal edges
I_mag = sqrt(I_X.^2 + I_Y.^2); % gradient magnitude
I_theta = atan2(I_Y,I_X+eps);
I_theta(find(isnan(I_theta))) = 0; % necessary????


% make orientation images
I_orientation = zeros([hgt, wid, num_angles], 'single');

% for each histogram angle
cosI = cos(I_theta);
sinI = sin(I_theta);
for a=1:num_angles
    % compute each orientation channel
    tmp = (cosI*cos(angles(a))+sinI*sin(angles(a))).^alpha;
    tmp = tmp .* (tmp > 0);

    % weight by magnitude
    I_orientation(:,:,a) = tmp .* I_mag;
end

% intergal image for each orientation
int_ori = zeros( size(I_orientation) );
for a = 1:num_angles
    int_ori(:,:,a) = cumsum(cumsum(double( I_orientation(:,:,a) )),2);
end


sift_arr = zeros( grid(1), grid(2), num_angles, 'single' );
sy = round( linspace( 1, hgt, grid(1)+1 ) );
sx = round( linspace( 1, wid, grid(2)+1 ) );

for i = 1:grid(1)
    for j = 1:grid(2)
        sift_arr(i,j,:) = int_ori(sy(i),sx(j),:) + int_ori(sy(i+1),sx(j+1),:) - int_ori(sy(i),sx(j+1),:) - int_ori(sy(i+1),sx(j),:);
    end
end

sift_arr = normalize_sift(sift_arr(:)');
sift_arr = sift_arr(:);



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function x = pad(x, D)

[nrows, ncols, cols] = size(sift_arr);
hgt = nrows+2*D;
wid = ncols+2*D;
PADVAL = 0;

x = [repmat(PADVAL, [hgt Dx cols]) ...
    [repmat(PADVAL, [Dy ncols cols]); x; repmat(PADVAL, [Dy-1 ncols cols])] ...
    repmat(PADVAL, [hgt Dx-1 cols])];

