function f = extr_patch_dfeat_linux( im, rect, para, net )

patch = im( rect(1):rect(2), rect(3):rect(4), : );

patch = imresize(patch, [256, 256]);

f = matcaffe_demo_linux(patch, net, para.IMAGE_MEAN, para.IMAGE_DIM, para.CROPPED_DIM);

f = f / (norm(f,2));

end