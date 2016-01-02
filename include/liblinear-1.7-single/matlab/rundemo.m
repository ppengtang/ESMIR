[y,xt] = libsvmread('../heart_scale');
model=train(single(y), single(full(xt)))
[l,a]=predict(y, xt, model);

