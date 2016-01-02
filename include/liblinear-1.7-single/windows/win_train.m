function win_train( label_vector, instance_matrix, options, model_name )

libsvmwrite(model_name, label_vector, instance_matrix);

eval(['!train ', options, ' ', model_name]);