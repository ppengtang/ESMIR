function win_predict(label_vector, instance_matrix, options, model_name)

libsvmwrite(model_name, label_vector, instance_matrix);