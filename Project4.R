#devtools::install_github("rstudio/reticulate")
#install.packages("keras")
library(tensorflow)
library(keras)

MR_list<-c("Tumorlu","Tumorsuz")

                        #DATA
training_path<-"D:/Staj2/DICOM1/train1"



train_datagen <- image_data_generator(rescale = 1/255)



training_set <- flow_images_from_directory(training_path,
                                           train_datagen,
                                           target_size = c(20,20),
                                           batch_size = 32,
                                           class_mode="categorical",
                                           classes = MR_list,
                                           seed = 42)


testing_path<-"D:/Staj2/DICOM1/test1"

test_datagen<-image_data_generator(rescale = 1/255)

test_set<-flow_images_from_directory(testing_path,
                                     test_datagen,
                                     target_size = c(20,20),
                                     batch_size = 32,
                                     class_mode="categorical",
                                     classes = MR_list,
                                     seed = 42
                                     )

training_set$class_indices

MR_classes_indices <- training_set$class_indices
save(MR_classes_indices,file = "/Staj2/DICOM1/MR_classes_indices.RData")

train_samples <- training_set$n
test_samples <- test_set$n

batch_size<-32
epochs<-25

                         #MODEL
model <- keras_model_sequential()

# add layers
model %>%
  layer_conv_2d(filter = 32, kernel_size = c(3,3), padding = "same", input_shape = c(20, 20,3)) %>%
  layer_activation("relu") %>%
  
  # Second hidden layer
  layer_conv_2d(filter = 16, kernel_size = c(3,3), padding = "same") %>%
  layer_activation_leaky_relu(0.5) %>%
  layer_batch_normalization() %>%
  
  # Use max pooling
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_dropout(0.25) %>%
  
  # Flatten max filtered output into feature vector 
  # and feed into dense layer
  layer_flatten() %>%
  layer_dense(100) %>%
  layer_activation("relu") %>%
  layer_dropout(0.5) %>%
  layer_dense(2) %>% 
  layer_activation("softmax")

# compile
model %>% compile(
  loss = "categorical_crossentropy",
  optimizer = optimizer_rmsprop(lr = 0.0001, decay = 1e-6),
  metrics = "accuracy"
)


                           ##BURADAN SONRASI HATALI.IN USER CODE HATASI ALIYORUM
hist <- model %>% fit_generator(
    training_set,
    steps_per_epoch = as.integer(train_samples / batch_size), 
    epochs = epochs, 
    validation_data = test_datagen,
    validation_steps = as.integer(test_samples / batch_size),
    verbose = 2,
  )

test_image<-image_load("D:/Staj2/DICOM1/prediction/1.jpg",target_size = c(64,64))
test_image<-image_to_array(test_image)
test_image<-k_expand_dims(test_image,axis=0)
result<-model %>% predict(test_image)
training_set$class_indices


if (result[0][0]==1){
  prediction = 'yes'
} else prediction='no'


