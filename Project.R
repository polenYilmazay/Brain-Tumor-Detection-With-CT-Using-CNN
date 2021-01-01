library(keras)

MR_list<-c("Tumorlu","Tumorsuz")

output_n <- length(MR_list)

img_width <- 20
img_height <- 20
target_size <- c(img_width, img_height)

channels <- 3

train_image_files_path<-"D:/Staj2/DICOM1/train1"


train_data_gen = image_data_generator(
  rescale = 1/255
)


train_image_array_gen <- flow_images_from_directory(train_image_files_path, 
                                                    train_data_gen,
                                                    target_size = target_size,
                                                    class_mode = "categorical",
                                                    classes = MR_list,
                                                    seed = 42)

valid_image_files_path<-"D:/Staj2/DICOM1/test1"

valid_data_gen <- image_data_generator(
  rescale = 1/255
)  


# validation images
valid_image_array_gen <- flow_images_from_directory(valid_image_files_path, 
                                                    valid_data_gen,
                                                    target_size = target_size,
                                                    class_mode = "categorical",
                                                    classes = MR_list,
                                                    seed = 42)

#cat("Number of images per class:")

#table(factor(train_image_array_gen$classes))

#cat("\nClass label vs index mapping:\n")

train_image_array_gen$class_indices

MR_classes_indices <- train_image_array_gen$class_indices
save(MR_classes_indices,file = "D:/Staj2/DICOM1/MR_classes_indices.RData")

train_samples <- train_image_array_gen$n
# number of validation samples

valid_samples <- valid_image_array_gen$n

# define batch size and number of epochs
batch_size <- 32
epochs <- 25


model <- keras_model_sequential()

# add layers
model %>%
  layer_conv_2d(filter = 32, kernel_size = c(3,3), padding = "same", input_shape = c(img_width, img_height, channels)) %>%
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
  layer_dense(output_n) %>% 
  layer_activation("softmax")

# compile
model %>% compile(
  loss = "categorical_crossentropy",
  optimizer = optimizer_rmsprop(lr = 0.0001, decay = 1e-6),
  metrics = "accuracy"
)

# fit
hist <- model %>% fit_generator(
  train_image_array_gen,
  steps_per_epoch = as.integer(train_samples / batch_size), 
  epochs = epochs, 
  validation_data = valid_image_array_gen,
  validation_steps = as.integer(valid_samples / batch_size),
  verbose = 2,
)

plot(hist)

