library(keras)   # for working with neural nets
library(lime)    # for explaining models
library(magick)  # for preprocessing images
library(ggplot2) # for additional plotting
library(EBImage)


MR_list<-c("Tumorlu","Tumorsuz")

train_image_files_path <- "D:/Staj2/DICOM1/train1"


train_data_gen = image_data_generator(
  rescale = 1/255
)

 

train_image_array_gen <- flow_images_from_directory(train_image_files_path, 
                                                    train_data_gen,
                                                    target_size = c(256,256),
                                                    class_mode = "categorical",
                                                    classes = MR_list,
                                                    seed = 42)



valid_image_files_path <- "D:/Staj2/DICOM1/test1"

valid_data_gen <- image_data_generator(
  rescale = 1/255,shear_range=0.2, zoom_range=0.2, horizontal_flip=TRUE
) 

# validation images
valid_image_array_gen <- flow_images_from_directory(valid_image_files_path, 
                                                    valid_data_gen,
                                                    target_size = c(256,256),
                                                    class_mode = "categorical",
                                                    classes = MR_list,
                                                    seed = 42)


## Class label vs index mapping:
train_image_array_gen$class_indices

MR_classes_indices <- train_image_array_gen$class_indices
save(MR_list,file = "/Staj2/DICOM1/MR_classes_indices.RData")

train_samples <- train_image_array_gen$n

valid_samples <- valid_image_array_gen$n


#model <- application_vgg16(weights = "imagenet", include_top = TRUE)
#model


model2 <- keras_model_sequential()
model2 %>%
  layer_conv_2d(filter = 32, kernel_size = c(3,3), padding = "same", input_shape = c(20, 20, 3)) %>%
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
  
  # Outputs from dense layer are projected onto output layer
  layer_dense(2) %>% 
  layer_activation("softmax")

# compile
model2 %>% compile(
  loss = "categorical_crossentropy",
  optimizer = optimizer_rmsprop(lr = 0.0001, decay = 1e-6),
  metrics = "accuracy"
)


hist<-model %>% 
  fit_generator(
    train_image_array_gen,
    steps_per_epoch = 220,
    epochs = 25,
    validation_data = valid_image_array_gen
  )

test_image_files_path <- "/Staj2/DICOM1/test1"

img <- image_read('D:/Staj2/DICOM1/test1/Tumorlu/IM000025.jpg')
img_path <- file.path(test_image_files_path, "Tumorlu", 'IM000025.jpg')
image_write(img, img_path)

img2<-image_read('D:/Staj2/DICOM1/test1/Tumorsuz/IM000029.jpg')
img_path2<-file.path(test_image_files_path,"Tumorsuz","IM000029.jpg")
image_write(img2,img_path2)

plot_superpixels(img_path, n_superpixels = 35, weight = 20)
plot_superpixels(img_path2, n_superpixels = 35, weight = 20)

image_prep <- function(x) {
  arrays <- lapply(x, function(path) {
    img <- image_load(path, target_size = c(224,224))
    x <- image_to_array(img)
    x <- array_reshape(x, c(1, dim(x)))
    x <- imagenet_preprocess_input(x)
  })
  do.call(abind::abind, c(arrays, list(along = 1)))
}


#res <- predict(model2, image_prep(c(img_path, img_path2)))
#imagenet_decode_predictions(res)

model2_labels <- readRDS(system.file('extdata', 'imagenet_labels.rds', package = 'lime'))


test_datagen <- image_data_generator(rescale = 1/255)

test_generator <- flow_images_from_directory(
  test_image_files_path,
  test_datagen,
  target_size = c(20, 20),
  class_mode = 'categorical')

predictions <- as.data.frame(predict_generator(model2, test_generator, steps = 1))

load("/Staj2/DICOM1/MR_classes_indices.RData")
MR_classes_indices <- train_image_array_gen$class_indices
MR_classes_indices_df <- data.frame(indices = unlist(MR_classes_indices))
MR_classes_indices_df <-MR_classes_indices_df[order(MR_classes_indices_df$indices), , drop = FALSE]
colnames(predictions) <- rownames(MR_classes_indices_df)

t(round(predictions, digits = 2))

for (i in 1:nrow(predictions)) {
  cat(i, ":")
  print(unlist(which.max(predictions[i ,])))
}

image_prep2 <- function(x) {
  arrays <- lapply(x, function(path) {
    img <- image_load(path, target_size = c(20, 20))
    x <- image_to_array(img)
    x <- reticulate::array_reshape(x, c(1, dim(x)))
    x <- x / 255
  })
  do.call(abind::abind, c(arrays, list(along = 1)))
}

MR_classes_indices_l <- rownames(MR_classes_indices_df)
names(MR_classes_indices_l) <- unlist(MR_classes_indices)
MR_classes_indices_l









