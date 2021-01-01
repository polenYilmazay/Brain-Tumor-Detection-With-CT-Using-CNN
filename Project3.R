library(oro.dicom)
library(oro.nifti)
library(DICOM)
library(keras)
library(EBImage)


setwd('D:/Staj2/DICOM1/train1')
train<-list()
  train <- readDICOM('D:/Staj2/DICOM1/train1',verbose = TRUE)

setwd('D:/Staj2/DICOM1/test1')
test<-list()
  test <- readDICOM('D:/Staj2/DICOM1/test1',verbose = TRUE)
  


train<-combine(train$img)
test<-combine(test$img)

trainy<-c(1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0)
testy<-c(01,1,1,1,1,0,0,0,0)

trainLabels<-to_categorical(trainy)
testLabels<-to_categorical(testy)


model <- keras_model_sequential() %>% 
  layer_conv_2d(filters = 32, kernel_size = c(3,3), activation = "relu", 
                input_shape = c(32,32,3)) %>% 
  layer_max_pooling_2d(pool_size = c(2,2)) %>% 
  layer_conv_2d(filters = 64, kernel_size = c(3,3), activation = "relu") %>% 
  layer_max_pooling_2d(pool_size = c(2,2)) %>% 
  layer_conv_2d(filters = 64, kernel_size = c(3,3), activation = "relu")

summary(model)

model %>% 
  layer_flatten() %>% 
  layer_dense(units = 64, activation = "relu") %>% 
  layer_dense(units = 10, activation = "softmax")

summary(model)

model %>% compile(
  optimizer = "adam",
  loss = "categorical_crossentropy",
  metrics = "accuracy"
)

# Fit model
history <- model %>%
  fit(train$img,
      trainLabels,
      epochs = 20,
      batch_size = 32,
      validation_split = 0.2)
#validation_data = list(test, testLabels))


plot(history)



# Evaluation & Prediction - train data
model %>% evaluate(train, trainLabels)
pred <- model %>% predict_classes(train)
table(Predicted = pred, Actual = trainy)

prob <- model %>% predict_proba(train)
cbind(prob, Predicted_class = pred, Actual = trainy)

# Evaluation & Prediction - test data
model %>% evaluate(test, testLabels)
pred <- model %>% predict_classes(test)
table(Predicted = pred, Actual = testy)

prob <- model %>% predict_proba(test)
cbind(prob, Predicted_class = pred, Actual = testy)




















