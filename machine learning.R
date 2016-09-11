#
#   Load required libraries
#
library(caret)
library(randomForest)
library(rpart)
library(knitr)
#
# Load Test data set
#
if (!exists("./pml-testing.csv")) {
    fileUrl <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
    download.file(fileUrl, destfile = "./pml-testing.csv", method = "curl")
}
testing <- read.table("./pml-testing.csv", sep = ",", header = TRUE, na.strings = c("NA", "#DIV/0!"))
#
# Load Training data set
#
if (!exists("./pml-training.csv")) {
    fileUrl <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
    download.file(fileUrl, destfile = "./pml-training.csv", method = "curl")
    dateDownloaded <- date()
    dateDownloaded
}
modelingData <- read.table("./pml-training.csv", sep = ",", header = TRUE, na.strings = c("NA", "#DIV/0!"))
#
#   Drop all empty columns (filled with NA's and hence not predicting anything)
#
modelingData <- modelingData[, !sapply(modelingData, function(x)all(is.na(x))), drop=F]
#
#   Drop all descriptive (non sensor) data, as these are obviously no predictors for the outcome
#
modelingData <- modelingData[,-(1:7)]
#
#   Drop near zero variance columns
#
nzv <- nearZeroVar(modelingData, saveMetrics=TRUE)
for(i in dim(nzv)[1]:1) {
    if(nzv[i,4]) {modelingData[,i] <- NULL}
}
#   Now we have a cleansed, trimmed dataset with only sensor data of sufficient variance
#
#   Replace NA sensor data with 0 (zero) to meet the input criteria of some modeling algorithms
#
modelingDataNNA <- modelingData
modelingDataNNA[is.na(modelingDataNNA)] <- 0
#
#   Devide the data into a training and validation set
#
set.seed(8)
inTrain = createDataPartition(modelingDataNNA$classe , p = 3/4) [[1]]
training = modelingDataNNA[ inTrain,] 
validate = modelingDataNNA[-inTrain,] 
#
# Which variables have most impact on the outcome (Class) (PCA) 
#   Do Principle Component Analysis
prComp <- prcomp(~ ., data = training[,-118], na.action=na.omit)
#   Print out the eigenvector/rotations first 5 rows and PCs
head(prComp$rotation[, 1:5], 5)
#
#   Which model (Try and compare)  
#
resultsTable <- data.frame(method = character(), accuracyTest = numeric(), accuracyVal = numeric())
#
#   Try Random Forest
#
modelRF <- train(classe ~ ., data = training, method = "rf")
predicRFtest <- predict(modelRF, training)
cmTestRF <- confusionMatrix(training$classe, predicRFtest)
# Validate the model with new input data  
predicRFval <- predict(modelRF, validate)
cmValRF <- confusionMatrix(validate$classe, predicRFval)
resultsTable <- data.frame(method = "RF", accuracyTest = cmTestRF$overall[1], accuracyVal = cmValRF$overall[1])
#
#   Try LDA with PCA pre-processing
#
modelLDAPCA <- train(classe ~ ., method="lda", preProcess="pca", data=training)
predictLDAPCAtest <- predict(modelLDAPCA, training)
cmTestLDAPCA <- confusionMatrix(training$classe, predictLDAPCAtest)
predictLDAPCAval <- predict(modelLDAPCA, validate)
cmValLDAPCA <- confusionMatrix(validate$classe, predictLDAPCAval)
resultsTable <- rbind(resultsTable, data.frame( method = "LDA-PCA", accuracyTest = cmTestLDAPCA$overall[1], accuracyVal = cmValLDAPCA$overall[1]))

#
#   Try RPART
#
modelRPART <- train(classe ~ ., method="rpart", data=training)
predictRPARTtest <- predict(modelRPART, training)
cmTestRPART <- confusionMatrix(training$classe, predictRPARTtest)
predictRPARTval <- predict(modelRPART, validate)
cmValRPART <- confusionMatrix(validate$classe, predictRPARTval)
resultsTable <- rbind(resultsTable, data.frame( method = "RPART", accuracyTest = cmTestRPART$overall[1], accuracyVal = cmValRPART$overall[1]))
#
#   Try LDA
#
modelLDA <- train(classe ~ ., method="lda", data=training)
predictLDAtest <- predict(modelLDA, training)
cmTestLDA <- confusionMatrix(training$classe, predictLDAtest)
predictLDAval <- predict(modelLDA, validate)
cmValLDA <- confusionMatrix(validate$classe, predictLDAval)
resultsTable <- rbind(resultsTable, data.frame( method = "LDA", accuracyTest = cmTestLDA$overall[1], accuracyVal = cmValLDA$overall[1]))
#
#   Try GBM
#
modelGBM <- train(classe ~ ., method="gbm", data=training, verbose=FALSE)
predictGBMtest <- predict(modelGBM, training)
cmTestGBM <- confusionMatrix(training$classe, predictGBMtest)
predictGBMval <- predict(modelGBM, validate)
cmValGBM <- confusionMatrix(validate$classe, predictGBMval)
resultsTable <- rbind(resultsTable, data.frame( method = "GBM", accuracyTest = cmTestGBM$overall[1], accuracyVal = cmValGBM$overall[1]))
#
#   Show results
#
kable(resultsTable[,1:3], align = "c", row.names = FALSE)
#
#   Show results confusion matrixes of best models
#
kable(cmValRF$table, align = "c", caption = "Confusion matrix RF. X = Reference, Y = Prediction")
kable(cmValGBM$table, align = "c", caption = "Confusion matrix GBM. X = Reference, Y = Prediction")
kable(cmValLDA$table, align = "c", caption = "Confusion matrix LDA. X = Reference, Y = Prediction")
#
# Stack the 3 top models with RF
# combine the prediction results and the true results into new data frame
combineData <- data.frame(RFpred=predicRFval, GBMpred=predictGBMval, LDApred=predictLDAval, classe=validate$classe)
# run a Random Forest (RF) model on the combined test data
modelCOMB <- train(classe ~., method="rf",data=combineData)
# use the resultant model to predict on the test set
predicCOMBval <- predict(modelCOMB, combineData)
cmCOMB <- confusionMatrix(combineData$classe, predicCOMBval)
#
#   Process the test data set with the best RF model.
#   Start with preprocessing the testing data to fit the model
#
testingPreped <- testing[,-(1:7)]
testingPreped[is.na(testingPreped)] <- 0
testingRF <- predict(modelRF, testingPreped)
testingRF
