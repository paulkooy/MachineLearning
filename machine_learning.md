# Machine Learning
Paul van der Kooy  
September 1, 2016  

## Synopsis

One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, our goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. The goal of this project is to predict the manner in which they did the exercise. Testing different models showed that the Random Forest model (RF) outperformed the other models on the validation set. Further it could not by improved by stacking. 


```r
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
```

```
## [1] "Sun Sep 11 20:00:56 2016"
```

```r
modelingData <- read.table("./pml-training.csv", sep = ",", header = TRUE, na.strings = c("NA", "#DIV/0!"))
```

## Data & Exploratory Analysis

The data for this project come from this source: <http://groupware.les.inf.puc-rio.br/har> (see the section on the Weight Lifting Exercise Dataset). The manner in which the participants performed barbell lifts is logged in variable "classe". Values of "classe" are:

Class A:  A exactly according to the specification  
Class B:  throwing the elbows to the front  
Class C:  lifting the dumbbell only halfway  
Class D:  lowering the dumbbell only halfway  
Class E:  and throwing the hips to the front  

###     Approach

Use part of training set to validata/test the results and prepare the training data set for modeling.


```r
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
```

```
##                              PC1           PC2           PC3           PC4
## roll_belt           2.690077e-04 -3.125922e-02  4.274859e-02 -3.586351e-03
## pitch_belt          1.042889e-03 -1.367687e-03  2.195945e-02  3.060731e-03
## yaw_belt           -3.963920e-03 -3.168017e-02 -2.864137e-03 -1.094883e-02
## total_accel_belt   -1.466038e-04 -4.018029e-03  6.103488e-03 -5.074286e-04
## kurtosis_roll_belt -5.205831e-06 -9.879917e-06 -1.064272e-05 -9.796771e-05
##                              PC5
## roll_belt           4.497935e-02
## pitch_belt         -6.866696e-03
## yaw_belt            6.268790e-02
## total_accel_belt    5.770183e-03
## kurtosis_roll_belt -7.680625e-06
```

```r
#
#   Which model (Try and compare)  
#
resultsTable <- data.frame(method = character(), accuracyTest = numeric(), accuracyVal = numeric())
```

## Processing

The training set is ordered by class, hence if we reserve a subset for model validation, this should be a randomly chosen subset. Furthermore, the PCA shows because of the ordering a correlation with the row- and window number and raw timestamps, which are obviously no predictors of the outcome (the manner in which the exercise was performed). For that reason we excluded these variables so that only accelerometers with available input remain as predictors.

Based on the resulting dataset and the nature of the data and problem (classification) we tested the following classification algoritmes:  
* Random Forest (RF)  
* Generalized Boosted Regression (GBM)  
* Linear Discriminant Analysis (LDA)  
* Linear Discriminant Analysis (LDA) with Principal Component Analysis (PCA) 
* Recursive Partitioning and Regression Trees (RPART)  
* A stacked model of (COMB) of the top 3 models (RF, GB and LDA)  
GLM was not used as this methods can only process 2-class outcomes, whereas our outcome contains 5 classes. Below the table with the accuracy of the models on the training and validation set for each testing method:

###     Which model (Try and compare)


```r
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
```


```r
#
#   Try GBM
#
modelGBM <- train(classe ~ ., method="gbm", data=training, verbose=FALSE)
```

```
## Loading required package: gbm
```

```
## Loading required package: survival
```

```
## 
## Attaching package: 'survival'
```

```
## The following object is masked from 'package:caret':
## 
##     cluster
```

```
## Loading required package: splines
```

```
## Loading required package: parallel
```

```
## Loaded gbm 2.1.1
```

```
## Loading required package: plyr
```

```r
predictGBMtest <- predict(modelGBM, training)
cmTestGBM <- confusionMatrix(training$classe, predictGBMtest)
predictGBMval <- predict(modelGBM, validate)
cmValGBM <- confusionMatrix(validate$classe, predictGBMval)
resultsTable <- rbind(resultsTable, data.frame( method = "GBM", accuracyTest = cmTestGBM$overall[1], accuracyVal = cmValGBM$overall[1]))
```


```r
#
#   Try LDA
#
modelLDA <- train(classe ~ ., method="lda", data=training)
```

```
## Loading required package: MASS
```

```
## Warning in lda.default(x, grouping, ...): variables are collinear

## Warning in lda.default(x, grouping, ...): variables are collinear

## Warning in lda.default(x, grouping, ...): variables are collinear

## Warning in lda.default(x, grouping, ...): variables are collinear

## Warning in lda.default(x, grouping, ...): variables are collinear

## Warning in lda.default(x, grouping, ...): variables are collinear

## Warning in lda.default(x, grouping, ...): variables are collinear

## Warning in lda.default(x, grouping, ...): variables are collinear

## Warning in lda.default(x, grouping, ...): variables are collinear

## Warning in lda.default(x, grouping, ...): variables are collinear

## Warning in lda.default(x, grouping, ...): variables are collinear

## Warning in lda.default(x, grouping, ...): variables are collinear

## Warning in lda.default(x, grouping, ...): variables are collinear

## Warning in lda.default(x, grouping, ...): variables are collinear

## Warning in lda.default(x, grouping, ...): variables are collinear

## Warning in lda.default(x, grouping, ...): variables are collinear

## Warning in lda.default(x, grouping, ...): variables are collinear

## Warning in lda.default(x, grouping, ...): variables are collinear

## Warning in lda.default(x, grouping, ...): variables are collinear

## Warning in lda.default(x, grouping, ...): variables are collinear

## Warning in lda.default(x, grouping, ...): variables are collinear

## Warning in lda.default(x, grouping, ...): variables are collinear

## Warning in lda.default(x, grouping, ...): variables are collinear

## Warning in lda.default(x, grouping, ...): variables are collinear

## Warning in lda.default(x, grouping, ...): variables are collinear

## Warning in lda.default(x, grouping, ...): variables are collinear
```

```r
predictLDAtest <- predict(modelLDA, training)
cmTestLDA <- confusionMatrix(training$classe, predictLDAtest)
predictLDAval <- predict(modelLDA, validate)
cmValLDA <- confusionMatrix(validate$classe, predictLDAval)
resultsTable <- rbind(resultsTable, data.frame( method = "LDA", accuracyTest = cmTestLDA$overall[1], accuracyVal = cmValLDA$overall[1]))
```


```r
#
#   Try LDA with PCA pre-processing
#
modelLDAPCA <- train(classe ~ ., method="lda", preProcess="pca", data=training)
predictLDAPCAtest <- predict(modelLDAPCA, training)
cmTestLDAPCA <- confusionMatrix(training$classe, predictLDAPCAtest)
predictLDAPCAval <- predict(modelLDAPCA, validate)
cmValLDAPCA <- confusionMatrix(validate$classe, predictLDAPCAval)
resultsTable <- rbind(resultsTable, data.frame( method = "LDA-PCA", accuracyTest = cmTestLDAPCA$overall[1], accuracyVal = cmValLDAPCA$overall[1]))
```


```r
#
#   Try RPART
#
modelRPART <- train(classe ~ ., method="rpart", data=training)
predictRPARTtest <- predict(modelRPART, training)
cmTestRPART <- confusionMatrix(training$classe, predictRPARTtest)
predictRPARTval <- predict(modelRPART, validate)
cmValRPART <- confusionMatrix(validate$classe, predictRPARTval)
resultsTable <- rbind(resultsTable, data.frame( method = "RPART", accuracyTest = cmTestRPART$overall[1], accuracyVal = cmValRPART$overall[1]))
```

###     How good are the results (Accuracy and Confusion matrix)  


 method     accuracyTest    accuracyVal 
---------  --------------  -------------
   RF        1.0000000       0.9951060  
   GBM       0.9740454       0.9661501  
   LDA       0.7077728       0.6984095  
 LDA-PCA     0.5214703       0.5179445  
  RPART      0.4943606       0.4991843  

Above table clearly shows that RF, GBM and LDA score higher than the others. The preferred method is RF as this delivers superior results. To determine whether stacking the top 3 methods is worthwhile the confusion matrix was observed.


Table: Confusion matrix RF. X = Reference, Y = Prediction

       A       B      C      D      E  
---  ------  -----  -----  -----  -----
A     1393     1      1      0      0  
B      4      944     1      0      0  
C      0       6     848     1      0  
D      0       0      3     800     1  
E      0       0      2      4     895 



Table: Confusion matrix GBM. X = Reference, Y = Prediction

       A       B      C      D      E  
---  ------  -----  -----  -----  -----
A     1371    14      4      4      2  
B      29     891    27      0      2  
C      0      28     817    10      0  
D      3       3     19     776     3  
E      1       5      4      8     883 



Table: Confusion matrix LDA. X = Reference, Y = Prediction

       A       B      C      D      E  
---  ------  -----  -----  -----  -----
A     1122    34     122    111     6  
B     149     602    116    31     51  
C      80     71     583    101    20  
D      54     41     105    567    37  
E      33     168    72     77     551 

Because of the high initial score of RF a model improvement by stacking looks doubtful. We do however give it a try with the top 3 models.


```r
# Stack the 3 top models with RF
# combine the prediction results and the true results into new data frame
combineData <- data.frame(RFpred=predicRFval, GBMpred=predictGBMval, LDApred=predictLDAval, classe=validate$classe)
# run a Random Forest (RF) model on the combined test data
modelCOMB <- train(classe ~., method="rf",data=combineData)
# use the resultant model to predict on the test set
predicCOMBval <- predict(modelCOMB, combineData)
cmCOMB <- confusionMatrix(combineData$classe, predicCOMBval)
```
The accuracy of the stacked/combine model is 0.9957178 does not improve the accuracy of the RF model of  0.995106  

## Conclusion

The Random Forest model (RF) is clearly the best model for the analysis of the sensor data of the Weight Lifting Exercise Dataset. 
Finally the best model was used on the testing set with the following result:


```r
#
#   Process the test data set with the best RF model.
#   Start with preprocessing the testing data to fit the model
#
testingPreped <- testing[,-(1:7)]
testingPreped[is.na(testingPreped)] <- 0
testingRF <- predict(modelRF, testingPreped)
testingRF
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```

##  Appendix

R-code, Rmd file and HTML can be found in the following repository: 
<https://github.com/paulkooy/MachineLearning>


```r
sessionInfo()
```

```
## R version 3.3.1 (2016-06-21)
## Platform: x86_64-apple-darwin13.4.0 (64-bit)
## Running under: OS X 10.11.6 (El Capitan)
## 
## locale:
## [1] en_US.UTF-8/en_US.UTF-8/en_US.UTF-8/C/en_US.UTF-8/en_US.UTF-8
## 
## attached base packages:
## [1] parallel  splines   stats     graphics  grDevices utils     datasets 
## [8] methods   base     
## 
## other attached packages:
##  [1] MASS_7.3-45         plyr_1.8.4          gbm_2.1.1          
##  [4] survival_2.39-4     knitr_1.14          rpart_4.1-10       
##  [7] randomForest_4.6-12 caret_6.0-71        ggplot2_2.1.0      
## [10] lattice_0.20-33    
## 
## loaded via a namespace (and not attached):
##  [1] Rcpp_0.12.6        compiler_3.3.1     highr_0.6         
##  [4] formatR_1.4        nloptr_1.0.4       class_7.3-14      
##  [7] iterators_1.0.8    tools_3.3.1        digest_0.6.10     
## [10] lme4_1.1-12        evaluate_0.9       nlme_3.1-128      
## [13] gtable_0.2.0       mgcv_1.8-12        Matrix_1.2-6      
## [16] foreach_1.4.3      yaml_2.1.13        SparseM_1.7       
## [19] e1071_1.6-7        stringr_1.1.0      MatrixModels_0.4-1
## [22] stats4_3.3.1       grid_3.3.1         nnet_7.3-12       
## [25] rmarkdown_1.0      minqa_1.2.4        reshape2_1.4.1    
## [28] car_2.1-3          magrittr_1.5       scales_0.4.0      
## [31] codetools_0.2-14   htmltools_0.3.5    pbkrtest_0.4-6    
## [34] colorspace_1.2-6   quantreg_5.26      stringi_1.1.1     
## [37] munsell_0.4.3
```

