Titanic\_submission
================
Avirat Gaikwad
October 22, 2017

Installing the required packages

``` r
library(rpart)
```

    ## Warning: package 'rpart' was built under R version 3.4.2

``` r
library(randomForest)
```

    ## Warning: package 'randomForest' was built under R version 3.4.2

    ## randomForest 4.6-12

    ## Type rfNews() to see new features/changes/bug fixes.

``` r
library(tree)
```

    ## Warning: package 'tree' was built under R version 3.4.2

``` r
library(e1071)
```

    ## Warning: package 'e1071' was built under R version 3.4.2

``` r
train<- read.csv("train.csv")
test<- read.csv("test.csv")
str(train)
```

    ## 'data.frame':    891 obs. of  12 variables:
    ##  $ PassengerId: int  1 2 3 4 5 6 7 8 9 10 ...
    ##  $ Survived   : int  0 1 1 1 0 0 0 0 1 1 ...
    ##  $ Pclass     : int  3 1 3 1 3 3 1 3 3 2 ...
    ##  $ Name       : Factor w/ 891 levels "Abbing, Mr. Anthony",..: 109 191 358 277 16 559 520 629 417 581 ...
    ##  $ Sex        : Factor w/ 2 levels "female","male": 2 1 1 1 2 2 2 2 1 1 ...
    ##  $ Age        : num  22 38 26 35 35 NA 54 2 27 14 ...
    ##  $ SibSp      : int  1 1 0 1 0 0 0 3 0 1 ...
    ##  $ Parch      : int  0 0 0 0 0 0 0 1 2 0 ...
    ##  $ Ticket     : Factor w/ 681 levels "110152","110413",..: 524 597 670 50 473 276 86 396 345 133 ...
    ##  $ Fare       : num  7.25 71.28 7.92 53.1 8.05 ...
    ##  $ Cabin      : Factor w/ 148 levels "","A10","A14",..: 1 83 1 57 1 1 131 1 1 1 ...
    ##  $ Embarked   : Factor w/ 4 levels "","C","Q","S": 4 2 4 4 4 3 4 4 4 2 ...

``` r
summary(test)
```

    ##   PassengerId         Pclass     
    ##  Min.   : 892.0   Min.   :1.000  
    ##  1st Qu.: 996.2   1st Qu.:1.000  
    ##  Median :1100.5   Median :3.000  
    ##  Mean   :1100.5   Mean   :2.266  
    ##  3rd Qu.:1204.8   3rd Qu.:3.000  
    ##  Max.   :1309.0   Max.   :3.000  
    ##                                  
    ##                                         Name         Sex     
    ##  Abbott, Master. Eugene Joseph            :  1   female:152  
    ##  Abelseth, Miss. Karen Marie              :  1   male  :266  
    ##  Abelseth, Mr. Olaus Jorgensen            :  1               
    ##  Abrahamsson, Mr. Abraham August Johannes :  1               
    ##  Abrahim, Mrs. Joseph (Sophie Halaut Easu):  1               
    ##  Aks, Master. Philip Frank                :  1               
    ##  (Other)                                  :412               
    ##       Age            SibSp            Parch             Ticket   
    ##  Min.   : 0.17   Min.   :0.0000   Min.   :0.0000   PC 17608:  5  
    ##  1st Qu.:21.00   1st Qu.:0.0000   1st Qu.:0.0000   113503  :  4  
    ##  Median :27.00   Median :0.0000   Median :0.0000   CA. 2343:  4  
    ##  Mean   :30.27   Mean   :0.4474   Mean   :0.3923   16966   :  3  
    ##  3rd Qu.:39.00   3rd Qu.:1.0000   3rd Qu.:0.0000   220845  :  3  
    ##  Max.   :76.00   Max.   :8.0000   Max.   :9.0000   347077  :  3  
    ##  NA's   :86                                        (Other) :396  
    ##       Fare                     Cabin     Embarked
    ##  Min.   :  0.000                  :327   C:102   
    ##  1st Qu.:  7.896   B57 B59 B63 B66:  3   Q: 46   
    ##  Median : 14.454   A34            :  2   S:270   
    ##  Mean   : 35.627   B45            :  2           
    ##  3rd Qu.: 31.500   C101           :  2           
    ##  Max.   :512.329   C116           :  2           
    ##  NA's   :1         (Other)        : 80

We need to add a column to test in order to combine it with train

``` r
test$Survived <- 0
titanic <- rbind(train,test)
```

Cleaning the dataset: Since we have only one missing value in Fare, we replace it with mean

``` r
mean_fare<- mean(na.omit(titanic$Fare))
titanic$Fare <- ifelse(is.na(titanic$Fare), mean_fare, titanic$Fare)

titanic[c(62, 830), "Embarked"] <- "S" #Cleaning the Embarked column 
titanic$Embarked <- as.factor(as.character(titanic$Embarked))
table(titanic$Embarked)
```

    ## 
    ##   C   Q   S 
    ## 270 123 916

Feature Extraction - We know that usually people with superior titles get saved earlier. Under that assumption; Extracting the title from the name. Some of the titles have very low count, we combine them with to common ones

``` r
titanic$Title <- gsub('(.*, )|(\\..*)', '', titanic$Name)

rare_title <- c('Dona', 'Lady', 'the Countess','Capt', 'Col', 'Don', 
                'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer')

titanic$Title[titanic$Title == 'Mlle']        <- 'Miss' 
titanic$Title[titanic$Title == 'Ms']          <- 'Miss'
titanic$Title[titanic$Title == 'Mme']         <- 'Mrs' 
titanic$Title[titanic$Title %in% rare_title]  <- 'Rare Title'

table(titanic$Sex, titanic$Title)
```

    ##         
    ##          Master Miss  Mr Mrs Rare Title
    ##   female      0  264   0 198          4
    ##   male       61    0 757   0         25

Adding a Child column to the dataset.

``` r
titanic$Child <- 0
titanic$Child[titanic$Age<10] <- 1
```

Let's also compute the Missing data in Age column using Random Forest

``` r
age_fit <- rpart(Age ~ Pclass + Sex + SibSp + Parch + Fare + Embarked,
               data=titanic[!is.na(titanic$Age),], 
               method="anova")
titanic$Age[is.na(titanic$Age)] <- predict(age_fit, titanic[is.na(titanic$Age),])
table(is.na(titanic$Age))
```

    ## 
    ## FALSE 
    ##  1309

Removing the variables which I will not be using during analysis

``` r
titanic_actual<- titanic[,c(-6,-4,-9,-11)]
titanic_actual$Pclass<- as.factor(titanic_actual$Pclass)
titanic_actual$SibSp <- as.factor(titanic_actual$SibSp)
titanic_actual$Parch<- as.factor(titanic_actual$Parch)
titanic_actual$Child<- as.factor(titanic_actual$Child)

train <- titanic_actual[1:891, ]
test <- titanic_actual[892:1309,]
```

Using SVM

``` r
svm_fit<- svm(Survived~., cost = 5, data = train)
summary(svm_fit)
```

    ## 
    ## Call:
    ## svm(formula = Survived ~ ., data = train, cost = 5)
    ## 
    ## 
    ## Parameters:
    ##    SVM-Type:  eps-regression 
    ##  SVM-Kernel:  radial 
    ##        cost:  5 
    ##       gamma:  0.03846154 
    ##     epsilon:  0.1 
    ## 
    ## 
    ## Number of Support Vectors:  455

``` r
pred<-predict(svm_fit, test)
pred<-round(pred)
table(pred)
```

    ## pred
    ##   0   1 
    ## 265 153

``` r
submission <- data.frame(PassengerId=test$PassengerId, Survived=pred)
table(submission$Survived)
```

    ## 
    ##   0   1 
    ## 265 153

``` r
write.csv(submission, "myfile.csv", row.names=FALSE)
```

In order to solve the problem of extra level in test set.We can see that factor of level 9 isn't present in train set which leads to a problem in predicting the test set

``` r
titanic[titanic$Parch == "9",]
```

    ##      PassengerId Survived Pclass                           Name    Sex
    ## 1234        1234        0      3          Sage, Mr. John George   male
    ## 1257        1257        0      3 Sage, Mrs. John (Annie Bullen) female
    ##           Age SibSp Parch   Ticket  Fare Cabin Embarked Title Child
    ## 1234 37.35294     1     9 CA. 2343 69.55              S    Mr     0
    ## 1257 37.35294     1     9 CA. 2343 69.55              S   Mrs     0

``` r
extra_obs<- test[test$PassengerId == "1234",]
train <- rbind(train, extra_obs)
```

Using Logistic Regression

``` r
fit_log<- glm(Survived~., data = train, family = "binomial")
summary(fit_log)
```

    ## 
    ## Call:
    ## glm(formula = Survived ~ ., family = "binomial", data = train)
    ## 
    ## Deviance Residuals: 
    ##     Min       1Q   Median       3Q      Max  
    ## -2.5218  -0.5707  -0.3963   0.5420   2.4885  
    ## 
    ## Coefficients:
    ##                   Estimate Std. Error z value Pr(>|z|)    
    ## (Intercept)      2.074e+01  1.384e+03   0.015 0.988038    
    ## PassengerId      4.350e-05  3.718e-04   0.117 0.906855    
    ## Pclass2         -9.064e-01  3.114e-01  -2.911 0.003604 ** 
    ## Pclass3         -1.793e+00  2.977e-01  -6.023 1.71e-09 ***
    ## Sexmale         -1.730e+01  1.384e+03  -0.013 0.990022    
    ## SibSp1          -2.093e-01  2.485e-01  -0.842 0.399755    
    ## SibSp2          -1.506e-01  5.587e-01  -0.270 0.787524    
    ## SibSp3          -2.441e+00  7.076e-01  -3.449 0.000562 ***
    ## SibSp4          -3.104e+00  8.625e-01  -3.599 0.000320 ***
    ## SibSp5          -1.749e+01  8.922e+02  -0.020 0.984356    
    ## SibSp8          -1.622e+01  7.293e+02  -0.022 0.982258    
    ## Parch1          -4.819e-01  3.368e-01  -1.431 0.152499    
    ## Parch2          -4.193e-01  4.088e-01  -1.026 0.304970    
    ## Parch3          -4.503e-01  1.168e+00  -0.386 0.699836    
    ## Parch4          -1.675e+01  1.011e+03  -0.017 0.986780    
    ## Parch5          -2.253e+00  1.205e+00  -1.870 0.061445 .  
    ## Parch6          -1.758e+01  2.400e+03  -0.007 0.994154    
    ## Parch9          -1.416e+01  2.400e+03  -0.006 0.995291    
    ## Fare             4.308e-03  2.764e-03   1.559 0.119079    
    ## EmbarkedQ       -4.764e-02  3.990e-01  -0.119 0.904969    
    ## EmbarkedS       -3.662e-01  2.534e-01  -1.445 0.148489    
    ## TitleMiss       -1.815e+01  1.384e+03  -0.013 0.989535    
    ## TitleMr         -3.829e+00  6.844e-01  -5.595 2.20e-08 ***
    ## TitleMrs        -1.759e+01  1.384e+03  -0.013 0.989855    
    ## TitleRare Title -4.109e+00  8.790e-01  -4.675 2.94e-06 ***
    ## Child1           8.709e-01  5.424e-01   1.606 0.108333    
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## (Dispersion parameter for binomial family taken to be 1)
    ## 
    ##     Null deviance: 1187.62  on 891  degrees of freedom
    ## Residual deviance:  716.64  on 866  degrees of freedom
    ## AIC: 768.64
    ## 
    ## Number of Fisher Scoring iterations: 15

``` r
predict_log<- predict(fit_log,test, type = "response")
predict_log<- round(predict_log)

submission <- data.frame(PassengerId=test$PassengerId, Survived=predict_log)
table(submission$Survived)
```

    ## 
    ##   0   1 
    ## 249 169

``` r
write.csv(submission, "Logistic_reg.csv", row.names=FALSE)
```

Using Random Forest Regression

``` r
train$Title<- as.factor(train$Title)
test$Title<- as.factor(test$Title)
set.seed(1234)
fit_rf <- randomForest(as.factor(Survived) ~ .,
                    data=train, 
                    importance=TRUE, 
                    ntree=2000)
summary(fit_rf)
```

    ##                 Length Class  Mode     
    ## call               5   -none- call     
    ## type               1   -none- character
    ## predicted        892   factor numeric  
    ## err.rate        6000   -none- numeric  
    ## confusion          6   -none- numeric  
    ## votes           1784   matrix numeric  
    ## oob.times        892   -none- numeric  
    ## classes            2   -none- character
    ## importance        36   -none- numeric  
    ## importanceSD      27   -none- numeric  
    ## localImportance    0   -none- NULL     
    ## proximity          0   -none- NULL     
    ## ntree              1   -none- numeric  
    ## mtry               1   -none- numeric  
    ## forest            14   -none- list     
    ## y                892   factor numeric  
    ## test               0   -none- NULL     
    ## inbag              0   -none- NULL     
    ## terms              3   terms  call

``` r
plot(fit_rf)
```

![](Titanic_submission_files/figure-markdown_github-ascii_identifiers/unnamed-chunk-11-1.png)

``` r
Prediction <- predict(fit_rf, test)
submit <- data.frame(PassengerId = test$PassengerId, Survived = Prediction)
write.csv(submit, file = "Randomforest.csv", row.names = FALSE)
```
