library(rpart)
library(randomForest)
library(tree)
library(e1071)

train<- read.csv("train.csv")
test<- read.csv("test.csv")

str(train) #To check the structure of dataset
summary(test)

test$Survived <- 0
titanic <- rbind(train,test)

mean_fare<- mean(na.omit(titanic$Fare))
#Since we have only one missing value in Fare, we replace it with mean
titanic$Fare <- ifelse(is.na(titanic$Fare), mean_fare, titanic$Fare)

titanic[c(62, 830), "Embarked"] <- "S" #Cleaning the Embarked column 
titanic$Embarked <- as.factor(as.character(titanic$Embarked))
table(titanic$Embarked)

#Feature Extraction - Learned from Kernels on the internet
#Extracting the title from the name
titanic$Title <- gsub('(.*, )|(\\..*)', '', titanic$Name)

#Some of the titles have very low count, we combine them with to common ones
rare_title <- c('Dona', 'Lady', 'the Countess','Capt', 'Col', 'Don', 
                'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer')

titanic$Title[titanic$Title == 'Mlle']        <- 'Miss' 
titanic$Title[titanic$Title == 'Ms']          <- 'Miss'
titanic$Title[titanic$Title == 'Mme']         <- 'Mrs' 
titanic$Title[titanic$Title %in% rare_title]  <- 'Rare Title'

# Title counts by sex
table(titanic$Sex, titanic$Title)

#Adding a Child column to the dataset 
titanic$Child <- 0
titanic$Child[titanic$Age<10] <- 1

#Let's also compute the Missing data in Age column using Random Forest
age_fit <- rpart(Age ~ Pclass + Sex + SibSp + Parch + Fare + Embarked,
               data=titanic[!is.na(titanic$Age),], 
               method="anova")
titanic$Age[is.na(titanic$Age)] <- predict(age_fit, titanic[is.na(titanic$Age),])
table(is.na(titanic$Age))

#Removing the variables which I will not be using during analysis
titanic_actual<- titanic[,c(-6,-4,-9,-11)]

#Converting variables to factors
titanic_actual$Pclass<- as.factor(titanic_actual$Pclass)
titanic_actual$SibSp <- as.factor(titanic_actual$SibSp)
titanic_actual$Parch<- as.factor(titanic_actual$Parch)
titanic_actual$Child<- as.factor(titanic_actual$Child)

#Split the data back into Test and Train sets
train <- titanic_actual[1:891, ]
test <- titanic_actual[892:1309,]

#Using SVM 
svm_fit<- svm(Survived~., cost = 5, data = train)
summary(svm_fit)

pred<-predict(svm_fit, test)
pred<-round(pred)
table(pred)

submission <- data.frame(PassengerId=test$PassengerId, Survived=pred)
table(submission$Survived)
write.csv(submission, "myfile.csv", row.names=FALSE)

#In order to solve the problem of extra level in test set.We can see that factor of level 9 isn't present 
#in train set which leads to a problem in predicting the test set
titanic[titanic$Parch == "9",]
extra_obs<- test[test$PassengerId == "1234",]
train <- rbind(train, extra_obs)

#Using Logistic Regression
fit_log<- glm(Survived~., data = train, family = "binomial")
summary(fit_log)

predict_log<- predict(fit_log,test, type = "response")
predict_log<- round(predict_log)

submission <- data.frame(PassengerId=test$PassengerId, Survived=predict_log)
table(submission$Survived)
write.csv(submission, "Logistic_reg.csv", row.names=FALSE)

#Using Random Forest Regression
train$Title<- as.factor(train$Title)
test$Title<- as.factor(test$Title)
set.seed(1234)
fit_rf <- randomForest(as.factor(Survived) ~ .,
                    data=train, 
                    importance=TRUE, 
                    ntree=2000)
summary(fit_rf)
plot(fit_rf)

Prediction <- predict(fit_rf, test)
submit <- data.frame(PassengerId = test$PassengerId, Survived = Prediction)
write.csv(submit, file = "Randomforest.csv", row.names = FALSE)

