setwd("C:/Users/Mayukha/Desktop/Data Science/Loan Prediction")
rm(list = ls(all = TRUE))

loan_train <- read.csv("Load Prediction Train.csv")
loan_test <- read.csv("LoadPredictionTest.csv")

summary(loan_train)
summary(loan_test)
colnames(loan_train)

#To find all the unique values
apply(loan_train, 2, function(x){length(unique(x))})
#To find the percent of each unique value
prop.table(table(loan_train$Self_Employed))
#Number of unique values in descending order
head(sort(table(loan_train$Dependents),decreasing = TRUE),20)

head(round(sort(prop.table(table(loan_train$Property_Area)),decreasing = TRUE),6),20)

str(loan_train)

loan_train[loan_train==""] <- NA
loan_test[loan_test==""] <- NA

#To remove blank level from columns

for (i in c(2,3,4,6)){
  loan_train[,i] <- factor(loan_train[,i])
}

for (i in c(2,3,4,6)){
  loan_test[,i] <- factor(loan_test[,i])
}

sort(colSums(is.na(loan_train)),decreasing = TRUE)
sort(colSums(is.na(loan_test)),decreasing = TRUE)

loan_train$Credit_History <- as.factor(loan_train$Credit_History)
loan_test$Credit_History <- as.factor(loan_test$Credit_History)

getmode <- function(v) {
  uniqv <- unique(v)
  uniqv[which.max(tabulate(match(v, uniqv)))]
}

attach(loan_train)
library(ggplot2)

ggplot(loan_train, aes(x=ApplicantIncome, y=Credit_History, color=Loan_Status)) + 
  geom_jitter(position = position_jitter(height = .1)) +
  scale_color_manual(values=c("red", "blue")) + facet_grid(Gender~.)
  ggtitle("Applucant income and credit history as loan Factors") + 
  ylab("Credit History")



ggplot(loan_train, aes(x = Dependents, y = ApplicantIncome, fill = factor(Loan_Status))) +
  geom_boxplot()


ggplot(loan_train,aes(Gender,color=income$X)) + geom_bar()


getmode(loan_train$Credit_History)
loan_train$Credit_History[is.na(loan_train$Credit_History)] <- 1
loan_test$Credit_History[is.na(loan_test$Credit_History)] <- 1

ggplot(loan_test, aes(Dependents, fill = Married)) + 
  geom_bar(stat = "count", position = "dodge")

View(loan_train[is.na(loan_train$Dependents),])

loan_train$Dependents[is.na(loan_train$Dependents)] <- 0
loan_train$Married[is.na(loan_train$Married)] <- "No"

loan_test$Dependents[is.na(loan_test$Dependents)] <- 0
loan_test$Married[is.na(loan_test$Married)] <- "No"

ggplot(loan_train, aes(ApplicantIncome,Self_Employed,fill = Loan_Status))
+ geom_bar()

loan_train[is.na(loan_train$Self_Employed) & loan_train$Loan_Status =="N",]
getmode(loan_train$Self_Employed[loan_train$Loan_Status=="N"])

#To find median in the plot
ggplot(loan_train,aes(x=ApplicantIncome))+geom_density() +geom_vline(aes(xintercept=median(ApplicantIncome,na.rm=T)),color='red',linetype='dashed')

barplot(table(loan_train$Gender,loan_train$Credit_History))
loan_train$Gender[is.na(loan_train$Gender)] <- "Male"
loan_test$Gender[is.na(loan_test$Gender)] <- "Male"

getmode(loan_train$Self_Employed)

loan_train$Self_Employed[is.na(loan_train$Self_Employed)] <- "No"
loan_test$Self_Employed[is.na(loan_test$Self_Employed)] <- "No"

library(caret)
library(MASS)
library(DMwR)

loan_train <- knnImputation(loan_train, k=3)
loan_test <- knnImputation(loan_test, k=3)

loan_train$Loan_ID <- NULL

#Scatter plot 

ggplot(loan_train, aes(ApplicantIncome,LoanAmount, color=Loan_Status)) + 
  #geom_point(aes(color=Loan_Status)) + 
  geom_jitter(position = position_jitter(height = .2))+
  scale_x_continuous("Applicant Income") +
  scale_y_continuous("Loan Amount") + 
  theme_bw() + labs(title= "Scatterplot")

summary(loan_train)


#Histogram
ggplot(loan_train, aes(ApplicantIncome)) + geom_histogram(binwidth = 10)+
  scale_x_continuous(breaks = seq(0,700,by=100)) +
  labs(title="Histogram")
median(loan_train$LoanAmount) #We can consider median for NA values as it has
#greater count

#Stacked Bar chart 
ggplot(loan_train,aes(Property_Area,fill = Loan_Status)) + geom_bar()+
  labs(title = "Stacked Bar chart")


set.seed(10)
train_rows <- createDataPartition(loan_train$Loan_Status, p=0.75, list = FALSE)
train_data <- loan_train[train_rows,]
test_data <- loan_train[-train_rows,]
colnames(train_data)

logistic <- glm(Loan_Status~.,data=train_data,family = "binomial")
step<- stepAIC(logistic)
#predictglm <- predict(logistic,test_data[,-13],type = "response")
predictglm <- predict(step,train_data,type = "response")
library(ROCR)
pred1= prediction(predictglm,train_data$Loan_Status)
perf=performance(pred1,measure = "tpr",x.measure = "fpr")
plot(perf,col=rainbow(10),colorize=T,print.cutoffs.at=seq(0,1,0.05))

perfauc<-performance(pred1,measure = "auc");perfauc
tab<-table(train_data$Loan_Status,ifelse(predictglm>0.5,"Y","N"));tab
ac<-sum(diag(tab))/sum(tab);ac

pred_test <-ifelse(predict(step,loan_test,type="response")>0.5,"Y","N")

loanpred <- data.frame(Loan_ID = loan_test$Loan_ID, Loan_Status = pred_test, row.names = NULL)

####SVM####
loan2 <- data.frame(predict(dummyVars(~.,data=loan_train[,-12]),loan_train))
loan2$Loan_Status <- loan_train$Loan_Status
par(mfrow=c(1,1))
library(corrplot)
corrplot(cor(loan2[,-22]),type = "lower")
summary(loan2)
colnames(loan2)

loan2_train <- loan2[train_rows,]
loan2_test <- loan2[-train_rows,]

ctrl <- trainControl(method = "repeatedcv",repeats = 5,classProbs = TRUE, search = "random")
grid <- expand.grid(C=10^(-4:1))

linear_svm <- train(x=loan2_train[,-22],y=loan2_train$Loan_Status,
                    method = "svmLinear", metric = "Accuracy", 
                    tuneLength = 10, trControl = ctrl)

grid <- expand.grid(C=10^(-4:1),sigma=10^(-4:1))

radial_svm <- train(x=loan2_train[,-22],y=loan2_train$Loan_Status,
                    method = "svmRadial", metric = "Accuracy",
                    tuneLength = 10, trControl = ctrl)
svms <- resamples(list(linear_svm,radial_svm))

bwplot(svms,metric = "Accuracy",ylab=c("linear kernel","radial kernel"))
########

#####Naive Bayes#####
library(infotheo)
library(e1071)

hist(loan_train$ApplicantIncome)
hist(loan_train$CoapplicantIncome)
hist(loan_train$LoanAmount)
hist(loan_train$Loan_Amount_Term)
loan_cat <- loan_train
loan_cat_test <- loan_test
loan_cat$incomelevel <- cut(loan_cat$ApplicantIncome, 3, include.lowest=TRUE, labels=c("Low", "Med","High"))
loan_cat$coincome <- cut(loan_cat$CoapplicantIncome, 3, include.lowest=TRUE, labels=c("Low", "Med","High"))
loan_cat$termlevel <- cut(loan_cat$Loan_Amount_Term, 3, include.lowest=TRUE, labels=c("Short", "Medium","Long"))
loan_cat_test$incomelevel <- cut(loan_cat_test$ApplicantIncome, 3, include.lowest=TRUE, labels=c("Low", "Med","High"))
loan_cat_test$coincome <- cut(loan_cat_test$CoapplicantIncome, 3, include.lowest=TRUE, labels=c("Low", "Med","High"))
loan_cat_test$termlevel <- cut(loan_cat_test$Loan_Amount_Term, 3, include.lowest=TRUE, labels=c("Short", "Medium","Long"))


loan_cat <- loan_cat[,-c(6,7,9)]
summary(loan_cat)
summary(loan_cat_test)
loan_cat_test$Gender <- factor(loan_cat_test$Gender)
loan_cat_test$Dependents <- factor(loan_cat_test$Dependents)
loan_cat_test$Self_Employed <- factor(loan_cat_test$Self_Employed)
cat_train <- loan_cat[train_rows,]
cat_test <- loan_cat[-train_rows,]

summary(cat_train)
colnames(cat_train)
income<- discretize(loan_train$ApplicantIncome,disc = "equalwidth", nbins = 2)
table(income$X)

naive <- naiveBayes(Loan_Status~.,data = cat_train)
pred <- predict(naive,cat_test[,-12])
confusionMatrix(cat_test$Loan_Status,pred)
pred_test <- predict(naive,loan_cat_test)

##########


loanpred <- data.frame(Loan_ID = loan_test$Loan_ID, Loan_Status = pred_test, row.names = NULL)

write.csv(loanpred,"predictions.csv", quote=F, row.names = F)
