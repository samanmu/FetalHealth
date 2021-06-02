#-------------------------------------------------------------------------
#---- downloading the data and creating 'train' and 'validation' sets
#-------------------------------------------------------------------------

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(gridExtra)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(rpart)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(rpart.plot)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(matrixStats)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(randomForest)) install.packages("data.table", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(data.table)
library(gridExtra)
library(rpart)
library(rpart.plot)
library(matrixStats)
library(randomForest)


# fetal_health dataset:
# https://www.kaggle.com/andrewmvd/fetal-health-classification/download
# the "fetal_health.csv" file is available in the github repository
# Extracting data from the downloaded file:

data <- read.csv("fetal_health.csv")

#-------------------------------------------------------------------------
#---- DATA SUMMARY
#-------------------------------------------------------------------------

#summary(data)
colnames(data)
head(data)

#-------------------------------------------------------------------------
#---- Fetal_health column as the outcome in the data set
#-------------------------------------------------------------------------

data$fetal_health <- as.factor(data$fetal_health)
print("number of each health condition in CTG results:")
table(data$fetal_health)
print("health conditions Proportions:")
prop.table(table(data$fetal_health))

#-------------------------------------------------------------------------
#---- Data exploration plots
#-------------------------------------------------------------------------

p1 <- data %>% ggplot(aes(fetal_health,baseline.value)) + geom_boxplot() + theme_bw()
p2 <- data %>% ggplot(aes(fetal_health,accelerations)) + geom_boxplot() + theme_bw() #INSIGHT
p3 <- data %>% ggplot(aes(fetal_health,fetal_movement)) + geom_boxplot() + theme_bw()
p4 <- data %>% ggplot(aes(fetal_health,uterine_contractions)) + geom_boxplot() + theme_bw()
p5 <- data %>% ggplot(aes(fetal_health,light_decelerations)) + geom_boxplot() + theme_bw()
p6 <- data %>% ggplot(aes(fetal_health,severe_decelerations)) + geom_boxplot() + theme_bw()
p7 <- data %>% ggplot(aes(fetal_health,prolongued_decelerations)) + geom_boxplot() + theme_bw()
p8 <- data %>% ggplot(aes(fetal_health,abnormal_short_term_variability)) + geom_boxplot() + theme_bw()
p9 <- data %>% ggplot(aes(fetal_health,mean_value_of_short_term_variability)) + geom_boxplot() + theme_bw()
p10 <- data %>% ggplot(aes(fetal_health,percentage_of_time_with_abnormal_long_term_variability)) + geom_boxplot() + theme_bw()
p11 <- data %>% ggplot(aes(fetal_health,mean_value_of_long_term_variability)) + geom_boxplot() + theme_bw() #INSIGHT
p12 <- data %>% ggplot(aes(fetal_health,histogram_width)) + geom_boxplot() + theme_bw()
p13 <- data %>% ggplot(aes(fetal_health,histogram_min)) + geom_boxplot() + theme_bw()
p14 <- data %>% ggplot(aes(fetal_health,histogram_max)) + geom_boxplot() + theme_bw()
p15 <- data %>% ggplot(aes(fetal_health,histogram_number_of_peaks)) + geom_boxplot() + theme_bw()
p16 <- data %>% ggplot(aes(fetal_health,histogram_number_of_zeroes)) + geom_boxplot() + theme_bw()
p17 <- data %>% ggplot(aes(fetal_health,histogram_mode)) + geom_boxplot() + theme_bw()
p18 <- data %>% ggplot(aes(fetal_health,histogram_mean)) + geom_boxplot() + theme_bw()
p19 <- data %>% ggplot(aes(fetal_health,histogram_median)) + geom_boxplot() + theme_bw() #INSIGHT
p20 <- data %>% ggplot(aes(fetal_health,histogram_variance)) + geom_boxplot() + theme_bw() #INSIGHT
p21 <- data %>% ggplot(aes(fetal_health,histogram_tendency)) + geom_boxplot() + theme_bw()

grid.arrange(p1,p2,p3,p4, ncol = 2)
grid.arrange(p5,p6,p7,p8, ncol = 2)
grid.arrange(p9,p10,p11,p12, ncol = 2)
grid.arrange(p13,p14,p15,p16, ncol = 2)
grid.arrange(p17,p18,p19,p20,p21, ncol = 2)
# grid.arrange(p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13,p14,p15,p16,p17,p18,p19,p20,p21)
rm(p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13,p14,p15,p16,p17,p18,p19,p20,p21)


#-------------------------------------------------------------------------
#---- Spliting data to 'fetalhealth' and 'validation' sets
#-------------------------------------------------------------------------
# Validation set will be 10% of the data
set.seed(1, sample.kind = "Rounding" )
test_index <- createDataPartition(y = data$fetal_health, times = 1, p = 0.1, list = FALSE)
fetalhealth <- data[-test_index,]
validation <- data[test_index,]
rm(test_index)
dim(fetalhealth)
dim(validation)

#-------------------------------------------------------------------------
#---- Splitting 'fetalhealth' to train_set and test_set
#-------------------------------------------------------------------------
set.seed(1, sample.kind = "Rounding" )
index <- createDataPartition(fetalhealth$fetal_health, 1, p = 0.1, list = F)
train_set <- fetalhealth[-index,]
test_set <- fetalhealth[index,]
rm(index)
dim(train_set)
dim(test_set)

#-------------------------------------------------------------------------
#---- Knn
#-------------------------------------------------------------------------

x <- train_set[,-22]    # 22 is the "fetal_health" column number
y <- train_set[,22]

set.seed(1, sample.kind = "Rounding")
control <- trainControl(method = "cv", number = 3, p = .9)
fit_knn <- train(x,y,method = "knn",
                 tuneGrid = data.frame(k=seq(3,11,2)),
                                       trControl = control)
ggplot(fit_knn, highlight = TRUE)
fit_knn$bestTune
y_hat_knn <- predict(fit_knn, test_set)

confusionMatrix(predict(fit_knn,test_set), test_set$fetal_health)
confusionMatrix(predict(fit_knn,test_set), test_set$fetal_health)$overal["Accuracy"]


#-------------------------------------------------------------------------
#---- decision tree
#-------------------------------------------------------------------------

fit_tree <- rpart(fetal_health ~ ., data = train_set)
rpart.plot(fit_tree, tweak = 1.2,
           digits = 2, box.palette = list("Greens","Blues","Reds"),
           shadow.col="gray", nn=F)
#summary(fit_tree)
varImp(fit_tree)
y_hat_t <- predict(fit_tree, test_set)
# rownames(y_hat_t) <- 1:length(test_set$fetal_health)

y_hat_tree <- 0
for (i in 1:length(test_set$fetal_health)){
  y_hat_tree[i] <- which.max(y_hat_t[i,])
}
y_hat_tree <- factor(y_hat_tree)
confusionMatrix(y_hat_tree,test_set$fetal_health)$overal["Accuracy"]

# which(y_hat_tree != test_set$fetal_health)

#-------------------------------------------------------------------------
#---- Random Forest
#-------------------------------------------------------------------------

x <- train_set[,-22]    # 22 is the "fetal_health" column number
y <- train_set[,22]

fit_rf <- randomForest(x, y, ntree = 1000)
plot(fit_rf)

imp <- importance(fit_rf)
imp

y_hat_rf <- predict(fit_rf, test_set)
confusionMatrix(y_hat_rf,test_set$fetal_health)$overal["Accuracy"]

#-------------------------------------------------------------------------
#---- Final model on the whole data, Random Forest
#-------------------------------------------------------------------------

data_x <- fetalhealth[,-22]
data_y <- fetalhealth[,22]

final_rf <- randomForest(data_x, data_y, ntree = 1000)
y_hat_final_rf <- predict(final_rf, validation)
confusionMatrix(y_hat_final_rf,validation$fetal_health)$overal["Accuracy"]

#-------------------------------------------------------------------------
#---- NOTE
#-------------------------------------------------------------------------

which(y_hat_knn != test_set$fetal_health)
which(y_hat_tree != test_set$fetal_health)
which(y_hat_rf != test_set$fetal_health)
