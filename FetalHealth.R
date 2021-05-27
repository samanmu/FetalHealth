#-------------------------------------------------------------------------
#---- downloading the data and creating 'train' and 'validation' sets
#-------------------------------------------------------------------------

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(data.table)
library(rpart)
library(matrixStats)

# fetal_health dataset:
# https://www.kaggle.com/andrewmvd/fetal-health-classification/download
# the "fetal_health.csv" file is available in the github repository
# Extracting data from the downloaded file:

data <- read.csv("fetal_health.csv")

# exploring the data:

summary(data)
colnames(data)
head(data)
print("number of each health condition in CTG results:")
table(data$fetal_health)
prop.table(table(data$fetal_health))

# print("number of each normal health CTG results:")
# sum(data$fetal_health == "1")
# print('number of "suspect" health CTG results:')
# sum(data$fetal_health == "2")
# print('number of "pathological" health CTG results:')
# sum(data$fetal_health == "3")

class(data$fetal_health)
data$fetal_health <- as.factor(data$fetal_health)

barplot(data$baseline.value)

p1 <- data %>% ggplot(aes(fetal_health,baseline.value)) + geom_boxplot()
p2 <- data %>% ggplot(aes(fetal_health,accelerations)) + geom_boxplot() #GOOD INSIGHT
p3 <- data %>% ggplot(aes(fetal_health,fetal_movement)) + geom_boxplot()
p4 <- data %>% ggplot(aes(fetal_health,uterine_contractions)) + geom_boxplot()
p5 <- data %>% ggplot(aes(fetal_health,light_decelerations)) + geom_boxplot()
p6 <- data %>% ggplot(aes(fetal_health,severe_decelerations)) + geom_boxplot()
p7 <- data %>% ggplot(aes(fetal_health,prolongued_decelerations)) + geom_boxplot()
p8 <- data %>% ggplot(aes(fetal_health,abnormal_short_term_variability)) + geom_boxplott()
p9 <- data %>% ggplot(aes(fetal_health,mean_value_of_short_term_variability)) + geom_boxplot()
p10 <- data %>% ggplot(aes(fetal_health,percentage_of_time_with_abnormal_long_term_variability)) + geom_boxplot()
p11 <- data %>% ggplot(aes(fetal_health,mean_value_of_long_term_variability)) + geom_boxplot() # INSIGHTS HERE
p12 <- data %>% ggplot(aes(fetal_health,histogram_width)) + geom_boxplot()
p13 <- data %>% ggplot(aes(fetal_health,histogram_min)) + geom_boxplot()
p14 <- data %>% ggplot(aes(fetal_health,histogram_max)) + geom_boxplot()
p15 <- data %>% ggplot(aes(fetal_health,histogram_number_of_peaks)) + geom_boxplot()
p16 <- data %>% ggplot(aes(fetal_health,histogram_number_of_zeroes)) + geom_boxplot()
p17 <- data %>% ggplot(aes(fetal_health,histogram_mode)) + geom_boxplot()
p18 <- data %>% ggplot(aes(fetal_health,histogram_mean)) + geom_boxplot()
p19 <- data %>% ggplot(aes(fetal_health,histogram_median)) + geom_boxplot()
p20 <- data %>% ggplot(aes(fetal_health,histogram_variance)) + geom_boxplot() #INSIGHTS
p21 <- data %>% ggplot(aes(fetal_health,histogram_tendency)) + geom_boxplot()
p2
p11
p20


rm(p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13,p14,p15,p16,p17,p18,p19,p20,p21)

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind = "Rounding" )
test_index <- createDataPartition(y = data$fetal_health, times = 1, p = 0.1, list = FALSE)
fetalhealth <- data[-test_index,]
validation <- data[test_index,]
rm(test_index)

#-------------------------------------------------------------------------
#---- Splitting 'fetalhealth' to train_set and test_set
#-------------------------------------------------------------------------
set.seed(1, sample.kind = "Rounding" )
index <- createDataPartition(fetalhealth$fetal_health, 1, p = 0.1, list = F)
train_set <- fetalhealth[-index,]
test_set <- fetalhealth[index,]
rm(index)


x <- train_set[,-22]    # 22 is the "fetal_health" column number
y <- train_set[,22]
#-------------------------------------------------------------------------
#---- logistic regression
#-------------------------------------------------------------------------

#install.packages("mbr")
library(mbr)
# colScale(x, add_attr = TRUE)

sds <- colSds(as.matrix(x))
qplot(sds, bins=256, color = I("black"))

mean(class(train_set$accelerations)=="numeric")

fit_glm <- glm(fetal_health~., data = train_set, family='binomial')
summary(fit_glm)
y_hat_glm <- factor(predict(fit_glm, test_set))
levels(y_hat_glm)

#-------------------------------------------------------------------------
#---- Knn
#-------------------------------------------------------------------------

fit_knn <- train(x,y,method = "knn", tuneGrid = data.frame(k=seq(3,11,2)))
ggplot(fit_knn, highlight = TRUE)
fit_knn$bestTune
y_hat_knn <- predict(fit_knn, test_set)
mean(y_hat_knn == test_set$fetal_health)
confusionMatrix(predict(fit_knn,test_set), test_set$fetal_health)$overal["Accuracy"]

#-------------------------------------------------------------------------
#---- decision tree
#-------------------------------------------------------------------------

fit_tree <- rpart(fetal_health ~ ., data = train_set)
plot(fit_tree, scale = 0.6)
text(fit_tree, cex = -0.85)
summary(fit_tree)
varImp(fit_tree)
y_hat_t <- predict(fit_tree, test_set)
rownames(y_hat_t) <- 1:length(test_set$fetal_health)

y_hat_tree <- 0
for (i in 1:length(test_set$fetal_health)){
  y_hat_tree[i] <- which.max(y_hat_t[i,])
}
y_hat_tree
y_hat_tree <- factor(y_hat_tree)
levels(y_hat_tree)
levels(test_set$fetal_health)
confusionMatrix(y_hat_tree,test_set$fetal_health)
confusionMatrix(y_hat_tree,test_set$fetal_health)$overal["Accuracy"]

which(y_hat_tree != test_set$fetal_health)
# y_hat_tree[19]
# test_set$fetal_health[19]

#-------------------------------------------------------------------------
#---- Random Forest
#-------------------------------------------------------------------------
