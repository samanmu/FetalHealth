#-------------------------------------------------------------------------
#---- downloading the data and creating 'train' and 'validation' sets
#-------------------------------------------------------------------------

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(data.table)

# fetal_health dataset:
# https://www.kaggle.com/andrewmvd/fetal-health-classification/download
# the "fetal_health.csv" file is available in the github repository
# Extracting data from the downloaded file:

data <- read.csv("fetal_health.csv")

# exploring the data:

head(data)
print("number of normal health CTG results:")
sum(data$fetal_health == "1")
print('number of "suspect" health CTG results:')
sum(data$fetal_health == "2")
print('number of "pathological" health CTG results:')
sum(data$fetal_health == "3")
