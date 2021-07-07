library(dplyr)
library(tidyr)
library(ggplot2)
library(readxl)
library(zoo)
library(lubridate)
library(stringr)
library(fpp2)
library(leaps)
library(caTools)
library(caret)

## clean data ####
dat <- read.csv("12-2019.csv") # import august flight

# Delete unused columns, remove national and security
flight <- dat[-c(2,5,7:9,12,13,15,16,17,22:24,30)]

# transform delay reason into binary variables
flight$delay_carrier <- ifelse(flight$delay_carrier>15, 1, 0)
flight$delay_weather <- ifelse(flight$delay_weather>15, 1, 0)
flight$delay_late_aircarft_arrival <- ifelse(flight$delay_late_aircarft_arrival>15, 1, 0)


# change schedule departure and arrival dt to numeric
dp_time <- strftime(flight$scheduled_departure_dt, format="%H:%M")
ar_time <- strftime(flight$scheduled_arrival_dt, format="%H:%M")
flight$scheduled_departure_dt <- as.numeric(sub("^(\\d+):(\\d+).*", "\\1.\\2", dp_time))
flight$scheduled_arrival_dt <- as.numeric(sub("^(\\d+):(\\d+).*", "\\1.\\2", ar_time))

# choose top 30 airports
top <- c('MDW','BNA','PDX','TPA','IAD','SAN','MIA','BWI','FLL','SLC',  
         'JFK','DCA','BOS','PHL','MSP','MCO','EWR','LAS','LGA','SFO',
         'DTW','IAH','PHX','SEA','LAX','CLT','DEN','DFW','ATL','ORD')

top30 <- flight %>% filter(origin_airport %in% top & destination_airport %in% top)

# remove na
top30 <- na.omit(top30)

table(top30$delay_weather)
table(top30$delay_carrier)
table(top30$delay_late_aircarft_arrival)


# subset delay and ontime for carrier delay
delay <- top30 %>% filter(delay_carrier==1)
ontime <- top30 %>% filter(delay_carrier==0)

set.seed(123456)

train.delay <- sample(nrow(delay),5000)
train.ontime <- sample(nrow(ontime),5000)
train1 <- rbind(delay[train.delay,], ontime[train.ontime,])

test.delay <- sample(nrow(delay[-train.delay]), 1250)
test.ontime <- sample(nrow(ontime[-train.ontime]), 1250)
test1 <- rbind(delay[test.delay,], ontime[test.ontime,])


# subset delay and ontime for late aircraft delay
delay <- top30 %>% filter(delay_late_aircarft_arrival==1)
ontime <- top30 %>% filter(delay_late_aircarft_arrival==0)

set.seed(123456)

train.delay <- sample(nrow(delay),5000)
train.ontime <- sample(nrow(ontime),5000)
train2 <- rbind(delay[train.delay,], ontime[train.ontime,])

test.delay <- sample(nrow(delay[-train.delay]), 1250)
test.ontime <- sample(nrow(ontime[-train.ontime]), 1250)
test2 <- rbind(delay[test.delay,], ontime[test.ontime,])



# subset delay and ontime for weather delay
delay <- top30 %>% filter(delay_weather==1)
ontime <- top30 %>% filter(delay_weather==0)

set.seed(123456)

train.delay <- sample(nrow(delay),914)
train.ontime <- sample(nrow(ontime),914)
train3 <- rbind(delay[train.delay,], ontime[train.ontime,])

test.delay <- sample(nrow(delay[-train.delay]), 229)
test.ontime <- sample(nrow(ontime[-train.ontime]), 229)
test3 <- rbind(delay[test.delay,], ontime[test.ontime,])





## Random forest ####
library(randomForest)

### Run for training and testing - 1 ###
train1 <- train1[,-c(2:3,6:7)]
train1$carrier_code <- as.factor(train1$carrier_code)
train1$day <- as.factor(train1$day)
train1$weekday <- as.factor(train1$weekday)
train1$delay_carrier <- as.factor(train1$delay_carrier)

test1 <- test1[,-c(2:3,6:7)]
test1$carrier_code <- as.factor(test1$carrier_code)
test1$day <- as.factor(test1$day)
test1$weekday <- as.factor(test1$weekday)
test1$delay_carrier <- as.factor(test1$delay_carrier)


#  Now we compute a bagged model default
set.seed(123456)
bag.train1 <- randomForest(delay_carrier ~ ., data = train1, importance = TRUE)
bag.train1 # OOB estimate of error rate: 40.83%
plot(bag.train1)
legend("topright", colnames(bag.train1$err.rate),col=1:4,cex=0.8,fill=1:4)

plot(bag.train1$err.rate)

# find the lowest error at ntree=200
set.seed(123456)
bag.train1.150 <- randomForest(delay_carrier ~ ., data = train1, ntree = 150, importance = TRUE)
bag.train1.150 # OOB estimate of error rate: 40.89%
plot(bag.train1.150)
legend("topright", colnames(bag.train1.150$err.rate),col=1:4,cex=0.8,fill=1:4)


# Compute testing data
yhat.bag.150 <- predict(bag.train1.150, test1)
tab.bag.150 <- table(test1$delay_carrier, yhat.bag.150)
tab.bag.150
err.bag.150 <- mean(test1$delay_carrier != yhat.bag.150)
err.bag.150 # 33.96%


# plot 
#par(mfrow=c(1,2))
importance(bag.train1.150,type = 1)[,1] %>% barplot(cex.names=0.8,main = "MeanDecreaseAccuracy")
importance(bag.train1.150,type = 2)[,1] %>% barplot(cex.names=0.8,main = "MeanDecreaseGini")
varImpPlot(bag.train1.150, main = "Variable Importance Plot")
importance(bag.train1.150)
sort(importance(bag.train1.150)[,3], decreasing = TRUE)[1:10] #MeanDecreaseAccuracy
sort(importance(bag.train1.150)[,4], decreasing = TRUE)[1:10] #MeanDecreaseGini



### Run for training and testing - 2 ###
train2 <- train2[,-c(2:3,5:6)]
train2$day <- as.factor(train2$day)
train2$weekday <- as.factor(train2$weekday)
train2$carrier_code <- as.factor(train2$carrier_code)
train2$delay_late_aircarft_arrival <- as.factor(train2$delay_late_aircarft_arrival)

test2 <- test2[,-c(2:3,5:6)]
test2$day <- as.factor(test2$day)
test2$weekday <- as.factor(test2$weekday)
test2$carrier_code <- as.factor(test2$carrier_code)
test2$delay_late_aircarft_arrival <- as.factor(test2$delay_late_aircarft_arrival)


#  Now we compute a bagged model default
set.seed(123456)
bag.train2 <- randomForest(delay_late_aircarft_arrival ~ ., data = train2, importance = TRUE)
bag.train2 # OOB estimate of  error rate: 31.53%
plot(bag.train2)
legend("topright", colnames(bag.train2$err.rate),col=1:4,cex=0.8,fill=1:4)
plot(bag.train2$err.rate)

# find the lowest error at ntree=100
set.seed(123456)
bag.train2.100 <- randomForest(delay_late_aircarft_arrival ~ ., data = train2, 
                               ntree = 100, importance = TRUE)
bag.train2.100 # OOB estimate of error rate: 32.6%
plot(bag.train2.100)
legend("topright", colnames(bag.train2.100$err.rate),col=1:4,cex=0.8,fill=1:4)

# Compute testing data
yhat.bag.100 <- predict(bag.train2.100, test2)
tab.bag.100 <- table(test2$delay_late_aircarft_arrival, yhat.bag.100)
tab.bag.100
err.bag.100 <- mean(test2$delay_late_aircarft_arrival != yhat.bag.100)
err.bag.100 # 28.28%

# plot 
#par(mfrow=c(1,2))
importance(bag.train2.100,type = 1)[,1] %>% barplot(cex.names=0.8,main = "MeanDecreaseAccuracy")
importance(bag.train2.100,type = 2)[,1] %>% barplot(cex.names=0.8,main = "MeanDecreaseGini")
varImpPlot(bag.train2.100, main = "Variable Importance Plot")
importance(bag.train2.100)
sort(importance(bag.train2.100)[,3], decreasing = TRUE)[1:10] #MeanDecreaseAccuracy
sort(importance(bag.train2.100)[,4], decreasing = TRUE)[1:10] #MeanDecreaseGini





### Run for training and testing - 3 ###
train3 <- train3[,-c(2:3,5,7)]
train3$day <- as.factor(train3$day)
train3$weekday <- as.factor(train3$weekday)
train3$carrier_code <- as.factor(train3$carrier_code)
train3$delay_weather <- as.factor(train3$delay_weather)

test3 <- test3[,-c(2:3,5,7)]
test3$day <- as.factor(test3$day)
test3$weekday <- as.factor(test3$weekday)
test3$carrier_code <- as.factor(test3$carrier_code)
test3$delay_weather <- as.factor(test3$delay_weather)

#  Now we compute a bagged model default
set.seed(123456)
bag.train3 <- randomForest(delay_weather ~ ., data = train3, importance = TRUE)
bag.train3 # OOB estimate of  error rate: 17.83%
plot(bag.train3)
legend("topright", colnames(bag.train3$err.rate),col=1:4,cex=0.8,fill=1:4)
plot(bag.train3$err.rate)

# find the lowest error at ntree=100
set.seed(123456)
bag.train3.80 <- randomForest(delay_weather ~ ., data = train3, 
                              ntree = 80, importance = TRUE)

bag.train3.80 # OOB estimate of error rate: 18.38%
plot(bag.train3.80)
legend("topright", colnames(bag.train3.80$err.rate),col=1:4,cex=0.8,fill=1:4)


# Compute testing data
yhat.bag.80 <- predict(bag.train3.80, test3)
tab.bag.80 <- table(test3$delay_weather, yhat.bag.80)
tab.bag.80
err.bag.80 <- mean(test3$delay_weather != yhat.bag.80)
err.bag.80 # 9.38%

# plot 
#par(mfrow=c(1,2))
importance(bag.train3.80,type = 1)[,1] %>% barplot(cex.names=0.8,main = "MeanDecreaseAccuracy")
importance(bag.train3.80,type = 2)[,1] %>% barplot(cex.names=0.8,main = "MeanDecreaseGini")
varImpPlot(bag.train3.80, main = "Variable Importance Plot")
importance(bag.train3.80)
sort(importance(bag.train3.80)[,3], decreasing = TRUE)[1:10] #MeanDecreaseAccuracy
sort(importance(bag.train3.80)[,4], decreasing = TRUE)[1:10] #MeanDecreaseGini















