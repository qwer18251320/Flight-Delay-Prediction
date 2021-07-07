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


# change airport code from text to number 
code <- data.frame(airport = unique(flight$destination_airport))
code$num <- as.numeric(code$airport)

flight$origin_num <- as.numeric(flight$origin_airport)
flight$dest_num <- as.numeric(flight$destination_airport)

write.csv(flight, 'flight.csv')
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


## export for NN ####
train1$carrier_num <- as.numeric(train1$carrier_code)
test1$carrier_num <- as.numeric(test1$carrier_code)
train1 <- train1[c(5,4,8:24)]
test1 <- test1[c(5,4,8:24)]
write.csv(train1, 'train1.csv')
write.csv(test1, 'test1.csv')

train2$carrier_num <- as.numeric(train2$carrier_code)
test2$carrier_num <- as.numeric(test2$carrier_code)
train2 <- train2[c(7,4,8:24)]
test2 <- test2[c(7,4,8:24)]
write.csv(train2, 'train2.csv')
write.csv(test2, 'test2.csv')

train3$carrier_num <- as.numeric(train3$carrier_code)
test3$carrier_num <- as.numeric(test3$carrier_code)
train3 <- train3[c(6,4,8:24)]
test3 <- test3[c(6,4,8:24)]
write.csv(train3, 'train3.csv')
write.csv(test3, 'test3.csv')


## logistic regression ####

## carrier delay
# factor variables
train1 <- train1[c(5,1:4,8:21)]
test1 <- test1[c(5,1:4,8:21)]
train1$origin_airport <- as.factor(train1$origin_airport)
train1$destination_airport <- as.factor(train1$destination_airport)
train1$carrier_code <- as.factor(train1$carrier_code)
train1$weekday <- as.factor(train1$weekday)
train1$day <- as.factor(train1$day)
test1$day <- as.factor(test1$day)
test1$carrier_code <- as.factor(test1$carrier_code)
test1$weekday <- as.factor(test1$weekday)

# use all variables and choose most significant variables
glm1 <- glm(delay_carrier~., train1, family = "binomial")
sum_glm1 <- summary(glm1)
sum_glm1$coefficients[sum_glm1$coefficients[,4]<0.05,]

# build glm
glm_carrier <- glm(delay_carrier ~ carrier_code + scheduled_elapsed_time + day + scheduled_departure_dt +
              scheduled_arrival_dt + HourlyDryBulbTemperature_x + HourlyPrecipitation_x+
              HourlyStationPressure_x + HourlyVisibility_x + HourlyWindSpeed_x +
              HourlyDryBulbTemperature_y + HourlyPrecipitation_y + HourlyStationPressure_y +
              HourlyVisibility_y + HourlyWindSpeed_y, train1, family = "binomial")


summary(glm_carrier)
1 - (glm_carrier$deviance / glm_carrier$null.deviance)

# predict test
pred1 <- predict(glm_carrier, test1, type = "response")  
pred1 <- ifelse(pred1>0.5,1,0)
confusionMatrix(table(pred1, test1[,"delay_carrier"]))  # 0.6132

err <- mean(test1$delay_carrier != pred1)
err  #  0.3868

write.csv(data.frame(exp(summary(glm_carrier)$coefficients[,1])), 'carrier.csv')


## late aircraft delay
train2 <- train2[c(7,1:4,8:21)]
test2 <- test2[c(7,1:4,8:21)]

train2$origin_airport <- as.factor(train2$origin_airport)
train2$destination_airport <- as.factor(train2$destination_airport)
train2$carrier_code <- as.factor(train2$carrier_code)
train2$weekday <- as.factor(train2$weekday)
train2$day <- as.factor(train2$day)

test2$origin_airport <- as.factor(test2$origin_airport)
test2$destination_airport <- as.factor(test2$destination_airport)
test2$carrier_code <- as.factor(test2$carrier_code)
test2$weekday <- as.factor(test2$weekday)
test2$day <- as.factor(test2$day)

# use all variables and choose the most significant variables
glm2 <- glm(delay_late_aircarft_arrival~.-destination_airport, train2, 
            family = "binomial")
sum_glm2 <- summary(glm2)
sum_glm2$coefficients[sum_glm2$coefficients[,4]<0.05,]

# build model 
glm_aircraft <- glm(delay_late_aircarft_arrival ~ carrier_code + origin_airport + scheduled_elapsed_time + day +
                      scheduled_departure_dt + scheduled_arrival_dt + HourlyDryBulbTemperature_x +
                      HourlyPrecipitation_x+ HourlyStationPressure_x + HourlyVisibility_x +
                      HourlyWindSpeed_x + HourlyPrecipitation_y + HourlyWindSpeed_y, 
                    train2, family = "binomial")

summary(glm_aircraft)
1 - (glm_aircraft$deviance / glm_aircraft$null.deviance)

# predict test
pred2 <- predict(glm_aircraft, test2, type = "response")  
pred2 <- ifelse(pred2>0.5,1,0)
confusionMatrix(table(pred2, test2[,"delay_late_aircarft_arrival"]))  # 0.6784

err <- mean(test2$delay_late_aircarft_arrival != pred2)
err  #  0.3216

write.csv(data.frame(exp(summary(glm_aircraft)$coefficients[,1])), 'aircraft.csv')


## weather delay
train3 <- train3[c(6,1:4,8:21)]
test3 <- test3[c(6,1:4,8:21)]

train3$origin_airport <- as.factor(train3$origin_airport)
train3$destination_airport <- as.factor(train3$destination_airport)
train3$carrier_code <- as.factor(train3$carrier_code)
train3$weekday <- as.factor(train3$weekday)
train3$day <- as.factor(train3$day)

test3$origin_airport <- as.factor(test3$origin_airport)
test3$destination_airport <- as.factor(test3$destination_airport)
test3$carrier_code <- as.factor(test3$carrier_code)
test3$weekday <- as.factor(test3$weekday)
test3$day <- as.factor(test3$day)

# use all variables and choose the most significant variables
glm3 <- glm(delay_weather~.-carrier_code, train3, 
            family = "binomial")
sum_glm3 <- summary(glm3)
sum_glm3$coefficients[sum_glm3$coefficients[,4]<0.05,]


# build model 
glm_weather <- glm(delay_weather ~ origin_airport + scheduled_elapsed_time + day +
                     scheduled_departure_dt + scheduled_arrival_dt + HourlyDryBulbTemperature_x +
                     HourlyPrecipitation_x+ HourlyStationPressure_x + HourlyVisibility_x +
                     HourlyWindSpeed_x + HourlyPrecipitation_y + HourlyWindSpeed_y +
                     HourlyStationPressure_y + HourlyVisibility_y,
                   train3, family = "binomial")

summary(glm_weather)
1 - (glm_weather$deviance / glm_weather$null.deviance)

# predict test
pred3 <- predict(glm_weather, test3, type = "response")  
pred3 <- ifelse(pred3>0.5,1,0)
confusionMatrix(table(pred3, test3[,"delay_weather"]))  # 0.821    

err <- mean(test3$delay_weather != pred3)
err  # 0.1790393


a <- summary(glm_weather)$coefficients[summary(glm_weather)$coefficients[,4]<0.05,]
write.csv(data.frame(exp(a[,1])), 'weather.csv')

## Correlation ####
library(Hmisc)
library(corrplot)

# carrier delay
carrier <- top30[c(5,4,8:21)]
corr <- rcorr(as.matrix(carrier))
corrplot(corr$r, type="upper", order="alphabet", 
         p.mat = corr$P, sig.level = 0.05, insig = "blank")

# aircraft delay
aircraft <- top30[c(7,4,8:21)]
corr <- rcorr(as.matrix(aircraft))
corrplot(corr$r, type="upper", order="alphabet", 
         p.mat = corr$P, sig.level = 0.05, insig = "blank")


# weather delay
weather <- top30[c(6,4,8:21)]
corr <- rcorr(as.matrix(weather))
corrplot(corr$r, type="upper", order="alphabet", 
         p.mat = corr$P, sig.level = 0.05, insig = "blank")

# all together
test <- top30[c(5,7,6,4,8:21)]
corr <- rcorr(as.matrix(test))
corrplot(corr$r, type="upper", order="alphabet", 
         p.mat = corr$P, sig.level = 0.05, insig = "blank")


