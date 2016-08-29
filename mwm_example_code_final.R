
## Install Libraries (1 time)

#install.packages("VGAM")
#install.packages("pbkrtest")
#install.packages("e1071")
#install.packages("caret")
#install.packages("mda")
#install.packages("MASS")
#install.packages("klaR")
#install.packages("nnet")
#install.packages("kernlab")
#install.packages("rpart")
#install.packages("RWeka")
#install.packages("ipred")
#install.packages("randomForest")
#install.packages("gbm")
#install.packages("C50")
#install.packages("plotly")
#install.packages(reshape2)
#install.packages(ggplot2)
#install.packages(mlbench)
#install.packages("caret")
#install.packages("caretEnsemble")

## Load Libraries

library(C50)
library(gbm)
library(randomForest)
library(ipred)
library(RWeka)
library(rpart)
library(kernlab)
library(nnet)
library(klaR)
library(mda)
library(VGAM)
library(MASS)
library(e1071)
library(plotly)
library(reshape2)
library(ggplot2)
library(mlbench)
library(caret)
library(caretEnsemble)

###############################################################
# FUNCTION: resample_model                                    #
# run models on data, with optional resamples                 #
# Input: number of resamples, rate of split, models to use    #
# Output: Data frame of model performance on test set         #
###############################################################

resample_model <- function(n_sample=1,r_sample=0.75,model_use=list(1,16)) {
  
  perf_df_test <-  perf_df_train <- data.frame(model_id= integer(0)
                                               ,model_name= character(0)
                                               ,model_it= integer(0)
                                               ,Accuracy= numeric(0)
                                               ,Kappa = numeric(0)
                                               ,Mean_Sensitivity = numeric(0)
                                               ,Mean_Specificity = numeric(0)
                                               ,Mean_Pos_Pred_Value = numeric(0)
                                               ,Mean_Neg_Pred_Value = numeric(0)
                                               ,Mean_Detection_Rate = numeric(0)
                                               ,Mean_Balanced_Accuracy = numeric(0))
  
  
  for(i in 1:n_sample){
    #  i<-1
    
    model_it <- i
    train_ratio <- r_sample
    #define % of training and test set
    trainindex <- sample(1:nrow(data_in),floor(nrow(data_in)*train_ratio))     #Random sample of rows for training set      
    
    data_in_train <- data_in[trainindex, ]   #get training set
    data_in_test <- data_in[-trainindex, ]     #get test set                                   
    
    
    x_train <- data_in_train[,pred_vars]
    x_test <- data_in_test[,pred_vars]
    
    ###################################
    ###model  mn logistic regression###
    ###################################
    
    #Score model
    if(1 %in% model_use){
      
      model_id <- 1
      model_name <- 'mn logistic regression'
      model <- vglm(Formula,family = "multinomial",data=data_in_train)
      
      probability <- predict(model,x_train,type="response")
      data_in_train$pred_mod<-apply(probability,1,which.max)
      data_in_train$pred_mod <- factor(data_in_train$pred_mod,
                                       levels = fact_levels,
                                       labels = fact_labels)
      
      
      probability <- predict(model,x_test,type="response")
      data_in_test$pred_mod<-apply(probability,1,which.max)
      data_in_test$pred_mod <- factor(data_in_test$pred_mod,
                                      levels = fact_levels,
                                      labels = fact_labels)
      
      #TRAIN performance
      
      data_tmp <- data_in_train[,c(resp_var, "pred_mod")]
      names(data_tmp) <- c("obs", "pred")
      tmp <- cbind(model_id , model_name, model_it, data.frame(
        t(multiClassSummary(data_tmp, lev = levels(eval(parse(text=paste0('data_in_train$',resp_var))))))
      ))
      perf_df_train <- rbind(perf_df_train,tmp)
      
      #TEST performance
      
      data_tmp <- data_in_test[,c(resp_var, "pred_mod")]
      names(data_tmp) <- c("obs", "pred")
      tmp <- cbind(model_id , model_name, model_it, data.frame(
        t(multiClassSummary(data_tmp, lev = levels(eval(parse(text=paste0('data_in_test$',resp_var))))))
      ))
      perf_df_test <- rbind(perf_df_test,tmp)
      
    }  
    
    ###################################
    ###model  linear discriminant  ###
    ###################################
    
    
    if(2 %in% model_use){
      
      model_id <- 2
      model_name= 'linear discriminant'
      
      model<-lda(Formula,data=data_in_train)
      
      data_in_train$pred_mod <-data.frame(predict(model,x_train, type="response"))[,c('class')]
      data_in_test$pred_mod <-data.frame(predict(model,x_test, type="response"))[,c('class')]
      
      #TRAIN performance
      
      data_tmp <- data_in_train[,c(resp_var, "pred_mod")]
      names(data_tmp) <- c("obs", "pred")
      tmp <- cbind(model_id , model_name, model_it, data.frame(
        t(multiClassSummary(data_tmp, lev = levels(eval(parse(text=paste0('data_in_train$',resp_var))))))
      ))
      perf_df_train <- rbind(perf_df_train,tmp)
      
      #TEST performance
      
      data_tmp <- data_in_test[,c(resp_var, "pred_mod")]
      names(data_tmp) <- c("obs", "pred")
      tmp <- cbind(model_id , model_name, model_it, data.frame(
        t(multiClassSummary(data_tmp, lev = levels(eval(parse(text=paste0('data_in_test$',resp_var))))))
      ))
      perf_df_test <- rbind(perf_df_test,tmp)
      
    } 
    
    ###################################
    ###model mixture discriminant ###
    ###################################
    #Score model
    if(3 %in% model_use){ 
      
      model_id <- 3
      model_name= 'mixture discriminant'
      model<-mda(Formula,data=data_in_train)
      
      data_in_train$pred_mod <-predict(model,x_train)
      data_in_test$pred_mod <-predict(model,x_test)
      
      #TRAIN performance
      
      data_tmp <- data_in_train[,c(resp_var, "pred_mod")]
      names(data_tmp) <- c("obs", "pred")
      tmp <- cbind(model_id , model_name, model_it, data.frame(
        t(multiClassSummary(data_tmp, lev = levels(eval(parse(text=paste0('data_in_train$',resp_var))))))
      ))
      perf_df_train <- rbind(perf_df_train,tmp)
      
      #TEST performance
      
      data_tmp <- data_in_test[,c(resp_var, "pred_mod")]
      names(data_tmp) <- c("obs", "pred")
      tmp <- cbind(model_id , model_name, model_it, data.frame(
        t(multiClassSummary(data_tmp, lev = levels(eval(parse(text=paste0('data_in_test$',resp_var))))))
      ))
      perf_df_test <- rbind(perf_df_test,tmp)
      
    } 
    
    ###################################
    ###model regularized discriminant ###
    ###################################
    
    #Score model
    if(4 %in% model_use){
      
      model_id <- 4
      model_name= 'regularized discriminant'
      
      model <- rda(Formula,data=data_in_train,gamma = 0.05,lambda = 0.01)
      
      data_in_train$pred_mod <-data.frame(predict(model,x_train, type="response"))[,c('class')]
      data_in_test$pred_mod <-data.frame(predict(model,x_test, type="response"))[,c('class')]
      
      #TRAIN performance
      ?multiClassSummary
      data_tmp <- data_in_train[,c(resp_var, "pred_mod")]
      
      
      names(data_tmp) <- c("obs", "pred")
      tmp <- cbind(model_id , model_name, model_it, data.frame(
        t(multiClassSummary(data_tmp, lev = levels(eval(parse(text=paste0('data_in_train$',resp_var))))))
      ))
      perf_df_train <- rbind(perf_df_train,tmp)
      
      #TEST performance
      
      data_tmp <- data_in_test[,c(resp_var, "pred_mod")]
      names(data_tmp) <- c("obs", "pred")
      tmp <- cbind(model_id , model_name, model_it, data.frame(
        t(multiClassSummary(data_tmp, lev = levels(eval(parse(text=paste0('data_in_test$',resp_var))))))
      ))
      perf_df_test <- rbind(perf_df_test,tmp)
      
    }
    
    
    ###################################
    ###model neural network ###
    ###################################
    #Score model
    if(5 %in% model_use){
      
      model_id <- 5
      model_name= 'neural network'
      
      model <- nnet(Formula,data=data_in_train,size = 4,decay = 0.0001,maxit = 500)
      data_in_train$pred_mod <- as.factor(predict(model,x_train,type="class"))
      data_in_test$pred_mod <-  as.factor(predict(model,x_test,type="class"))
      
      #TRAIN performance
      
      data_tmp <- data_in_train[,c(resp_var, "pred_mod")]
      names(data_tmp) <- c("obs", "pred")
      tmp <- cbind(model_id , model_name, model_it, data.frame(
        t(multiClassSummary(data_tmp, lev = levels(eval(parse(text=paste0('data_in_train$',resp_var))))))
      ))
      perf_df_train <- rbind(perf_df_train,tmp)
      
      #TEST performance
      
      data_tmp <- data_in_test[,c(resp_var, "pred_mod")]
      names(data_tmp) <- c("obs", "pred")
      
      tmp <- cbind(model_id , model_name, model_it, data.frame(
        t(multiClassSummary(data_tmp, lev = levels(eval(parse(text=paste0('data_in_test$',resp_var))))))
      ))
      perf_df_test <- rbind(perf_df_test,tmp)
      
    }
    
    ###################################
    ###model flexible discriminant ###
    ###################################
    
    #Score model
    if(6 %in% model_use){
      
      model_id <- 6  
      model_name= 'flexible discriminant'
      model6 <- fda(Formula,data=data_in_train)
      data_in_train$pred_mod6 <-predict(model6,x_train,type="class")
      data_in_test$pred_mod6 <-predict(model6,x_test,type="class")
      
      #TRAIN performance
      
      data_tmp <- data_in_train[,c(resp_var, "pred_mod")]
      names(data_tmp) <- c("obs", "pred")
      tmp <- cbind(model_id , model_name, model_it, data.frame(
        t(multiClassSummary(data_tmp, lev = levels(eval(parse(text=paste0('data_in_train$',resp_var))))))
      ))
      perf_df_train <- rbind(perf_df_train,tmp)
      
      #TEST performance
      
      data_tmp <- data_in_test[,c(resp_var, "pred_mod")]
      names(data_tmp) <- c("obs", "pred")
      tmp <- cbind(model_id , model_name, model_it, data.frame(
        t(multiClassSummary(data_tmp, lev = levels(eval(parse(text=paste0('data_in_test$',resp_var))))))
      ))
      perf_df_test <- rbind(perf_df_test,tmp)
      
    }
    
    ###################################
    ###model Support Vector Machine ###
    ###################################
    
    #Score model
    if(7 %in% model_use){
      
      model_id <- 7
      model_name <- 'support vector machine'
      model <- ksvm(Formula,data=data_in_train)
      data_in_train$pred_mod<-predict(model,x_train,type="response")
      data_in_test$pred_mod<-predict(model,x_test,type="response")
      
      #TRAIN performance
      
      data_tmp <- data_in_train[,c(resp_var, "pred_mod")]
      names(data_tmp) <- c("obs", "pred")
      tmp <- cbind(model_id , model_name, model_it, data.frame(
        t(multiClassSummary(data_tmp, lev = levels(eval(parse(text=paste0('data_in_train$',resp_var))))))
      ))
      perf_df_train <- rbind(perf_df_train,tmp)
      
      #TEST performance
      
      data_tmp <- data_in_test[,c(resp_var, "pred_mod")]
      names(data_tmp) <- c("obs", "pred")
      tmp <- cbind(model_id , model_name, model_it, data.frame(
        t(multiClassSummary(data_tmp, lev = levels(eval(parse(text=paste0('data_in_test$',resp_var))))))
      ))
      perf_df_test <- rbind(perf_df_test,tmp)
      
    }
    
    ###################################
    ###model k-nearest neighbors    ###
    ###################################
    
    
    #Score model
    if(8 %in% model_use){
      
      model_id <- 8
      model_name <- 'k-nearest neighbors'
      model <- knn3(Formula,data=data_in_train,k=5)
      data_in_train$pred_mod <-predict(model,x_train,type="class")
      data_in_test$pred_mod <-predict(model,x_test,type="class")
      
      #TRAIN performance
      
      data_tmp <- data_in_train[,c(resp_var, "pred_mod")]
      names(data_tmp) <- c("obs", "pred")
      tmp <- cbind(model_id , model_name, model_it, data.frame(
        t(multiClassSummary(data_tmp, lev = levels(eval(parse(text=paste0('data_in_train$',resp_var))))))
      ))
      perf_df_train <- rbind(perf_df_train,tmp)
      
      
      #TEST performance
      
      data_tmp <- data_in_test[,c(resp_var, "pred_mod")]
      names(data_tmp) <- c("obs", "pred")
      tmp <- cbind(model_id , model_name, model_it, data.frame(
        t(multiClassSummary(data_tmp, lev = levels(eval(parse(text=paste0('data_in_test$',resp_var))))))
      ))
      perf_df_test <- rbind(perf_df_test,tmp)
      
    }
    
    ###################################
    ###model naive bayes   ###
    ###################################
    
    
    #Score model
    if(9 %in% model_use){
      
      model_id <- 9  
      model_name= 'naive bayes'
      model9 <-naiveBayes(Formula,data=data_in_train,k=5)
      data_in_train$pred_mod9 <- predict(model9,x_train)
      data_in_test$pred_mod9 <- predict(model9,x_test)
      
      #TRAIN performance
      
      data_tmp <- data_in_train[,c(resp_var, "pred_mod")]
      names(data_tmp) <- c("obs", "pred")
      tmp <- cbind(model_id , model_name, model_it, data.frame(
        t(multiClassSummary(data_tmp, lev = levels(eval(parse(text=paste0('data_in_train$',resp_var))))))
      ))
      perf_df_train <- rbind(perf_df_train,tmp)
      
      
      #TEST performance
      
      data_tmp <- data_in_test[,c(resp_var, "pred_mod")]
      names(data_tmp) <- c("obs", "pred")
      tmp <- cbind(model_id , model_name, model_it, data.frame(
        t(multiClassSummary(data_tmp, lev = levels(eval(parse(text=paste0('data_in_test$',resp_var))))))
      ))
      perf_df_test <- rbind(perf_df_test,tmp)
      
    }
    
    
    ###################################
    ###model classification and regression tree   ###
    ###################################
    
    
    #Score model
    if(10 %in% model_use){
      
      model_id <- 10
      
      model_name= 'classification and regression tree'
      model <- rpart(Formula,data=data_in_train)
      data_in_train$pred_mod <-predict(model,x_train,type="class")
      data_in_test$pred_mod <-predict(model,x_test,type="class")
      
      
      #TRAIN performance
      
      data_tmp <- data_in_train[,c(resp_var, "pred_mod")]
      names(data_tmp) <- c("obs", "pred")
      tmp <- cbind(model_id , model_name, model_it, data.frame(
        t(multiClassSummary(data_tmp, lev = levels(eval(parse(text=paste0('data_in_train$',resp_var))))))
      ))
      perf_df_train <- rbind(perf_df_train,tmp)
      
      
      #TEST performance
      
      data_tmp <- data_in_test[,c(resp_var, "pred_mod")]
      names(data_tmp) <- c("obs", "pred")
      tmp <- cbind(model_id , model_name, model_it, data.frame(
        t(multiClassSummary(data_tmp, lev = levels(eval(parse(text=paste0('data_in_test$',resp_var))))))
      ))
      perf_df_test <- rbind(perf_df_test,tmp)
      
    }
    
    ###################################
    ###model tree C4.5   ###
    ###################################
    
    
    #Score model
    if(11 %in% model_use){
      
      model_id <- 11
      
      
      model_name= 'tree C4.5'
      model <-J48(Formula,data=data_in_train)
      data_in_train$pred_mod <-predict(model ,x_train)
      data_in_test$pred_mod <-predict(model ,x_test)
      
      #TRAIN performance
      
      data_tmp <- data_in_train[,c(resp_var, "pred_mod")]
      names(data_tmp) <- c("obs", "pred")
      tmp <- cbind(model_id , model_name, model_it, data.frame(
        t(multiClassSummary(data_tmp, lev = levels(eval(parse(text=paste0('data_in_train$',resp_var))))))
      ))
      perf_df_train <- rbind(perf_df_train,tmp)
      
      
      #TEST performance
      
      data_tmp <- data_in_test[,c(resp_var, "pred_mod")]
      names(data_tmp) <- c("obs", "pred")
      tmp <- cbind(model_id , model_name, model_it, data.frame(
        t(multiClassSummary(data_tmp, lev = levels(eval(parse(text=paste0('data_in_test$',resp_var))))))
      ))
      perf_df_test <- rbind(perf_df_test,tmp)
      
    }
    
    
    ###################################
    ###model tree C4.5   ###
    ###################################
    
    
    #Score model
    if(12 %in% model_use){
      
      model_id <- 12
      model_name= 'PART'
      
      model <-PART(Formula,data=data_in_train)
      data_in_train$pred_mod <-predict(model,x_train)
      data_in_test$pred_mod <-predict(model,x_test)
      
      #TRAIN performance
      
      data_tmp <- data_in_train[,c(resp_var, "pred_mod")]
      names(data_tmp) <- c("obs", "pred")
      tmp <- cbind(model_id , model_name, model_it, data.frame(
        t(multiClassSummary(data_tmp, lev = levels(eval(parse(text=paste0('data_in_train$',resp_var))))))
      ))
      perf_df_train <- rbind(perf_df_train,tmp)
      
      
      #TEST performance
      
      data_tmp <- data_in_test[,c(resp_var, "pred_mod")]
      names(data_tmp) <- c("obs", "pred")
      tmp <- cbind(model_id , model_name, model_it, data.frame(
        t(multiClassSummary(data_tmp, lev = levels(eval(parse(text=paste0('data_in_test$',resp_var))))))
      ))
      perf_df_test <- rbind(perf_df_test,tmp)
      
    }
    
    ###################################
    ###model bagging CART   ###
    ###################################
    
    
    #Score model
    if(13 %in% model_use){
      
      model_id <- 13
      
      model_name= 'bagging CART'
      model <-ipred:::bagging(Formula,data=data_in_train)
      data_in_train$pred_mod <- predict(model ,x_train)
      data_in_test$pred_mod <- predict(model ,x_test)
      
      #TRAIN performance
      
      data_tmp <- data_in_train[,c(resp_var, "pred_mod")]
      names(data_tmp) <- c("obs", "pred")
      str(data_tmp)
      tmp <- cbind(model_id , model_name, model_it, data.frame(
        t(multiClassSummary(data_tmp, lev = levels(eval(parse(text=paste0('data_in_train$',resp_var))))))
      ))
      perf_df_train <- rbind(perf_df_train,tmp)
      
      
      #TEST performance
      
      data_tmp <- data_in_test[,c(resp_var, "pred_mod")]
      names(data_tmp) <- c("obs", "pred")
      tmp <- cbind(model_id , model_name, model_it, data.frame(
        t(multiClassSummary(data_tmp, lev = levels(eval(parse(text=paste0('data_in_test$',resp_var))))))
      ))
      perf_df_test <- rbind(perf_df_test,tmp)
      
    }
    
    ###################################
    ###model random forest   ###
    ###################################
    
    
    #Score model
    if(14 %in% model_use){
      
      model_id <- 14  
      
      model_name= 'random forest'
      model <-randomForest(Formula,data=data_in_train)
      data_in_train$pred_mod <-predict(model ,x_train)
      data_in_test$pred_mod <-predict(model ,x_test)
      
      #TRAIN performance
      
      data_tmp <- data_in_train[,c(resp_var, "pred_mod")]
      names(data_tmp) <- c("obs", "pred")
      tmp <- cbind(model_id , model_name, model_it, data.frame(
        t(multiClassSummary(data_tmp, lev = levels(eval(parse(text=paste0('data_in_train$',resp_var))))))
      ))
      perf_df_train <- rbind(perf_df_train,tmp)
      
      
      #TEST performance
      
      data_tmp <- data_in_test[,c(resp_var, "pred_mod")]
      names(data_tmp) <- c("obs", "pred")
      tmp <- cbind(model_id , model_name, model_it, data.frame(
        t(multiClassSummary(data_tmp, lev = levels(eval(parse(text=paste0('data_in_test$',resp_var))))))
      ))
      perf_df_test <- rbind(perf_df_test,tmp)
      
    }
    
    
    ###################################
    ###model gradient boosted machine  ###
    ###################################
    
    
    #Score model
    if(15 %in% model_use){
      
      model_id <- 15  
      
      
      model_name= 'gradient boosted machine'
      model <-gbm(Formula,data=data_in_train,distribution="multinomial")
      probability<-predict(model ,x_train,n.trees=1)
      data_in_train$pred_mod  <- as.factor(colnames(probability)[apply(probability,1,which.max)])
      
      probability<-predict(model ,x_test,n.trees=1)
      data_in_test$pred_mod  <-  as.factor(colnames(probability)[apply(probability,1,which.max)])
      
      #TRAIN performance
      
      data_tmp <- data_in_train[,c(resp_var, "pred_mod")]
      names(data_tmp) <- c("obs", "pred")
      str(data_tmp)
      tmp <- cbind(model_id , model_name, model_it, data.frame(
        t(multiClassSummary(data_tmp, lev = levels(eval(parse(text=paste0('data_in_train$',resp_var))))))
      ))
      perf_df_train <- rbind(perf_df_train,tmp)
      
      
      #TEST performance
      
      data_tmp <- data_in_test[,c(resp_var, "pred_mod")]
      names(data_tmp) <- c("obs", "pred")
      tmp <- cbind(model_id , model_name, model_it, data.frame(
        t(multiClassSummary(data_tmp, lev = levels(eval(parse(text=paste0('data_in_test$',resp_var))))))
      ))
      perf_df_test <- rbind(perf_df_test,tmp)
      
    }
    
    ###################################
    ###model boosted C5.0  ###
    ###################################
    
    
    #Score model
    if(16 %in% model_use){
      
      model_id <- 16  
      
      model_name= 'boosted C5.0'
      
      model  <- C5.0(Formula,data=data_in_train,trials=10)
      data_in_train$pred_mod  <- predict(model ,x_train)
      data_in_test$pred_mod <- predict(model ,x_test)
      
      #TRAIN performance
      
      data_tmp <- data_in_train[,c(resp_var, "pred_mod")]
      names(data_tmp) <- c("obs", "pred")
      tmp <- cbind(model_id , model_name, model_it, data.frame(
        t(multiClassSummary(data_tmp, lev = levels(eval(parse(text=paste0('data_in_train$',resp_var))))))
      ))
      perf_df_train <- rbind(perf_df_train,tmp)
      
      
      #TEST performance
      
      data_tmp <- data_in_test[,c(resp_var, "pred_mod")]
      names(data_tmp) <- c("obs", "pred")
      tmp <- cbind(model_id , model_name, model_it, data.frame(
        t(multiClassSummary(data_tmp, lev = levels(eval(parse(text=paste0('data_in_test$',resp_var))))))
      ))
      perf_df_test <- rbind(perf_df_test,tmp)
      
    }
    ###  
    
  }
  
  return(perf_df_test)
  
}


# Models available to model_use argument
model_refs <- data.frame(model_id = seq(1,16,by=1), model_name = c('mn logistic regression'
                                                                   ,'linear discriminant'
                                                                   ,'mixture discriminant'
                                                                   ,'regularized discriminant'
                                                                   ,'neural network'
                                                                   ,'flexible discriminant'
                                                                   ,'support vector machine'
                                                                   ,'k-nearest neighbors'
                                                                   ,'naive bayes'
                                                                   ,'classification and regression tree'
                                                                   ,'tree C4.5'
                                                                   ,'PART'
                                                                   ,'bagging CART'
                                                                   ,'random forest'
                                                                   ,'gradient boosted machine'
                                                                   ,'boosted C5.0'))
model_refs # show the table

## Get the data


data(PimaIndiansDiabetes)
data_in <- PimaIndiansDiabetes

#summary(data_in)
#str(data_in)

## Identify the predictor and response variables

#colnames(data_in) <- tolower(colnames(data_in))
resp_var <- c("diabetes") #class
pred_vars <- unlist(colnames(data_in[,!(colnames(data_in) %in% resp_var)])) # c("a","b","c")

## Make formula function

Formula <- formula(paste(resp_var,"~ ",  paste(pred_vars, collapse=" + ")))

##response variable values
fact_labels <-levels(eval(parse(text=paste0('data_in$',resp_var))))
fact_levels <- seq(1,length(fact_labels), 1)


######################################################
#           Example 1                                #
#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX#
#1 sample: Did you find the right signal in the noise?
#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX#


set.seed(111) # linear discriminant > logistic > CART >>>
set.seed(1111) # CART > logistic > SVM > linear discriminant >>>
set.seed(11111) # CART > SVM > linear dscriminant > logistic >>>
#set.seed(Sys.time())
out_test_perf <- resample_model(n_sample = 1 ,r_sample = 0.75, model_use=list(
  1,2,7,10))

# Make a quick chart to compare
ggplot(data=out_test_perf, aes(x=model_name, y=Pos_Pred_Value, fill=model_name)) +
  geom_bar(stat="identity") +
  geom_label(aes(label = floor(Pos_Pred_Value*10000)/100, label.size = 0.25))


######################################################
#           Example 2                                #
#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX#
#Many resamples: Which model best describes this data?
#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX#

set.seed(Sys.time())
out_test_perf <- resample_model(n_sample = 1000 ,r_sample = 0.75, model_use=list(
  1,2,7,10))

# Show a boxplot with means
means <- aggregate(Pos_Pred_Value ~  model_name, data=out_test_perf, mean)
ggplot(data=out_test_perf, aes(x=model_name, y=Pos_Pred_Value, fill=model_name)) + geom_boxplot() +
  stat_summary(fun.y=mean, colour="darkred", geom="point", 
               shape=18, size=3) + 
  geom_text(data = means, aes(label = floor(Pos_Pred_Value*10000)/100, y = Pos_Pred_Value*1.01))



# single variable box plots - can extend to all models at once
#d <- melt(out_test_perf[,c("model_name", "Pos_Pred_Value")])
#ggplot(d, aes(x=model_name, y=value, fill=model_name)) + geom_boxplot()

# histogram comparisons
#d <- melt(out_test_perf[,-c(1,3)])
#ggplot(d,aes(x = value, fill=model_name)) + 
#  facet_wrap(~variable,scales = "free_x") +  
#  geom_histogram()

################################################
#           Example 3                          #
#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX#
#           Bagging                            #
#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX#


###########################
# Exisiting bagging model #
###########################

set.seed(111) # Bagging CART > CART
set.seed(1111) # Bagging CART > CART
set.seed(11111) # Bagging CART > CART
#set.seed(Sys.time())
out_test_perf <- resample_model(n_sample = 1 ,r_sample = 0.75, model_use=list(
  10,13))

# Make a quick chart to compare
ggplot(data=out_test_perf, aes(x=model_name, y=Pos_Pred_Value, fill=model_name)) +
  geom_bar(stat="identity") +
  geom_label(aes(label = floor(Pos_Pred_Value*10000)/100, label.size = 0.25))

##################
# Hand made bags #
##################

r_sample     <- 0.75
n_sample     <- 100

train_ratio <- r_sample
#define % of training and test set
trainindex <- sample(1:nrow(data_in),floor(nrow(data_in)*train_ratio))     #Random sample of rows for training set      

data_in_train <- data_in[trainindex, ]   #get training set
data_in_test <- data_in[-trainindex, ]   #get test set                                   

x_train <- data_in_train[,pred_vars]
x_test <- data_in_test[,pred_vars]

predictions <- foreach(m=1:n_sample, .combine=rbind) %do% {
  training_positions <- sample(nrow(data_in_train), size=floor((nrow(data_in_train)*r_sample)))
  
  ## Model to bag ##
  glm_fit <- glm(Formula ,
                 data=data_in_train[training_positions,],
                 family=binomial(logit),
                 control = list(maxit = 25))
  
  ## probability ##
  prob <- predict(glm_fit,
                  newdata=data_in_test, 
                  type="response")
  
  ## classification  (need to change for non-binomial##
  pred <- ifelse(prob>0.5,fact_labels[2],fact_labels[1])
  
  s <- summary(glm_fit) # actual model created
  p <- s$coeff[,4]     
  c <- s$coeff[,1]
  pvalues <- p[p<0.1]   #pval < 0.1
  coeffs  <- c[p<0.1]   #coeffs pval < 0.1
  return(list(prob,pred,pvalues,coeffs))
}

predictions

table(data_in_test$diabetes)

## Bagging w/ average of probability aggregation

prob_agg <- apply(data.frame(as.data.frame(predictions[,1]), row.names = NULL),1,mean)
cl_bagged_avg <- ifelse(prob_agg>0.5,fact_labels[2],fact_labels[1])

df_perf <- data.frame(data_in_test$diabetes,cl_bagged_avg)
confusionMatrix(table(df_perf))


## Bagging w/ voting of classification aggregation


## function to find mode from row of similar columns
Mode <- function(x) {
  ux <- unique(x)
  ux[which.max(tabulate(match(x, ux)))]
}


cl_bagged_vote <- apply(data.frame(as.data.frame(predictions[,2]), row.names = NULL),1,Mode)
table(cl_bagged_vote)

df_perf <- data.frame(data_in_test$diabetes,cl_bagged_vote)
confusionMatrix(table(df_perf))


###############################################
# Bagging from selected R packages
###############################################
# There are several : http://blog.revolutionanalytics.com/2014/04/ensemble-packages-in-r.html
# caret also has bagging processes


r_sample     <- 0.75

train_ratio <- r_sample
#define % of training and test set
trainindex <- sample(1:nrow(data_in),floor(nrow(data_in)*train_ratio))     #Random sample of rows for training set      

data_in_train <- data_in[trainindex, ]   #get training set
data_in_test <- data_in[-trainindex, ]   #get test set                                   

x_train <- data_in_train[,pred_vars]
x_test <- data_in_test[,pred_vars]



library(adabag)

data.train_bagging <- adabag:::bagging(Formula
                                       ,data=data_in_train
                                       ,mfinal=15
                                       ,control=rpart.control(maxdepth=5, minsplit=15))


summary(data.train_bagging)

#Using the pruning option
data.test_bagging.pred <- predict.bagging(data.train_bagging,newdata=data_in_test, newmfinal=10)
data.test_bagging.pred <- predict.bagging(data.train_bagging,newdata=data_in_test)
data.test_bagging.pred$confusion
data.test_bagging.pred$error



################################################
#           Example 4                          #
#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX#
#          Boosting                            #
#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX#


###########################
# Exisiting boosting model #
###########################
#4

set.seed(111) 
out_test_perf <- resample_model(n_sample = 100 ,r_sample = 0.75, model_use=list(10,13,16))

ggplot(data=out_test_perf, aes(x=model_name, y=Pos_Pred_Value, fill=model_name)) +
  geom_bar(stat="identity") +
  geom_label(aes(label = floor(Pos_Pred_Value*10000)/100, label.size = 0.25))


means <- aggregate(Pos_Pred_Value ~  model_name, data=out_test_perf, mean)
ggplot(data=out_test_perf, aes(x=model_name, y=Pos_Pred_Value, fill=model_name)) + geom_boxplot() +
  stat_summary(fun.y=mean, colour="darkred", geom="point", 
               shape=18, size=3) + 
  geom_text(data = means, aes(label = floor(Pos_Pred_Value*10000)/100, y = Pos_Pred_Value*1.01))


###############################################
# Boosting from selected R packages
###############################################



r_sample     <- 0.75

train_ratio <- r_sample
#define % of training and test set
trainindex <- sample(1:nrow(data_in),floor(nrow(data_in)*train_ratio))     #Random sample of rows for training set      

data_in_train <- data_in[trainindex, ]   #get training set
data_in_test <- data_in[-trainindex, ]   #get test set                                   

x_train <- data_in_train[,pred_vars]
x_test <- data_in_test[,pred_vars]


library(adabag)

data.train_boost <- adabag:::boosting(Formula
                                      ,data=data_in_train
                                      ,boos=TRUE
                                      ,mfinal=20
                                      ,coeflearn='Breiman')

summary(data.train_boost)

data.train_boost$formula
data.train_boost$trees
data.train_boost$weights
data.train_boost$importance

data.test_boost.pred <- predict.boosting(data.train_boost,newdata=data_in_test)
data.test_boost.pred$confusion
data.test_boost.pred$error



################################################
#           Example 5                          #
#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX#
#          stacking                            #
#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX#

r_sample     <- 0.75

train_ratio <- r_sample
#define % of training and test set
trainindex <- sample(1:nrow(data_in),floor(nrow(data_in)*train_ratio))     #Random sample of rows for training set      

data_in_train <- data_in[trainindex, ]   #get training set
data_in_test <- data_in[-trainindex, ]   #get test set                                   

x_train <- data_in_train[,pred_vars]
x_test <- data_in_test[,pred_vars]


##################
# Using Caret Package #
##################

# set seed
seed <- 111

####

# create sub-models
control <- trainControl(method="repeatedcv", number=10, repeats=3, savePredictions=TRUE, classProbs=TRUE)
algorithmList <- c('lda', 'rpart', 'knn','rf', 'svmRadial') #, 
set.seed(seed)
models <- caretList(Formula, data=data_in_train, trControl=control, methodList=algorithmList)
results <- resamples(models)
summary(results)
dotplot(results)

mods_eval <- NULL

mods_eval <- data.frame(ensemble_step= rep('sub_mod',length(algorithmList))
                        ,model_type = rownames(summary(results)$statistics$Accuracy)
                        ,model_eval = summary(results)$statistics$Accuracy[,'Mean'])

# correlation between results
modelCor(results)
splom(results)

# supervisor model: glm
stackControl <- trainControl(method="repeatedcv", number=10, repeats=3, savePredictions=TRUE, classProbs=TRUE)
set.seed(seed)
stack.glm <- caretStack(models, method="glm", metric="Accuracy",trControl=stackControl)
print(stack.glm)


mods_eval <- rbind( mods_eval,data.frame(ensemble_step= 'sup_mod'
                                         ,model_type = 'glm'
                                         ,model_eval = max(stack.glm$error['Accuracy'])))

# supervisor model: randomforest
set.seed(seed)
stack.rf <- caretStack(models, method="rf", metric="Accuracy", trControl=stackControl)
print(stack.rf)

mods_eval <- rbind( mods_eval,data.frame(ensemble_step= 'sup_mod'
                                         ,model_type = 'rf'
                                         ,model_eval = max(stack.rf$error['Accuracy'])))

rownames(mods_eval) <- NULL

# show the model evaluation sub and supervisor models
mods_eval$step_label <-paste(mods_eval$ensemble_step,mods_eval$model_type, sep='-')
str(mods_eval)

# visualize the model evaluations
ggplot(data=mods_eval, aes(x=step_label, y=model_eval, group=1, fill=ensemble_step)) +
  geom_line() +
  geom_point() +
  geom_label(aes(label=floor(model_eval*10000)/100))
