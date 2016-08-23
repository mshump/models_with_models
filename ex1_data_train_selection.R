          
#xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx#
# 1) Sampling and resampleing can be powerful (or dangerous)#
# 2) Investigate multiple models                            #
#xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx#




############################
# Install Libraries (1 time)
###########################

#install.packages("VGAM")
#install.packages("caret")
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
#install.packagesmlbench)

############################
# Load Libraries
###########################
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
library(caret)
library(VGAM)
library(MASS)
library(e1071)
library(plotly)
library(reshape2)
library(ggplot2)
library(mlbench)


###############################################################
# FUNCTION: resample_model                                    #
# run models on data, with optional resamples                 #
# Input: number of resamples, rate of split, models to use    #
# Output: Data frame of model performance on test set         #
###############################################################


resample_model <- function(n_sample=1,r_sample=1,model_use=list(1,2)) {
  
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
      model <-bagging(Formula,data=data_in_train)
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
model_refs




################
# Get the data #
################

data(PimaIndiansDiabetes)
data_in <- PimaIndiansDiabetes

#summary(data_in)
#str(data_in)

#data(Vehicle)
#str(Vehicle)
#data_in <- Vehicle

###############################################
# Identify the predictor and response variables
###############################################
#colnames(data_in) <- tolower(colnames(data_in))
resp_var <- c("diabetes") #class
pred_vars <- unlist(colnames(data_in[,!(colnames(data_in) %in% resp_var)])) # c("a","b","c")


###############################################
# Make formula function
###############################################

Formula <- formula(paste(resp_var,"~ ",  paste(pred_vars, collapse=" + ")))

#response variable values
fact_labels <-levels(eval(parse(text=paste0('data_in$',resp_var))))
fact_levels <- seq(1,length(fact_labels), 1)

######################################################
# Example: 1 sample                                  #
# Show: Did you find the right signal in the noise?  #
######################################################

set.seed(111) # logistic > linear discriminant> CART >>>
set.seed(1111) # CART > logistic > SVM > linear discriminant >>>
set.seed(11111) # CART > SVM > linear dscriminant > logistic >>>
#set.seed(Sys.time())
out_test_perf <- resample_model(n_sample = 1 ,r_sample = 0.75, model_use=list(
  1,2,5,7,8,9,10))

# Make a quick chart to compare
ggplot(data=out_test_perf, aes(x=model_name, y=Pos_Pred_Value, fill=model_name)) +
  geom_bar(stat="identity") +
  geom_label(aes(label = floor(Pos_Pred_Value*10000)/100, label.size = 0.25))


################################################
# Example: Many resamples                      #
# Show: Which model best describes this data?  #
################################################

set.seed(Sys.time())
out_test_perf <- resample_model(n_sample = 100 ,r_sample = 0.75, model_use=list(
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





