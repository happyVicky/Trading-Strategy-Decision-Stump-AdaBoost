rm(list = ls(all = TRUE))

install.packages("quantmod")
install.packages("e1071")
install.packages("PerformanceAnalytics")

require(quantmod)
require(PerformanceAnalytics)
require(e1071)


getSymbols("^GSPC",from = "2000-01-01", to = "2015-01-01")
Data <- GSPC
period <- 10
MAData <- SMA(Cl(Data),n = period)

Price <- lag(MAData,-1)-lag(MAData,0)
Class <- ifelse(Price>0,1,-1)

cci <- CCI(HLC(Data),n=20,c=0.015)
cmf <- CMF(HLC(Data),Vo(Data),n=20)
cmo <- CMO(Cl(Data),n=14)
dvi <- DVI(Cl(Data))$dvi
macd <- MACD(Cl(Data),nFast=12,nSlow=26,nSig=9)$macd
mfi <- MFI(HLC(Data),Vo(Data),n=14)
obv <- OBV(Cl(Data),Vo(Data))
Momentum <- momentum(Cl(Data))
rsi <- RSI(Cl(Data),n=14)
Stoch <- stoch(HLC(Data),nFastK=14,nFastD=3,nSlowD=3)
fastk <- Stoch$fastK
fastd <- Stoch$fastD
slowd <- Stoch$slowD
tdi <- TDI(Cl(Data),n=20,multiple=2)$tdi
williamsad <- williamsAD(HLC(Data))
wpr <- WPR(HLC(Data),n=14)
#Features from TTR package


DataSet <- data.frame(cci,cmf,cmo,dvi,macd,mfi,obv,Momentum,rsi,fastk,fastd,slowd,tdi,williamsad,wpr,Class)
DataSet <- DataSet[-c(1:251),]
Data <- Data[-c(1:251),]
if(nrow(Data)!=nrow(DataSet)){
  print("Error: different rows of Data and DataSet")
}
#check
select <- seq(1,nrow(DataSet),10)
DataSet.Select <- DataSet[select,]
colnames(DataSet) <- c("CCI","CMF","CMO","DVI","MACD","MFI","OBV","MOMENTUM","RSI","FASTK","FASTD","SLOWD","TDI","WILLIAMSAD","WPR","Class")
colnames(DataSet.Select) <- c("CCI","CMF","CMO","DVI","MACD","MFI","OBV","MOMENTUM","RSI","FASTK","FASTD","SLOWD","TDI","WILLIAMSAD","WPR","Class")

w <- 0.7
trainstart <- 1
trainend <- round(w*length(DataSet.Select[,1]))
teststart <- round(w*length(DataSet.Select[,1]))+1
testend <- length(DataSet.Select[,1])

Training <- DataSet.Select[trainstart:trainend,]
Test <- DataSet.Select[teststart:testend,]
#Separate the data into 80% training set to build our model, 20% test set to test the patterns we found


## function <adaBoost >
adaBoost <- function(X,y,B) {
  n <- dim(X)[1]
  w <- rep(1/n,times=n)
  alpha <- rep(0,times=B)
  allPars <- rep(list(list()),B)
  # boost base classifiers
  for(b in 1:B) {
    # step a) train base classifier
    allPars[[b]] <- train(X,w,y)
    # step b) compute error
    missClass <- (y != classify(X,allPars[[b]]))
    e <- (w %*% missClass/sum(w))[1]
    # step c) compute voting weight
    alpha[b] <- log((1-e)/e)
    # step d) recompute weights
    w <- w*exp(alpha[b]*missClass)
  }
  return(list(allPars=allPars , alpha=alpha))
}

## function <agg_class > for AdaBoost implementation
agg_class <- function(X,alpha ,allPars) { 
  n <- dim(X)[1]
  B <- length(alpha)
  Labels <- matrix(0,nrow=n,ncol=B)
  # determine labeling for each base classifier
  for(b in 1:B) {
    Labels[,b] <- classify(X,allPars[[b]])
  }
  # weight classifier response with respective alpha coefficient
  Labels <- Labels %*% alpha
  c_hat <- sign(Labels)
  return(c_hat) 
}


## function <train > for AdaBoost implementation
train <- function(X,w,y) {
  n <- dim(X)[1]
  p <- dim(X)[2]
  mode <- rep(0,times=p)
  theta <- rep(0,times=p)
  loss <- rep(0,times=p)
  # find optimal theta for every dimension j
  for(j in 1:p) {
    # sort datapoints along dimension
    indx <- order(X[,j])
    x_j <- X[indx,j]
    # using a cumulative sum, count the weight when progressively # shifting the threshold to the right
    w_cum <- cumsum(w[indx] * y[indx])
    # handle multiple occurrences of same x_j value: threshold # point must not lie between elements of same value w_cum[duplicated(x_j)==1] <- NA
    # find the optimum threshold and classify accordingly
    m <- max(abs(w_cum), na.rm=TRUE)
    maxIndx <- min(which(abs(w_cum)==m))
    mode[j] <- (w_cum[maxIndx] < 0)*2 - 1
    theta[j] <- x_j[maxIndx]
    c <- ((x_j > theta[j])*2 - 1)*mode[j]
    loss[j] <- w %*% (c != y)
  }
  # determine optimum dimension, threshold and comparison mode
  m <- min(loss)
  j_star <- min(which(loss==m))
  pars <- list(j=j_star, theta=theta[j_star], mode=mode[j_star])
  return(pars)
}

## function <classify > for AdaBoost implementation
classify <- function(X,pars) {
  label <- (2*(X[,pars$j] > pars$theta) - 1)*pars$mode
  return(label) 
}


## file to run AdaBoost
B_max <- 60
# load datasets
X <- DataSet.Select[,-16]
y <- DataSet.Select[,16]
n <- dim(X)[1]
testErrorRate <- rep(0,B_max)
trainErrorRate <- rep(0,B_max)

ada <- adaBoost(X[trainstart:trainend,], y[trainstart:trainend], B_max)
allPars <- ada$allPars
alpha <- ada$alpha
#determine error rate, depending on the number of base classifiers
for(B in 1:B_max)
{
  c_hat_test <- agg_class(X[teststart:testend,],alpha[1:B],allPars[1:B])
  testErrorRate[B] <- mean(y[teststart:testend] != c_hat_test)
  c_hat_train <- agg_class(X[trainstart:trainend,], alpha[1:B], allPars[1:B])
  trainErrorRate[B] <- mean(y[trainstart:trainend] != c_hat_train)
}
  # plot results
plot(trainErrorRate, type = "l", main = "training error", xlab = "number of classifiers",
        ylab = "error_rate", ylim = c(0, 0.5))
  
plot(testErrorRate, type = "l", main = "test error", xlab = "number of classifiers",
        ylab = "error_rate", ylim = c(0, 0.5))

#get best boost
B.best <- which.min(trainErrorRate)
best.boost <- adaBoost(X[trainstart:trainend,], y[trainstart:trainend], B.best)

modifyMyPos <- function(pos,periods){
  modifypos <- rep(pos,each=periods)
  modifypos <- modifypos[-(1:(periods-1))]
  return(modifypos)
}

myPosition <- function(x,model,periods){
  alpha <- model$alpha
  allPars <- model$allPars
  position <- ifelse(agg_class(x,alpha,allPars)==1,1,0)
  modipos <- modifyMyPos(position,periods)
  return(modipos)
}
#Strategy: if the svm model says "Up", long it; else short it.

myStock <- Test
test.start.row <- which(rownames(DataSet) == rownames(Test[1,]))
test.end.row <- which(rownames(DataSet) == rownames(Test[nrow(Test),]))
test <- Cl(Data)[test.start.row:test.end.row,]

if(nrow(test)!=1+(nrow(Test)-1)*period){
  print("Error: test or Test has wrong #row")
}

myposition <- myPosition(x = myStock,model = best.boost,periods=period)
bmkReturns <- dailyReturn(test, type = "arithmetic")
myReturns <- bmkReturns*Lag(myposition,1)
myReturns[1] <- 0

names(bmkReturns) <- 'SP500'
names(myReturns) <- 'My Strategy'

charts.PerformanceSummary(cbind(bmkReturns,myReturns))

Performance <- function(x) {
  
  cumRetx = Return.cumulative(x)
  annRetx = Return.annualized(x, scale=252)
  sharpex = SharpeRatio.annualized(x, scale=252)
  winpctx = length(x[x > 0])/length(x[x != 0])
  annSDx = sd.annualized(x, scale=252)
  
  DDs <- findDrawdowns(x)
  maxDDx = min(DDs$return)
  maxLx = max(DDs$length)
  
  Perf = c(cumRetx, annRetx, sharpex, winpctx, annSDx, maxDDx, maxLx)
  names(Perf) = c("Cumulative Return", "Annual Return","Annualized Sharpe Ratio",
                  "Win %", "Annualized Volatility", "Maximum Drawdown", "Max Length Drawdown")
  return(Perf)
}
cbind(Me=Performance(myReturns),SP500=Performance(bmkReturns))  


#Conclusion:
##################################################
#w <- 0.7
# B_max <- 60
# Me       SP500
# Cumulative Return        2.44373208   0.7593556
# Annual Return            0.34513048   0.1450616
# Annualized Sharpe Ratio  3.45691322   0.9475619
# Win %                    0.62500000   0.5609524
# Annualized Volatility    0.09983776   0.1530893
# Maximum Drawdown        -0.06721517  -0.1938824
# Max Length Drawdown     47.00000000 207.0000000
##################################################









