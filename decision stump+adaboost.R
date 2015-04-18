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
set.seed(10)
B_max <- 60
nCV <- 5
# load datasets
X <- read.table("hw1.txt")
y <- read.table("hw1label.txt")[,1]
n <- dim(X)[1]
testErrorRate <- matrix(0,nrow=B_max,ncol=nCV)
trainErrorRate <- matrix(0,nrow=B_max,ncol=nCV)
for(i in 1:nCV) {
  # randomly split data in training and test half
  p <- sample.int(n)
  trainIndx <- p[1:round(n/2)]
  testIndx <- p[-(1:round(n/2))]
  ada <- adaBoost(X[trainIndx ,], y[trainIndx], B_max)
  allPars <- ada$allPars
  alpha <- ada$alpha
  #determine error rate, depending on the number of base classifiers
  for(B in 1:B_max)
  {
    c_hat_test <- agg_class(X[testIndx,],alpha[1:B],allPars[1:B])
    testErrorRate[B,i] <- mean(y[testIndx] != c_hat_test)
    c_hat_train <- agg_class(X[trainIndx ,], alpha[1:B], allPars[1:B])
    trainErrorRate[B,i] <- mean(y[trainIndx] != c_hat_train)
  }
# plot results
matplot(trainErrorRate, type = "l", lty = 1:nCV, main = "training error", xlab = "number of classifiers",
        ylab = "error_rate", ylim = c(0, 0.5))

matplot(testErrorRate, type = "l", lty = 1:nCV, main = "test error", xlab = "number of classifiers",
        ylab = "error_rate", ylim = c(0, 0.5))
}







