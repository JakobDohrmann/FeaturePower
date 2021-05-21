args = commandArgs(trailingOnly=TRUE)

library(inTrees)
library(randomForest)

# Set parameters
n <- as.numeric(args[1])
m <- as.numeric(args[2])
datadir <- args[3]
cols <- as.numeric(args[4])

dataset <- paste(getwd(), datadir, sep="/")
dataset <- paste(dataset, datadir, sep="/")
dataset <- paste(dataset, '.csv', sep="")
forest <- paste(datadir, 'forest.csv', sep="/")
importances <- paste(datadir, 'importances.csv', sep="/")
accuracy <- paste(datadir, 'accuracy.csv', sep="/")

# Load and set up data
data <- read.csv(file=dataset, header=FALSE)
split = .66
all <- data[sample(nrow(data)),]
train <- all[1:(nrow(data)*split),]
test <- all[(nrow(data)*split+1):nrow(data),]
X_train <- train[,1:ncol(data)-1]
y_train <- train[,ncol(data)]
X_test <- test[,1:ncol(data)-1]
y_test <- test[,ncol(data)]

# Train forest
rf <- randomForest(X_train, as.factor(y_train), ntree=n, mtry=m, importance=TRUE)
ntree <- rf$ntree
oob <- mean(rf$err.rate[,1])

# Write tree structure
tree = getTree(rf, 1, labelVar=FALSE)
tree = t(tree)
write("left_daughter right_daughter split_var split_point status prediction",file=forest, append=FALSE)
write(tree,file=forest, ncolumns=6, append=TRUE)
for(i in 2:ntree){
tree = getTree(rf, i, labelVar=FALSE)
tree = t(tree)
write('', file=forest, ncolumns=6, append=TRUE)
write(tree,file=forest, ncolumns=6, append=TRUE)
}

# Write feature importance values
write("-1 1 MeanDecreaseAccuracy MeanDecreaseGini",file=importances, append=FALSE)
write(t(importance(rf)), importances, ncolumns=cols, append=TRUE)

# Write OOB error
write(oob, file=accuracy, append=TRUE)
