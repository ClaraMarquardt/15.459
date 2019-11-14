# -------------------------------------------------
# Dependencies ----------------------------------
# -------------------------------------------------

## Libraries

# Development tools
install.packages("devtools")
install.packages("Rcpp")

# Generally useful data packages
install.packages(c("tidyverse"))

install.packages(c("ggplot2","ggrepel","corrplot"))
install.packages(c("quadprog","tseries"))

# Machine learning packages
install.packages(c("e1071","ISLR"))
install.packages("ROCR")

# Text analysis packages
install.packages("tidytext")
install.packages("tm")
install.packages("wordcloud")
install.packages("gutenbergr")
install.packages("textdata")
install.packages("quanteda")

# RTextTools no longer supported, requires manual build 
install.packages("maxent")
install.packages("SparseM")
install.packages(c("randomForest","tree","ipred","caTools","glmnet","tau"))


