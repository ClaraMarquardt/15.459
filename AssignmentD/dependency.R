# R Packages installed 10/31/2019
# Some packages have overlapping content (e.g., tidyr, tidyverse, tidyquant, xts)

# Preliminaries:
# (1) Install R itself (currently version 3.6.1)
#     https://cran.rstudio.com
# (2) Install RStudio (currently version 1.2.5001)
#     https://rstudio.com/products/rstudio/download/
# (3) Windows ONLY:
#     Install RTools (currently version 3.5)
#     https://cran.r-project.org/bin/windows/Rtools/
#     During install, check box to add RTools to PATH variable.

# Development tools
install.packages("devtools")
# install.packages("Rcpp")

# # Generally useful data packages
# install.packages(c("tidyverse"))

# install.packages(c("ggplot2","ggrepel","corrplot"))
# install.packages(c("quadprog","tseries"))


# # Quant finance packages. Tidyquant includes xts
# install.packages(c("tidyquant","quantmod","RQuantLib"))

# # Quantstrat requires separate build
# library(devtools)
# install_github("braverock/blotter")
# install_github("braverock/quantstrat")

# # Machine learning packages
# install.packages(c("e1071","ISLR"))
# install.packages("ROCR")

# # Text analysis packages
# install.packages("tidytext")
# install.packages("tm")
# install.packages("wordcloud")
# install.packages("gutenbergr")
# install.packages("textdata")
# install.packages("quanteda")

# # RTextTools no longer supported, requires manual build after loading pre-reqs:
# #install.packages("RTextTools")
# #install.packages("maxent")
# install.packages("SparseM")
# install.packages(c("randomForest","tree","ipred","caTools","glmnet","tau"))

# # Links to sources to be built manually ("tree" only if R version < 3.6)
# # https://cran.r-project.org/src/contrib/Archive/tree/tree_1.0-37.tar.gz
# # https://cran.r-project.org/src/contrib/Archive/maxent/maxent_1.3.3.1.tar.gz
# # https://cran.r-project.org/src/contrib/Archive/RTextTools/RTextTools_1.4.2.tar.gz

# # Database connectivity
# install.packages(c("RODBC","DBI","RPostgres"))


# # Advanced:  manual compilation and installation
# #   MacOS: Install XCode (from the App Store)
# #   Windows: Install RTools (currently version 3.5)
# #     https://cran.r-project.org/bin/windows/Rtools/
# #     Edit PATH environment variable to include R and RTools.
# #         Under Control Panel > User Accounts > User Accounts [sic],
# #         click on "Change my environment variables" and add these:
# #         C:\RTools\bin
# #         C:\Program Files\R\R-3.6.1\bin\x64
# #   For manual package compilation/install (i.e., maxent and RTextTools)
# #     (a) Open a CMD window and verify PATH (Windows) or a Terminal (MacOS)
# #     (b) Run these commands from src or download directory:
# #            R CMD INSTALL maxent_1.3.3.1.tar.gz
# #            R CMD INSTALL RTextTools_1.4.2.tar.gz
# #     (c) If you want the installation goes to a *system* directory (for all users of the computer)
# #         then use the option, after "R CMD INSTALL" above,
# #         (i)  MacOS:  --library=/Library/Frameworks/R.framework/Versions/3.6/Resources/library
# #         (ii) Windows --library="C:/Program Files/R/R-3.6.1/library"

