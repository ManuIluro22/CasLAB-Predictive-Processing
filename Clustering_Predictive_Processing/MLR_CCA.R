library(readxl)
library(CCA)
library(stats)
library(heplots)
library(dplyr)
library(car) 

# Read the Excel files
scales <- read_excel("C:/Users/manue/OneDrive/Documentos/CasLAB_Test/scales_transformed.xlsx")
features <- read_excel("C:/Users/manue/OneDrive/Documentos/CasLAB_Test/All_Features_dataset.xlsx")

# Convert the sorting columns to character if they are not already
scales$EPRIME_CODE <- as.character(scales$EPRIME_CODE)
features$Subject <- as.character(features$Subject)

# Sort both datasets based on the respective columns
scales_sorted <- scales %>% arrange(EPRIME_CODE)
features_sorted <- features %>% arrange(Subject)

# Define the lists of features and metrics
list_features <- c("Mean_Rating0","Mean_Rating0_Match","Mean_Rating0_No_Match","Dif_Match","Cor_Pred_Like","Cor_Pred_Like_Match","Cor_Pred_Like_No_Match","Mean_Rating0_Match_Negative","Mean_Rating0_No_Match_Negative","Dif_Negative","Trend_Match","Trend_No_Match",
                   "Trend_No_Match_Negative", "Trend_Match_Negative",
                   "Cor_Pred_Like_Match_Negative", "Cor_Pred_Like_No_Match_Negative","Mean_Rating0_Match_Happy","Mean_Rating0_No_Match_Happy","Dif_Happy",
                   "Trend_No_Match_Happy", "Trend_Match_Happy",
                   "Cor_Pred_Like_Match_Happy", "Cor_Pred_Like_No_Match_Happy","Cor_Pred_Like_Negative","Mean_Rating0_Negative","Cor_Pred_Like_Happy","Mean_Rating0_Happy")
list_metrics <- c('PA', 'NA.', 'ERQ_CR', 'ERQ_ES', 'UPPSP_NU', 'UPPSP_PU', 'UPPSP_SS', 'UPPSP_PMD', 'UPPSP_PSV', 'BIS', 'BAS_RR', 'BAS_D', 'BAS_FS', 'TEPS_AF', 'TEPS_CF', 'SHS', 'FS', 'LOT_R', 'RRQ_Rum', 'RRQ_Ref', 'ASI_P', 'ASI_C', 'ASI_S', 'ASI_T', 'SPQ', 'SPQ_IR', 'MSSB_POS', 'MSSB_NEG', 'MSSB_DES')

# Ensure the relevant columns exist in both datasets
common_columns <- intersect(colnames(scales_sorted), colnames(features_sorted))
feature_scales <- merge(scales_sorted, features_sorted, by.x = "EPRIME_CODE", by.y = "Subject")


# Define the formula and fit the model
formula <- UPPSP_SS  + PA  + ERQ_CR + BIS + ERQ_ES + ASI_P ~  Mean_Rating0_Match + Mean_Rating0_No_Match + Cor_Pred_Like_No_Match  
model <- lm(formula, data = feature_scales)

coef(model)
summary_aov <- summary.aov(model)
summary_aov
Anova(model)
# Perform ANOVA using car package
anova_results <- Anova(model)
print(anova_results)

##CCA, NEEDS TO BE DONE WITH THE IMPUTATION DATASET

# Extract predictor and response variables from the dataset
X <- feature_scales[, c("UPPSP_SS","PA","ERQ_CR","BIS","ERQ_ES","ASI_P")]
Y <- feature_scales[, c("Mean_Rating0_Match", "Mean_Rating0_No_Match","Cor_Pred_Like_No_Match")]

# Perform CCA
cca_result <- cancor(X, Y)
summary(cca_result)
cca_result

X_can1 <- cca_result$coef$X[,1]
Y_can1 <- cca_result$coef$Y[,1]

loadings_X_squared_1 = X_can1^2
loadings_Y_squared_1 = Y_can1^2

loadings_X_squared_1
loadings_Y_squared_1

total_sum_X_1 <- sum(loadings_X_squared_1)
total_sum_Y_1 <- sum(loadings_Y_squared_1)

proportions_X_1 <- loadings_X_squared_1 / total_sum_X_1
proportions_Y_1 <- loadings_Y_squared_1 / total_sum_Y_1

X_can2 <- cca_result$coef$X[,2]
Y_can2 <- cca_result$coef$Y[,2]

loadings_X_squared_2 = X_can2^2
loadings_Y_squared_2 = Y_can2^2

total_sum_X_2 <- sum(loadings_X_squared_2)
total_sum_Y_2 <- sum(loadings_Y_squared_2)

proportions_X_2 <- loadings_X_squared_2 / total_sum_X_2
proportions_Y_2 <- loadings_Y_squared_2 / total_sum_Y_2

X_can3 <- cca_result$coef$X[,3]
Y_can3 <- cca_result$coef$Y[,3]

loadings_X_squared_3 = X_can3^2
loadings_Y_squared_3 = Y_can3^2

total_sum_X_3 <- sum(loadings_X_squared_3)
total_sum_Y_3 <- sum(loadings_Y_squared_3)

proportions_X_3 <- loadings_X_squared_3 / total_sum_X_3
proportions_Y_3 <- loadings_Y_squared_3 / total_sum_Y_3


proportions_X_1*0.96535+proportions_X_2*0.03465
proportions_Y_1*0.96535+proportions_Y_2*0.03465




