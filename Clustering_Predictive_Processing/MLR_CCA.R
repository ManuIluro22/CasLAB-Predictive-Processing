library(readxl)
library(CCA)
library(stats)
library(heplots)

# Load the data
feature_scales <- read_excel("C:/Users/manue/OneDrive/Documentos/CasLAB_Test/features_scales.xlsx")

#formula 1

formula <- cbind(PA,BIS)  ~ Mean_Rating0 + Cor_Pred_Like
model <- lm(formula, data = feature_scales)
coef(model)
summary_aov <- summary.aov(model)
summary_aov
Anova(model)


# Extract predictor and response variables from the dataset
X <- feature_scales[, c("Mean_Rating0","Cor_Pred_Like")]
Y <- feature_scales[, c("PA", "BIS")]

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


