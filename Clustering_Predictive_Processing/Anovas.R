library("readxl")
library(dplyr)
library(car)

scales <- read_excel("C:/Users/manue/OneDrive/Documentos/CasLAB_Test/scales_transformed.xlsx")

features <- read_excel("C:/Users/manue/OneDrive/Documentos/CasLAB_Test/All_Features_dataset.xlsx")
# Read the Excel files
scales <- read_excel("C:/Users/manue/OneDrive/Documentos/CasLAB_Test/scales.xlsx")
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


# Initialize a dataframe to store significant ANOVA results
significant_anova_results <- data.frame(
  Feature = character(),
  Metric = character(),
  P_Value = numeric(),
  Coefficient = numeric(),
  stringsAsFactors = FALSE
)

# Perform ANOVA for each combination of feature and metric
for (feature in list_features) {
  for (metric in list_metrics) {
    # Combine the relevant columns from both datasets
    combined_data <- data.frame(
      Feature = features_sorted[[feature]],
      Metric = scales_sorted[[metric]]
    )
    
    # Remove rows with NaN values in the current combination of feature and metric
    combined_data <- combined_data %>%
      filter(!is.na(Feature) & !is.na(Metric))
    
    # Check if there are enough data points to perform ANOVA
    if (nrow(combined_data) > 1) {
      # Perform ANOVA
      anova_result <- aov(Feature ~ Metric, data = combined_data)
      anova_summary <- summary(anova_result)
      
      # Extract the p-value and coefficient
      p_value <- anova_summary[[1]][["Pr(>F)"]][1]
      coefficient <- coef(anova_result)[[2]]
      
      # Check if the p-value is less than 0.05
      if (!is.na(p_value) && p_value < 0.05) {
        # Save the result
        significant_anova_results <- rbind(significant_anova_results, data.frame(
          Feature = feature,
          Metric = metric,
          P_Value = p_value,
          Coefficient = coefficient,
          stringsAsFactors = FALSE
        ))
      }
    }
  }
}

# Display the significant ANOVA results
significant_anova_results
