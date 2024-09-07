rm(list = ls())

# location of unprocessed data
data_loc = "/Users/sucra/Desktop/DPCCA/data_orig/stocks"

setwd(data_loc)
filenames <- list.files(pattern = "\\.csv$")
filenames <- gsub(".csv", "", filenames)
# print(filenames)
# writeLines(filenames, "filenames.txt")

#install.packages("readr")
#install.packages("dplyr")
library(readr)
library(dplyr)

finance_tickers <- c("BAC", "C", "JPM", "WFC", "GS", "MS", "AXP", "COF", "USB", "FITB")
energy_tickers <- c("XOM", "CVX", "COP", "EOG", "OXY", "SLB", "HAL", "BKR", "MRO", "XEC")
tech_tickers <- c("AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "FB", "TSLA", "NVDA", "ADBE", "INTC")
helthcare_tickers <- c("ABBV", "ABT", "AMGN", "ANTM", "BIIB", "BMY", "CI", "CVS", "GILD", "ISRG")
industrial_tickers <- c("BA", "CAT", "CSX", "DE", "DHR", "EMR", "ETN", "FDX", "GD", "GE")

tickers <- c(finance_tickers, energy_tickers, tech_tickers, helthcare_tickers, industrial_tickers)

# Initialize an empty data frame for closed prices
df_closed_prices <- data.frame(Date = character())

# Loop through each ticker
for(ticker in tickers) {
  file_path <- paste0(data_loc, "/", ticker, ".csv")
  
  # Read the CSV file
  df_stock <- read_csv(file_path)
  
  # Filter for 2018 and 2019
  df_stock_filtered <- df_stock %>%
    filter(Date >= '2018-01-01', Date <= '2019-12-31')
  
  # Select Date and Close, renaming Close to the ticker name
  df_close <- df_stock_filtered %>%
    select(Date, Close) %>%
    rename(!!ticker := Close)
  
  # Merge with the main data frame
  if(ncol(df_closed_prices) == 1) {
    df_closed_prices <- df_close
  } else {
    df_closed_prices <- merge(df_closed_prices, df_close, by = "Date", all = TRUE)
  }
}

# Remove rows with any NA values
df_closed_prices <- na.omit(df_closed_prices)

# Check for missing values in each column
missing_values <- sapply(df_closed_prices, function(x) sum(is.na(x)))

# Print the number of missing values for each column
print(missing_values)

# Check if there are any missing values in the entire DataFrame
any_missing <- any(is.na(df_closed_prices))
print(any_missing)

# Specify the file path where you want to save the CSV
file_path <- "/Users/sucra/Desktop/DPCCA/data_orig/stock_prices.csv"

# Save df_closed_prices DataFrame to CSV
write.csv(df_closed_prices, file_path, row.names = FALSE)