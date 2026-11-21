# Part 1: Loading Libraries

library(tidyverse)
library(fpp3)
library(BVAR)


# Part 2: A Business Forecasting Horse Race: ARIMA vs. ETS

# 2.1a

liquor_raw <- read_csv(
  "data/MRTSSM4453USN.csv",
  na = ".",
  col_types = cols(
    observation_date = col_date(format = "%Y-%m-%d"),
    MRTSSM4453USN = col_double()
  )
)

liquor <- liquor_raw |>
  transmute(Month = yearmonth(observation_date), sales = MRTSSM4453USN) |>
  as_tsibble(index = Month) |>
  filter(Month <= yearmonth("2019 Dec"))

has_gaps(liquor)


# 2.1b.

liquor |>
  autoplot(sales) +
  labs(
    title = "U.S. Beer, Wine, and Liquor Retail Sales",
    x = NULL, y = "Millions of dollars", color = NULL
  ) +
  theme_minimal()

liquor |>
  gg_season(log(sales)) +
  labs(
    title = "Seasonal Pattern of Logged Sales",
    y = "Millions of dollars"
  ) +
  theme_minimal()

# Alcohol sales have an upward trend since 1992, when the data started being collected. The seasonal pattern shows a reliable peak in sales in the month of December, with January being the consistently lowest month.

# The seasonal swings in sales do tend to grow with the level, which suggests that using the log of sales is a good decision. This will keep seasonal swings constant in size throughout the horizon


# 2.2. Fit Both Models

# 2.2a.

liquor_fit <- liquor |>
  model(arima = ARIMA(log(sales)),
        ets   = ETS(log(sales)))

liquor_fit |> select(arima) |> report()
liquor_fit |> select(ets) |> report()

# The chosen ARIMA specification was ARIMA(1,0,2)(2,1,1). This means that we have 1 non-seasonal AR lag, 0 non-seasonal differences, and 2 non-seasonal MA error lags. We also have 2 seasonal AR lags, 1 seasonal difference, and 1 seasonal MA error lag.

# The chosen ETS specification was ETS(A, A, A). This means that we have an additive trend, additive error, and additive seasonality, consistent with the additive Holt-Winters model.

liquor_fit |> select(arima) |> gg_tsresiduals()

liquor_fit |>
  select(arima) |>
  augment() |>
  features(.innov, ljung_box, lag = 24, dof = 6)

# Several of the residuals have ACF's that stretch beyond the significance bands, suggesting that there are residuals that are not white noise.


# 2.3. The Horse Race

# 2.3a.

liquor_cv <- liquor |>
  stretch_tsibble(.init = 120, .step = 1) |>
  filter(.id < max(.id))

liquor_cv_fit <- liquor_cv |>
  model(snaive = SNAIVE(sales),
        ets    = ETS(log(sales)),
        arima  = ARIMA(log(sales)))

saveRDS(liquor_cv_fit, "results/liquor_cv_fit.rds")

# This TS-CV will consider 120 forecast origins, since .init = 120.


# 2.3b.

liquor_cv_fc <- liquor_cv_fit |>
  forecast(h = 1)

liquor_cv_fc |>
  accuracy(liquor, measures = list(RMSE = RMSE, MAE = MAE))

head(liquor_cv_fc)

tail(liquor_cv_fc)

# The dates of the first and last forecast are January 2002 and December 2019, respectively.

# The ARIMA model has an RMSE of 81.4 and an MAE of 65.7.

# The ETS model has an RSMSE of 94.4 and an MAE of 72.3.

# The seasonal random walk model has an RMSE of 161.0 and an MAE of 137.0.

# It is clear that the ARIMA model forecasts most accurately, with both metrics in agreement.


# 2.4. Are the Differences Significant?

# Diebold-Mariano Test for ARIMA vs. ETS

liquor_errs <- liquor_cv_fc |>
  as_tibble() |>
  left_join(as_tibble(liquor) |> select(Month, actual = sales),
            by = "Month") |>
  mutate(e = actual - .mean) |>
  select(Month, .model, e) |>
  pivot_wider(names_from = .model, values_from = e)

forecast::dm.test(liquor_errs$arima, liquor_errs$ets, h = 1, power = 2)

# Diebold-Mariano Test for ARIMA vs. Seasonal Random Walk


liquor_errs <- liquor_cv_fc |>
  as_tibble() |>
  left_join(as_tibble(liquor) |> select(Month, actual = sales),
            by = "Month") |>
  mutate(e = actual - .mean) |>
  select(Month, .model, e) |>
  pivot_wider(names_from = .model, values_from = e)

forecast::dm.test(liquor_errs$arima, liquor_errs$snaive, h = 1, power = 2)

# The first DM test returned a DM statistic of -2.9062 and a p-value of 0.004.

# The second DM test returned a DM statistic of -10.755 and a p-value of 2.2e-16.

# Both tests in tandem confirm that the ARIMA model is the most accurate, since both DM statistics are negative. If I were the liquor store chain, I would choose the ARIMA model to forecast sales.


# Part 3: Forecast Combination

# 3a.

liquor_combo_fc <- liquor_cv_fit |>
  mutate(combo = (arima + ets) / 2) |>
  forecast(h = 1)

liquor_combo_fc |>
  accuracy(liquor, measures = list(RMSE = RMSE, MAE = MAE))

# The combo model outperforms the ETS and seasonal random walk models in accuracy, but is still beat by the ARIMA model, which remains the best one.


# 3b. 

cor(liquor_errs$arima, liquor_errs$ets)

# The correlation between the errors of the ARIMA and ETS forecasts is very high, at 0.879. This explains why the combination model did not beat the ARIMA model in terms of accuracy. The high correlation of the errors means they do not cancel much in the combination model.




