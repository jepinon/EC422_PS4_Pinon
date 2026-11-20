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












