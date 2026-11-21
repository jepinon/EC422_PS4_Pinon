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


# Part 4: A Medium-Scale VAR vs. a BVAR

# 4.1. The Data

# 4.1a.

qd_raw <- read_csv("data/fredqd_2026-09.csv", show_col_types = FALSE) |>
  slice(-(1:2))

# 4.1b.

growth <- function(x) 100 * difference(log(x))

qd <- qd_raw |>
  mutate(date = as.Date(sasdate, "%m/%d/%Y")) |>
  transmute(date,
            gdp    = growth(GDPC1),     # real GDP
            cons   = growth(PCECC96),   # real consumption
            inv    = growth(GPDIC1),    # real investment
            emp    = growth(PAYEMS),    # payroll employment
            infl   = growth(GDPCTPI),   # GDP price index
            unrate = difference(UNRATE),     # unemployment rate
            ffr    = difference(FEDFUNDS),   # federal funds rate
            gs10   = difference(GS10)) |>    # 10-year Treasury yield
  filter(date >= as.Date("1960-03-01"), date <= as.Date("2019-12-01"))

y <- qd |> select(-date) |> as.matrix()

nrow(y)
head(y)


# 4.1c.

N <- nrow(y)
origins <- 120:(N - 1)
actual <- y[121:N, ]

# There will be 120 forecasts produced for each variable, meaning the total number of forecasts will be 960. The first forecast will be for March 1, 1990, and the last forecast will be for December 1, 2019.


# 4.2 OLS VARs

var_ols_fc <- function(z, p) {
  e <- embed(z, p + 1)            # columns: y_t, then its lags 1 to p
  M <- ncol(z)                    # number of variables
  B <- coef(lm(e[, 1:M] ~ e[, -(1:M)]))
  x <- c(1, t(z[nrow(z):(nrow(z) - p + 1), ]))   # constant, latest p lags
  x %*% B
}

fc_var1 <- NULL
fc_var4 <- NULL
for (T in origins) {
  fc_var1 <- rbind(fc_var1, var_ols_fc(y[1:T, ], 1))
  fc_var4 <- rbind(fc_var4, var_ols_fc(y[1:T, ], 4))
}

rmse_by_var <- function(f) sqrt(colMeans((actual - f)^2))

rmse_by_var(fc_var1)
rmse_by_var(fc_var4)

# The VAR(1) forecasts better, with a lower RMSE for seven of the eight variables. The only variable that VAR(4) forecasts more accurately is GDP price index, and the difference in accuracy is very small.

# VAR(4) contains VAR(1) as a special case, and it's inclusion of more coefficients leads to overfitting and a greater estimation error.


# 4.3 A BVAR

# 4.3a.

bvar_cv <- function(lambda) {
  pr <- bv_priors(hyper = "lambda",
                  mn = bv_mn(lambda = bv_lambda(mode = lambda, sd = 0.4,
                                                min = 0.999 * lambda,
                                                max = 1.001 * lambda),
                             b = 0))
  fc <- NULL
  for (T in origins) {
    fit_T <- bvar(y[1:T, ], lags = 4, priors = pr,
                  n_draw = 2000, n_burn = 1000, verbose = FALSE)
    x <- c(1, t(y[T:(T - 3), ]))         # constant and 4 lags of each variable
    fc <- rbind(fc, x %*% coef(fit_T))   # forecast of y_{T+1}
  }
  fc
}

set.seed(422)
fc_bvar_03 <- bvar_cv(0.3)
saveRDS(fc_bvar_03, "results/fc_bvar_03.rds")


# 4.3b.

set.seed(422)
fc_bvar_005 <- bvar_cv(0.05)

var_results <- tibble(
  Variable = colnames(y),
  `VAR(1)` = rmse_by_var(fc_var1),
  `VAR(4)` = rmse_by_var(fc_var4),
  `BVAR(4), lambda = 0.3` = rmse_by_var(fc_bvar_03),
  `BVAR(4), lambda = 0.05` = rmse_by_var(fc_bvar_005)
)

saveRDS(var_results, "results/var_results.rds")

var_results |> knitr::kable(digits = 3)

# The BVAR(4) with lambda = 0.3 had a lower RMSE than the OLS VAR(4) for every variable. The moderate shrinkage from the lamda of 0.3 reduces estimation error, and in doing so, reduces the RMSE.

# The BVAR(4) only outperforms the VAR(1) in RMSE for three of the eight variables.

# The BVAR(4) with lamda = 0.05 shrinks the model so close to the prior mean that it underfits, so that estimation error is reduced and bias is increased. The winning RMSE here would not be a fair estimate of future forecast accuracy because of selection bias.


# 4.3c.

e_var4 <- actual[, "gdp"] - fc_var4[, 1]
e_bvar <- actual[, "gdp"] - fc_bvar_03[, 1]

forecast::dm.test(e_var4, e_bvar, h = 1, power = 2)


e_var4 <- actual[, "cons"] - fc_var4[, 2]
e_bvar <- actual[, "cons"] - fc_bvar_03[, 2]

forecast::dm.test(e_var4, e_bvar, h = 1, power = 2)

# For the first DM test, the DM statistic is 2.14 and the p-value is 0.035, suggesting that the BVAR(4) is a more accurate forecast for GDP growth.

# For the second DM test, the DM statistic is 3.03 and the p-value is 0.003, suggesting that the BVAR(4) is a more accurate forecast for real consumption growth.

# This is in alignment with my results from 4.3b., where I determined that the BVAR(4) with lambda = 0.3 was more accurate than VAR(4) for every variable.






















