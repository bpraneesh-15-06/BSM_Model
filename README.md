# Black-Scholes-Merton Option Pricing Model

This project implements the classical **Black-Scholes-Merton (BSM)** model for estimating the prices of **European call options** based on key financial features. While the traditional BSM framework assumes **constant volatility (σ)** and **risk-free interest rate (r)**, this implementation enhances realism by automatically estimating these parameters from the data using numerical optimization.


## Key Features

- **BSM Formula Implementation**  
  Implements the analytical Black-Scholes formula for pricing European-style call options.

- **Automatic Calibration of Parameters**  
  Estimates `σ` (volatility) and `r` (risk-free rate) using `scipy.optimize.minimize` to minimize the mean squared error between predicted and actual option prices.

- **Model Assumptions**  
  Operates under the standard BSM assumption that volatility and interest rate remain constant during the option's life — though this project fits them dynamically per dataset.

- **Model Evaluation**  
  Predicts theoretical prices and evaluates model performance using metrics such as:
  - Mean Squared Error (MSE)
  - Mean Absolute Error (MAE)
  - R² Score

- **Result Export**  
  Saves model predictions to `predicted_options_test.csv` for backtesting or further analytics.


## Input Features

| Feature | Description |
|--------|-------------|
| `S` | Current Stock Price |
| `K` | Option Strike Price |
| `t` | Time to Maturity (in years) |
| `option_price` | Actual Market Price of the Call Option (used during training) |
| `r` and `σ` | Learned automatically via optimization |

---

## Output

- **Predicted Option Prices** (vector)
- **CSV File**: `predicted_options_test.csv`
- **Model Metrics**:
  - `Mean Squared Error`
  - `Mean Absolute Error`
  - `R² Score`

---

## Example Use Cases

- Financial modeling education and teaching the foundations of option pricing.
- Prototyping a derivative pricing engine.
- Estimating the implied volatility surface from observed prices.
- Demonstrating risk-neutral valuation techniques.

---

## Tech Stack

- **Programming Language**: Python 3.x
- **Libraries**:
  - `NumPy`, `Pandas`: Data manipulation and analysis
  - `SciPy`: Optimization (`minimize`)
  - `scikit-learn`: Data splitting and evaluation metrics

---

## Theoretical Background

The Black-Scholes-Merton model is a cornerstone of modern financial engineering and provides a closed-form solution for pricing European options. The formula assumes log-normal price distribution, no dividends, continuous trading, and constant `r` and `σ`.

