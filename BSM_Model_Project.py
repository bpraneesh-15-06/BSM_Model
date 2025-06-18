import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error, r2_score
 


'''
S == Current Stock Price
K == Strike price
t == time to maturity
r == risk free rate
sigma == Volatility of the underlying asset
'''

#loading train_test data...
file_path = r'C:\Users\bpran\Downloads\BSM_dataset.csv'
data = pd.read_csv(file_path)

X = data.drop('option_price',axis = 1)
y = data['option_price']

y = y.dropna()

# Black-Scholes call option pricing function
def black_scholes_call(S, K, t, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * t) / (sigma * np.sqrt(t))
    d2 = d1 - sigma * np.sqrt(t)
    call_price = S * norm.cdf(d1) - K * np.exp(-r * t) * norm.cdf(d2)
    return call_price

def cost_function(params, X, y):
    r, sigma = params 
    S = X['stock_price']
    K = X['strike_price']
    t = X['time_to_maturity']
    predicted_prices = black_scholes_call(S, K, t, r, sigma)
    mse = np.mean((predicted_prices - y) ** 2)
    return mse

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initial guess...
initial_guess = [0.01, 0.2]  # r = 1%, sigma = 20%

# scipy's minimize!!!
result = minimize(cost_function, initial_guess, args=(X_train, y_train), bounds=[(0, 0.1), (0.01, 1)])

# estimated parameters for the BSM model
optimized_r = result.x[0]
optimized_sigma = result.x[1]

print(f"Optimized Risk-Free Rate: {optimized_r}")
print(f"Optimized Volatility: {optimized_sigma}")

# Predict option prices for the test dataset
S_test = X_test['stock_price']
K_test = X_test['strike_price']
t_test = X_test['time_to_maturity']
predicted_test_prices = black_scholes_call(S_test, K_test, t_test, optimized_r, optimized_sigma)


predicted_test_prices.to_csv('predicted_options_test.csv')

print("Predictions saved to 'predicted_options_test.csv'.")


#metrics to calculate error
mse = mean_squared_error(y_test, predicted_test_prices)
mae = mean_absolute_error(y_test, predicted_test_prices)
r2 = r2_score(y_test, predicted_test_prices)

print(f"Mean Squared Error: {mse}")
print(f"Mean Absolute Error: {mae}")
print(f"R-squared: {r2}")