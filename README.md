# 3M Stock (MMM) Analysis Dashboard
Welcome to the 3M Stock (MMM) Analysis Dashboard, a comprehensive and interactive platform designed to provide valuable insights and risk metrics for investors interested in 3M stock. My primary goal is to empower investors to make informed decisions by analyzing various risk factors associated with 3M stock, including Expected Shortfall, Value at Risk (VaR), and semi-deviation.

# Repository Structure
In this repository, you will find all the files and folders required to make my Dashboard:
#### 1. app.py:
Contains scripts for preprocessing and cleaning the 3M stock data.
Includes the implementation of various risk metrics such as Expected Shortfall, Value at Risk, and semi-deviation.
Contains the code for creating the interactive dashboard that displays the results of my analysis.
#### 2. script.sh:
Hosts the script that automates the daily update of the dashboard with the latest stock data.
#### 3. stock_price.txt:
Provides an overview of how I made my stock datas file.
#### 4. dashboard.md:
Provides a direct access to the 3M Stock (MMM) Analysis Dashboard
#### 5. README.md: 
Provides an overview of the project and its components.

# Methodology
To provide a comprehensive analysis, I have employed several advanced techniques and methodologies:

### Non-parametric VaR: 
I have used the Epanechnikov Kernel to implement the non-parametric VaR, which offers a more flexible and robust approach to assessing potential losses.
### Annualized Volatility: 
I have calculated the annualized volatility using the Hurst exponent, which takes into account the long-term memory of the time series and provides a more accurate measure of the stock's volatility.

### Trading Hours Data: 
The analysis is based solely on data collected during trading hours, specifically between Monday and Friday, and between the Opening and Closing Prices. This ensures that the risk metrics are more representative of the actual trading dynamics.


# Daily Report Updates
To keep the 3M Stock Analysis Dashboard up-to-date, I automatically update the platform with the latest stock data every day at 22:00 (10 PM which is 8 PM in France). By updating the data after the Closing Price is determined, I ensure that my risk metrics and analysis are always based on the most recent and relevant information.

# Contributing
I welcome contributions and suggestions from the community to improve and expand the capabilities of the 3M Stock Analysis Dashboard. Feel free to submit issues, pull requests, or reach out to me directly if you have any ideas or recommendations.

Thank you for choosing the 3M Stock (MMM) Analysis Dashboard as your go-to source for insightful and actionable 3M stock data analysis. I am committed to helping you make the best investment choices possible.
