# [JPX Tokyo Stock Exchange Prediction](https://www.kaggle.com/competitions/jpx-tokyo-stock-exchange-prediction)

This project is focused on predicting stock performance on the JPX Tokyo Stock Exchange and enabling users to interact with stock data, explore historical trends, and create investment portfolios. The project includes deploying a user-friendly app, ranking stocks by expected returns, and implementing machine learning models for predictions.

## Project Overview

The project is divided into three main parts:

1. **Interactive UI for Data Exploration**  
   A Streamlit application was developed to enable users to interact with historical stock data (2017-2021). Through this interface, users can perform exploratory data analysis (EDA) on over 2,000 stocks listed on the JPX Tokyo Stock Exchange. Users can visualize trends, compare stocks, and select portfolio options based on historical performance.

2. **Market Portfolio Construction and Stock Ranking**  
   A portfolio of around 2,000 stocks was built, with each stock ranked by expected returns. This process involved using a **LightGBM** machine learning model to predict the expected returns of each stock, allowing users to see top-ranked stocks and build data-driven investment strategies.

3. **Extended Prediction Model with LSTM**  
   To extend the predictive capabilities, an **LSTM model** was applied to forecast stock prices for the next period. An additional prediction app was developed to allow users to make real-time predictions and gain insights into the next period's stock price trends.

## Features

- **Streamlit UI for Stock Data Exploration**  
  - Visualize and analyze historical stock performance.
  - Select stocks based on customizable filters for portfolio creation.

- **Stock Ranking with LightGBM Model**  
  - Predict stock returns and rank stocks accordingly.
  - Portfolio creation by selecting top-performing stocks.

- **LSTM Prediction Model for Stock Price Forecasting**  
  - Forecast the next period’s stock prices for enhanced decision-making.
  - Interactive prediction tool for testing various investment strategies.

## Tools and Technologies

- **Streamlit**: For building an interactive user interface.
- **LightGBM**: Machine learning model for ranking stocks by expected returns.
- **LSTM (Long Short-Term Memory)**: Recurrent neural network model for time-series prediction.
- **Python Libraries**: Including `pandas`, `numpy`, `matplotlib`, and `scikit-learn` for data manipulation, visualization, and analysis.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/JPX-Tokyo-Stock-Exchange-Prediction.git
   cd JPX-Tokyo-Stock-Exchange-Prediction
