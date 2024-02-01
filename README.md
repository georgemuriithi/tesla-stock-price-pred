# Tesla Stock Price Prediction

<a href="https://github.com/georgemuriithi/tesla-stock-price-pred/blob/main/LICENSE">
    <img alt="License" src="https://img.shields.io/github/license/georgemuriithi/tesla-stock-price-pred.svg?color=blue&cachedrop">
</a>

Tesla’s stock price is predicted over some months using an LSTM (Long Short-Term Memory) model. Tweets about Tesla are used to improve prediction accuracy.

First, the stock price is predicted over some months using an **LSTM Multivariate Time-Series Prediction** model. Then, tweets about Tesla are cleaned and their daily average sentiment scores are computed using **TextBlob.** Lastly, the daily average sentiment scores are added as a feature in the LSTM model and used for prediction.

***Disclaimer:** The LSTM model cannot be used to predict stock prices in real life because the stock market is highly unpredictable. In this project, the validation phase is used to test the model's performance. The purpose of the project is to implement Multivariate Time-Series Prediction using LSTM.*

## Problem Description
The task is to investigate the impact of tweets about Tesla on its stock price.

### Data
The csvs zip file and model states can be accessed from the *data* folder.

## Solution Approach
### <a href="https://github.com/georgemuriithi/tesla-stock-price-pred/blob/main/1-TSLA-Stock-Price-Prediction-Without-Sentiment-Scores.ipynb">Prediction Without Sentiment Scores</a>
<a href="https://colab.research.google.com/drive/1lH4_PTMBuH_K2vdHvJsanxvbyNoX4bQi?usp=sharing">
    <img alt="Open In Colab" src="https://colab.research.google.com/assets/colab-badge.svg">
</a>

*Adj close price* is used as the **response** and the following are used as the **features:**

- High price
- Low price
- Open price
- Close price
- Volume

The features are **normalized** because an LSTM model is sensitive to the scale of data, and then **converted to tensors.**

LSTM model **parameters:**

- `input_size=5`
- `batch_first=True`
- `num_classes=1`
- `optimizer=Adam`
- `loss_function=MSELoss()`

LSTM model **hyperparameters** after tuning with **Ray Tune** using Grid Search Algorithm:

- `hidden_size=3`
- `num_layers=1`
- `learning_rate=0.001`
- `num_epochs=8000`

***GPU** is leveraged.*

**MSE (Mean Squared Error)** results:

- Training and Validation: **108.98853599149807**
- Training: **0.37573209398539453**
- Validation: **429.7357225013402**

### <a href="https://github.com/georgemuriithi/tesla-stock-price-pred/blob/main/2-Cleaning-TSLA-Tweets.ipynb">Cleaning Tweets</a>
<a href="https://colab.research.google.com/drive/1yZPE1YhPoZ9aCzuhd1tqF75J1CHxhSkM?usp=sharing">
    <img alt="Open In Colab" src="https://colab.research.google.com/assets/colab-badge.svg">
</a>

The tweets are **cleaned** and **pre-processed** in the following ways:

- Adding whitespaces to the ends to join them
- Lowercasing
- Removing http links
- Removing special characters and numbers
- Removing stop words
- Removing unnecessary words and characters

They are then **lemmatized**, and a **frequency analysis** is conducted on the words.

### <a href="https://github.com/georgemuriithi/tesla-stock-price-pred/blob/main/3-Sentiment-Analysis-On-Cleaned-TSLA-Tweets.ipynb">Sentiment Analysis</a>
<a href="https://colab.research.google.com/drive/1CUspmd06sUzBiiEife9YmuXX2OjWlLrp?usp=sharing">
    <img alt="Open In Colab" src="https://colab.research.google.com/assets/colab-badge.svg">
</a>

**Sentiment scores** for the tweets are calculated using **TextBlob.** The polarity range is [-1.0, 1.0], with -1.0 as the most negative polarity, 1.0 as the most positive polarity and 0.0 as the neutral polarity. Then, a **frequency analysis** is conducted on the sentiment scores. Lastly, the **daily average sentiment scores** are computed.

### <a href="https://github.com/georgemuriithi/tesla-stock-price-pred/blob/main/4-TSLA-Stock-Price-Prediction-With-Sentiment-Scores.ipynb">Prediction With Sentiment Scores</a>
<a href="https://colab.research.google.com/drive/1w1OSOoh5ab2jB8S6Devm-o7zDMaXnMnj?usp=sharing">
    <img alt="Open In Colab" src="https://colab.research.google.com/assets/colab-badge.svg">
</a>

Finally, the daily average sentiment scores are added as a feature to our LSTM model.

LSTM model **parameters:**

- `input_size=6`
- `num_classes=1`
- `optimizer=Adam`
- `loss_function=MSELoss()`

LSTM model **hyperparameters** after tuning with **Ray Tune** using Grid Search Algorithm:

- `hidden_size=5`
- `num_layers=1`
- `learning_rate=0.002`
- `num_epochs=8000`

***GPU** is leveraged.*

**MSE (Mean Squared Error)** results:

- Training and Validation: **35.028627722561374**
- Training: **0.11838243676105603**
- Validation: **138.12294583219045**

## Conclusion
From the MSE results of Prediction Without and With Sentiment Scores, it is clear that adding the daily average sentiment scores of tweets as a feature to the LSTM model improves its prediction accuracy. This means that tweets about Tesla have a level of impact on its stock price.
