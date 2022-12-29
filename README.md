# Tesla Stock Price Prediction

<a href="https://github.com/georgemuriithi/tesla-stock-price-pred/blob/main/LICENSE">
    <img alt="License" src="https://img.shields.io/github/license/georgemuriithi/tesla-stock-price-pred.svg?color=blue&cachedrop">
</a>

Tesla’s stock price was predicted over some months using an LSTM model. Tweets about Tesla were used to improve prediction accuracy.

First, the stock price was predicted over some months using an **LSTM Multivariate Time-Series Prediction** model. Then, tweets about Tesla were cleaned and their daily average sentiment scores were computed using **TextBlob.** Lastly, the daily average sentiment scores were added as a feature in the LSTM model and used for prediction.

***Disclaimer:** The LSTM model cannot be used to predict Tesla's stock price in real life. The stock market is highly unpredictable. Therefore, prediction models do not work. In this project, the validation phase is used to test the model's performance.*

## Problem Description
The task was to investigate the impact of tweets about Tesla on its stock price.

### Data
The csv files zip and model states can be accessed from the *data* folder.

## Solution Approach
### <a href="https://github.com/georgemuriithi/tesla-stock-price-pred/blob/main/1-TSLA-Stock-Price-Prediction-Without-Sentiment-Scores.ipynb">Prediction Without Sentiment Scores</a>
<a href="https://colab.research.google.com/drive/1lH4_PTMBuH_K2vdHvJsanxvbyNoX4bQi?usp=sharing">
    <img alt="Open In Colab" src="https://colab.research.google.com/assets/colab-badge.svg">
</a>

**Adj Close price** was used as the response and the following were used as the features:

- High price
- Low price
- Open price
- Close price
- Volume

The features were **normalized** because an LSTM model is sensitive to the scale of data and then **converted to tensors.**

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

***GPU** was leveraged.*

Results (**Mean Squared Error**):

- Training and Validation: **108.98853599149807**
- Training: **0.37573209398539453**
- Validation: **429.7357225013402**

### <a href="https://github.com/georgemuriithi/tesla-stock-price-pred/blob/main/2-Cleaning-TSLA-Tweets.ipynb">Cleaning Tweets</a>
<a href="https://colab.research.google.com/drive/1yZPE1YhPoZ9aCzuhd1tqF75J1CHxhSkM?usp=sharing">
    <img alt="Open In Colab" src="https://colab.research.google.com/assets/colab-badge.svg">
</a>

The tweets are **cleaned** in the following ways:

- Adding whitespaces to the ends
- Removing http links
- Removing special characters and numbers
- Lowercasing
- Removing stop words

After that, they are **lemmatized** and a **frequency analysis** is performed on the words.

### <a href="https://github.com/georgemuriithi/tesla-stock-price-pred/blob/main/3-Sentiment-Analysis-On-Cleaned-TSLA-Tweets.ipynb">Sentiment Analysis</a>
<a href="https://colab.research.google.com/drive/1CUspmd06sUzBiiEife9YmuXX2OjWlLrp?usp=sharing">
    <img alt="Open In Colab" src="https://colab.research.google.com/assets/colab-badge.svg">
</a>

**Sentiment scores** are calculated for each tweet using **TextBlob,** within the range [-1.0, 1.0], where -1.0 is a negative polarity and 1.0 is positive. A score of 0.0 refers to a neutral evaluation of the tweet. Afterwards, a **frequency analysis** is performed on the sentiment scores. Lastly, the **daily average sentiment scores** are computed and those that are 0.0 removed.

### <a href="https://github.com/georgemuriithi/tesla-stock-price-pred/blob/main/4-TSLA-Stock-Price-Prediction-With-Sentiment-Scores.ipynb">Prediction With Sentiment Scores</a>
<a href="https://colab.research.google.com/drive/1w1OSOoh5ab2jB8S6Devm-o7zDMaXnMnj?usp=sharing">
    <img alt="Open In Colab" src="https://colab.research.google.com/assets/colab-badge.svg">
</a>

Lastly, we add Sentiment Scores of tweets as a feature in our LSTM model.

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

Results (**Mean Squared Error**):

- Training and Validation: **35.028627722561374**
- Training: **0.11838243676105603**
- Validation: **138.12294583219045**

### Conclusion
As we can see from the Mean Squared Error results of Prediction With and Without Sentiment Scores, adding the Sentiment Scores of tweets as a feature in the LSTM model improves its prediction accuracy.
