
from alpha_vantage.timeseries import TimeSeries
import pandas as pd
from datetime import datetime
import datetime as dt
import yfinance as yf
from dateutil.relativedelta import relativedelta
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
import math, random
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt2
import praw
import pandas as pd
import warnings
import mpld3
import plotly.graph_objects as go
warnings.filterwarnings("ignore")
import csv
import time
import re
from textblob import TextBlob
import preprocessor as p
import os
import json
from datetime import datetime, timedelta
from dotenv import load_dotenv
load_dotenv()

class SentimentAnalysis():
    def __init__(self):
        pass
    
    def get_dataframe(self, quote):

        # filepath=f"{quote}.csv"
        # if os.path.exists(filepath):
        #     os.remove(filepath) 

        self.get_historical(quote)

        df=pd.read_csv(f'{quote}.csv')
        os.remove(f'{quote}.csv')
        return df

    def preprocess(self, df,quote):
        df = df.dropna()
        code_list=[]
        for i in range(0,len(df)):
            code_list.append(quote)
        df2=pd.DataFrame(code_list,columns=['Code'])
        df2 = pd.concat([df2, df], axis=1)
        df=df2
        return df

    def get_historical(self, quote):
        end = datetime.now()
        if end.month <= 6:
            start = datetime(end.year - 1, end.month + 6, end.day)  # Adjust year when subtracting
        else:
            start = datetime(end.year, end.month - 6, end.day)

        data = yf.download(quote, start=start, end=end)
        df = pd.DataFrame(data=data)
        # folder_path = 'data'
        # df.to_csv(f'{folder_path}/{quote}.csv')
        df.to_csv(f'{quote}.csv')
        if(df.empty):
            ts = TimeSeries(key='N6A6QT6IBFJOPJ70',output_format='pandas')
            data, meta_data = ts.get_daily_adjusted(symbol='NSE:'+quote, outputsize='full')
            data=data.head(503).iloc[::-1]
            data=data.reset_index()
            df=pd.DataFrame()
            df['Date']=data['date']
            df['Open']=data['1. open']
            df['High']=data['2. high']
            df['Low']=data['3. low']
            df['Close']=data['4. close']
            df['Adj Close']=data['5. adjusted close']
            df['Volume']=data['6. volume']
            df.to_csv(''+quote+'.csv',index=False)
        return

    def LIN_REG_ALGO(self, df):
        #No of days to be forcasted in future
        forecast_out = int(7)
        #Price after n days
        df['Close after n days'] = df['Close'].shift(-forecast_out)
        #New df with only relevant data
        df_new=df[['Close','Close after n days']]

        #Structure data for train, test & forecast
        #lables of known data, discard last 35 rows
        y =np.array(df_new.iloc[:-forecast_out,-1])
        y=np.reshape(y, (-1,1))
        #all cols of known data except lables, discard last 35 rows
        X=np.array(df_new.iloc[:-forecast_out,0:-1])
        #Unknown, X to be forecasted
        X_to_be_forecasted=np.array(df_new.iloc[-forecast_out:,0:-1])
        
        #Traning, testing to plot graphs, check accuracy
        X_train=X[0:int(0.8*len(df)),:]
        X_test=X[int(0.8*len(df)):,:]
        y_train=y[0:int(0.8*len(df)),:]
        y_test=y[int(0.8*len(df)):,:]
        
        # Feature Scaling===Normalization
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)
        
        X_to_be_forecasted=sc.transform(X_to_be_forecasted)
        
        #Training
        clf = LinearRegression(n_jobs=-1)
        clf.fit(X_train, y_train)
        
        #Testing
        y_test_pred=clf.predict(X_test)
        y_test_pred=y_test_pred*(1.04)
        
        # fig, ax = plt.subplots(figsize=(7.2, 4.8), dpi=65)
        # ax.plot(y_test, label='Actual Price')
        # ax.plot(y_test_pred, label='Predicted Price')
        # ax.legend()

        # html_str = mpld3.fig_to_html(fig)
        # plt.close(fig) 
        
        # fig = go.Figure()
        # fig.add_trace(go.Scatter(y=y_test,mode='lines',name='Actual Price'))
        # fig.add_trace(go.Scatter(y=y_test_pred,mode='lines',name='Predicted Price'))
        # fig.update_layout(title="Actual vs. Predicted Price",xaxis_title="Time",yaxis_title="Price")
        # html_str = fig.to_html(full_html=False, include_plotlyjs=False)

        error_lr = math.sqrt(mean_squared_error(y_test, y_test_pred))
        
        forecast_set = clf.predict(X_to_be_forecasted)
        forecast_set=forecast_set*(1.04)
        mean=forecast_set.mean()
        lr_pred=forecast_set[0,0]
        
        # print(forecast_set)
        forecast_data = np.array(forecast_set).flatten()
        start_date = datetime.today()+ timedelta(days=1)
        dates = [(start_date + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(len(forecast_data))]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dates, y=forecast_data, name='Forecast Price'))
        fig.update_layout(title='Forecasted Prices for the Next 7 Days', xaxis_title='Date', yaxis_title='Price')
        html_str = fig.to_html(full_html=False, include_plotlyjs=False)

        # print(html_str)
        return df, lr_pred, forecast_set, mean, error_lr,html_str

    def retrieving_reddit_polarity(self, quote,num_of_posts):
        reddit = praw.Reddit(client_id=os.getenv("client_id"),
                            client_secret=os.getenv("client_secret"),
                            user_agent=f'Scraper 1.0 by /u/{os.getenv("user_id")}')
        # Search for posts mentioning the stock symbol in a subreddit
        subreddit = reddit.subreddit('stocks')  # You can change 'stocks' to another subreddit if needed
        posts = subreddit.search(quote, limit=num_of_posts,sort='new')  # Fetch Reddit posts with the stock symbol

        # Lists to store the post text and their polarity
        post_list = []  # List of posts alongside their polarity
        post_text_list = []  # List of raw post texts to be displayed on web page
        global_polarity = 0  # Sum of polarities of all posts
        pos = 0  # Number of positive posts
        neg = 0  # Number of negative posts
        neutral = 0  # Number of neutral posts

        # Process each post
        for post in posts:
            post_text = post.title + " " + post.selftext  # Combine post title and body

            # Clean post using preprocessor and regex for better sentiment analysis
            # post_text_cleaned = p.clean(post_text)  # Clean using tweet preprocessor
            post_text_cleaned = re.sub('&amp;', '&', post_text)
            post_text_cleaned = re.sub(':', '', post_text_cleaned)
            post_text_cleaned = post_text_cleaned.encode('ascii', 'ignore').decode('ascii')  # Remove non-ASCII chars

            # Perform sentiment analysis on the cleaned post
            blob = TextBlob(post_text_cleaned)
            polarity = blob.sentiment.polarity  # Sentiment polarity from TextBlob

            # Count positive, negative, and neutral posts
            if polarity > 0:
                pos += 1
            elif polarity < 0:
                neg += 1
            else:
                neutral += 1

            global_polarity += polarity
            post_list.append((post_text_cleaned, polarity))  # Append cleaned post and polarity
            post_text_list.append(post_text)  # Store raw post text for display
        #
        # Calculate the overall polarity by averaging
        if len(post_list) != 0:
            global_polarity = global_polarity / len(post_list)

        # Adjust for any discrepancies in neutral count (in case negative values happen)
        neutral = num_of_posts - pos - neg
        if neutral < 0:
            neutral = 0  # Ensure neutral is non-negative
            neg = num_of_posts - pos - neutral


        
        labels = ['Positive', 'Negative', 'Neutral']
        sizes = [pos, neg, neutral]
        explode = (0, 0, 0)  # No explosion in the pie chart

        # Create the pie chart
        # fig, ax1 = plt.subplots(figsize=(7.2, 4.8), dpi=65)
        # ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%', startangle=90)
        # ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
        # plt.tight_layout()

        # # Convert plot to HTML using mpld3
        # html_str = mpld3.fig_to_html(fig)
        # plt.close(fig)  # Close the figure after conversion
        fig = go.Figure(data=[go.Pie(labels=labels,values=sizes,hole=0, title='Sentiment Analysis')])
        html_str = fig.to_html(full_html=False, include_plotlyjs=False) 
        # Save the HTML string to a file
        # Print the overall sentiment result
        if global_polarity > 0:
            post_pol = "Overall Positive"
        else:
            post_pol = "Overall Negative"

        return global_polarity, post_text_list, post_pol, pos, neg, neutral,html_str

    def recommending(self, global_polarity,today_stock,mean):
        # -1 -0.5 => string sell
        # -0.3 to - 0.2 => sell 
        # -0.2 to 0.2 => hold

        # 0.2 to 0.5 => buy
        # 0.5 to 1 -> Strong buy

        if today_stock.iloc[-1]['Close'] < mean:
            if -1 <= global_polarity < -0.3:
                idea = "FALL"
                decision = "STRONG SELL"
            elif -0.3 <= global_polarity < -0.2:
                idea = "FALL"
                decision = "STRONG SELL"
            elif -0.2 <= global_polarity < 0.2:
                idea = "UNCERTAIN RISE"
                decision = "HOLD"
            elif 0.2 <= global_polarity <= 0.5:
                idea = "RISE"
                decision = "BUY"
            elif 0.5 < global_polarity <= 1:
                idea = "RISE"
                decision = "STRONG BUY"
            else:
                idea = "UNCERTAIN"
                decision = "HOLD"
        else:
            idea = "FALL"
            decision = "SELL"

        return idea, decision
    
    def analysis(self, quote,num_of_posts):
        dataframe=self.get_dataframe(quote)
        today_stock=dataframe.iloc[-1:]
        df=self.preprocess(dataframe,quote)
        df, prediction,forecast_set,mean,rmse,html_str=self.LIN_REG_ALGO(df)
        global_polarity, post_text_list, post_pol, pos, neg, neutral,html_str_pie=self.retrieving_reddit_polarity(quote,num_of_posts) 
        idea, decision=self.recommending(global_polarity,today_stock,mean)
        json_data1 = {
            "today_stock":today_stock['Close'].tolist(),
            "prediction": prediction,
            "forecast_set": [item[0] for item in forecast_set],
            "mean": mean,
            "rmse": rmse,
            "global_polarity": global_polarity,
            "post_pol": post_pol,
            "pos": pos,
            "neg": neg,
            "neutral": neutral,
            "idea":idea,
            "decision":decision,
            "posts":post_text_list,
            "LR_graph":html_str,
            "PIE_chart":html_str_pie,
        }
        # file_path1 = 'lin_reg_data.json'
        # with open(file_path1, 'w') as json_file:
        #     json.dump(json_data1, json_file, indent=4)
        # return json_data1
        file_path1 = 'lin_reg_data.json'
    
        # Write json_data1 to file
        with open(file_path1, 'w') as json_file:
            json.dump(json_data1, json_file, indent=4)
        
        # Load json_data1 contents to return
        with open(file_path1, 'r') as json_file:
            json_data1 = json.load(json_file)

        # Delete the file after loading the data
        os.remove(file_path1)
        
        return json_data1
    
if __name__ == "__main__":
    SA=SentimentAnalysis()
    quote='BAC'
    num_of_posts=10 
    output=SA.analysis(quote,num_of_posts)
    print(output)
    