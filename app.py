# -*- coding: utf-8 -*-

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import pandas as pd
import datetime
import plotly.graph_objs as go
# Load data from CSV file
global df 
df = pd.read_csv('stock_price.txt', header=None, names=['Date', 'Price'])
df['Date'] = pd.to_datetime(df['Date'])

start_date = pd.to_datetime('2023-03-05 12:35:02')

import datetime
import time

time_low = datetime.time(13, 30, 0)
time_high = datetime.time(20, 5, 0)

df2 = df[(df['Date'].dt.time > time_low) & (df['Date'].dt.time < time_high)]

df3 = df2[(df2['Date'].dt.dayofweek < 5)]

df3.index = range(0, len(df3))

def generate_daily_report(open_price, close_price, min_price, max_price, mean_price, daily_change, daily_change_7d, vol, x_dicho, vol_daily, vol_anu, e_s, l_sd, u_sd, alpha):
    report = html.Div([
        html.H2(style={'textAlign': 'center', 'color': '#2C3E50'}),
        html.Table([
            html.Tr([html.Td('Opening'), html.Td(f'{open_price}$')]),
            html.Tr([html.Td('Closing'), html.Td(f'{close_price}$')]),
            html.Tr([html.Td('+Highest'), html.Td(f'{max_price}$')]),
            html.Tr([html.Td('-Lowest'), html.Td(f'{min_price}$')]),
            html.Tr([html.Td('Average'), html.Td(f'{round(mean_price,3)}$')]),
            html.Tr([html.Td('Daily Change'), html.Td(f'{daily_change}%')]),
            html.Tr([html.Td('Weekly Change'), html.Td(f'{daily_change_7d}%')]),
            html.Tr([html.Td('Daily Volatility'), html.Td(f'{round(vol_daily*100,3)}%')]),
            html.Tr([html.Td('Annualized Volatility'), html.Td(f'{round(vol_anu*100,3)}%')]),
            html.Tr([html.Td('Lower Semi-Deviation'), html.Td(f'{round(l_sd*100,3)}%')]),
            html.Tr([html.Td('Upper Semi-Deviation'), html.Td(f'{round(u_sd*100,3)}%')]),
            html.Tr([html.Td(f'Value at Risk (α={alpha})'), html.Td(f'{round(x_dicho*100,3)}%')]),
            html.Tr([html.Td(f'Expected Shortfall (α={alpha})'), html.Td(f'{e_s}%')])
        ], style={'margin': '0 auto'}),
    ], style={'width': '60%', 'display': 'inline-block', 'vertical-align': 'top'})
    return report

def update_daily_report():
    df = pd.read_csv('stock_price.txt', header=None, names=['Date', 'Price'])
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Filtering the DataFrame for the given time range
    time_low = datetime.time(13, 30, 0)
    time_high = datetime.time(20, 5, 0)
    filtered_time_df = df[(df['Date'].dt.time >= time_low) & (df['Date'].dt.time <= time_high)]
    
    # Getting the last day with a time between 13:30:00 and 20:00:00
    last_day = filtered_time_df['Date'].dt.date.max()
    day_to_subtract = 7
    week_ago =last_day - datetime.timedelta(days=day_to_subtract)
    
    
    
    # Check if the last day is a Saturday or Sunday
    last_day_weekday = last_day.weekday()
    
    ls = week_ago.weekday()
    
    if last_day_weekday < 5:  # If the last day is a weekday
        filtered_last_day_df = filtered_time_df[filtered_time_df['Date'].dt.date == last_day]
        filtered_last_day_df_7 = filtered_time_df[filtered_time_df['Date'].dt.date == week_ago]
    else:  # If the last day is a Saturday or Sunday
        # Find the last Friday
        days_to_subtract = last_day_weekday - 4
        d_t_s = ls - 4
        last_friday = last_day - datetime.timedelta(days=days_to_subtract)
        d_t_s_f = week_ago - datetime.timedelta(days=d_t_s)
        filtered_last_day_df = filtered_time_df[filtered_time_df['Date'].dt.date == last_friday]
        filtered_last_day_df_7 = filtered_time_df[filtered_time_df['Date'].dt.date == d_t_s_f]
    
    # Display the 'Price' value of the first date of the new DataFrame
    open_price = filtered_last_day_df['Price'].iloc[0]
    open_price_7d = filtered_last_day_df_7['Price'].iloc[0]
    
    
    # Display the 'Price' value of the last date of the new DataFrame
    close_price = filtered_last_day_df['Price'].iloc[-1]
    close_price_7d = filtered_last_day_df_7['Price'].iloc[-1]
    
    min_price = filtered_last_day_df['Price'].min()
    max_price = filtered_last_day_df['Price'].max()
    
    daily_change = round((close_price/open_price - 1)*100,2)
    daily_change_7d = round((close_price/open_price_7d - 1)*100,2)
    
    def daily_prices(data):
        data_daily = []
        i = 0
        while(i < len(data)):
            # Gets current date
            date = data.loc[i][0]
            # Only keeps the date, erases the time
            date = date[:10]
            # Get the average of that day
            price_data = data[data['Date'].str.contains(date)].Price
            # Computes the mean
            data_daily.append([date, np.mean(price_data)])
            i += len(price_data)
        data_daily = pd.DataFrame(data_daily)
        data_daily.columns = ['Date', 'Price']
        return data_daily
    
    import numpy as np
    df3_copy = df3.copy()
    df3_copy['Date'] = df3_copy['Date'].dt.strftime('%d/%m/%Y %H:%M:%S')
    
    stock_daily = daily_prices(df3_copy)
    
    stock_daily['Date'] = pd.to_datetime(stock_daily['Date'], format='%d/%m/%Y')
    
    weekday_stock_daily = stock_daily[~stock_daily['Date'].dt.dayofweek.isin([5, 6])]
    
    weekday_stock_daily.loc[:, 'Date'] = weekday_stock_daily['Date'].dt.strftime('%d/%m/%Y')
    
    mean_price = weekday_stock_daily['Price'].iloc[-1]
    
    def compute_returns(data):
        returns = []
        for i in range(0, len(data) - 1):
            returns.append((data.iloc[i+1,1] - data.iloc[i,1])/data.iloc[i,1])
        return returns
    
    returns = compute_returns(weekday_stock_daily)
    
    def Hurst(data):
        mean_returns = np.mean(data)
        centered_returns = data - mean_returns
        cumulative_returns = np.cumsum(centered_returns)
        i = len(cumulative_returns)
        std_returns = np.std(data)
        R = max(cumulative_returns) - min(cumulative_returns)
        ratio = R/std_returns
        return np.log(ratio)/np.log(i)
    
    hurst = Hurst(returns)
    
    def annualize_volatility(data, hurst):
        std_daily = np.std(data)
        return std_daily, std_daily*(pow(252,hurst))
    
    vola = annualize_volatility(returns, hurst)
    vol_daily = vola[0]
    vol_anu = vola[1]
    
    # VaR non-paramétrique
    
    # Paramètres
    n =  len(returns)
    vol = vola[0] # estimation de la vol empirique 
    h = 1.06*(n**(-1/5))*vol
    
    def k(x):
        return (3/4)*(1-x**2)
        
    # Fonction de répartition
    def K(x) : 
        return (1/4)*(2 + 3*x -x**3)
    
    
    
    def f(data,x,h,n):
        somme = 0
        for i in range(0,len(data)-1) :
            u = (x-returns[i])/h
            if u<1 and u>-1:
                somme = somme + k(u)
            else:
                somme += 0
        somme = somme*(1/h*len(returns))
        return somme
    
    def F(data,x,h,n):
        somme = 0
        for i in range(0,len(data)-1) :
            u = (x-returns[i])/h
            if u<1 and u>-1:
                somme = somme + K(u)
            else:
                somme += 0
        somme = somme*(1/len(returns))
        return somme
    
    def dichotomy_method(F, alpha, start, end, epsilon):
        x = (start + end) / 2
        while end - start > epsilon:
            if F(returns,x, h, n) == 1 - alpha:
                return x
            elif F(returns,x, h, n) < 1 - alpha:
                end = x
            else:
                start = x
            x = (start + end) / 2
        return x
    
    alpha = 0.95
    start = 0
    end = 1
    epsilon = 10**(-10)
    
    # Dichotomy Method
    x_dicho = dichotomy_method(F, alpha, start, end, epsilon)
    round(x_dicho,4)*100
    
    returns = np.array(returns)
    rdt_pos = returns[returns >0]
    rdt_neg = returns[returns < 0]
    
    u_sd = np.sqrt(rdt_pos.var(ddof=0))
    l_sd = np.sqrt(rdt_neg.var(ddof=0))
    
    # Trapezoidal
    def trapezoidal_method(F, a, b, n2): 
        h2 = (b-a)/n2 
        x = a 
        sum = 0.0
        for i in range(1,n2):
            sum = sum + dichotomy_method(F, x, start, end, epsilon)
            x = x + h2
        sum = sum + (dichotomy_method(F, a, start, end, epsilon) + dichotomy_method(F, b, start, end, epsilon)) / 2.0
        return h2 * sum
    
    a = alpha
    b = 1
    n2 = 100
    it2 = trapezoidal_method(F, a, b, n2)
    e = it2/(1-alpha)
    e_s = round(e*100,3)
    
    report = generate_daily_report(open_price, close_price, min_price, max_price, mean_price, daily_change, daily_change_7d, vol, x_dicho, vol_daily, vol_weekly, vol_anu, e_s, l_sd, u_sd, alpha)
    return report


def time_until_next_8pm():
    now = datetime.datetime.now()
    next_8pm = now.replace(hour=20, minute=0, second=0, microsecond=0)
    if now.hour >= 20:
        next_8pm += datetime.timedelta(days=1)
    time_until = next_8pm - now
    return time_until.total_seconds()


app = dash.Dash(__name__, external_stylesheets=['https://stackpath.bootstrapcdn.com/bootstrap/4.4.1/css/bootstrap.min.css'])


### APP ###

app.layout = html.Div(children=[
    html.H1(children='3M Stock MMM (NYSE) Dashboard', style={'textAlign': 'center'}),
    html.Div([
        dcc.Graph(id='3M-stock-price-graph', style={'width': '80%', 'height': '600px', 'margin': '0 auto'}),
        html.Div(children=[
            html.Div(children=[
                html.H3('Current Price', style={'textAlign': 'center', 'color': '#2C3E50'}),
                html.H2(id='current-value', style={'textAlign': 'center'}, children="${:,.2f}".format(df['Price'].iloc[-1]))
            ], className='card', style={'width': '100%', 'textAlign': 'center'}),
        ], className='row', style={'display': 'flex', 'justifyContent': 'center'}),
        dcc.Interval(
            id='interval-component',
            interval=5*60*1000,  # in milliseconds (5 minutes)
            n_intervals=0
        )
    ]),
    
    html.Div(style={'display': 'flex', 'justifyContent': 'center'}, children=[
        html.Div([
            html.H2('Risk and Performance Metrics', style={'textAlign': 'center', 'color': '#2C3E50'}),
            html.Div(id='daily-report', children=update_daily_report(), style={'textAlign': 'center'}),
            dcc.Interval(id='daily-report-update', interval=time_until_next_8pm() * 1000, max_intervals=-1),
        ], style={'width': '40%', 'display': 'inline-block', 'vertical-align': 'top', 'backgroundColor': '#EBF5FB', 'textAlign': 'center', 'justifyContent': 'center', 'alignItems': 'center'}),
    ]),
])



@app.callback(Output('daily-report', 'children'),
              Input('daily-report-update', 'n_intervals'))
def update_report(n):
    return update_daily_report()


@app.callback(
    Output('3M-stock-price-graph', 'figure'),
    Output('current-value', 'children'),
    Input('interval-component', 'n_intervals')
)
def update_dashboard(n):
    df = pd.read_csv('stock_price.txt', header=None, names=['Date', 'Price'])
    df['Date'] = pd.to_datetime(df['Date'])
    
    first_date = df['Date'].iloc[0].strftime('%Y-%m-%d')
    last_date = df['Date'].iloc[-1].strftime('%Y-%m-%d')
    figure = {
        'data': [
            go.Scatter(
                x=df['Date'],
                y=df['Price'],
                mode='lines'
            )
        ],
        'layout': go.Layout(
            title=f"3M Company (MMM) Stock Price from ({first_date}) to ({last_date})",
            xaxis={'title': 'Date'},
            yaxis={'title': 'Price'}
        )
    }
    
    current_value = "${:,.2f}".format(df['Price'].iloc[-1])

    return figure, current_value

if __name__ == '__main__':
        app.run_server(debug=True, host='0.0.0.0', port=8050)
