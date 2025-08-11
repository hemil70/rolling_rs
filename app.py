from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from flask_caching import Cache
from werkzeug.datastructures import FileStorage
from typing import List

app = Flask(__name__)
cache = Cache(app, config={'CACHE_TYPE': 'simple'})

DEFAULT_DAILY_RS = 55
DEFAULT_WEEKLY_RS = 21
DEFAULT_MONTHLY_RS = 12

def parse_uploaded_file(file: FileStorage) -> List[str]:
    if not file:
        raise ValueError("No file uploaded. Please upload a file containing stock tickers.")
    content = file.read().decode('utf-8')
    stocks_to_check = [entry.split(':')[-1].strip().upper() for entry in content.split(',') if entry.strip()]
    if not stocks_to_check:
        raise ValueError("No valid stock tickers found in the uploaded file.")
    return stocks_to_check

def resample_data(data, interval):
    resampled_data = {}
    for ticker in data.columns.levels[0]:
        ticker_data = data[ticker]
        if interval == "weekly":
            resampled = ticker_data.resample('W-FRI').agg({
                'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
            })
        elif interval == "monthly":
            resampled = ticker_data.resample('ME').agg({
                'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
            })
        resampled_data[ticker] = resampled
    return pd.concat(resampled_data, axis=1)

def compute_rs(stock_close, index_close, rs_length):
    stock_growth = stock_close / stock_close.shift(rs_length)
    index_growth = index_close / index_close.shift(rs_length)
    rs = stock_growth / index_growth - 1
    return rs.ffill().iloc[-1]

def get_rs_series(stock_close, index_close, rs_length):
    stock_growth = stock_close / stock_close.shift(rs_length)
    index_growth = index_close / index_close.shift(rs_length)
    rs = stock_growth / index_growth - 1
    return rs.ffill()

@cache.memoize(timeout=3600)
def get_stock_data(tickers):
    return yf.download(tickers, period="2y", interval="1d", group_by="ticker")
    # return yf.download(tickers, period="15d", interval="90m", group_by="ticker")

def determine_wins(pairs, rs_values):
    wins = {ticker: 0 for ticker in set([t for pair in pairs for t in pair])}
    for (stock, index), rs in zip(pairs, rs_values):
        if rs > 0:
            wins[stock] += 1
        else:
            wins[index] += 1
    return wins

def rank_tickers(wins):
    ranking = sorted(wins.items(), key=lambda x: x[1], reverse=True)
    return {ticker: i+1 for i, (ticker, _) in enumerate(ranking)}

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        message = request.form['message']
        print(f"Contact Form: {name}, {email}, {message}")
        return redirect(url_for('home'))
    return render_template('contact.html')

@app.route('/screener', methods=['GET', 'POST'])
def screener():
    if request.method == 'POST':
        try:
            stocks_to_check = parse_uploaded_file(request.files['file'])
        except ValueError as e:
            return str(e), 400

        index_symbol = request.form.get('index_symbol', '^GSPC').strip().upper()
        if not index_symbol:
            return "Index symbol is required. Please provide a valid index ticker (e.g., '^GSPC').", 400

        timeframes = request.form.getlist('timeframes')
        periods = {}
        for tf in timeframes:
            period_input = request.form.get(f'{tf}_rs_lengths', '')
            if period_input:
                periods[tf] = [int(p) for p in period_input.split(',') if p.isdigit() and int(p) > 0]
                if not periods[tf]:
                    return f"Please provide valid positive integer RS lengths for {tf.capitalize()} timeframe.", 400
            else:
                periods[tf] = [DEFAULT_DAILY_RS if tf == 'daily' else DEFAULT_WEEKLY_RS if tf == 'weekly' else DEFAULT_MONTHLY_RS]

        calculate_outperformance = request.form.get('calculate_outperformance')
        reference_date = request.form.get('reference_date') if calculate_outperformance else None

        if not timeframes and not calculate_outperformance:
            return "Please select at least one feature: RS calculation for a timeframe or outperformance analysis.", 400

        all_tickers = list(set(stocks_to_check + [index_symbol]))
        daily_data = get_stock_data(all_tickers)

        table_html = None
        chart_html = None
        stock_list = None
        outperformance_df = None
        outperformance_chart = None
        crossover_stocks = {}

        if timeframes:
            weekly_data = resample_data(daily_data, "weekly") if 'weekly' in timeframes else None
            monthly_data = resample_data(daily_data, "monthly") if 'monthly' in timeframes else None

            daily_closes = daily_data.xs("Close", level=1, axis=1) if 'daily' in timeframes else None
            weekly_closes = weekly_data.xs("Close", level=1, axis=1) if 'weekly' in timeframes else None
            monthly_closes = monthly_data.xs("Close", level=1, axis=1) if 'monthly' in timeframes else None

            rs_results = []
            for stock in stocks_to_check:
                if stock in daily_data.columns:
                    rs_values = {"Stock": stock}
                    if 'daily' in timeframes:
                        for p in periods['daily']:
                            rs_series = get_rs_series(daily_closes[stock], daily_closes[index_symbol], p)
                            if len(rs_series) >= 2:
                                last_rs = rs_series.iloc[-1]
                                prev_rs = rs_series.iloc[-2]
                                rs_values[f"Daily_RS_{p}"] = last_rs
                                if prev_rs <= 0 < last_rs:
                                    key = f"Daily_{p}"
                                    if key not in crossover_stocks:
                                        crossover_stocks[key] = []
                                    crossover_stocks[key].append(stock)
                    if 'weekly' in timeframes:
                        for p in periods['weekly']:
                            rs_series = get_rs_series(weekly_closes[stock], weekly_closes[index_symbol], p)
                            if len(rs_series) >= 2:
                                last_rs = rs_series.iloc[-1]
                                prev_rs = rs_series.iloc[-2]
                                rs_values[f"Weekly_RS_{p}"] = last_rs
                                if prev_rs <= 0 < last_rs:
                                    key = f"Weekly_{p}"
                                    if key not in crossover_stocks:
                                        crossover_stocks[key] = []
                                    crossover_stocks[key].append(stock)
                    if 'monthly' in timeframes:
                        for p in periods['monthly']:
                            rs_series = get_rs_series(monthly_closes[stock], monthly_closes[index_symbol], p)
                            if len(rs_series) >= 2:
                                last_rs = rs_series.iloc[-1]
                                prev_rs = rs_series.iloc[-2]
                                rs_values[f"Monthly_RS_{p}"] = last_rs
                                if prev_rs <= 0 < last_rs:
                                    key = f"Monthly_{p}"
                                    if key not in crossover_stocks:
                                        crossover_stocks[key] = []
                                    crossover_stocks[key].append(stock)
                    rs_results.append(rs_values)

            rs_df = pd.DataFrame(rs_results)
            rs_columns = [col for col in rs_df.columns if col != "Stock"]
            if rs_columns:
                filtered_df = rs_df[rs_df[rs_columns].gt(0.1).all(axis=1)]
            else:
                filtered_df = rs_df

            if not filtered_df.empty:
                timeframe_order = ['Monthly', 'Weekly', 'Daily']
                sort_columns = []
                for tf in timeframe_order:
                    tf_columns = [col for col in rs_columns if col.startswith(f"{tf}_RS_")]
                    if tf_columns:
                        tf_columns.sort(key=lambda x: int(x.split('_')[-1]), reverse=True)
                        sort_columns.extend(tf_columns)

                if sort_columns:
                    filtered_df = filtered_df.sort_values(by=sort_columns, ascending=False)

                stock_list = filtered_df['Stock'].tolist()

                bars = []
                for col in sort_columns:
                    timeframe, period = col.split('_RS_')
                    bars.append(go.Bar(name=f"{timeframe} {period}", y=filtered_df['Stock'], x=filtered_df[col], orientation='h'))

                title_parts = [f"{tf.capitalize()} {', '.join(map(str, periods[tf]))}" for tf in timeframes]
                chart_title = f"Relative Strength ({', '.join(title_parts)}) by Stock"

                fig = go.Figure(data=bars)
                fig.update_layout(
                    autosize=True,
                    title=chart_title,
                    yaxis_title="Stock",
                    xaxis_title="Relative Strength",
                    barmode='group',
                    height=300 + 20 * len(filtered_df),
                    legend_title="Timeframe"
                )
                chart_html = fig.to_html(full_html=False, include_plotlyjs='cdn')
                table_html = filtered_df.to_html(classes='table table-striped', index=False)
            else:
                table_html = "<p>No stocks met the RS criteria (RS > 0.1 for all selected periods).</p>"

            if crossover_stocks:
                all_crossover_sets = [set(stocks) for stocks in crossover_stocks.values() if stocks]
                if all_crossover_sets:
                    common_crossover_stocks = list(set.intersection(*all_crossover_sets))
                    common_crossover_stocks.sort()
                else:
                    common_crossover_stocks = []
            else:
                common_crossover_stocks = []

        if calculate_outperformance:
            if not reference_date:
                return "Please provide a reference date for outperformance calculation.", 400

            close_prices = daily_data.xs('Close', level=1, axis=1)
            if reference_date not in close_prices.index:
                return "The reference date is not in the data range. Please choose a trading day within the last 2 years.", 400

            close_prices = close_prices.loc[reference_date:]
            start_close = close_prices.iloc[0]
            end_close = close_prices.iloc[-1]
            pct_change = (end_close / start_close - 1) * 100
            index_pct_change = pct_change[index_symbol]

            stock_outperformance = {
                stock: (pct_change[stock] - index_pct_change)
                for stock in stocks_to_check
                if stock in pct_change and not pd.isna(pct_change[stock]) and pct_change[stock] > index_pct_change
            }

            if stock_outperformance:
                outperformance_df = pd.DataFrame(
                    list(stock_outperformance.items()),
                    columns=['Stock', 'Outperformance']
                ).sort_values('Outperformance', ascending=False)

                hovertext = [
                    f"Stock: {stock}<br>Change: {pct_change[stock]:.2f}%<br>Index Change: {index_pct_change:.2f}%<br>Outperformance: {outperformance:.2f}%"
                    for stock, outperformance in zip(outperformance_df['Stock'], outperformance_df['Outperformance'])
                ]

                fig = go.Figure([go.Bar(
                    x=outperformance_df['Outperformance'],
                    y=outperformance_df['Stock'],
                    orientation='h',
                    hovertext=hovertext,
                    hoverinfo='text'
                )])
                fig.update_layout(
                    autosize=True,
                    title=f"Outperformance Since {reference_date}",
                    xaxis_title="Outperformance (%)",
                    yaxis_title="Stock",
                    height=max(300, 300 + 20 * len(outperformance_df)),
                    margin=dict(l=100, r=20, t=50, b=50)
                )
                fig.update_yaxes(categoryorder='array', categoryarray=outperformance_df['Stock'].tolist())
                outperformance_chart = fig.to_html(full_html=False, include_plotlyjs='cdn')
            else:
                outperformance_chart = "<p>No stocks outperformed the index since the reference date.</p>"

        outperforming_stocks = outperformance_df['Stock'].tolist() if outperformance_df is not None else []
        common_stocks = sorted(list(set(stock_list or []) & set(outperforming_stocks)))

        return render_template(
            'results.html',
            table=table_html,
            chart=chart_html,
            stock_list=stock_list,
            outperforming_stocks=outperforming_stocks,
            common_stocks=common_stocks,
            outperformance_chart=outperformance_chart,
            reference_date=reference_date,
            crossover_stocks=crossover_stocks,
            common_crossover_stocks=common_crossover_stocks
        )

    return render_template('screener.html')

@app.route('/asset_comparison', methods=['GET', 'POST'])
def asset_comparison():
    if request.method == 'POST':
        rs_lengths_input = request.form.get('rs_lengths', '21,55,123')
        rs_lengths = []
        for x in rs_lengths_input.split(','):
            try:
                val = int(x.strip())
                if val > 0:
                    rs_lengths.append(val)
            except ValueError:
                pass
        if not rs_lengths:
            return "Please provide at least one valid positive integer RS length (e.g., '21,55,123').", 400

        tickers = ['GLD', '^GSPC', '^NSEI', 'BTC-USD',  # Your original tickers
                   '^DJI', '^IXIC', '^FTSE', '^GDAXI',  # Major indices
                   'SI=F', 'CL=F', 'NG=F',              # Major commodities
                   'EURUSD=X', 'JPY=X', 'GBPUSD=X',     # Major currencies
                   'ETH-USD',                           # Major cryptocurrency
                   'VNQ',                               # Major real estate ETF
                   'TLT', 'LQD'                         # Major bond ETFs
                   ]
        ticker_names = {'GLD': 'Gold',
                        '^GSPC': 'S&P 500',
                        '^NSEI': 'Nifty 50',
                        'BTC-USD': 'Bitcoin',
                        '^DJI': 'Dow Jones Industrial Average',
                        '^IXIC': 'Nasdaq Composite',
                        '^FTSE': 'FTSE 100',
                        '^GDAXI': 'DAX',
                        'SI=F': 'Silver',
                        'CL=F': 'Crude Oil',
                        'NG=F': 'Natural Gas',
                        'EURUSD=X': 'Euro',
                        'JPY=X': 'Japanese Yen',
                        'GBPUSD=X': 'British Pound',
                        'ETH-USD': 'Ethereum',
                        'VNQ': 'Vanguard Real Estate ETF',
                        'TLT': 'iShares 20+ Year Treasury Bond ETF',
                        'LQD': 'iShares iBoxx $ Investment Grade Corporate Bond ETF'
                        }
        pairs = [
            ('GLD', '^GSPC'), ('GLD', '^NSEI'), ('GLD', 'BTC-USD'), ('^GSPC', '^NSEI'),  # Original pairs
            ('SI=F', '^GSPC'),  # Silver vs. S&P 500
            ('CL=F', '^GSPC'),  # Crude Oil vs. S&P 500
            ('BTC-USD', '^DJI'),  # Bitcoin vs. Dow Jones
            ('ETH-USD', '^GSPC'),  # Ethereum vs. S&P 500
            ('VNQ', '^GSPC'),  # Real Estate ETF vs. S&P 500
            ('TLT', '^GSPC'),  # Treasury Bond ETF vs. S&P 500
            ('LQD', '^GSPC'),  # Corporate Bond ETF vs. S&P 500
            ('EURUSD=X', '^GSPC'),  # Euro vs. S&P 500
            ('JPY=X', '^GSPC'),  # Japanese Yen vs. S&P 500
            ('^GSPC', '^FTSE'),  # S&P 500 vs. FTSE 100
            ('^GSPC', '^GDAXI'),  # S&P 500 vs. DAX
            ('GLD', 'SI=F'),  # Gold vs. Silver
            ('BTC-USD', 'ETH-USD')  # Bitcoin vs. Ethereum
]
        data = get_stock_data(tickers)
        close_prices = data.xs('Close', level=1, axis=1)

        summary_data = []
        charts = []
        skipped_lengths = []

        for rs_length in rs_lengths:
            if len(close_prices) < rs_length + 1:
                skipped_lengths.append(rs_length)
                continue

            rs_values = []
            for stock, index in pairs:
                rs = compute_rs(close_prices[stock], close_prices[index], rs_length)
                rs_values.append(rs)

            wins = determine_wins(pairs, rs_values)
            ranking = rank_tickers(wins)

            rs_df = pd.DataFrame({
                'Pair': [f"{ticker_names[stock]} vs {ticker_names[index]}" for stock, index in pairs],
                'RS': rs_values
            })

            fig = go.Figure()
            colors = ['green' if rs > 0 else 'red' for rs in rs_df['RS']]
            fig.add_trace(go.Bar(
                x=rs_df['Pair'],
                y=rs_df['RS'],
                marker_color=colors,
                text=rs_df['RS'].round(4),
                textposition='auto'
            ))
            fig.add_hline(y=0, line_dash="dash", line_color="black")
            fig.update_layout(
                title=f"Relative Strength for {rs_length}-day period",
                xaxis_title="Pair",
                yaxis_title="Relative Strength",
                height=400
            )
            chart_html = fig.to_html(full_html=False, include_plotlyjs='cdn')
            charts.append({'rs_length': rs_length, 'chart': chart_html})

            summary_data.append({
                'RS Length': rs_length,
                'Rank 1': next((ticker_names[t] for t, r in ranking.items() if r == 1), 'N/A'),
                'Rank 2': next((ticker_names[t] for t, r in ranking.items() if r == 2), 'N/A'),
                'Rank 3': next((ticker_names[t] for t, r in ranking.items() if r == 3), 'N/A'),
                'Rank 4': next((ticker_names[t] for t, r in ranking.items() if r == 4), 'N/A'),
                'Rank 5': next((ticker_names[t] for t, r in ranking.items() if r == 5), 'N/A'),
                'Rank 6': next((ticker_names[t] for t, r in ranking.items() if r == 6), 'N/A'),
                'Rank 7': next((ticker_names[t] for t, r in ranking.items() if r == 7), 'N/A'),
                'Rank 8': next((ticker_names[t] for t, r in ranking.items() if r == 8), 'N/A'),
                'Rank 9': next((ticker_names[t] for t, r in ranking.items() if r == 9), 'N/A'),
                'Rank 10': next((ticker_names[t] for t, r in ranking.items() if r == 10), 'N/A'),
                'Rank 11': next((ticker_names[t] for t, r in ranking.items() if r == 11), 'N/A'),
                'Rank 12': next((ticker_names[t] for t, r in ranking.items() if r == 12), 'N/A'),
                'Rank 13': next((ticker_names[t] for t, r in ranking.items() if r == 13), 'N/A'),
                'Rank 14': next((ticker_names[t] for t, r in ranking.items() if r == 14), 'N/A'),
                'Rank 15': next((ticker_names[t] for t, r in ranking.items() if r == 15), 'N/A'),
                'Rank 16': next((ticker_names[t] for t, r in ranking.items() if r == 16), 'N/A'),
                'Rank 17': next((ticker_names[t] for t, r in ranking.items() if r == 17), 'N/A'),
                'Rank 18': next((ticker_names[t] for t, r in ranking.items() if r == 18), 'N/A')
            })

        summary_df = pd.DataFrame(summary_data)
        summary_html = summary_df.to_html(classes='table table-striped', index=False)

        return render_template('asset_comparison_results.html', summary=summary_html, charts=charts, skipped_lengths=skipped_lengths)

    return render_template('asset_comparison.html')

if __name__ == '__main__':
    app.run(debug=True)