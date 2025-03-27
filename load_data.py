import pandas as pd
import datetime
import re

def load_snp_data(csv_path):
    df = pd.read_csv(csv_path)

    # Парсимо тільки дату і час, без таймзони
    def parse_datetime(x):
        match = re.match(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", str(x))
        return datetime.datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S")

    df["Date"] = df["Date"].apply(parse_datetime)

    df["Price"] = df["Close"]
    df["Returns"] = df["Price"].pct_change().fillna(0)
    df["AbsChange"] = df["Price"].diff().fillna(0)
    df["Weekday"] = df["Date"].apply(lambda x: x.weekday())

    print(df.tail())
    return df

load_snp_data("snp500.csv")

