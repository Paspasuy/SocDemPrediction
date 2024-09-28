from utils.get_localtime import add_tz_and_localtime_column

def prepare_train_events(df):
    df = add_tz_and_localtime_column(df)
    return df
