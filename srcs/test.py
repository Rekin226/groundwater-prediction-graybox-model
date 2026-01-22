import pandas as pd
import matplotlib.pyplot as plt


# read rf_data.csv and plot the first rf1 column as a time series
def plot_rf_data(rf_data_path: str):
    df_rf = pd.read_csv(rf_data_path, parse_dates=['date time'])
    df_rf = df_rf.set_index('date time')
    if 'rf1' not in df_rf.columns:
        print("Column 'rf1' not found in rf_data.csv")
        return
    plt.figure(figsize=(12, 6))
    plt.plot(df_rf.index, df_rf['rf1'], label='Rainfall (rf1)', color='blue')
    plt.xlabel('Date')
    plt.ylabel('Rainfall')
    plt.title('Rainfall Time Series for rf1')
    plt.legend()
    plt.grid()
    plt.show()
if __name__ == "__main__":
    plot_rf_data('../data/rf_data.csv')