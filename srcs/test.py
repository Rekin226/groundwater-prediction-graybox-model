import pandas as pd

# gw_fit_results.csv path
csv_path = "../workspace/gw_fit_results.csv"


def load_gw_fit_results(path: str) -> pd.DataFrame:
	return pd.read_csv(path)


def sort_by_r2(df: pd.DataFrame) -> pd.DataFrame:
	return df.sort_values(by="r2", ascending=True)


def filter_r2_below(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
	return df[df["r2"] < threshold]


if __name__ == "__main__":
	df = load_gw_fit_results(csv_path)
	df_sorted = sort_by_r2(df)
	df_filtered = filter_r2_below(df_sorted, threshold=0.5)
	print(df_filtered)
	df_filtered.to_csv("../workspace/gw_fit_results_r2_below_0p5.csv", index=False)

