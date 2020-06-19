import pandas as pd
def load_data(dataset_dir):
	return pd.read_csv(dataset_dir)
def dataset_cleanup(dataset):
