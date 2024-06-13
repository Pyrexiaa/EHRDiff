import numpy as np
import pandas as pd

def convert_csv_to_npy(df):
    data = df.to_numpy()
    np.save('datasets/ppum/tri3_add_on_fits_india.npy', data)

def main():
    try:
        data = np.load('datasets/mimic/X_num_train.npy')
        print(f"Shape of the data: {data.shape}")
        if data.ndim == 2:
            num_rows, num_features = data.shape
            print(f"Number of rows: {num_rows}")
            print(f"Number of features: {num_features}")
        else:
            print("Data is not 2-dimensional.")
    except FileNotFoundError:
        print("The file was not found.")
    except IOError as e:
        print(f"An I/O error occurred: {e}")
    except ValueError:
        print("The file is corrupted or not a valid .npy file.")

if __name__ == '__main__':
    dataset = pd.read_csv('datasets/ppum/tri3_add_on_fits_india.csv')
    convert_csv_to_npy(dataset)
    # main()
