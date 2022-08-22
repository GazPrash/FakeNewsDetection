# Loading, Cleaning & Preparing dataset for model training
import os
import pandas as pd
import numpy as np
import random as rd

def load_data(args:list, external_signature, orientation:str = "rows") -> pd.DataFrame:
    file_signature = "FalseNewsClassf.csv"
    if not os.path.exists(f"data/{file_signature}"):
        try:
            datasets = []
            for arg in args:
                target_val = 1 if arg == "Fake.csv" else 0
                temp_data = pd.read_csv(f"data/{arg}")
                temp_data["fraudulent"] = [target_val for _ in range(len(temp_data))]
                datasets.append(temp_data)

            main_data = pd.concat(datasets, axis = orientation)
            
            file_signature:str = external_signature
            main_data.to_csv(f"data/{file_signature}")
        except Exception as e:
            print(e)
            return

    else:
        main_data = pd.read_csv(f"data/{file_signature}")

    return main_data

def prepare_data_for_t_n_t(data:pd.DataFrame, samples = 2):
    data_samples = []
    sample_max_count = int(1e4)

    for _ in range(samples):
        new_sample = data.sample(sample_max_count, replace= False, random_state= rd.randint(0, 1000))
        data_samples.append(new_sample)

    return data_samples


def main():
    primary_dataset = ["Fake.csv", "True.csv"]
    main_data = load_data(primary_dataset,"FakeNewsClassficationData.csv", orientation="rows")
    samples = prepare_data_for_t_n_t(main_data, samples=2)

    return samples;

if __name__ == "__main__":
    main()