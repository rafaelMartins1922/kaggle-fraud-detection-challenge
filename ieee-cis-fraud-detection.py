import pandas as pd
import numpy as np

#importação de dados dos arquivos csv

data_type_conversions = {
    "TransactionDT": np.int16,
    "TransactionAmt": np.float16,
    "ProductCD": "category",
    "addr1": np.float16,
    "addr2": np.float16,
    "P_emaildomain": "category",
    "R_emaildomain": "category",
    "card1": np.int16,
    "card2": np.float16,
    "card3": np.float16,
    "card4": "category",
    "card5": np.float16,
    "card6": "category",
    # **{f"C{k}": np.float16 for k in range(1,15)},
    # **{f"D{k}": np.float16 for k in range(1,16)},
    #**{f"M{k}": "category" for k in range(1,10)},
    # **{f"V{k}": np.float16 for k in range(1,340)}
}

data = pd.read_csv("./inputs/train_transaction.csv", engine="python", nrows=100000, dtype=data_type_conversions, usecols = list(data_type_conversions.keys()))


print(f" ** Memory usage of the file - {sum(data.memory_usage()) * 0.000001} MB for {len(data.index)} Rows")
print(f" {data.info()}")
print(f" ** Summarize the data types and count of columns \n{data.dtypes.value_counts()}")

print(f" ** File has {len(data) - len(data.drop_duplicates())} duplicate rows off the total {len(data)}  ")