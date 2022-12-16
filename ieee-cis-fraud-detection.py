import pandas as pd
import numpy as np

#importação de dados dos arquivos csv

def print_memory_info(dataset):
    print(f" ** Memory usage of the file - {sum(dataset.memory_usage()) * 0.000001} MB for {len(dataset.index)} Rows")
    print(f" {dataset.info(verbose=True)}")
    print(f" ** Summarize the dataset types and count of columns \n{dataset.dtypes.value_counts()}")
    print(f" ** File has {len(dataset) - len(dataset.drop_duplicates())} duplicate rows off the total {len(dataset)}  ")

transaction_type_conversions = {
    "TransactionID": np.int64,
    "isFraud": np.int8,
    "TransactionDT": np.int16,
    "TransactionAmt": np.float16,
    "ProductCD": "category",
    "addr1": np.float16,
    "addr2": np.float16,
    "dist1": np.float16,
    "dist2": np.float16,
    "P_emaildomain": "category",
    "R_emaildomain": "category",
    "card1": np.int16,
    "card2": np.float16,
    "card3": np.float16,
    "card4": "category",
    "card5": np.float16,
    "card6": "category",
    **{f"C{k}": np.float16 for k in range(1,15)},
    **{f"D{k}": np.float16 for k in range(1,16)},
    **{f"M{k}": "category" for k in range(1,10)},
    **{f"V{k}": np.float16 for k in range(1,340)}
}

train_identity_type_conversions = {
    "TransactionID": np.int64,
    "id_01": np.float16,
    "id_02": np.float16,
    "id_03": np.float16,
    "id_04": np.float16,
    "id_05": np.float16,
    "id_06": np.float16,
    "id_07": np.float16,
    "id_08": np.float16,
    "id_09": np.float16,
    "id_10": np.float16,
    "id_11": np.float16,
    "id_12": "category",
    "id_13": np.float16,
    "id_14": np.float16,
    "id_15": "category",
    "id_16": "category",
    "id_17": np.float16,
    "id_18": np.float16,
    "id_19": np.float16,
    "id_20": np.float16,
    "id_21": np.float16,
    "id_22": np.float16,
    "id_23": "category",
    "id_24": np.float16,
    "id_25": np.float16,
    "id_26": np.float16,
    "id_27": "category",
    "id_28": "category",
    "id_29": "category",
    "id_30": "category",
    "id_31": "category",
    "id_32": np.float16,
    "id_33": "category",
}

test_identity_type_conversions = {
    "TransactionID": np.int64,
    "id-01": np.float16,
    "id-02": np.float16,
    "id-03": np.float16,
    "id-04": np.float16,
    "id-05": np.float16,
    "id-06": np.float16,
    "id-07": np.float16,
    "id-08": np.float16,
    "id-09": np.float16,
    "id-10": np.float16,
    "id-11": np.float16,
    "id-12": "category",
    "id-13": np.float16,
    "id-14": np.float16,
    "id-15": "category",
    "id-16": "category",
    "id-17": np.float16,
    "id-18": np.float16,
    "id-19": np.float16,
    "id-20": np.float16,
    "id-21": np.float16,
    "id-22": np.float16,
    "id-23": "category",
    "id-24": np.float16,
    "id-25": np.float16,
    "id-26": np.float16,
    "id-27": "category",
    "id-28": "category",
    "id-29": "category",
    "id-30": "category",
    "id-31": "category",
    "id-32": np.float16,
    "id-33": "category",
}



train_transaction = pd.read_csv("./inputs/train_transaction.csv", engine="python", nrows=10000, dtype = transaction_type_conversions, usecols = list(transaction_type_conversions.keys()))
# print_memory_info(train_transaction)

test_transaction = pd.read_csv("./inputs/test_transaction.csv", engine="python", nrows=10000, dtype=transaction_type_conversions, usecols = [x for x in list(transaction_type_conversions.keys()) if x != "isFraud"])
# print_memory_info(train_transaction)

train_identity = pd.read_csv("./inputs/train_identity.csv", engine="python", dtype=train_identity_type_conversions)
# print_memory_info(train_identity)

test_identity = pd.read_csv("./inputs/test_identity.csv", engine="python", dtype=test_identity_type_conversions)
# print_memory_info(test_identity)

train = pd.merge(train_transaction, train_identity, on = 'TransactionID', how = 'left')
test = pd.merge(test_transaction,test_identity, on = 'TransactionID', how = 'left')

print('train:',train.shape)
print('test:', test.shape)

print(test.columns)

print(list(test.columns).index("id-01"))
print(list(test.columns).index("id-38"))


# fazer analise exploratoria
# fazer modelos
# usar modelos para fazer predições e usar métricas para medir acuidade das predições