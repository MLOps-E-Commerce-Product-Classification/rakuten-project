import pandas as pd
import json

with open("configs/label_encoding.json") as f:
    label_encoding = json.load(f)

# idx_to_code hat Keys als Strings!
idx_to_code = {int(k): v for k, v in label_encoding["idx_to_code"].items()}

y_new = pd.read_csv("data/raw/Y_train_new.csv", index_col=0)
print(y_new.head())  # erst mal schauen was drin ist
y_new["prdtypecode"] = y_new["prdtypecode"].map(idx_to_code)
y_new.to_csv("data/raw/Y_train_new.csv")
print("Done!")
