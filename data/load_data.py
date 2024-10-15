import pandas as pd
from sklearn.model_selection import train_test_split

def load_data():
    data = pd.read_csv("data/cs-training.csv", na_values = "NA")
    data.dropna(inplace = True)

    X = data.drop(columns = ["SeriousDlqin2yrs"])
    y = data["SeriousDlqin2yrs"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=777, stratify=y)

    print("Success data load")
    return X_train, X_test, y_train, y_test