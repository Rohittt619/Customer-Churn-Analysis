import pandas as pd

def load_data():
    df = pd.read_csv(
        "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    )
    return df


def clean_data(df):

    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )

    df["TotalCharges"] = df["TotalCharges"].fillna(
        df["TotalCharges"].median()
    )

    return df


if __name__ == "__main__":

    df = load_data()

    df = clean_data(df)

    df.to_csv(
        "data/cleaned_churn.csv",
        index=False
    )

    print("Preprocessing Done")
    
    
    