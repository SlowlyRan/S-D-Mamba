import pandas as pd
import matplotlib.pyplot as plt
def clean(fp):
    df= pd.read_csv(fp)
    df = df[[col for col in df if col != 'power_output'] + ['power_output']]
    df = df.rename(columns={'date_time': "date", "power_output": "OT"})
    df_raw = df
    abnormal = df_raw[df_raw['direction32']==-999].index
    df_raw.date= pd.to_datetime(df_raw.date)
    day = 96
    for i in abnormal:
        p1 = df_raw.loc[i-day:i+day:2*day]
        p2 = p1.mean()
        df_raw.loc[i] = p2
    rp = "C:\Repo\S-D-Mamba\Data/"
    df_raw.to_csv(rp+fp.split("_")[-1],index=False)

if __name__ == "__main__":
    import os
    rp = r"C:\Repo\pv_data\Data/"
    ll = os.listdir(rp)
    for i in ll:
        print(i)
        fp = rp+i
        clean(fp)