import pandas as pd


df = pd.read_csv('Russian_data.csv', sep=';', encoding='utf-8', index_col='index')
df_ch = pd.read_csv('Chinese_data.csv', sep=';', encoding='utf-8', index_col='index')

df['text_ch'] = df_ch['text']

print(df)
