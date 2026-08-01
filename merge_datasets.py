import pandas as pd
import os

print("Menggabungkan dataset...")
df1 = pd.read_csv("dataset_dampak_lingkungan.csv")
df2 = pd.read_csv("dataset_dampak_lingkungan_lengkap.csv")
df3 = pd.read_csv("dataset_dampak_lingkungan_dekade.csv")

# Pastikan kolom yang sama (Suhu, Humid, Hujan, NDVI, Luas Konversi, Kecamatan, Label)
cols = ['Suhu', 'Humid', 'Hujan', 'NDVI', 'Luas Konversi', 'Kecamatan', 'Label']

df_all = pd.concat([df1[cols], df2[cols], df3[cols]], ignore_index=True)

# Drop duplicate rows if any
df_all.drop_duplicates(inplace=True)

# Tambahkan ID lagi
df_all.insert(0, 'ID', ['D' + str(i).zfill(5) for i in range(1, len(df_all) + 1)])

output_file = "dataset_gabungan.csv"
df_all.to_csv(output_file, index=False)
print(f"Selesai! Total baris unik setelah digabung: {len(df_all)}")
