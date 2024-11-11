import pandas as pd

# Membaca file CSV dengan hanya mengambil kolom yang dibutuhkan
df = pd.read_csv('../Dataset/styles.csv', usecols=['id', 'subCategory'])

# Mengurutkan data berdasarkan subCategory secara alfabetis
df_sorted = df.sort_values('subCategory')

# Menampilkan hasil
print("\nData yang telah diurutkan:")
print(df_sorted)

# Optional: Menyimpan hasil ke file CSV baru
df_sorted.to_csv('sorted_categories.csv', index=False)
