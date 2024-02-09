
### Kütüphane İşlemleri ###

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
import sklearn
from sklearn.preprocessing import MinMaxScaler
import plotly.express as px

### Görüntü Ayarları ve Veri Setini Yükleme ###

# Tüm sütunları görmek için.
pd.set_option('display.max_columns', None)

# Tüm satırları görmek için.
# pd.set_option('display.max_rows', None)

# Sayıları beş basamaklı verir.
pd.set_option('display.float_format', lambda x: '%.5f' % x)

# Veri setini yükleme.
df_ = pd.read_excel("datasets/online_retail_II.xlsx")

# Veri setinin bir kopyasını oluşturma.
df = df_.copy()

### Değişkenler ###

# InvoiceNo: Fatura numarası. Her işleme yani faturaya ait eşsiz numara. C ile başlıyorsa iptal edilen işlem.
# StockCode: Ürün kodu. Her bir ürün için eşsiz numara.
# Description: Ürün ismi
# Quantity: Ürün adedi. Faturalardaki ürünlerden kaçar tane satıldığını ifade etmektedir.
# InvoiceDate: Fatura tarihi ve zamanı.
# UnitPrice: Ürün fiyatı (Sterlin cinsinden)
# CustomerID: Eşsiz müşteri numarası
# Country: Ülke ismi. Müşterinin yaşadığı ülke.

df.head()
df.isnull().sum()
df.shape
df.describe().T


df["Invoice"].nunique()  # Toplam 19215 fatura kesilmiş

### "Total Price" ekleme ###
# Toplam fiyat özelliği eklendi.
df["TotalPrice"] = df["Quantity"] * df["Price"]

# En çok para ödeyen fatura numarlarının büyükten küçüğe sıralanması
df.groupby("Invoice").agg({"TotalPrice": "sum"}).sort_values(by="TotalPrice", ascending=False).head()




############ VERİ ÖNİŞLEME ############

# 0'dan büyük olanları seçme
df = df[(df['Quantity'] > 0)]

# Eksik değerleri silme
df.dropna(inplace=True)

# Boyutu kontrol etme
df.shape  # 407695, 11

# Invoice'u stringe çevirme
df["Invoice"] = df["Invoice"].astype(str)

# C (iade) olanları çıkarma
df = df[~df["Invoice"].str.contains("C", na=False)]

# 'InvoiceDate' sütununu datetime formatına çevirme
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

# Saati silme
# Aslında saati silmeyip alışveriş saatlerini analiz ederek saat bazlı indirim/kampanya tavsiyelerinde bulunulabilir.
df['InvoiceDate'] = df['InvoiceDate'].dt.floor('d')




##### 1.Hangi ürünler en yüksek ve en düşük satışları gösteriyor? #####

df["Description"].describe().T

# Hangi üründen kaç tane var?
df["Description"].value_counts().head()

# 4444 description var.
df["Description"].nunique()

# En çok satılan ürünler.
top_products = df.groupby("Description").agg({"Quantity": "sum"}).sort_values("Quantity", ascending=False).head()

# Pasta grafiği için verileri hazırlama
labels = top_products.index
sizes = top_products['Quantity']

# Pasta grafiğini çizme
plt.figure(figsize=(8, 8))
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
plt.axis('equal')
plt.title('EN ÇOK SATILAN İLK 5 ÜRÜN')
plt.show()

# En az satılan ürünler.
worst_products = df.groupby("Description").agg({"Quantity": "sum"}).sort_values("Quantity", ascending=True).head()

# Pasta grafiği için verileri hazırlama
labels = worst_products.index
sizes = worst_products['Quantity']

# Pasta grafiğini çizme
plt.figure(figsize=(8, 8))
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
plt.axis('equal')
plt.title('EN AZ SATILAN İLK 5 ÜRÜN')
plt.show()




## Ülkelere Göre En Çok Satılan Ürünler ##

# Ürünlerin her ülkede toplam satış miktarını hesaplama
top_products = df.groupby(['Country', 'Description']).agg({'Quantity': 'sum'}).reset_index()

top_products_per_country = top_products.sort_values(['Country', 'Quantity'], ascending=[True, False]).groupby('Country').head(1)

# Bar grafiği için verileri hazırlama
countries = top_products_per_country['Country']
quantities = top_products_per_country['Quantity']
products = top_products_per_country['Description']

# Bar grafiğini çizme
plt.figure(figsize=(10, 8))
plt.barh(countries, quantities, color='skyblue')
plt.xlabel('Ürün')
plt.ylabel('Ülke')
plt.title('Ülkelere Göre En Çok Satılan Ürünler')
for index, value in enumerate(products):
    plt.text(quantities.iloc[index], index, str(value))
plt.show()


##### 2. Sezonluk satış trendleri nelerdir? #####

### "Season" ekleme ###
def Season(month):
    if month in [11, 12, 1, 2, 3, 4]:  # 11-4 arası aylar kış aylarıdır
        return "Winter"
    else:
        return "Summer"

# "Season" sütununu oluşturup sezon etiketlerini ekliyoruz
df["Season"] = df["InvoiceDate"].dt.month.apply(Season)

# Sonucu kontrol etme
df.head()

# Ne kadar yaz/kış var? Kışın daha fazla satış yapılmıştır.
df["Season"].value_counts()

# Her bir sezon ve ürün için satışların toplamı
Seasonal_sales = df.groupby(["Season", "Description"]).agg({"Quantity": "sum"}).reset_index()

# Mevsimlere göre toplam satış miktarını hesaplama
seasonal_totals = df.groupby('Season')['Quantity'].sum().reset_index()

# Grafik oluşturma
plt.figure(figsize=(10, 6))
sns.barplot(x='Season', y='Quantity', data=seasonal_totals)
plt.title('Toplam Satış Miktarları (Mevsimlere Göre)')
plt.xlabel('Mevsim')
plt.ylabel('Toplam Satış Miktarı')
plt.show()

# Hangi sezonun en çok hangi ürünlerin satıldığı
Most_sold_products = Seasonal_sales.loc[Seasonal_sales.groupby("Season")["Quantity"].idxmax()]

# Kışın en çok satılan ilk beş ürün
winter_sales = df[df["Season"] == "Winter"].groupby("Description").agg({"Quantity": "sum"}).reset_index()
most_sold_winter_products = winter_sales.sort_values(by="Quantity", ascending=False).head()
print(most_sold_winter_products)

# Yazın en çok satılan ilk beş ürün
summer_sales = df[df["Season"] == "Summer"].groupby("Description").agg({"Quantity": "sum"}).reset_index()
most_sold_summer_products = summer_sales.sort_values(by="Quantity", ascending=False).head()
print(most_sold_summer_products)

# Kışın en çok satılan ürünler için bar grafiği
plt.figure(figsize=(10, 6))
plt.bar(most_sold_winter_products['Description'], most_sold_winter_products['Quantity'], color='green')
plt.xlabel('Ürün')
plt.ylabel('Miktar')
plt.title('Kışın En Çok Satılan İlk 5 Ürün')
plt.xticks(rotation=15)
plt.show()

# Yazın en çok satılan ürünler için bar grafiği
plt.figure(figsize=(10, 6))
plt.bar(most_sold_summer_products['Description'], most_sold_summer_products['Quantity'], color='orange')
plt.xlabel('Ürün')
plt.ylabel('Miktar')
plt.title('Yazın En Çok Satılan İlk 5 Ürün')
plt.xticks(rotation=15)
plt.show()






##### 3. En çok alışveriş yapan ülkeler hangileridir? #####

# Her bir ülkenin toplam satış miktarı
Country_sales_sorted = df.groupby("Country")["Quantity"].sum().reset_index().sort_values(by="Quantity", ascending=False)

# En çok alışveriş yapan ilk beş ülke;
# United Kingdom, Denmark, EIRE, Netherlands, Germany

fig = px.choropleth(Country_sales_sorted,
                    locations='Country',
                    locationmode='country names',
                    color='Quantity',
                    color_continuous_scale='Viridis',
                    title='Ülkelerin Toplam Satış Miktarı')
fig.show()



##### 4. İndirimler satışlara nasıl etki ediyor? #####

# Fiyatlardaki değişiklikleri kontrol etme
price_changes = df.groupby("Description")["Price"].nunique()

# Her ürün için fiyatın değişip değişmediğini kontrol etme
for Description, unique_prices in price_changes.items():
    if unique_prices > 1:
        print(f"{Description} ürününde fiyat değişikliği var.")
    else:
        print(f"{Description} ürününde fiyat değişikliği yok.")

# Fiyat değişikliği olan ve olmayan ürün sayıları
products_with_changes = (price_changes > 1).sum()
products_without_changes = (price_changes == 1).sum()

# Sonucu göster
print(f"Fiyat değişikliği olan ürün sayısı: {products_with_changes}")
print(f"Fiyat değişikliği olmayan ürün sayısı: {products_without_changes}")

# Fiyat değişikliği olan ürünlerin isimleri
products_with_changes = price_changes[price_changes > 1].index.tolist()
print(products_with_changes)

# Indirim yapılmış ürünler
discounted_products = price_changes[price_changes > 1].index

# Indirim yapılmış ürünlerin bilgileri
df_discounted_products = df[df["Description"].isin(discounted_products)]
print(df_discounted_products)

# Indirim yapılmış ürünlerin toplam sayısı
total_discounted_products = df_discounted_products.shape[0]
print("Toplam indirim yapılmış ürün sayısı:", total_discounted_products)  # 376734

# İndirim yapılmış ürünlerin "StockCode" sütunlarının başlarına "D" harfi ekleme
df.loc[df["Description"].isin(discounted_products), "StockCode"] = 'D' + df["StockCode"].astype(str)

# "StockCode" değeri "D" ile başlayan ürünlerin indirim başlangıç tarihlerini bul
discount_start_dates = df[df["StockCode"].str.startswith("D")]["Date"]


# "D" ile başlayan "StockCode" değeri içeren satırları kontrol et
mask = df['StockCode'].str.startswith('D')

df.dropna(subset=['InvoiceDate'], inplace=True)

# Devam edemedim, ama edeceğim için silmedim :)



##### 5. Günlerin satışlar üzerinde nasıl bir etkisi var? #####

# "Day Name" ekleme
df["DayName"] = df["InvoiceDate"].dt.day_name()

# Her günün toplam satış miktarını hesapla
daily_sales = df.groupby("DayName")["Quantity"].sum().reset_index()

# En çok satış yapılan günleri belirle
Most_sold_days = daily_sales[daily_sales["Quantity"] == daily_sales["Quantity"].max()]
sorted_sales_by_day = daily_sales.sort_values('Quantity', ascending=False)

# Sonucu gösterme
print(sorted_sales_by_day)

# Histogram grafiği oluşturma
plt.figure(figsize=(10, 6))
plt.bar(sorted_sales_by_day['DayName'], sorted_sales_by_day['Quantity'], color='purple', edgecolor='yellow')
plt.xlabel('Günler')
plt.ylabel('Satış Miktarı')
plt.title('Toplam Satış Miktarının Haftanın Günlerine Göre Dağılımı')
plt.xticks(rotation=35)
plt.show()


### 2009 YILI İÇİN GÜNLERE BAKMA ###

# 2009 yılındaki satışların filtrelenmesi
df_2009 = df[df["InvoiceDate"].dt.year == 2009]

# Gün isimlerini ekleme
df_2009 = df_2009.copy()
df_2009["Day_name"] = df_2009["InvoiceDate"].dt.day_name()

# Her günün toplam satış miktarını hesaplama
daily_sales_2009 = df_2009.groupby("Day_name")["Quantity"].sum().reset_index()

# En çok satış yapılan günü bulma (Thursday 85913)
most_sold_day_2009 = daily_sales_2009[daily_sales_2009["Quantity"] == daily_sales_2009["Quantity"].max()]

# Günleri haftanın gün sırasına göre sıralama
ordered_days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
daily_sales_2009['Day_name'] = pd.Categorical(daily_sales_2009['Day_name'], categories=ordered_days, ordered=True)
daily_sales_2009 = daily_sales_2009.sort_values('Day_name')

plt.figure(figsize=(10, 6))
plt.plot(daily_sales_2009["Day_name"], daily_sales_2009["Quantity"], marker='o', linestyle='-')
plt.title("2009 Yılına Ait Haftalık Satış Miktarları")
plt.xlabel("Gün")
plt.ylabel("Satış Miktarı")
plt.xticks(rotation=45)
plt.grid(True)
plt.show()

### 2010 YILI İÇİN GÜNLERE BAKMA ###

# 2010 yılındaki satışların filtrelenmesi
df_2010 = df[df["InvoiceDate"].dt.year == 2010]

# Gün isimlerini ekleme
df_2010 = df_2010.copy()
df_2010["Day_name"] = df_2010["InvoiceDate"].dt.day_name()

# Her günün toplam satış miktarını hesaplama
daily_sales_2010 = df_2010.groupby("Day_name")["Quantity"].sum().reset_index()

# En çok satış yapılan günü bulma (Thursday 1043347)
most_sold_day_2010 = daily_sales_2010[daily_sales_2010["Quantity"] == daily_sales_2010["Quantity"].max()]

# Günleri haftanın gün sırasına göre sıralama
ordered_days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
daily_sales_2010['Day_name'] = pd.Categorical(daily_sales_2010['Day_name'], categories=ordered_days, ordered=True)
daily_sales_2010 = daily_sales_2010.sort_values('Day_name')

plt.figure(figsize=(10, 6))
plt.plot(daily_sales_2010["Day_name"], daily_sales_2010["Quantity"], marker='o', linestyle='-')
plt.title("2010 Yılına Ait Haftalık Satış Miktarları")
plt.xlabel("Gün")
plt.ylabel("Satış Miktarı")
plt.xticks(rotation=45)
plt.grid(True)
plt.show()








# 2009 ve 2010 yıllarındaki en çok satış yapılan günler için grafik
most_sold_day_2009['Year'] = 2009
most_sold_day_2010['Year'] = 2010
combined_data = pd.concat([most_sold_day_2009, most_sold_day_2010])
plt.figure(figsize=(10, 6))
sns.barplot(x='Day_name', y='Quantity', hue='Year', data=combined_data)
plt.xlabel('Günler')
plt.ylabel('Toplam Satış')
plt.title('2009 ve 2010 Yıllarının En Çok Satış Yapılan Günlerin Karşılaştırması')
plt.show()



