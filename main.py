import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,roc_auc_score
from sklearn.model_selection import GridSearchCV, cross_validate, validation_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz, export_text
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
from skompiler import skompile
import matplotlib.pyplot as plt
import graphviz
import joblib
from sklearn.tree import export_graphviz
import pydotplus
import warnings
warnings.simplefilter(action="ignore")

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

data = pd.read_csv('../OkulProje/database/diabetes.csv')

#Burada diabetes veri setini checkup ediyoruz.
def check_df(dataframe, head=8):
  print("##### Shape #####") #kac kolon kac satir
  print(dataframe.shape)
  print("##### Types #####") #kolonların, yani özelliklerin tipleri (int,float,str..)
  print(dataframe.dtypes)
  print("##### Tail #####") #Veri setinin son 5 degerini inceliyoruz.
  print(dataframe.tail(head))
  print("##### Head #####") #Veri setinin ilk 5 degerini inceliyoruz.
  print(dataframe.head(head))
  print("##### Null Analysis #####") #Bos deger olup olmadigini kontrol ediyoruz.
  print(dataframe.isnull().sum())
  print("##### Quantiles #####") #sayısal verilere sahip olan sütunların istatiksel değerlerine baktık.
  #Hamilelik max 17 olarak girilmis, 17 kere hamilelik kulaga biraz imkansız geliyor. Verimizde Outlier degerler bulunuyor.
  # Mesela glucose degeri min olarak 0 gosterilmis, glucose 0 olamaz. Demek ki null kısımlara 0 girilmis.
  print(dataframe.describe([0,0.05, 0.50, 0.95, 0.99, 1]).T)

check_df(data)

# Kolon isimlerini büyültüyorum ki sorgulama vs yaparken yazmak, okumak kolay olsun.
data.columns = [col.upper() for col in data.columns]
data.head()

#Outlier değerleri grafik üzerinde görebilmek için
f, ax = plt.subplots(figsize=(20,20)) #f->figure and ax->axis
fig = sns.boxplot(data=data, orient="h") #horizontally (grafiği yatayda alabilmek için)
#plt.show()


#Korelasyon analizi için
#Mesela buradaki korelasyona bakarak, doğum ve yaş arasında 0.54 pozitif korelasyon var.
#Outcome ı en çok etkileyen glikoz değeriymiş. 0.47
#Glikoz değerinden sonra en çok etki eden BMI olmuş. 0.29
#Yaş ile deri kalınlığı arasında da negatif korelasyon bulunmakta. -0.11
sns.clustermap(data.corr(), annot = True, fmt = ".2f")
#plt.show()

# BASE MODEL KURULUMU


y = data["OUTCOME"]
X = data.drop("OUTCOME", axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=28)

rf_model = RandomForestClassifier(random_state=46).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)


print(f"Accuracy: {round(accuracy_score(y_pred, y_test), 2)}")
print(f"Recall: {round(recall_score(y_pred,y_test),3)}")
print(f"Precision: {round(precision_score(y_pred,y_test), 2)}")
print(f"F1: {round(f1_score(y_pred,y_test), 2)}")
print(f"Auc: {round(roc_auc_score(y_pred,y_test), 2)}")

# Accuracy: 0.77 for random_state=17
# Recall: 0.706
# Precision: 0.59
# F1: 0.64
# Auc: 0.75

# Accuracy: 0.74 for random_state=32
# Recall: 0.667
# Precision: 0.57
# F1: 0.62
# Auc: 0.72

# Accuracy: 0.8 for random_state=12
# Recall: 0.768
# Precision: 0.63
# F1: 0.69
# Auc: 0.79

# Accuracy: 0.81 for random_state=25
# Recall: 0.68
# Precision: 0.72
# F1: 0.7
# Auc: 0.78

# Accuracy: 0.73 for random_statement=28
# Recall: 0.632
# Precision: 0.54
# F1: 0.58
# Auc: 0.7


#Degiskenleri kategorik, kardinal ve nümerik olarak ayırıyoruz.
# Kardinal Degisken : kategorik degiskenin 20den fazla (buna biz karar veriyoruz 5 da  yazabilirdik.) sınıfı varsa kategorik gibi gorunen degiskenlerdir.
"""
Kardinal değişkenler, sayılarla ifade edilebilen ve belirli bir sıralamaya sahip olmayan değişkenlerdir. 
Örneğin, bir ankette katılımcıların yaşları, eğitim seviyeleri veya gelir düzeyleri kardinal değişkenlere örnek olarak verilebilir.
Aşağıda, katılımcıların yaşlarını içeren basit bir veri seti örneği bulunmaktadır:

{25,30,35,40,22,28,32,37,45,29}

Bu veri setindeki her bir sayı, bir katılımcının yaşı olarak temsil edilir. 
Bu veri seti, kardinal bir değişkeni gösterir çünkü her bir değer sayısal olarak ifade edilebilir ve bu değerler arasında bir sıralama mevcuttur.
"""
def grab_col_names(dataframe, cat_th=10,
                   car_th=20):  # essiz deger sayisi 10dan kucukse kategorik degisken, 5 den buyukse de kardinal degisken gibi dusunucez.
  # Veri setimiz küçük olduğundan ben 5 ile sınırlandırdım.
  cat_cols = [col for col in dataframe.columns if str(dataframe[col].dtypes) in ["category", "object", "bool"]]

  num_but_cat = [col for col in dataframe.columns if
                 dataframe[col].nunique() < cat_th and dataframe[col].dtypes in ["int", "float"]]

  cat_but_car = [col for col in dataframe.columns if
                 dataframe[col].nunique() > car_th and str(dataframe[col].dtypes) in ["category", "object"]]

  cat_cols = num_but_cat + cat_cols
  cat_cols = [col for col in cat_cols if col not in cat_but_car]

  num_cols = [col for col in dataframe.columns if dataframe[col].dtypes in ["int", "float"]]
  num_cols = [col for col in num_cols if col not in cat_cols]

  print(f"Observations: {dataframe.shape[0]}")
  print(f"Variables: {dataframe.shape[1]}")
  print(f"Categorical Columns: {len(cat_cols)}")
  print(f"Numerical Columns: {len(num_cols)}")
  print(f"Categoric but Cardinal: {len(cat_but_car)}")
  print(f"Numeric but Categoric: {len(num_but_cat)}")

  return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(data)

"""
Observations: 768
Variables: 9
Categorical Columns: 1
Numerical Columns: 8
Categoric but Cardinal: 0
Numeric but Categoric: 1
(['OUTCOME'], ['PREGNANCIES', 'GLUCOSE', 'BLOODPRESSURE', 'SKINTHICKNESS', 'INSULIN', 'BMI', 'DIABETESPEDIGREEFUNCTION', 'AGE'], [])
#Çıktıda aldığımız değerlere göre kardinal bir değişkenimiz bulunmamakta, 1 kategorik, 8 numerik, 1 tane de numerik ama kategorik (OUTCOME) değerimiz bulunmakta.
"""

#numerik değiskenler ve target analizi
def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")

for col in num_cols:
    target_summary_with_num(data, "OUTCOME", col)

#nümerik değişken analizi
def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show()

for col in num_cols:
    num_summary(data, col, plot=True)

#kategorik degisken analizi yani sadece outcome
def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()

cat_summary(data, "OUTCOME")

#0 değerlerini NaN olarak değiştirelim.
nan_col=['GLUCOSE','BLOODPRESSURE','SKINTHICKNESS', 'INSULIN', 'BMI']
data[nan_col]=data[nan_col].replace(0, np.NaN)
#kaç boş değer olduğunu gördük ki deri kalınlığı ve insülin değerinde fazlasıyla boş değer var.
data.isnull().sum()


# "Outcome" değeri 1 olan gözlemler için medyan değer hesaplama
median_value_outcome_1 = data.loc[data['OUTCOME'] == 1].median()
# "Outcome" değeri 0 olan gözlemler için medyan değer hesaplama
median_value_outcome_0 = data.loc[data['OUTCOME'] == 0].median()
# Boş değerleri doldurma
data.loc[data['OUTCOME'] == 1] = data.loc[data['OUTCOME'] == 1].fillna(median_value_outcome_1)
data.loc[data['OUTCOME'] == 0] = data.loc[data['OUTCOME'] == 0].fillna(median_value_outcome_0)

data.isnull().sum()

#Aykırı değerimizi (Outlier) saptama işlemi için:
def outlier_thresholds(dataframe, col_name, q1 = 0.25, q3 = 0.75):
  quantile1 = dataframe[col_name].quantile(q1)
  quantile3 = dataframe[col_name].quantile(q3)
  interquantile_range = quantile3 - quantile1
  up_limit = quantile3 + 1.5 * interquantile_range
  low_limit = quantile1 - 1.5 * interquantile_range
  return low_limit, up_limit

#Thresholdlara göre outlier var mı yok mu diye kontrol etmek için:
def check_outlier(dataframe, col_name):
  low_limit, up_limit = outlier_thresholds(dataframe, col_name)
  if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
    return True
  else:
    return False

#Var olan outlierları görmek için:
def grab_outliers(dataframe, col_name, index=False):
  low, up = outlier_thresholds(dataframe, col_name)
  if dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].shape[0] > 10:
    print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].head())
  else:
    print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))])
  if index:
    outlier_index = dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].index
    return outlier_index

for col in num_cols:
  print(col,"için LOW LIMIT, UP LIMIT değerleri = ", outlier_thresholds(data,col),"\n")

for col in num_cols:
  print(col,"için outlier kontrolü", check_outlier(data,col),"\n")


#Bu çıktıdan alınan değerlere göre hamilelik (PREGNANCIES) 14-17 aralığındakiler outlier,
#Glikoz değerleri için 0 görünen kısımlar null (glikoz 0 olamaz),
#Kan basıncı değeri (BLOODPRESSURE) 0 olamaz null değerler var.
#BMI değerleri 0 olamaz null değerler var.
#Deri kalınlığı min 0.6 mm, max 2.4mm dir. 0 olan değerler null değerdir.
for col in num_cols:
  print(col,"için var olan outlierlar\n", grab_outliers(data,col))

#replace_with_thresholds fonksiyonunu uygulamadan önce yukarıda kararını verdiğimiz 0 değerleri null yapmalıyız ki onları da outlier olarak görmesin.
def replace_with_thresholds(dataframe, variable):
  low_limit, up_limit = outlier_thresholds(dataframe, variable)
  dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
  dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

#şimdi kalan aykırı değerlerin değişimini sağlayabiliriz.
for col in num_cols:
  print(replace_with_thresholds(data, col))


plt.hist(data, bins=10, edgecolor='black')  # 'bins' parametresiyle aralık sayısını belirleyebilirsiniz
plt.xlabel('Değer Aralığı')
plt.ylabel('Frekans')
plt.title('Veri Seti Histogramı')
plt.show()

#FEATURE ENGINEERING

# Yaş değişkenini kategorilere ayırıp yeni yaş değişkeni oluşturulması
data.loc[(data["AGE"] >= 18) & (data["AGE"] <= 32), "NEW_AGE_CAT"] = "Young"
data.loc[(data["AGE"] >  32) & (data["AGE"] <  50), "NEW_AGE_CAT"] = "Adult"
data.loc[(data["AGE"] >= 50), "NEW_AGE_CAT"] = "Mature"

# BMI 18,5 aşağısı underweight, 18.5 ile 24.9 arası normal, 24.9 ile 29.9 arası Overweight ve 30 üstü obez
data['NEW_BMI'] = pd.cut(x=data['BMI'], bins=[0, 18.5, 24.9, 29.9, 100],labels=["Underweight", "Healthy", "Overweight", "Obese"])

# Glukoz degerini kategorik değişkene çevirme
data["NEW_GLUCOSE"] = pd.cut(x=data["GLUCOSE"], bins=[0, 70, 140, 200, 300], labels=["Low", "Healthy", "Prediabetes", "Diabetes"])

# Deri Kalinligi degerini kategorik degiskene cevirme mm turunden
data["NEW_SKIN_THIC"] = pd.cut(x=data["SKINTHICKNESS"], bins=[1.25, 2, 2.5, 3.25], labels=["Thin", "Healthy", "Thick"])

# # Yaş ve beden kitle indeksini bir arada düşünerek kategorik değişken oluşturma 3 kırılım yakalandı
data.loc[(data["BMI"] < 18.5) & ((data["AGE"] >= 18) & (data["AGE"] <= 32)), "NEW_AGE_BMI_NOM"] = "UnderweightYoung"
data.loc[(data["BMI"] < 18.5) & ((data["AGE"] > 32) & (data["AGE"] < 50)), "NEW_AGE_BMI_NOM"] = "UnderweightAdult"
data.loc[(data["BMI"] < 18.5) & (data["AGE"] >= 50), "NEW_AGE_BMI_NOM"] = "UnderweightMature"
data.loc[((data["BMI"] >= 18.5) & (data["BMI"] < 25)) & ((data["AGE"] >= 18) & (data["AGE"] <= 32)), "NEW_AGE_BMI_NOM"] = "HealthyYoung"
data.loc[((data["BMI"] >= 18.5) & (data["BMI"] < 25)) & ((data["AGE"] > 32) & (data["AGE"] < 50)), "NEW_AGE_BMI_NOM"] = "HealthyAdult"
data.loc[((data["BMI"] >= 18.5) & (data["BMI"] < 25)) & (data["AGE"] >= 50), "NEW_AGE_BMI_NOM"] = "HealthyMature"
data.loc[((data["BMI"] >= 25) & (data["BMI"] < 30)) & ((data["AGE"] >= 18) & (data["AGE"] <= 32)), "NEW_AGE_BMI_NOM"] = "OverweightYoung"
data.loc[((data["BMI"] >= 25) & (data["BMI"] < 30)) & ((data["AGE"] > 32) & (data["AGE"] < 50)), "NEW_AGE_BMI_NOM"] = "OverweightAdult"
data.loc[((data["BMI"] >= 25) & (data["BMI"] < 30)) & (data["AGE"] >= 50), "NEW_AGE_BMI_NOM"] = "OverweightMature"
data.loc[(data["BMI"] > 30) & ((data["AGE"] >= 18) & (data["AGE"] <= 32)), "NEW_AGE_BMI_NOM"] = "ObeseYoung"
data.loc[(data["BMI"] > 30) & ((data["AGE"] > 32) & (data["AGE"] < 50)), "NEW_AGE_BMI_NOM"] = "ObeseAdult"
data.loc[(data["BMI"] > 30) & (data["AGE"] >= 50), "NEW_AGE_BMI_NOM"] = "ObeseMature"

# Yaş ve Glikoz değerlerini bir arada düşünerek kategorik değişken oluşturma
data.loc[(data["GLUCOSE"] < 70) & ((data["AGE"] >= 18) & (data["AGE"] <= 32)), "NEW_AGE_GLUCOSE_NOM"] = "LowYoung"
data.loc[(data["GLUCOSE"] < 70) & ((data["AGE"] > 32) & (data["AGE"] < 50)), "NEW_AGE_GLUCOSE_NOM"] = "LowAdult"
data.loc[(data["GLUCOSE"] < 70) & (data["AGE"] >= 50), "NEW_AGE_GLUCOSE_NOM"] = "LowMature"
data.loc[((data["GLUCOSE"] >= 70) & (data["GLUCOSE"] < 140)) & ((data["AGE"] >= 18) & (data["AGE"] <= 32)), "NEW_AGE_GLUCOSE_NOM"] = "HealthyYoung"
data.loc[((data["GLUCOSE"] >= 70) & (data["GLUCOSE"] < 140)) & ((data["AGE"] > 32) & (data["AGE"] < 50)), "NEW_AGE_GLUCOSE_NOM"] = "HealthyAdult"
data.loc[((data["GLUCOSE"] >= 70) & (data["GLUCOSE"] < 140)) & (data["AGE"] >= 50), "NEW_AGE_GLUCOSE_NOM"] = "HealthyMature"
data.loc[((data["GLUCOSE"] >= 140) & (data["GLUCOSE"] <= 200)) & ((data["AGE"] >= 18) & (data["AGE"] <= 32)), "NEW_AGE_GLUCOSE_NOM"] = "PrediabetesYoung"
data.loc[((data["GLUCOSE"] >= 140) & (data["GLUCOSE"] <= 200)) & ((data["AGE"] > 32) & (data["AGE"] < 50)), "NEW_AGE_GLUCOSE_NOM"] = "PrediabetesAdult"
data.loc[((data["GLUCOSE"] >= 140) & (data["GLUCOSE"] <= 200)) & (data["AGE"] >= 50), "NEW_AGE_GLUCOSE_NOM"] = "PrediabetesMature"
data.loc[(data["GLUCOSE"] > 200) & ((data["AGE"] >= 18) & (data["AGE"] <= 32)), "NEW_AGE_GLUCOSE_NOM"] = "DiabetesYoung"
data.loc[(data["GLUCOSE"] > 200) & ((data["AGE"] > 32) & (data["AGE"] < 50)), "NEW_AGE_GLUCOSE_NOM"] = "DiabetesAdult"
data.loc[(data["GLUCOSE"] > 200) & (data["AGE"] >= 50), "NEW_AGE_GLUCOSE_NOM"] = "DiabetesMature"

# Deri Kalinligi ve Beden Kitle Indeksi degerlerini bir arada dusunerek kategorik degisken olusturma
data.loc[(data["BMI"] < 18.5) & ((data["SKINTHICKNESS"] > 1.25) & (data["SKINTHICKNESS"] <= 2)), "NEW_BMI_SKIN_THIC_NOM"] = "UnderweightThin"
data.loc[(data["BMI"] < 18.5) & ((data["SKINTHICKNESS"] > 2) & (data["SKINTHICKNESS"] <= 2.5)), "NEW_BMI_SKIN_THIC_NOM"] = "UnderweightHealthy"
data.loc[(data["BMI"] < 18.5) & ((data["SKINTHICKNESS"] > 2.5) & (data["SKINTHICKNESS"] <= 3.25)), "NEW_BMI_SKIN_THIC_NOM"] = "UnderweightThick"
data.loc[((data["BMI"] >= 18.5) & (data["BMI"] < 25)) & ((data["SKINTHICKNESS"] > 1.25) & (data["SKINTHICKNESS"] <= 2)), "NEW_BMI_SKIN_THIC_NOM"] = "HealthyThin"
data.loc[((data["BMI"] >= 18.5) & (data["BMI"] < 25)) & ((data["SKINTHICKNESS"] > 2) & (data["SKINTHICKNESS"] <= 2.5)), "NEW_BMI_SKIN_THIC_NOM"] = "Healthy_Healthy"
#data.loc[((data["BMI"] >= 18.5) % (data["BMI"] < 25)) & ((data["SKINTHICKNESS"] > 2 & (data["SKINTHICKNESS"] <= 2.5)), "NEW_BMI_SKIN_THIC_NOM"] = "Healthy_Healthy"
data.loc[((data["BMI"] >= 18.5)) & (data["BMI"] < 25) & ((data["SKINTHICKNESS"] > 2.5) & (data["SKINTHICKNESS"] <= 3.25)), "NEW_BMI_SKIN_THIC_NOM"] = "HealthyThick"
data.loc[((data["BMI"] >= 25) % (data["BMI"] < 30)) & ((data["SKINTHICKNESS"] > 1.25) & (data["SKINTHICKNESS"] <= 2)), "NEW_BMI_SKIN_THIC_NOM"] = "OverweightThin"
data.loc[((data["BMI"] >= 25) % (data["BMI"] < 30)) & ((data["SKINTHICKNESS"] > 2) & (data["SKINTHICKNESS"] <= 2.5)), "NEW_BMI_SKIN_THIC_NOM"] = "OverweightHealthy"
data.loc[((data["BMI"] >= 25) % (data["BMI"] < 30)) & ((data["SKINTHICKNESS"] > 2.5) & (data["SKINTHICKNESS"] <= 3.25)), "NEW_BMI_SKIN_THIC_NOM"] = "OverweightThick"
data.loc[(data["BMI"] > 30) & ((data["SKINTHICKNESS"] > 1.25) & (data["SKINTHICKNESS"] <= 2)), "NEW_BMI_SKIN_THIC_NOM"] = "ObeseThin"
data.loc[(data["BMI"] > 30) & ((data["SKINTHICKNESS"] > 2) & (data["SKINTHICKNESS"] <= 2.5)), "NEW_BMI_SKIN_THIC_NOM"] = "ObeseHealthy"
data.loc[(data["BMI"] > 30) & ((data["SKINTHICKNESS"] > 2.5) & (data["SKINTHICKNESS"] <= 3.25)), "NEW_BMI_SKIN_THIC_NOM"] = "ObeseThick"

# Deri Kalinligi ve Insulin degerlerini bir arada dusunerek kategorik degisken olusturma
data.loc[((data["SKINTHICKNESS"] > 1.25) & (data["SKINTHICKNESS"] <= 2)) & ((data["INSULIN"] >= 16) & (data["INSULIN"] <=160)), "NEW_BMI_SKIN_THIC_NOM"] = "ThinNormal"
data.loc[((data["SKINTHICKNESS"] > 1.25) & (data["SKINTHICKNESS"] <= 2)) & (data["INSULIN"] >=160), "NEW_BMI_SKIN_THIC_NOM"] = "ThinAbNormal"
data.loc[((data["SKINTHICKNESS"] > 2) & (data["SKINTHICKNESS"] <= 2.5)) & ((data["INSULIN"] >= 16) & (data["INSULIN"] <=160)), "NEW_BMI_SKIN_THIC_NOM"] = "HealthyNormal"
data.loc[((data["SKINTHICKNESS"] > 2) & (data["SKINTHICKNESS"] <= 2.5)) & (data["INSULIN"] >=160), "NEW_BMI_SKIN_THIC_NOM"] = "HealthyAbNormal"
data.loc[((data["SKINTHICKNESS"] > 2.5) & (data["SKINTHICKNESS"] <= 3)) & ((data["INSULIN"] >= 16) & (data["INSULIN"] <=160)), "NEW_BMI_SKIN_THIC_NOM"] = "ThickNormal"
data.loc[((data["SKINTHICKNESS"] > 2.5) & (data["SKINTHICKNESS"] <= 3)) & (data["INSULIN"] >=160), "NEW_BMI_SKIN_THIC_NOM"] = "ThickAbNormal"



# İnsulin Değeri ile Kategorik değişken türetmek

def set_insulin(dataframe, col_name="INSULIN"):
    if 16 <= dataframe[col_name] <= 166:
        return "Normal"
    else:
        return "Abnormal"


# ENCODING

# Değişkenlerin tiplerine göre ayrılması işlemi
cat_cols, num_cols, cat_but_car = grab_col_names(data)

# LABEL ENCODING

def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe


binary_cols = [col for col in data.columns if data[col].dtypes == "O" and data[col].nunique() == 2]
binary_cols

for col in binary_cols:
    data = label_encoder(data, col)

# One-Hot Encoding İşlemi
# cat_cols listesinin güncelleme işlemi
cat_cols = [col for col in cat_cols if col not in binary_cols and col not in ["OUTCOME"]]
cat_cols


def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

data = one_hot_encoder(data, cat_cols, drop_first=True)

data.head()
print(data.columns)



# Model Kurulumu CART

data = pd.read_csv("database/diabetes.csv")
data.columns = [col.upper() for col in data.columns]

y = data["OUTCOME"]
X = data.drop(["OUTCOME"], axis=1)

cart_model = DecisionTreeClassifier(random_state=28).fit(X, y)

# Confusion Matrix icin y_pred
y_pred = cart_model.predict(X)

# AUC icin y_prob
y_prob = cart_model.predict_proba(X)[:, 1]

# Confusion Matrix
print(classification_report(y, y_prob))

# AUC
roc_auc_score(y, y_prob)

# CROSS VALIDATION yontemi ile basari degerlendirme

cart_model = DecisionTreeClassifier(random_state=28).fit(X, y)

cv_results =cross_validate(cart_model, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc", "precision", "recall"])

print("Test Accuracy:", cv_results['test_accuracy'])
cv_results['test_accuracy'].mean()

print("Test F1:", cv_results['test_f1'])
cv_results['test_f1'].mean()

print("Test AUC:", cv_results['test_roc_auc'])
cv_results['test_roc_auc'].mean()

print("Test Precision:", cv_results['test_precision'])
cv_results['test_precision'].mean()

print("Test Recall:", cv_results['test_recall'])
cv_results['test_recall'].mean()

# HiperParametre Optimizasyonu GridSearchCV ile

cart_model.get_params()

cart_params = {"max_depth": range(1, 11), "min_samples_split": range(2, 20)}

cart_best_grid = GridSearchCV(cart_model, cart_params, cv=5, n_jobs=1, verbose=1).fit(X, y)

cart_best_grid.best_params_
print("En Iyi Parametreler:", cart_best_grid.best_params_)

cart_best_grid.best_score_
print("En Iyi Skor:", cart_best_grid.best_score_)

random = X.sample(1, random_state=45)

cart_best_grid.predict(random)

# FINAL MODEL

cart_final= DecisionTreeClassifier(**cart_best_grid.best_params_, random_state=28).fit(X, y)
cart_final.get_params()

cart_final = cart_model.set_params(**cart_best_grid.best_params_).fit(X, y)

cv_results = cross_validate(cart_final, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc", "precision", "recall"])

print("Test Accuracy:", cv_results['test_accuracy'])
cv_results['test_accuracy'].mean()

print("Test F1:", cv_results['test_f1'])
cv_results['test_f1'].mean()

print("Test AUC:", cv_results['test_roc_auc'])
cv_results['test_roc_auc'].mean()

print("Test Precision:", cv_results['test_precision'])
cv_results['test_precision'].mean()

print("Test Recall:", cv_results['test_recall'])
cv_results['test_recall'].mean()

# Model Analiz Etme

def val_curve_params(model, X, y, param_name, param_range, scoring="roc_auc", cv=5):
    train_score, test_score = validation_curve(
        model, X=X, y=y, param_name=param_name, param_range=param_range, scoring=scoring, cv=cv)

    mean_train_score = np.mean(train_score, axis=1)
    mean_test_score = np.mean(test_score, axis=1)

    plt.plot(param_range, mean_train_score, label="Egitim Skoru", color='#ff4d00')

    plt.plot(param_range, mean_test_score, label="Test Skoru", color='#000080')

    plt.title(f"{type(model).__name__} icin Dogrulama Egrisi")
    plt.xlabel(f"{param_name} Sayisi")
    plt.ylabel(f"{scoring}")
    plt.legend(loc='best')
    plt.show(block=True)

cart_val_params =[["max_depth", range(1, 11)], ["min_samples_split", range(2, 20)]]

for i in range(len(cart_val_params)):
    val_curve_params(cart_model, X, y, cart_val_params[i][0], cart_val_params[i][1])


# Karar Agacini Goruntuleme
def tree_graph(model, col_names, file_name):
    tree_str = export_graphviz(model, feature_name=col_names, filled=True, out_file=None)
    graph = pydotplus.graph_from_dot_data(tree_str)
    graph.write_png(file_name)


tree_graph(model=cart_final, col_names=X.columns, file_name="cart_final.png")

cart_final.get_params()

# Modeli Kayit Etme ve Yukleme

joblib.dump(cart_final, "cart_final.pkl")

cart_model_from_disc = joblib.load("cart_final.pkl")


