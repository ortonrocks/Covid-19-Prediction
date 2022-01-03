import pandas  as pd
df=pd.read_csv('train_covid.csv')

#對各州進行標籤
df['state_code'].unique()

states_dict={
    'az':1,'fl':2,'ia':3,'in':4,'la':5,
    'mo':6,'nj':7,'nm':8,'nv':9,'ny':10,
    'pa':11,'va':12,'al':13,'ar':14,'ca':15,
    'ct':16,'il':17, 'ky':18,'ma':19,'mi':20,
    'mn':21,'ms':22,'mt':23,'nc':24,'nh':25,
    'oh':26,'ri':27,'sc':28,'sd':29,'tx':30,
    'ut':31,'wi':32,'co':33,'ga':34,'id':35,
    'ks':36, 'md':37,'ne':38,'ok':39,'or':40,
    'ak':41,'de':42,'wa':43,'tn':44,  'hi':45,
    'nd':46,'me':47,'wv':48,'vt':49 }

df=df.replace({'state_code':states_dict})

#對性別進行標籤
gender_dict={'female':0,"male":1 }

df=df.replace({'gender':gender_dict})

#對年齡進行標籤
age_dict={'18-34':0,"35-54":1,'55+':2 }

#取代
df=df.replace({'age_bucket':age_dict})

#刪除多於列
df=df.drop(['date','n','weight_sums','pct_worried_finances'],axis=1)

#把na值轉換為0
df.fillna(0, inplace=True)

#weight標準化
def minmax_norm(df_input):
    return (df_input - df_input.min()) / ( df_input.max() - df_input.min())
df['pct_cli']=minmax_norm(df['pct_cli'])

df['pct_cli']=minmax_norm(df['pct_cli'])

#將資料欄轉化為list查看欄數
column_list=df.columns.tolist()
len(column_list)

#對第三列以後的資料進行標準化
column_list=df.columns.tolist()
column_list=column_list[4:93]
for column in column_list:
    df[column]=minmax_norm(df[column])

#輸出csv檔
df.to_csv('data_washing.csv',encoding='utf-8')