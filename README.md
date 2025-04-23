# EXNO2DS
register number :212224230126
# AIM:
      To perform Exploratory Data Analysis on the given data set.
      
# EXPLANATION:
  The primary aim with exploratory analysis is to examine the data for distribution, outliers and anomalies to direct specific testing of your hypothesis.
  
# ALGORITHM:
STEP 1: Import the required packages to perform Data Cleansing,Removing Outliers and Exploratory Data Analysis.

STEP 2: Replace the null value using any one of the method from mode,median and mean based on the dataset available.

STEP 3: Use boxplot method to analyze the outliers of the given dataset.

STEP 4: Remove the outliers using Inter Quantile Range method.

STEP 5: Use Countplot method to analyze in a graphical method for categorical data.

STEP 6: Use displot method to represent the univariate distribution of data.

STEP 7: Use cross tabulation method to quantitatively analyze the relationship between multiple variables.

STEP 8: Use heatmap method of representation to show relationships between two variables, one plotted on each axis.

## CODING AND OUTPUT
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns  
df=pd.read_csv("/content/titanic_dataset.csv")
df
```
![426956563-7ce4fecf-2e76-41ee-8f2e-c12f404b060b](https://github.com/user-attachments/assets/4b653fe9-8339-4679-bf39-5df8c06a8bd9)
```
df.info()
```
![426956774-198b0af4-a286-40f0-a066-c041a2f79c37](https://github.com/user-attachments/assets/45f3509a-69bd-4059-adac-bf84436eb777)
```
df.shape
```
![426957147-f74f8c8f-4593-4144-b9ed-e22563d678a8](https://github.com/user-attachments/assets/9ae144d1-9a6d-4454-a9f9-56e36232c3c0)
```
df.set_index("PassengerId",inplace=True)
df.describe()
```
![426957374-d684de01-719e-4b79-996b-bcc84383c3ee](https://github.com/user-attachments/assets/c63800ea-4bfa-4212-a788-dfd72e7f8f56)
```
df.shape
```
![426957572-cf1f16f7-f5dc-4af5-b7f5-182838dbd335](https://github.com/user-attachments/assets/f1c6385d-690f-47b3-9132-fb4d4bf18a5e)
```
df.nunique()
```
![426957809-0979d962-9a1a-4f1d-8161-aa150b333758](https://github.com/user-attachments/assets/2f8174d2-c194-452c-9cfb-3600e7da2a04)
```
 df["Survived"].value_counts()
```
![426958025-d5d68db1-b097-47d9-bc52-5b6a1a436613](https://github.com/user-attachments/assets/907e5118-81c2-4208-b122-bfe4db8ff8e7)
```
per=(df["Survived"].value_counts()/df.shape[0]*100).round(2)
per
```
![426958300-c95f7796-5a6f-40e8-bed3-c26a7c809638](https://github.com/user-attachments/assets/a22d5732-7f56-4dcf-8ce7-011cfa72e728)
```
sns.countplot(data=df,x="Survived")
```
![426958510-848bee5d-383d-462c-b7af-858f163c234b](https://github.com/user-attachments/assets/c8e6859a-4835-45d2-81f7-e3038419cd2c)
```
df
```
![426958767-90855b04-f263-40e5-b22b-70623ab636bf](https://github.com/user-attachments/assets/c567977d-e172-488f-8b93-b21131cf0a8e)
```
df.Pclass.unique()
```
![426958987-e61afe2a-63a0-447d-bb6e-3e5ab9d445ea](https://github.com/user-attachments/assets/ef57876e-7a9c-4c55-a9e8-2053c9b788d3)
```
df.rename(columns={'Sex':'Gender'},inplace=True)
df
```
![426959278-e44a4788-52a4-4b64-9b56-963aad47e8b4](https://github.com/user-attachments/assets/18e493b1-e9a7-4e1a-9873-700c07c725cf)
```
sns.catplot(x="Gender",col="Survived",kind="count",data=df,height=5,aspect=.7)
```
![426959517-f5c739f4-d7fa-4b0b-853e-d8fa16b044d0](https://github.com/user-attachments/assets/bfce2bd6-3ed1-452c-bc67-b464c55115d9)
```
sns.catplot(x="Survived",hue="Gender",data=df,kind="count")
```
![426959948-46654825-2452-422d-a336-e9ece839eace](https://github.com/user-attachments/assets/bf161f84-42dc-4b35-9442-a21b6f3ee223)
```
df.boxplot(column="Age",by="Survived")
```
![426960377-f8117598-4b76-4fa8-9e2a-22b6d3ad0d4b](https://github.com/user-attachments/assets/b05bdd3c-6106-468f-a063-d9d0f2c1fee1)
```
sns.scatterplot(x=df["Age"],y=df["Fare"])
```
![426960753-d6b4a751-9c17-492f-bce7-23abd165e5b3](https://github.com/user-attachments/assets/257801e9-b2f6-4136-9de3-1e65f8844630)
```
sns.jointplot(x="Age",y="Fare",data=df)
```
![426961036-8ce4f093-3613-4fde-adb1-0010d3d4805d](https://github.com/user-attachments/assets/ee89bd3f-a03a-4b01-bffe-4ebf7339c6de)
```
 fig, ax1 = plt.subplots(figsize=(8,5))
 plt = sns.boxplot(ax=ax1,x='Pclass',y='Age',hue='Gender',data=df)
```
![426961353-a8454fd3-5810-4e61-8ae5-581bb00f38a0](https://github.com/user-attachments/assets/fca385ac-38e2-4bc4-bf1d-9443e0286864)
```
sns.catplot(data=df,col="Survived",x="Gender",hue="Pclass",kind="count")
```
![426961735-e08c0fc2-ec10-4b7f-a54e-1f1fccb429c9](https://github.com/user-attachments/assets/173b97a6-b50f-4135-ac14-c291079139b0)
```
numerical_df = df.select_dtypes(include=['number'])
corr = numerical_df.corr()
sns.heatmap(corr, annot=True)
```
![426961998-9ce5e00f-a0d1-4d21-bcfc-81545d720094](https://github.com/user-attachments/assets/5f06dda4-5d23-4628-a04c-5ad110ffd6ec)
```
 sns.pairplot(df)
```
![426965057-99c75f40-1337-4195-b1de-74b4a2c9162c](https://github.com/user-attachments/assets/9f7631d1-a6da-4db9-8e26-846c66d29f63)

# RESULT
Thus, the Exploratory Data Analysis on the given data set was performed successfully.
