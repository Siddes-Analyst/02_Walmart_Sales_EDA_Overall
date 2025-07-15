# 🛒 Walmart Null Value Handling - Full Project Journal (One Example)

# <b> <font color= #ABFF00> Walmart EDA(Exploratory Data Analysis)

### <b><font color= #FFFF00> General Topics:
- #### *Import the libraries*
- #### *Load the Dataset*
- #### *Drop Duplicate Rows*
- #### *Change column format(if need)*


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
```


```python
df = pd.read_csv(r"D:\B_Data_Anlysist_Project\Python_Projects\02_Walmart_EDA\eda_walmart_sales_dataset.csv")
df.head()
```

```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1000 entries, 0 to 999
    Data columns (total 10 columns):
     #   Column         Non-Null Count  Dtype  
    ---  ------         --------------  -----  
     0   Order ID       1000 non-null   object 
     1   Order Date     1000 non-null   object 
     2   Customer ID    1000 non-null   object 
     3   Customer Name  1000 non-null   object 
     4   City           1000 non-null   object 
     5   Region         1000 non-null   object 
     6   Category       1000 non-null   object 
     7   Quantity       1000 non-null   int64  
     8   Sales          1000 non-null   float64
     9   Profit         1000 non-null   float64
    dtypes: float64(2), int64(1), object(7)
    memory usage: 78.3+ KB



```python
df.drop_duplicates(inplace= True)
```


```python
df["Order Date"] = pd.to_datetime(df["Order Date"])
```


```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1000 entries, 0 to 999
    Data columns (total 10 columns):
     #   Column         Non-Null Count  Dtype         
    ---  ------         --------------  -----         
     0   Order ID       1000 non-null   object        
     1   Order Date     1000 non-null   datetime64[ns]
     2   Customer ID    1000 non-null   object        
     3   Customer Name  1000 non-null   object        
     4   City           1000 non-null   object        
     5   Region         1000 non-null   object        
     6   Category       1000 non-null   object        
     7   Quantity       1000 non-null   int64         
     8   Sales          1000 non-null   float64       
     9   Profit         1000 non-null   float64       
    dtypes: datetime64[ns](1), float64(2), int64(1), object(6)
    memory usage: 78.3+ KB


### <b><font color= #FFFF00> Q1.Customer Segmentation Challenge:
#### *Identify the top 10% of customers who contributed the most to the total profit. What common characteristics (region, category, city) do they share?*


```python
customer_profit = df.groupby("Customer ID")["Profit"].sum().reset_index()
customer_profit
```

```python
customer_profit = customer_profit.sort_values(by= "Profit", ascending= False)
customer_profit
```

```python
top_10_percent_num = int(0.10 * customer_profit.shape[0])
top_10_percent_num
```

```python
top_10_percent_df = customer_profit.head(top_10_percent_num)
top_10_percent_df
```

```python
top_df = df[df["Customer ID"].isin(top_10_percent_df["Customer ID"])]
top_df
```

### <b><font color= #FFFF00> Region column:


```python
region = top_df["Region"].value_counts()
region_index = region.index
region_values = region.values
```

```python
region = top_df["Region"].value_counts()
region
```


    Region
    West     30
    East     26
    North    25
    South    25
    Name: count, dtype: int64

```python
sns.barplot(x= region_index, y= region_values)
plt.show
```

![png](output_17_1.png)
    


### <b><font color= #FFFF00> Category column:


```python
category = top_df["Category"].value_counts()
category_index = category.index
category_value = category.values
```


```python
print(top_df["Category"].value_counts())
```

    Category
    Furniture          39
    Office Supplies    37
    Technology         30
    Name: count, dtype: int64



```python
sns.barplot(x= category_index,y= category_value)
plt.show()
```


    
![png](output_21_0.png)
    



```python
print(top_df["City"].value_counts().head(10))
```

    City
    South Megan          2
    South Alyssamouth    1
    Reyesmouth           1
    North Sandyfurt      1
    South Ashleyhaven    1
    Craigport            1
    New Stephanie        1
    Ricefurt             1
    Melissatown          1
    New Rachaelhaven     1
    Name: count, dtype: int64


### <b> <font color= #ABFF00> Conclusion:
- #### `Region`: Distribution is fairly even, but [East] has a slight edge.
- #### `Category`: [Furniture] appears more frequently.
- #### `City`: One or two cities like [South Megan] show up more than once, but no strong city dominance.

### <b><font color= #FFFF00> Q2. Monthly Sales Recovery Strategy:
#### *Determine which month in the past year had the lowest overall profit. What specific product category and region contributed most to this loss?*


```python
df_loss = pd.DataFrame(df)
df_loss.head()
```

```python
df_loss["Order Date"].nunique()
```


```python
df_loss["Year"] = df_loss["Order Date"].dt.year
```


```python
df_loss["Month"] = df_loss["Order Date"].dt.month
```


```python
df_loss["Year"].unique()
```
array([2024, 2023, 2025])

```python
df_loss["Year"].value_counts()
```
    Year
    2024    492
    2025    275
    2023    233
    Name: count, dtype: int64

```python
df_2024 = df_loss[df_loss["Year"] == 2024]
df_2024
```

### <b><font color= #FFFF00> Monthly profit:

```python
df_2024_months = df_2024.groupby("Month")["Profit"].sum().reset_index()
df_2024_months
```

```python
df_2024_months = df_2024_months.sort_values(by= "Profit", ascending= False)
df_2024_months
```


```python
sns.barplot(x= df_2024_months["Month"], y= df_2024_months["Profit"])
plt.show()
```
    
![png](output_35_0.png)
    

```python
df_march = df_2024[df_2024["Month"] == 3]
df_march.head()
```


```python
df_march["Profit"].info()
```

    <class 'pandas.core.series.Series'>
    Index: 41 entries, 8 to 993
    Series name: Profit
    Non-Null Count  Dtype  
    --------------  -----  
    41 non-null     float64
    dtypes: float64(1)
    memory usage: 656.0 bytes

```python
df_march["Profit"].sum()
```

### <b><font color= #FFFF00> Region wise Distribution:


```python
df_march["Region"].value_counts()
```




    Region
    North    13
    South    11
    West     10
    East      7
    Name: count, dtype: int64




```python
region = df_march.groupby("Region")["Profit"].sum()
region
```




    Region
    East    -282.76
    North    574.55
    South   -486.86
    West     -57.15
    Name: Profit, dtype: float64




```python
region.index
region.values
sns.barplot(x= region.index, y= region.values)
plt.ylabel("Profit")
plt.show()
```


    
![png](output_42_0.png)
    


### <b><font color= #FFFF00> Category wise Distribution:


```python
df_march["Category"].value_counts()
```




    Category
    Office Supplies    18
    Furniture          12
    Technology         11
    Name: count, dtype: int64




```python
Category = df_march.groupby("Category")["Profit"].sum()
Category
```




    Category
    Furniture          199.04
    Office Supplies   -307.96
    Technology        -143.30
    Name: Profit, dtype: float64




```python
Category.index
Category.values
sns.barplot(x= Category.index, y= Category.values)
plt.ylabel("Profit")
plt.show()
```
    
![png](output_46_0.png)
    
### <b><font color= #FFFF00> Both wise Distribution:


```python
df_march.groupby(["Category","Region"])["Profit"].sum()
```

    Category         Region
    Furniture        East     -150.12
                     North     340.85
                     South     -16.23
                     West       24.54
    Office Supplies  East     -187.05
                     North     266.55
                     South    -365.34
                     West      -22.12
    Technology       East       54.41
                     North     -32.85
                     South    -105.29
                     West      -59.57
    Name: Profit, dtype: float64

```python
grouped = df_march.groupby(["Category", "Region"])["Profit"].sum().reset_index()
```


```python
sns.barplot(x="Category", y="Profit", hue="Region", data=grouped)
plt.show()
```
    
![png](output_50_0.png)
    

### <b> <font color= #ABFF00> Conclusion:
- #### Past year is `2024` – 492 records.
- #### `March` Month make the least amount of loss profit. Loss is `-252.22`.
- #### March month dissection Region wise `South` made a more amount of lose. Loss is `-486.86`.
- #### Category wise `Office Supplies` made a more amount of lose. Loss is `-307.96`.
- #### Both `Region` and `Category` wise `south` & `Office Supplies` made a more amount of lose. Loss is `-365.34`.

### <b><font color= #FFFF00> Q3. Profitability Anomaly Detection:
#### *Identify any orders with high sales but negative profit. What patterns do you notice in terms of region, category, or quantity?*

```python
df_anomaly = pd.DataFrame(df)
df_anomaly
```

```python
df_neg_profit = df_anomaly[df_anomaly["Profit"] < 0]
df_neg_profit
```

```python
df_neg_profit.groupby("Category").agg({"Category" : "count"})
```


```python
df_neg_profit["Quantity"].unique()
```

array([7, 8, 6, 2, 1, 9, 3, 5, 4], dtype=int64)


### <b><font color= #FFFF00> Furniture Distribution:


```python
df_count_furniture = df_neg_profit[df_neg_profit["Category"] == "Furniture"].groupby("Quantity")["Profit"].count()
df_count_furniture
```
    Quantity
    1    16
    2    10
    3    14
    4     7
    5     9
    6    19
    7    14
    8    15
    9    11
    Name: Profit, dtype: int64

```python
df_neg_profit_furniture = df_neg_profit[df_neg_profit["Category"] == "Furniture"].groupby("Quantity")["Profit"].sum()
df_neg_profit_furniture
```

    Quantity
    1    -802.44
    2    -577.02
    3    -805.80
    4    -389.89
    5    -630.29
    6   -1181.15
    7    -607.38
    8    -688.04
    9    -532.80
    Name: Profit, dtype: float64

```python
df_furniture = df_neg_profit[df_neg_profit["Category"] == "Furniture"].groupby("Quantity")["Sales"].sum()
df_furniture
```
    Quantity
    1    9013.62
    2    5562.43
    3    7801.78
    4    3806.66
    5    4987.83
    6    9700.03
    7    5712.62
    8    5971.68
    9    5570.56
    Name: Sales, dtype: float64

```python
plt.figure(figsize = (9,6))
sns.barplot(x= df_furniture.index, y= df_furniture.values)
plt.title("Furniture sales on Negative Profit")
plt.ylabel("Sales")
plt.show()
```
   
![png](output_61_0.png)

### <b><font color= #FFFF00> Office_Suppliers Distribution:

```python
df_count_Office_Supplies = df_neg_profit[df_neg_profit["Category"] == "Office Supplies"].groupby("Quantity")["Profit"].count()
df_count_Office_Supplies
```
    Quantity
    1    16
    2    13
    3    21
    4    12
    5    17
    6    18
    7    14
    8    15
    9    16
    Name: Profit, dtype: int64

```python
df_neg_Office_Supplies = df_neg_profit[df_neg_profit["Category"] == "Office Supplies"].groupby("Quantity")["Profit"].sum()
df_neg_Office_Supplies
```
    Quantity
    1    -876.00
    2    -661.01
    3    -924.51
    4    -628.67
    5    -691.39
    6   -1191.58
    7    -562.95
    8    -725.70
    9    -582.02
    Name: Profit, dtype: float64

```python
df_Office_Supplies = df_neg_profit[df_neg_profit["Category"] == "Office Supplies"].groupby("Quantity")["Sales"].sum()
df_Office_Supplies
```

    Quantity
    1     8454.98
    2     5615.46
    3    10234.87
    4     6603.31
    5     8006.98
    6     9938.60
    7     6935.57
    8     7224.52
    9     8494.63
    Name: Sales, dtype: float64

```python
plt.figure(figsize = (9,6))
sns.barplot(x= df_Office_Supplies.index, y= df_Office_Supplies.values)
plt.title("Office_Supplies sales on Negative Profit")
plt.ylabel("Sales")
plt.show()
```
    
![png](output_66_0.png)
    

### <b><font color= #FFFF00> Technology Distribution:


```python
df_count_Technology = df_neg_profit[df_neg_profit["Category"] == "Technology"].groupby("Quantity")["Profit"].count()
df_count_Technology
```

    Quantity
    1    19
    2    14
    3    14
    4    16
    5    19
    6    20
    7    14
    8    11
    9    14
    Name: Profit, dtype: int64

```python
df_neg_Technology = df_neg_profit[df_neg_profit["Category"] == "Technology"].groupby("Quantity")["Profit"].sum()
df_neg_Technology
```
    Quantity
    1    -965.17
    2    -710.43
    3    -812.00
    4    -604.46
    5   -1283.44
    6    -960.39
    7    -625.54
    8    -518.92
    9    -884.20
    Name: Profit, dtype: float64

```python
df_Technology = df_neg_profit[df_neg_profit["Category"] == "Technology"].groupby("Quantity")["Sales"].sum()
df_Technology
```
    Quantity
    1    10400.94
    2     6155.91
    3     6783.78
    4     7235.92
    5    11130.52
    6    11033.73
    7     6800.03
    8     5772.87
    9     8434.57
    Name: Sales, dtype: float64

```python
plt.figure(figsize = (9,6))
sns.barplot(x= df_Technology.index, y= df_Technology.values)
plt.title("Technology sales on Negative Profit")
plt.ylabel("Sales")
plt.show()
```
  
![png](output_71_0.png)
    

### <b><font color= #FFFF00> East Region Distribution:


```python
df_neg_profit["Region"].unique()
```
    array(['South', 'East', 'North', 'West'], dtype=object)

```python
df_count_East = df_neg_profit[df_neg_profit["Region"] == "East"].groupby("Quantity")["Profit"].count()
df_count_East
```
    Quantity
    1    14
    2    10
    3    10
    4     7
    5    11
    6    19
    7    14
    8    13
    9    14
    Name: Profit, dtype: int64

```python
df_profit_East = df_neg_profit[df_neg_profit["Region"] == "East"].groupby("Quantity")["Profit"].sum()
df_profit_East
```
    Quantity
    1    -929.15
    2    -454.48
    3    -545.82
    4    -517.64
    5    -451.72
    6   -1240.21
    7    -704.56
    8    -669.10
    9    -681.08
    Name: Profit, dtype: float64

```python
df_sales_East = df_neg_profit[df_neg_profit["Region"] == "East"].groupby("Quantity")["Sales"].sum()
df_sales_East
```
    Quantity
    1     9344.71
    2     4328.10
    3     4958.74
    4     4776.51
    5     4709.89
    6    10578.75
    7     6412.53
    8     6265.65
    9     7954.16
    Name: Sales, dtype: float64

```python
plt.figure(figsize = (9,6))
sns.barplot(x= df_sales_East.index, y= df_sales_East.values)
plt.title("East sales on Negative Profit")
plt.ylabel("Sales")
plt.show()
```
  
![png](output_77_0.png)
    

### <b><font color= #FFFF00> West Region Distribution:


```python
df_profit_West = df_neg_profit[df_neg_profit["Region"] == "West"].groupby("Quantity")["Profit"].count()
df_profit_West
```
    Quantity
    1    12
    2     9
    3    15
    4    14
    5    13
    6    13
    7    10
    8    15
    9    10
    Name: Profit, dtype: int64

```python
df_profit_West = df_neg_profit[df_neg_profit["Region"] == "West"].groupby("Quantity")["Profit"].sum()
df_profit_West
```
    Quantity
    1   -329.90
    2   -150.25
    3   -735.72
    4   -701.62
    5   -590.07
    6   -688.36
    7   -389.78
    8   -794.10
    9   -305.97
    Name: Profit, dtype: float64

```python
df_sales_West = df_neg_profit[df_neg_profit["Region"] == "West"].groupby("Quantity")["Sales"].sum()
df_sales_West
```

    Quantity
    1    6215.65
    2    3118.75
    3    6925.78
    4    6878.59
    5    6895.15
    6    7319.91
    7    4470.00
    8    6783.22
    9    3736.70
    Name: Sales, dtype: float64

```python
plt.figure(figsize = (9,6))
sns.barplot(x= df_sales_West.index, y= df_sales_West.values)
plt.title("West sales on Negative Profit")
plt.ylabel("Sales")
plt.show()
```   
![png](output_82_0.png)
    
### <b><font color= #FFFF00> North Region Distribution:


```python
df_profit_North = df_neg_profit[df_neg_profit["Region"] == "North"].groupby("Quantity")["Profit"].count()
df_profit_North
```
    Quantity
    1    13
    2     6
    3    14
    4     5
    5     7
    6     9
    7     6
    8     9
    9    11
    Name: Profit, dtype: int64

```python
df_profit_North = df_neg_profit[df_neg_profit["Region"] == "North"].groupby("Quantity")["Profit"].sum()
df_profit_North
```
    Quantity
    1   -764.34
    2   -285.40
    3   -730.11
    4   -133.06
    5   -623.66
    6   -402.46
    7   -162.61
    8   -299.10
    9   -827.33
    Name: Profit, dtype: float64

```python
df_profit_North = df_neg_profit[df_neg_profit["Region"] == "North"].groupby("Quantity")["Sales"].sum()
df_profit_North
```
    Quantity
    1    5257.83
    2    3190.53
    3    6243.89
    4    2241.93
    5    4197.23
    6    3641.99
    7    1992.01
    8    3807.97
    9    6971.56
    Name: Sales, dtype: float64

```python
plt.figure(figsize = (9,6))
sns.barplot(x= df_profit_North.index, y= df_profit_North.values)
plt.title("North sales on Negative Profit")
plt.ylabel("Sales")
plt.show()
```
    
![png](output_87_0.png)
    

### <b><font color= #FFFF00> South Region Distribution:


```python
df_profit_South = df_neg_profit[df_neg_profit["Region"] == "South"].groupby("Quantity")["Profit"].count()
df_profit_South
```
    Quantity
    1    12
    2    12
    3    10
    4     9
    5    14
    6    16
    7    12
    8     4
    9     6
    Name: Profit, dtype: int64

```python
df_profit_South = df_neg_profit[df_neg_profit["Region"] == "South"].groupby("Quantity")["Profit"].sum()
df_profit_South
```
    Quantity
    1    -620.22
    2   -1058.33
    3    -530.66
    4    -270.70
    5    -939.67
    6   -1002.09
    7    -538.92
    8    -170.36
    9    -184.64
    Name: Profit, dtype: float64

```python
df_profit_South = df_neg_profit[df_neg_profit["Region"] == "South"].groupby("Quantity")["Sales"].sum()
df_profit_South
```
    Quantity
    1    7051.35
    2    6696.42
    3    6692.02
    4    3748.86
    5    8323.06
    6    9131.71
    7    6573.68
    8    2112.23
    9    3837.34
    Name: Sales, dtype: float64

```python
plt.figure(figsize = (9,6))
sns.barplot(x= df_profit_South.index, y= df_profit_South.values)
plt.title("South sales on Negative Profit")
plt.ylabel("Sales")
plt.show()
```
  
![png](output_92_0.png)
    

### <b><font color= #FFFF00> Overall Visualization:

```python
category_sales = df_neg_profit.pivot_table(index="Quantity", columns="Category", values="Sales", aggfunc="sum")
category_sales.plot(kind="bar", figsize=(10,6))
plt.title("Negative Profit: Sales by Quantity and Category")
plt.ylabel("Sales")
plt.show()
```
   
![png](output_94_0.png)
    

```python
region_quantity = df_neg_profit.pivot_table(index="Quantity", columns="Region", values="Sales", aggfunc="sum")
sns.heatmap(region_quantity, annot=True, cmap="YlOrRd", fmt=".0f")
plt.title("Sales with Negative Profit by Quantity and Region")
plt.show()
```
  
![png](output_95_0.png)
    
### <b> <font color= #ABFF00> Conclusion:
- #### The same quantities — especially `1`, `3`, and `6` units — are showing up again and again in loss-making orders.
- #### This happens in all product categories (Furniture, Office Supplies, Technology).
- #### It also happens in all regions (East, West, North, South).
- #### This tells us that small quantity orders, even when they have high sales, are still not profitable.

### <b><font color= #FFFF00> Q4. Optimizing Product Mix for Regions:
#### *For each region, find the best-selling category by volume and the most profitable category. Are they the same? What does this imply?*

```python
df_product_mix = pd.DataFrame(df)
df_product_mix.head()
```
