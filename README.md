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




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Order ID</th>
      <th>Order Date</th>
      <th>Customer ID</th>
      <th>Customer Name</th>
      <th>City</th>
      <th>Region</th>
      <th>Category</th>
      <th>Quantity</th>
      <th>Sales</th>
      <th>Profit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ORD-100000</td>
      <td>2024-08-10</td>
      <td>CUST-2824</td>
      <td>Jeffrey Jacobs</td>
      <td>Velezburgh</td>
      <td>East</td>
      <td>Furniture</td>
      <td>7</td>
      <td>238.76</td>
      <td>34.49</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ORD-100001</td>
      <td>2024-02-28</td>
      <td>CUST-1409</td>
      <td>Jacob Walker</td>
      <td>South Alyssamouth</td>
      <td>West</td>
      <td>Office Supplies</td>
      <td>4</td>
      <td>675.17</td>
      <td>159.79</td>
    </tr>
    <tr>
      <th>2</th>
      <td>ORD-100002</td>
      <td>2024-08-26</td>
      <td>CUST-5506</td>
      <td>Jennifer Baker</td>
      <td>East Anthonyburgh</td>
      <td>North</td>
      <td>Office Supplies</td>
      <td>8</td>
      <td>29.51</td>
      <td>3.49</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ORD-100003</td>
      <td>2023-11-26</td>
      <td>CUST-5012</td>
      <td>Maxwell Reed</td>
      <td>Penamouth</td>
      <td>East</td>
      <td>Technology</td>
      <td>5</td>
      <td>113.07</td>
      <td>20.42</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ORD-100004</td>
      <td>2024-11-04</td>
      <td>CUST-4657</td>
      <td>Nicole Cox</td>
      <td>Masonshire</td>
      <td>South</td>
      <td>Furniture</td>
      <td>7</td>
      <td>801.92</td>
      <td>-96.20</td>
    </tr>
  </tbody>
</table>
</div>




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




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Customer ID</th>
      <th>Profit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>CUST-1006</td>
      <td>-49.51</td>
    </tr>
    <tr>
      <th>1</th>
      <td>CUST-1009</td>
      <td>5.14</td>
    </tr>
    <tr>
      <th>2</th>
      <td>CUST-1018</td>
      <td>82.38</td>
    </tr>
    <tr>
      <th>3</th>
      <td>CUST-1027</td>
      <td>-3.49</td>
    </tr>
    <tr>
      <th>4</th>
      <td>CUST-1035</td>
      <td>-72.42</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>949</th>
      <td>CUST-9947</td>
      <td>23.48</td>
    </tr>
    <tr>
      <th>950</th>
      <td>CUST-9955</td>
      <td>25.25</td>
    </tr>
    <tr>
      <th>951</th>
      <td>CUST-9976</td>
      <td>-32.89</td>
    </tr>
    <tr>
      <th>952</th>
      <td>CUST-9977</td>
      <td>87.23</td>
    </tr>
    <tr>
      <th>953</th>
      <td>CUST-9998</td>
      <td>200.42</td>
    </tr>
  </tbody>
</table>
<p>954 rows × 2 columns</p>
</div>




```python
customer_profit = customer_profit.sort_values(by= "Profit", ascending= False)
customer_profit
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Customer ID</th>
      <th>Profit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>457</th>
      <td>CUST-5333</td>
      <td>344.03</td>
    </tr>
    <tr>
      <th>700</th>
      <td>CUST-7658</td>
      <td>303.31</td>
    </tr>
    <tr>
      <th>733</th>
      <td>CUST-7939</td>
      <td>285.50</td>
    </tr>
    <tr>
      <th>391</th>
      <td>CUST-4824</td>
      <td>273.34</td>
    </tr>
    <tr>
      <th>144</th>
      <td>CUST-2375</td>
      <td>264.70</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>931</th>
      <td>CUST-9824</td>
      <td>-165.87</td>
    </tr>
    <tr>
      <th>650</th>
      <td>CUST-7126</td>
      <td>-166.61</td>
    </tr>
    <tr>
      <th>92</th>
      <td>CUST-1949</td>
      <td>-168.89</td>
    </tr>
    <tr>
      <th>28</th>
      <td>CUST-1319</td>
      <td>-173.27</td>
    </tr>
    <tr>
      <th>882</th>
      <td>CUST-9270</td>
      <td>-177.30</td>
    </tr>
  </tbody>
</table>
<p>954 rows × 2 columns</p>
</div>




```python
top_10_percent_num = int(0.10 * customer_profit.shape[0])
top_10_percent_num
```




    95




```python
top_10_percent_df = customer_profit.head(top_10_percent_num)
top_10_percent_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Customer ID</th>
      <th>Profit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>457</th>
      <td>CUST-5333</td>
      <td>344.03</td>
    </tr>
    <tr>
      <th>700</th>
      <td>CUST-7658</td>
      <td>303.31</td>
    </tr>
    <tr>
      <th>733</th>
      <td>CUST-7939</td>
      <td>285.50</td>
    </tr>
    <tr>
      <th>391</th>
      <td>CUST-4824</td>
      <td>273.34</td>
    </tr>
    <tr>
      <th>144</th>
      <td>CUST-2375</td>
      <td>264.70</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>530</th>
      <td>CUST-5889</td>
      <td>160.23</td>
    </tr>
    <tr>
      <th>37</th>
      <td>CUST-1409</td>
      <td>159.79</td>
    </tr>
    <tr>
      <th>330</th>
      <td>CUST-4249</td>
      <td>157.66</td>
    </tr>
    <tr>
      <th>197</th>
      <td>CUST-2874</td>
      <td>157.24</td>
    </tr>
    <tr>
      <th>231</th>
      <td>CUST-3236</td>
      <td>156.78</td>
    </tr>
  </tbody>
</table>
<p>95 rows × 2 columns</p>
</div>




```python
top_df = df[df["Customer ID"].isin(top_10_percent_df["Customer ID"])]
top_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Order ID</th>
      <th>Order Date</th>
      <th>Customer ID</th>
      <th>Customer Name</th>
      <th>City</th>
      <th>Region</th>
      <th>Category</th>
      <th>Quantity</th>
      <th>Sales</th>
      <th>Profit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>ORD-100001</td>
      <td>2024-02-28</td>
      <td>CUST-1409</td>
      <td>Jacob Walker</td>
      <td>South Alyssamouth</td>
      <td>West</td>
      <td>Office Supplies</td>
      <td>4</td>
      <td>675.17</td>
      <td>159.79</td>
    </tr>
    <tr>
      <th>12</th>
      <td>ORD-100012</td>
      <td>2025-02-20</td>
      <td>CUST-2535</td>
      <td>Jessica Burch</td>
      <td>West Amandashire</td>
      <td>East</td>
      <td>Furniture</td>
      <td>3</td>
      <td>831.92</td>
      <td>166.41</td>
    </tr>
    <tr>
      <th>26</th>
      <td>ORD-100026</td>
      <td>2024-12-15</td>
      <td>CUST-6574</td>
      <td>Sarah Estrada</td>
      <td>New Julieville</td>
      <td>North</td>
      <td>Technology</td>
      <td>4</td>
      <td>819.27</td>
      <td>228.61</td>
    </tr>
    <tr>
      <th>32</th>
      <td>ORD-100032</td>
      <td>2023-12-25</td>
      <td>CUST-2519</td>
      <td>Michael Anderson</td>
      <td>Davidfurt</td>
      <td>East</td>
      <td>Technology</td>
      <td>5</td>
      <td>627.91</td>
      <td>187.22</td>
    </tr>
    <tr>
      <th>37</th>
      <td>ORD-100037</td>
      <td>2024-03-05</td>
      <td>CUST-5333</td>
      <td>Brian Wise</td>
      <td>Port Gary</td>
      <td>South</td>
      <td>Furniture</td>
      <td>9</td>
      <td>948.97</td>
      <td>91.60</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>958</th>
      <td>ORD-100958</td>
      <td>2024-06-15</td>
      <td>CUST-4804</td>
      <td>Andrew Smith</td>
      <td>New Carloshaven</td>
      <td>East</td>
      <td>Furniture</td>
      <td>4</td>
      <td>928.80</td>
      <td>167.65</td>
    </tr>
    <tr>
      <th>963</th>
      <td>ORD-100963</td>
      <td>2024-02-24</td>
      <td>CUST-1153</td>
      <td>Joshua Decker</td>
      <td>North Meganshire</td>
      <td>South</td>
      <td>Office Supplies</td>
      <td>7</td>
      <td>616.81</td>
      <td>169.91</td>
    </tr>
    <tr>
      <th>973</th>
      <td>ORD-100973</td>
      <td>2024-02-20</td>
      <td>CUST-5097</td>
      <td>Tracy Russell</td>
      <td>Lake Damon</td>
      <td>North</td>
      <td>Furniture</td>
      <td>1</td>
      <td>848.05</td>
      <td>219.75</td>
    </tr>
    <tr>
      <th>980</th>
      <td>ORD-100980</td>
      <td>2023-12-03</td>
      <td>CUST-4886</td>
      <td>Chris Flores</td>
      <td>Lake Denise</td>
      <td>West</td>
      <td>Technology</td>
      <td>6</td>
      <td>778.68</td>
      <td>181.73</td>
    </tr>
    <tr>
      <th>990</th>
      <td>ORD-100990</td>
      <td>2024-04-12</td>
      <td>CUST-9138</td>
      <td>Joan Washington</td>
      <td>Lake Maryfort</td>
      <td>West</td>
      <td>Office Supplies</td>
      <td>1</td>
      <td>819.61</td>
      <td>202.48</td>
    </tr>
  </tbody>
</table>
<p>106 rows × 10 columns</p>
</div>



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




    <function matplotlib.pyplot.show(close=None, block=None)>




    
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




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Order ID</th>
      <th>Order Date</th>
      <th>Customer ID</th>
      <th>Customer Name</th>
      <th>City</th>
      <th>Region</th>
      <th>Category</th>
      <th>Quantity</th>
      <th>Sales</th>
      <th>Profit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ORD-100000</td>
      <td>2024-08-10</td>
      <td>CUST-2824</td>
      <td>Jeffrey Jacobs</td>
      <td>Velezburgh</td>
      <td>East</td>
      <td>Furniture</td>
      <td>7</td>
      <td>238.76</td>
      <td>34.49</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ORD-100001</td>
      <td>2024-02-28</td>
      <td>CUST-1409</td>
      <td>Jacob Walker</td>
      <td>South Alyssamouth</td>
      <td>West</td>
      <td>Office Supplies</td>
      <td>4</td>
      <td>675.17</td>
      <td>159.79</td>
    </tr>
    <tr>
      <th>2</th>
      <td>ORD-100002</td>
      <td>2024-08-26</td>
      <td>CUST-5506</td>
      <td>Jennifer Baker</td>
      <td>East Anthonyburgh</td>
      <td>North</td>
      <td>Office Supplies</td>
      <td>8</td>
      <td>29.51</td>
      <td>3.49</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ORD-100003</td>
      <td>2023-11-26</td>
      <td>CUST-5012</td>
      <td>Maxwell Reed</td>
      <td>Penamouth</td>
      <td>East</td>
      <td>Technology</td>
      <td>5</td>
      <td>113.07</td>
      <td>20.42</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ORD-100004</td>
      <td>2024-11-04</td>
      <td>CUST-4657</td>
      <td>Nicole Cox</td>
      <td>Masonshire</td>
      <td>South</td>
      <td>Furniture</td>
      <td>7</td>
      <td>801.92</td>
      <td>-96.20</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_loss["Order Date"].nunique()
```




    537




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




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Order ID</th>
      <th>Order Date</th>
      <th>Customer ID</th>
      <th>Customer Name</th>
      <th>City</th>
      <th>Region</th>
      <th>Category</th>
      <th>Quantity</th>
      <th>Sales</th>
      <th>Profit</th>
      <th>Year</th>
      <th>Month</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ORD-100000</td>
      <td>2024-08-10</td>
      <td>CUST-2824</td>
      <td>Jeffrey Jacobs</td>
      <td>Velezburgh</td>
      <td>East</td>
      <td>Furniture</td>
      <td>7</td>
      <td>238.76</td>
      <td>34.49</td>
      <td>2024</td>
      <td>8</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ORD-100001</td>
      <td>2024-02-28</td>
      <td>CUST-1409</td>
      <td>Jacob Walker</td>
      <td>South Alyssamouth</td>
      <td>West</td>
      <td>Office Supplies</td>
      <td>4</td>
      <td>675.17</td>
      <td>159.79</td>
      <td>2024</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>ORD-100002</td>
      <td>2024-08-26</td>
      <td>CUST-5506</td>
      <td>Jennifer Baker</td>
      <td>East Anthonyburgh</td>
      <td>North</td>
      <td>Office Supplies</td>
      <td>8</td>
      <td>29.51</td>
      <td>3.49</td>
      <td>2024</td>
      <td>8</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ORD-100004</td>
      <td>2024-11-04</td>
      <td>CUST-4657</td>
      <td>Nicole Cox</td>
      <td>Masonshire</td>
      <td>South</td>
      <td>Furniture</td>
      <td>7</td>
      <td>801.92</td>
      <td>-96.20</td>
      <td>2024</td>
      <td>11</td>
    </tr>
    <tr>
      <th>5</th>
      <td>ORD-100005</td>
      <td>2024-09-25</td>
      <td>CUST-3286</td>
      <td>Aaron Duncan</td>
      <td>West Brookeburgh</td>
      <td>North</td>
      <td>Technology</td>
      <td>3</td>
      <td>186.76</td>
      <td>5.75</td>
      <td>2024</td>
      <td>9</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>993</th>
      <td>ORD-100993</td>
      <td>2024-03-29</td>
      <td>CUST-6772</td>
      <td>Carol Matthews</td>
      <td>Lawsonton</td>
      <td>East</td>
      <td>Furniture</td>
      <td>2</td>
      <td>842.30</td>
      <td>8.19</td>
      <td>2024</td>
      <td>3</td>
    </tr>
    <tr>
      <th>994</th>
      <td>ORD-100994</td>
      <td>2024-10-24</td>
      <td>CUST-4588</td>
      <td>Matthew Chambers</td>
      <td>South Stephenburgh</td>
      <td>North</td>
      <td>Technology</td>
      <td>7</td>
      <td>735.47</td>
      <td>139.87</td>
      <td>2024</td>
      <td>10</td>
    </tr>
    <tr>
      <th>995</th>
      <td>ORD-100995</td>
      <td>2024-04-11</td>
      <td>CUST-4115</td>
      <td>Kristi Larson</td>
      <td>Jenniferburgh</td>
      <td>West</td>
      <td>Technology</td>
      <td>8</td>
      <td>546.81</td>
      <td>20.87</td>
      <td>2024</td>
      <td>4</td>
    </tr>
    <tr>
      <th>996</th>
      <td>ORD-100996</td>
      <td>2024-02-13</td>
      <td>CUST-5106</td>
      <td>William Lopez</td>
      <td>Jensenport</td>
      <td>East</td>
      <td>Furniture</td>
      <td>1</td>
      <td>594.44</td>
      <td>28.99</td>
      <td>2024</td>
      <td>2</td>
    </tr>
    <tr>
      <th>999</th>
      <td>ORD-100999</td>
      <td>2024-10-01</td>
      <td>CUST-1645</td>
      <td>Samantha Combs</td>
      <td>Port Scottstad</td>
      <td>North</td>
      <td>Technology</td>
      <td>3</td>
      <td>569.37</td>
      <td>-60.88</td>
      <td>2024</td>
      <td>10</td>
    </tr>
  </tbody>
</table>
<p>492 rows × 12 columns</p>
</div>



### <b><font color= #FFFF00> Monthly profit:


```python
df_2024_months = df_2024.groupby("Month")["Profit"].sum().reset_index()
df_2024_months
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Month</th>
      <th>Profit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>149.47</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1795.80</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>-252.22</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1330.69</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>107.94</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6</td>
      <td>1213.24</td>
    </tr>
    <tr>
      <th>6</th>
      <td>7</td>
      <td>1052.02</td>
    </tr>
    <tr>
      <th>7</th>
      <td>8</td>
      <td>792.76</td>
    </tr>
    <tr>
      <th>8</th>
      <td>9</td>
      <td>1241.87</td>
    </tr>
    <tr>
      <th>9</th>
      <td>10</td>
      <td>1224.80</td>
    </tr>
    <tr>
      <th>10</th>
      <td>11</td>
      <td>1295.25</td>
    </tr>
    <tr>
      <th>11</th>
      <td>12</td>
      <td>649.50</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_2024_months = df_2024_months.sort_values(by= "Profit", ascending= False)
df_2024_months
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Month</th>
      <th>Profit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1795.80</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1330.69</td>
    </tr>
    <tr>
      <th>10</th>
      <td>11</td>
      <td>1295.25</td>
    </tr>
    <tr>
      <th>8</th>
      <td>9</td>
      <td>1241.87</td>
    </tr>
    <tr>
      <th>9</th>
      <td>10</td>
      <td>1224.80</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6</td>
      <td>1213.24</td>
    </tr>
    <tr>
      <th>6</th>
      <td>7</td>
      <td>1052.02</td>
    </tr>
    <tr>
      <th>7</th>
      <td>8</td>
      <td>792.76</td>
    </tr>
    <tr>
      <th>11</th>
      <td>12</td>
      <td>649.50</td>
    </tr>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>149.47</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>107.94</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>-252.22</td>
    </tr>
  </tbody>
</table>
</div>




```python
sns.barplot(x= df_2024_months["Month"], y= df_2024_months["Profit"])
plt.show()
```


    
![png](output_35_0.png)
    



```python
df_march = df_2024[df_2024["Month"] == 3]
df_march.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Order ID</th>
      <th>Order Date</th>
      <th>Customer ID</th>
      <th>Customer Name</th>
      <th>City</th>
      <th>Region</th>
      <th>Category</th>
      <th>Quantity</th>
      <th>Sales</th>
      <th>Profit</th>
      <th>Year</th>
      <th>Month</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>8</th>
      <td>ORD-100008</td>
      <td>2024-03-27</td>
      <td>CUST-2424</td>
      <td>Suzanne Johnston</td>
      <td>Johnsonstad</td>
      <td>West</td>
      <td>Office Supplies</td>
      <td>5</td>
      <td>108.45</td>
      <td>17.70</td>
      <td>2024</td>
      <td>3</td>
    </tr>
    <tr>
      <th>10</th>
      <td>ORD-100010</td>
      <td>2024-03-11</td>
      <td>CUST-1520</td>
      <td>Collin Anderson</td>
      <td>New Mandyburgh</td>
      <td>South</td>
      <td>Office Supplies</td>
      <td>8</td>
      <td>725.04</td>
      <td>-109.05</td>
      <td>2024</td>
      <td>3</td>
    </tr>
    <tr>
      <th>37</th>
      <td>ORD-100037</td>
      <td>2024-03-05</td>
      <td>CUST-5333</td>
      <td>Brian Wise</td>
      <td>Port Gary</td>
      <td>South</td>
      <td>Furniture</td>
      <td>9</td>
      <td>948.97</td>
      <td>91.60</td>
      <td>2024</td>
      <td>3</td>
    </tr>
    <tr>
      <th>45</th>
      <td>ORD-100045</td>
      <td>2024-03-12</td>
      <td>CUST-6925</td>
      <td>James Hodge</td>
      <td>New Johnmouth</td>
      <td>West</td>
      <td>Office Supplies</td>
      <td>3</td>
      <td>32.24</td>
      <td>-4.95</td>
      <td>2024</td>
      <td>3</td>
    </tr>
    <tr>
      <th>64</th>
      <td>ORD-100064</td>
      <td>2024-03-26</td>
      <td>CUST-3803</td>
      <td>Leah West</td>
      <td>Richardsonfort</td>
      <td>West</td>
      <td>Furniture</td>
      <td>2</td>
      <td>214.36</td>
      <td>41.65</td>
      <td>2024</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>




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




    -252.22000000000003



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




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Order ID</th>
      <th>Order Date</th>
      <th>Customer ID</th>
      <th>Customer Name</th>
      <th>City</th>
      <th>Region</th>
      <th>Category</th>
      <th>Quantity</th>
      <th>Sales</th>
      <th>Profit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ORD-100000</td>
      <td>2024-08-10</td>
      <td>CUST-2824</td>
      <td>Jeffrey Jacobs</td>
      <td>Velezburgh</td>
      <td>East</td>
      <td>Furniture</td>
      <td>7</td>
      <td>238.76</td>
      <td>34.49</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ORD-100001</td>
      <td>2024-02-28</td>
      <td>CUST-1409</td>
      <td>Jacob Walker</td>
      <td>South Alyssamouth</td>
      <td>West</td>
      <td>Office Supplies</td>
      <td>4</td>
      <td>675.17</td>
      <td>159.79</td>
    </tr>
    <tr>
      <th>2</th>
      <td>ORD-100002</td>
      <td>2024-08-26</td>
      <td>CUST-5506</td>
      <td>Jennifer Baker</td>
      <td>East Anthonyburgh</td>
      <td>North</td>
      <td>Office Supplies</td>
      <td>8</td>
      <td>29.51</td>
      <td>3.49</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ORD-100003</td>
      <td>2023-11-26</td>
      <td>CUST-5012</td>
      <td>Maxwell Reed</td>
      <td>Penamouth</td>
      <td>East</td>
      <td>Technology</td>
      <td>5</td>
      <td>113.07</td>
      <td>20.42</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ORD-100004</td>
      <td>2024-11-04</td>
      <td>CUST-4657</td>
      <td>Nicole Cox</td>
      <td>Masonshire</td>
      <td>South</td>
      <td>Furniture</td>
      <td>7</td>
      <td>801.92</td>
      <td>-96.20</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>995</th>
      <td>ORD-100995</td>
      <td>2024-04-11</td>
      <td>CUST-4115</td>
      <td>Kristi Larson</td>
      <td>Jenniferburgh</td>
      <td>West</td>
      <td>Technology</td>
      <td>8</td>
      <td>546.81</td>
      <td>20.87</td>
    </tr>
    <tr>
      <th>996</th>
      <td>ORD-100996</td>
      <td>2024-02-13</td>
      <td>CUST-5106</td>
      <td>William Lopez</td>
      <td>Jensenport</td>
      <td>East</td>
      <td>Furniture</td>
      <td>1</td>
      <td>594.44</td>
      <td>28.99</td>
    </tr>
    <tr>
      <th>997</th>
      <td>ORD-100997</td>
      <td>2025-07-04</td>
      <td>CUST-3240</td>
      <td>Paul Beck</td>
      <td>Samanthaview</td>
      <td>North</td>
      <td>Office Supplies</td>
      <td>9</td>
      <td>513.28</td>
      <td>-50.13</td>
    </tr>
    <tr>
      <th>998</th>
      <td>ORD-100998</td>
      <td>2025-03-15</td>
      <td>CUST-2591</td>
      <td>Tracy Flynn</td>
      <td>North Stephanietown</td>
      <td>South</td>
      <td>Technology</td>
      <td>7</td>
      <td>304.57</td>
      <td>29.11</td>
    </tr>
    <tr>
      <th>999</th>
      <td>ORD-100999</td>
      <td>2024-10-01</td>
      <td>CUST-1645</td>
      <td>Samantha Combs</td>
      <td>Port Scottstad</td>
      <td>North</td>
      <td>Technology</td>
      <td>3</td>
      <td>569.37</td>
      <td>-60.88</td>
    </tr>
  </tbody>
</table>
<p>1000 rows × 10 columns</p>
</div>




```python
df_neg_profit = df_anomaly[df_anomaly["Profit"] < 0]
df_neg_profit
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Order ID</th>
      <th>Order Date</th>
      <th>Customer ID</th>
      <th>Customer Name</th>
      <th>City</th>
      <th>Region</th>
      <th>Category</th>
      <th>Quantity</th>
      <th>Sales</th>
      <th>Profit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4</th>
      <td>ORD-100004</td>
      <td>2024-11-04</td>
      <td>CUST-4657</td>
      <td>Nicole Cox</td>
      <td>Masonshire</td>
      <td>South</td>
      <td>Furniture</td>
      <td>7</td>
      <td>801.92</td>
      <td>-96.20</td>
    </tr>
    <tr>
      <th>6</th>
      <td>ORD-100006</td>
      <td>2024-11-28</td>
      <td>CUST-2679</td>
      <td>Jamie Nguyen</td>
      <td>East Ernest</td>
      <td>East</td>
      <td>Furniture</td>
      <td>7</td>
      <td>656.22</td>
      <td>-128.18</td>
    </tr>
    <tr>
      <th>7</th>
      <td>ORD-100007</td>
      <td>2024-07-03</td>
      <td>CUST-9935</td>
      <td>Jessica Pitts</td>
      <td>West Michelle</td>
      <td>East</td>
      <td>Furniture</td>
      <td>8</td>
      <td>245.80</td>
      <td>-18.84</td>
    </tr>
    <tr>
      <th>10</th>
      <td>ORD-100010</td>
      <td>2024-03-11</td>
      <td>CUST-1520</td>
      <td>Collin Anderson</td>
      <td>New Mandyburgh</td>
      <td>South</td>
      <td>Office Supplies</td>
      <td>8</td>
      <td>725.04</td>
      <td>-109.05</td>
    </tr>
    <tr>
      <th>13</th>
      <td>ORD-100013</td>
      <td>2025-01-08</td>
      <td>CUST-4582</td>
      <td>Taylor Owens</td>
      <td>New Heidi</td>
      <td>East</td>
      <td>Furniture</td>
      <td>6</td>
      <td>403.21</td>
      <td>-39.51</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>981</th>
      <td>ORD-100981</td>
      <td>2024-04-17</td>
      <td>CUST-7248</td>
      <td>Nicholas Cruz</td>
      <td>Fisherborough</td>
      <td>North</td>
      <td>Technology</td>
      <td>3</td>
      <td>659.60</td>
      <td>-38.76</td>
    </tr>
    <tr>
      <th>988</th>
      <td>ORD-100988</td>
      <td>2024-12-31</td>
      <td>CUST-1132</td>
      <td>Eileen Vasquez</td>
      <td>Kimberlyville</td>
      <td>South</td>
      <td>Technology</td>
      <td>3</td>
      <td>140.41</td>
      <td>-8.25</td>
    </tr>
    <tr>
      <th>989</th>
      <td>ORD-100989</td>
      <td>2024-03-18</td>
      <td>CUST-1803</td>
      <td>Nathan Evans</td>
      <td>West Daniel</td>
      <td>West</td>
      <td>Furniture</td>
      <td>2</td>
      <td>728.72</td>
      <td>-17.11</td>
    </tr>
    <tr>
      <th>997</th>
      <td>ORD-100997</td>
      <td>2025-07-04</td>
      <td>CUST-3240</td>
      <td>Paul Beck</td>
      <td>Samanthaview</td>
      <td>North</td>
      <td>Office Supplies</td>
      <td>9</td>
      <td>513.28</td>
      <td>-50.13</td>
    </tr>
    <tr>
      <th>999</th>
      <td>ORD-100999</td>
      <td>2024-10-01</td>
      <td>CUST-1645</td>
      <td>Samantha Combs</td>
      <td>Port Scottstad</td>
      <td>North</td>
      <td>Technology</td>
      <td>3</td>
      <td>569.37</td>
      <td>-60.88</td>
    </tr>
  </tbody>
</table>
<p>398 rows × 10 columns</p>
</div>




```python
df_neg_profit.groupby("Category").agg({"Category" : "count"})
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Category</th>
    </tr>
    <tr>
      <th>Category</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Furniture</th>
      <td>115</td>
    </tr>
    <tr>
      <th>Office Supplies</th>
      <td>142</td>
    </tr>
    <tr>
      <th>Technology</th>
      <td>141</td>
    </tr>
  </tbody>
</table>
</div>




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




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Order ID</th>
      <th>Order Date</th>
      <th>Customer ID</th>
      <th>Customer Name</th>
      <th>City</th>
      <th>Region</th>
      <th>Category</th>
      <th>Quantity</th>
      <th>Sales</th>
      <th>Profit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ORD-100000</td>
      <td>2024-08-10</td>
      <td>CUST-2824</td>
      <td>Jeffrey Jacobs</td>
      <td>Velezburgh</td>
      <td>East</td>
      <td>Furniture</td>
      <td>7</td>
      <td>238.76</td>
      <td>34.49</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ORD-100001</td>
      <td>2024-02-28</td>
      <td>CUST-1409</td>
      <td>Jacob Walker</td>
      <td>South Alyssamouth</td>
      <td>West</td>
      <td>Office Supplies</td>
      <td>4</td>
      <td>675.17</td>
      <td>159.79</td>
    </tr>
    <tr>
      <th>2</th>
      <td>ORD-100002</td>
      <td>2024-08-26</td>
      <td>CUST-5506</td>
      <td>Jennifer Baker</td>
      <td>East Anthonyburgh</td>
      <td>North</td>
      <td>Office Supplies</td>
      <td>8</td>
      <td>29.51</td>
      <td>3.49</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ORD-100003</td>
      <td>2023-11-26</td>
      <td>CUST-5012</td>
      <td>Maxwell Reed</td>
      <td>Penamouth</td>
      <td>East</td>
      <td>Technology</td>
      <td>5</td>
      <td>113.07</td>
      <td>20.42</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ORD-100004</td>
      <td>2024-11-04</td>
      <td>CUST-4657</td>
      <td>Nicole Cox</td>
      <td>Masonshire</td>
      <td>South</td>
      <td>Furniture</td>
      <td>7</td>
      <td>801.92</td>
      <td>-96.20</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_product_mix.groupby(["Region","Category"])["Sales"].sum()
```




    Region  Category       
    East    Furniture          44515.40
            Office Supplies    43166.97
            Technology         51097.55
    North   Furniture          39738.54
            Office Supplies    40463.24
            Technology         39787.34
    South   Furniture          40073.94
            Office Supplies    46950.11
            Technology         37103.00
    West    Furniture          44470.98
            Office Supplies    39181.65
            Technology         41876.95
    Name: Sales, dtype: float64




```python
df_sales = df_product_mix.pivot_table(index= "Region", columns= "Category", values= "Sales", aggfunc= "sum")
df_sales
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Category</th>
      <th>Furniture</th>
      <th>Office Supplies</th>
      <th>Technology</th>
    </tr>
    <tr>
      <th>Region</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>East</th>
      <td>44515.40</td>
      <td>43166.97</td>
      <td>51097.55</td>
    </tr>
    <tr>
      <th>North</th>
      <td>39738.54</td>
      <td>40463.24</td>
      <td>39787.34</td>
    </tr>
    <tr>
      <th>South</th>
      <td>40073.94</td>
      <td>46950.11</td>
      <td>37103.00</td>
    </tr>
    <tr>
      <th>West</th>
      <td>44470.98</td>
      <td>39181.65</td>
      <td>41876.95</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_sales.plot(kind= "bar", figsize= (10,7))
plt.title("Region wise Category Sales")
plt.ylabel("Sales")
plt.tight_layout()
plt.show()
```


    
![png](output_101_0.png)
    



```python
df_profit = df_product_mix.pivot_table(index= "Region", columns= "Category", values= "Profit", aggfunc= "sum")
df_profit
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Category</th>
      <th>Furniture</th>
      <th>Office Supplies</th>
      <th>Technology</th>
    </tr>
    <tr>
      <th>Region</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>East</th>
      <td>2618.19</td>
      <td>538.38</td>
      <td>3137.44</td>
    </tr>
    <tr>
      <th>North</th>
      <td>3541.17</td>
      <td>2836.63</td>
      <td>1111.70</td>
    </tr>
    <tr>
      <th>South</th>
      <td>1089.64</td>
      <td>3156.00</td>
      <td>1052.92</td>
    </tr>
    <tr>
      <th>West</th>
      <td>3133.24</td>
      <td>1929.09</td>
      <td>1718.79</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_profit.plot(kind= "bar", figsize= (10,7))
plt.title("Region wise Category Profit")
plt.ylabel("Profit")
plt.tight_layout()
plt.show()
```


    
![png](output_103_0.png)
    



```python
df_north_office = df_product_mix[(df_product_mix["Region"] == "North") & (df_product_mix["Category"] == "Office Supplies")]
df_north_office
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Order ID</th>
      <th>Order Date</th>
      <th>Customer ID</th>
      <th>Customer Name</th>
      <th>City</th>
      <th>Region</th>
      <th>Category</th>
      <th>Quantity</th>
      <th>Sales</th>
      <th>Profit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>ORD-100002</td>
      <td>2024-08-26</td>
      <td>CUST-5506</td>
      <td>Jennifer Baker</td>
      <td>East Anthonyburgh</td>
      <td>North</td>
      <td>Office Supplies</td>
      <td>8</td>
      <td>29.51</td>
      <td>3.49</td>
    </tr>
    <tr>
      <th>19</th>
      <td>ORD-100019</td>
      <td>2025-04-19</td>
      <td>CUST-7873</td>
      <td>Amy Hernandez</td>
      <td>Smithland</td>
      <td>North</td>
      <td>Office Supplies</td>
      <td>5</td>
      <td>94.65</td>
      <td>18.16</td>
    </tr>
    <tr>
      <th>21</th>
      <td>ORD-100021</td>
      <td>2024-08-28</td>
      <td>CUST-8359</td>
      <td>Carol Doyle</td>
      <td>Christopherhaven</td>
      <td>North</td>
      <td>Office Supplies</td>
      <td>6</td>
      <td>36.27</td>
      <td>-7.16</td>
    </tr>
    <tr>
      <th>53</th>
      <td>ORD-100053</td>
      <td>2023-08-22</td>
      <td>CUST-2654</td>
      <td>Katherine Davidson</td>
      <td>Adammouth</td>
      <td>North</td>
      <td>Office Supplies</td>
      <td>6</td>
      <td>530.43</td>
      <td>53.97</td>
    </tr>
    <tr>
      <th>76</th>
      <td>ORD-100076</td>
      <td>2023-07-25</td>
      <td>CUST-6168</td>
      <td>James Davidson</td>
      <td>Port Kathyberg</td>
      <td>North</td>
      <td>Office Supplies</td>
      <td>8</td>
      <td>523.73</td>
      <td>63.52</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>950</th>
      <td>ORD-100950</td>
      <td>2024-11-04</td>
      <td>CUST-3950</td>
      <td>Michelle Carr</td>
      <td>East David</td>
      <td>North</td>
      <td>Office Supplies</td>
      <td>2</td>
      <td>980.17</td>
      <td>39.64</td>
    </tr>
    <tr>
      <th>974</th>
      <td>ORD-100974</td>
      <td>2025-03-28</td>
      <td>CUST-8484</td>
      <td>Anna Newman</td>
      <td>New Derekville</td>
      <td>North</td>
      <td>Office Supplies</td>
      <td>3</td>
      <td>919.74</td>
      <td>-137.13</td>
    </tr>
    <tr>
      <th>975</th>
      <td>ORD-100975</td>
      <td>2025-01-10</td>
      <td>CUST-5949</td>
      <td>Ricky Navarro</td>
      <td>East Heather</td>
      <td>North</td>
      <td>Office Supplies</td>
      <td>2</td>
      <td>259.72</td>
      <td>13.76</td>
    </tr>
    <tr>
      <th>991</th>
      <td>ORD-100991</td>
      <td>2023-08-31</td>
      <td>CUST-5689</td>
      <td>Michael Dawson</td>
      <td>Robertton</td>
      <td>North</td>
      <td>Office Supplies</td>
      <td>7</td>
      <td>221.38</td>
      <td>60.49</td>
    </tr>
    <tr>
      <th>997</th>
      <td>ORD-100997</td>
      <td>2025-07-04</td>
      <td>CUST-3240</td>
      <td>Paul Beck</td>
      <td>Samanthaview</td>
      <td>North</td>
      <td>Office Supplies</td>
      <td>9</td>
      <td>513.28</td>
      <td>-50.13</td>
    </tr>
  </tbody>
</table>
<p>82 rows × 10 columns</p>
</div>




```python
df_north_neg = df_north_office[df_north_office["Profit"] < 0]
df_north_neg
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Order ID</th>
      <th>Order Date</th>
      <th>Customer ID</th>
      <th>Customer Name</th>
      <th>City</th>
      <th>Region</th>
      <th>Category</th>
      <th>Quantity</th>
      <th>Sales</th>
      <th>Profit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>21</th>
      <td>ORD-100021</td>
      <td>2024-08-28</td>
      <td>CUST-8359</td>
      <td>Carol Doyle</td>
      <td>Christopherhaven</td>
      <td>North</td>
      <td>Office Supplies</td>
      <td>6</td>
      <td>36.27</td>
      <td>-7.16</td>
    </tr>
    <tr>
      <th>120</th>
      <td>ORD-100120</td>
      <td>2025-05-27</td>
      <td>CUST-8433</td>
      <td>Brad Adams</td>
      <td>Vargaschester</td>
      <td>North</td>
      <td>Office Supplies</td>
      <td>2</td>
      <td>193.28</td>
      <td>-13.36</td>
    </tr>
    <tr>
      <th>123</th>
      <td>ORD-100123</td>
      <td>2025-04-11</td>
      <td>CUST-9201</td>
      <td>Tasha Andrews</td>
      <td>Littlechester</td>
      <td>North</td>
      <td>Office Supplies</td>
      <td>6</td>
      <td>734.90</td>
      <td>-55.28</td>
    </tr>
    <tr>
      <th>188</th>
      <td>ORD-100188</td>
      <td>2024-03-15</td>
      <td>CUST-3167</td>
      <td>Jake Dean</td>
      <td>Laurieshire</td>
      <td>North</td>
      <td>Office Supplies</td>
      <td>3</td>
      <td>361.08</td>
      <td>-42.48</td>
    </tr>
    <tr>
      <th>225</th>
      <td>ORD-100225</td>
      <td>2024-01-10</td>
      <td>CUST-4872</td>
      <td>Jennifer Henson</td>
      <td>Martinville</td>
      <td>North</td>
      <td>Office Supplies</td>
      <td>8</td>
      <td>182.16</td>
      <td>-7.58</td>
    </tr>
    <tr>
      <th>287</th>
      <td>ORD-100287</td>
      <td>2024-06-03</td>
      <td>CUST-4492</td>
      <td>Cynthia Kline</td>
      <td>South Wayne</td>
      <td>North</td>
      <td>Office Supplies</td>
      <td>1</td>
      <td>512.39</td>
      <td>-70.13</td>
    </tr>
    <tr>
      <th>294</th>
      <td>ORD-100294</td>
      <td>2024-04-30</td>
      <td>CUST-7054</td>
      <td>John Johnson</td>
      <td>Leblancstad</td>
      <td>North</td>
      <td>Office Supplies</td>
      <td>8</td>
      <td>637.67</td>
      <td>-55.10</td>
    </tr>
    <tr>
      <th>323</th>
      <td>ORD-100323</td>
      <td>2024-08-31</td>
      <td>CUST-1058</td>
      <td>Heather Gomez</td>
      <td>Lake Laura</td>
      <td>North</td>
      <td>Office Supplies</td>
      <td>3</td>
      <td>330.37</td>
      <td>-59.13</td>
    </tr>
    <tr>
      <th>335</th>
      <td>ORD-100335</td>
      <td>2024-01-30</td>
      <td>CUST-1590</td>
      <td>Richard Wright</td>
      <td>Gilbertmouth</td>
      <td>North</td>
      <td>Office Supplies</td>
      <td>5</td>
      <td>228.56</td>
      <td>-10.89</td>
    </tr>
    <tr>
      <th>455</th>
      <td>ORD-100455</td>
      <td>2024-10-23</td>
      <td>CUST-2193</td>
      <td>Christine Henderson</td>
      <td>Rachelfurt</td>
      <td>North</td>
      <td>Office Supplies</td>
      <td>9</td>
      <td>653.82</td>
      <td>-5.28</td>
    </tr>
    <tr>
      <th>477</th>
      <td>ORD-100477</td>
      <td>2024-01-06</td>
      <td>CUST-9702</td>
      <td>Leah Ho</td>
      <td>South Kimberlyfurt</td>
      <td>North</td>
      <td>Office Supplies</td>
      <td>2</td>
      <td>427.50</td>
      <td>-71.30</td>
    </tr>
    <tr>
      <th>482</th>
      <td>ORD-100482</td>
      <td>2025-02-28</td>
      <td>CUST-9976</td>
      <td>Richard Porter</td>
      <td>Juantown</td>
      <td>North</td>
      <td>Office Supplies</td>
      <td>8</td>
      <td>626.74</td>
      <td>-32.89</td>
    </tr>
    <tr>
      <th>496</th>
      <td>ORD-100496</td>
      <td>2025-01-15</td>
      <td>CUST-5681</td>
      <td>Steve Reynolds</td>
      <td>New Christopher</td>
      <td>North</td>
      <td>Office Supplies</td>
      <td>9</td>
      <td>450.85</td>
      <td>-21.12</td>
    </tr>
    <tr>
      <th>506</th>
      <td>ORD-100506</td>
      <td>2024-08-11</td>
      <td>CUST-7275</td>
      <td>Todd May</td>
      <td>Calderonland</td>
      <td>North</td>
      <td>Office Supplies</td>
      <td>2</td>
      <td>775.74</td>
      <td>-11.18</td>
    </tr>
    <tr>
      <th>520</th>
      <td>ORD-100520</td>
      <td>2023-12-03</td>
      <td>CUST-1320</td>
      <td>Melissa Davis</td>
      <td>Jamesfurt</td>
      <td>North</td>
      <td>Office Supplies</td>
      <td>6</td>
      <td>175.37</td>
      <td>-5.21</td>
    </tr>
    <tr>
      <th>527</th>
      <td>ORD-100527</td>
      <td>2023-12-18</td>
      <td>CUST-7865</td>
      <td>Joseph Martin</td>
      <td>Port Laura</td>
      <td>North</td>
      <td>Office Supplies</td>
      <td>3</td>
      <td>807.61</td>
      <td>-73.54</td>
    </tr>
    <tr>
      <th>559</th>
      <td>ORD-100559</td>
      <td>2024-11-16</td>
      <td>CUST-6529</td>
      <td>James Rogers</td>
      <td>Riveratown</td>
      <td>North</td>
      <td>Office Supplies</td>
      <td>2</td>
      <td>676.70</td>
      <td>-83.19</td>
    </tr>
    <tr>
      <th>570</th>
      <td>ORD-100570</td>
      <td>2024-03-12</td>
      <td>CUST-4673</td>
      <td>Gina Walker</td>
      <td>Roberttown</td>
      <td>North</td>
      <td>Office Supplies</td>
      <td>6</td>
      <td>502.20</td>
      <td>-32.58</td>
    </tr>
    <tr>
      <th>624</th>
      <td>ORD-100624</td>
      <td>2024-03-20</td>
      <td>CUST-8136</td>
      <td>Rebecca Gomez</td>
      <td>Lake Josephton</td>
      <td>North</td>
      <td>Office Supplies</td>
      <td>7</td>
      <td>386.39</td>
      <td>-38.55</td>
    </tr>
    <tr>
      <th>721</th>
      <td>ORD-100721</td>
      <td>2024-05-19</td>
      <td>CUST-3485</td>
      <td>Amanda Evans</td>
      <td>Pachecofort</td>
      <td>North</td>
      <td>Office Supplies</td>
      <td>2</td>
      <td>330.14</td>
      <td>-57.06</td>
    </tr>
    <tr>
      <th>744</th>
      <td>ORD-100744</td>
      <td>2024-06-30</td>
      <td>CUST-3847</td>
      <td>Darryl Morales</td>
      <td>Gregoryside</td>
      <td>North</td>
      <td>Office Supplies</td>
      <td>7</td>
      <td>183.55</td>
      <td>-2.08</td>
    </tr>
    <tr>
      <th>764</th>
      <td>ORD-100764</td>
      <td>2023-08-19</td>
      <td>CUST-4683</td>
      <td>Christine Richards</td>
      <td>Timborough</td>
      <td>North</td>
      <td>Office Supplies</td>
      <td>6</td>
      <td>98.23</td>
      <td>-17.28</td>
    </tr>
    <tr>
      <th>796</th>
      <td>ORD-100796</td>
      <td>2023-08-23</td>
      <td>CUST-9502</td>
      <td>Joshua Ruiz DVM</td>
      <td>East Holly</td>
      <td>North</td>
      <td>Office Supplies</td>
      <td>9</td>
      <td>90.11</td>
      <td>-1.45</td>
    </tr>
    <tr>
      <th>806</th>
      <td>ORD-100806</td>
      <td>2024-06-07</td>
      <td>CUST-6460</td>
      <td>William Watson</td>
      <td>Sydneybury</td>
      <td>North</td>
      <td>Office Supplies</td>
      <td>5</td>
      <td>923.53</td>
      <td>-127.28</td>
    </tr>
    <tr>
      <th>859</th>
      <td>ORD-100859</td>
      <td>2024-06-29</td>
      <td>CUST-7807</td>
      <td>Kathleen Ibarra</td>
      <td>Derrickhaven</td>
      <td>North</td>
      <td>Office Supplies</td>
      <td>9</td>
      <td>419.99</td>
      <td>-68.12</td>
    </tr>
    <tr>
      <th>937</th>
      <td>ORD-100937</td>
      <td>2023-08-15</td>
      <td>CUST-5906</td>
      <td>Nancy Ruiz</td>
      <td>Lopezfort</td>
      <td>North</td>
      <td>Office Supplies</td>
      <td>1</td>
      <td>771.53</td>
      <td>-149.64</td>
    </tr>
    <tr>
      <th>974</th>
      <td>ORD-100974</td>
      <td>2025-03-28</td>
      <td>CUST-8484</td>
      <td>Anna Newman</td>
      <td>New Derekville</td>
      <td>North</td>
      <td>Office Supplies</td>
      <td>3</td>
      <td>919.74</td>
      <td>-137.13</td>
    </tr>
    <tr>
      <th>997</th>
      <td>ORD-100997</td>
      <td>2025-07-04</td>
      <td>CUST-3240</td>
      <td>Paul Beck</td>
      <td>Samanthaview</td>
      <td>North</td>
      <td>Office Supplies</td>
      <td>9</td>
      <td>513.28</td>
      <td>-50.13</td>
    </tr>
  </tbody>
</table>
</div>




```python
office_positive_profit = df_north_office.groupby("Quantity")["Profit"].sum()
sum(office_positive_profit.values)
```




    2836.63




```python
office_negative_profit = df_north_neg.groupby("Quantity")["Profit"].sum()
sum(office_negative_profit.values)
```




    -1306.1199999999997




```python
df_north_furniture = df_product_mix[(df_product_mix["Region"] == "North") & (df_product_mix["Category"] == "Furniture")]
df_north_furniture
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Order ID</th>
      <th>Order Date</th>
      <th>Customer ID</th>
      <th>Customer Name</th>
      <th>City</th>
      <th>Region</th>
      <th>Category</th>
      <th>Quantity</th>
      <th>Sales</th>
      <th>Profit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>30</th>
      <td>ORD-100030</td>
      <td>2024-07-19</td>
      <td>CUST-6514</td>
      <td>Leah Bell</td>
      <td>Wheelerport</td>
      <td>North</td>
      <td>Furniture</td>
      <td>3</td>
      <td>594.39</td>
      <td>-28.05</td>
    </tr>
    <tr>
      <th>39</th>
      <td>ORD-100039</td>
      <td>2023-11-25</td>
      <td>CUST-8527</td>
      <td>Nicole Ball</td>
      <td>New Steven</td>
      <td>North</td>
      <td>Furniture</td>
      <td>9</td>
      <td>148.71</td>
      <td>0.94</td>
    </tr>
    <tr>
      <th>48</th>
      <td>ORD-100048</td>
      <td>2024-10-28</td>
      <td>CUST-1750</td>
      <td>Evan Cochran</td>
      <td>East Williamshire</td>
      <td>North</td>
      <td>Furniture</td>
      <td>2</td>
      <td>637.14</td>
      <td>77.20</td>
    </tr>
    <tr>
      <th>49</th>
      <td>ORD-100049</td>
      <td>2025-03-04</td>
      <td>CUST-4733</td>
      <td>Tabitha Cole</td>
      <td>Stevenmouth</td>
      <td>North</td>
      <td>Furniture</td>
      <td>8</td>
      <td>265.31</td>
      <td>-44.46</td>
    </tr>
    <tr>
      <th>52</th>
      <td>ORD-100052</td>
      <td>2024-02-23</td>
      <td>CUST-4814</td>
      <td>James Hogan</td>
      <td>West Matthewfurt</td>
      <td>North</td>
      <td>Furniture</td>
      <td>6</td>
      <td>984.56</td>
      <td>79.09</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>945</th>
      <td>ORD-100945</td>
      <td>2024-03-31</td>
      <td>CUST-8056</td>
      <td>Robert Austin</td>
      <td>West Katherine</td>
      <td>North</td>
      <td>Furniture</td>
      <td>5</td>
      <td>535.00</td>
      <td>119.85</td>
    </tr>
    <tr>
      <th>949</th>
      <td>ORD-100949</td>
      <td>2025-04-07</td>
      <td>CUST-7751</td>
      <td>Rebecca Smith</td>
      <td>New Kyleview</td>
      <td>North</td>
      <td>Furniture</td>
      <td>6</td>
      <td>299.89</td>
      <td>-55.60</td>
    </tr>
    <tr>
      <th>970</th>
      <td>ORD-100970</td>
      <td>2025-07-04</td>
      <td>CUST-6661</td>
      <td>Laura Browning</td>
      <td>New Kelsey</td>
      <td>North</td>
      <td>Furniture</td>
      <td>8</td>
      <td>293.05</td>
      <td>37.60</td>
    </tr>
    <tr>
      <th>973</th>
      <td>ORD-100973</td>
      <td>2024-02-20</td>
      <td>CUST-5097</td>
      <td>Tracy Russell</td>
      <td>Lake Damon</td>
      <td>North</td>
      <td>Furniture</td>
      <td>1</td>
      <td>848.05</td>
      <td>219.75</td>
    </tr>
    <tr>
      <th>986</th>
      <td>ORD-100986</td>
      <td>2023-09-04</td>
      <td>CUST-7484</td>
      <td>Kathleen Webb</td>
      <td>Garrettside</td>
      <td>North</td>
      <td>Furniture</td>
      <td>7</td>
      <td>52.74</td>
      <td>3.08</td>
    </tr>
  </tbody>
</table>
<p>80 rows × 10 columns</p>
</div>




```python
df_north_fur_neg = df_north_office[df_north_office["Profit"] < 0]
df_north_fur_neg
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Order ID</th>
      <th>Order Date</th>
      <th>Customer ID</th>
      <th>Customer Name</th>
      <th>City</th>
      <th>Region</th>
      <th>Category</th>
      <th>Quantity</th>
      <th>Sales</th>
      <th>Profit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>21</th>
      <td>ORD-100021</td>
      <td>2024-08-28</td>
      <td>CUST-8359</td>
      <td>Carol Doyle</td>
      <td>Christopherhaven</td>
      <td>North</td>
      <td>Office Supplies</td>
      <td>6</td>
      <td>36.27</td>
      <td>-7.16</td>
    </tr>
    <tr>
      <th>120</th>
      <td>ORD-100120</td>
      <td>2025-05-27</td>
      <td>CUST-8433</td>
      <td>Brad Adams</td>
      <td>Vargaschester</td>
      <td>North</td>
      <td>Office Supplies</td>
      <td>2</td>
      <td>193.28</td>
      <td>-13.36</td>
    </tr>
    <tr>
      <th>123</th>
      <td>ORD-100123</td>
      <td>2025-04-11</td>
      <td>CUST-9201</td>
      <td>Tasha Andrews</td>
      <td>Littlechester</td>
      <td>North</td>
      <td>Office Supplies</td>
      <td>6</td>
      <td>734.90</td>
      <td>-55.28</td>
    </tr>
    <tr>
      <th>188</th>
      <td>ORD-100188</td>
      <td>2024-03-15</td>
      <td>CUST-3167</td>
      <td>Jake Dean</td>
      <td>Laurieshire</td>
      <td>North</td>
      <td>Office Supplies</td>
      <td>3</td>
      <td>361.08</td>
      <td>-42.48</td>
    </tr>
    <tr>
      <th>225</th>
      <td>ORD-100225</td>
      <td>2024-01-10</td>
      <td>CUST-4872</td>
      <td>Jennifer Henson</td>
      <td>Martinville</td>
      <td>North</td>
      <td>Office Supplies</td>
      <td>8</td>
      <td>182.16</td>
      <td>-7.58</td>
    </tr>
    <tr>
      <th>287</th>
      <td>ORD-100287</td>
      <td>2024-06-03</td>
      <td>CUST-4492</td>
      <td>Cynthia Kline</td>
      <td>South Wayne</td>
      <td>North</td>
      <td>Office Supplies</td>
      <td>1</td>
      <td>512.39</td>
      <td>-70.13</td>
    </tr>
    <tr>
      <th>294</th>
      <td>ORD-100294</td>
      <td>2024-04-30</td>
      <td>CUST-7054</td>
      <td>John Johnson</td>
      <td>Leblancstad</td>
      <td>North</td>
      <td>Office Supplies</td>
      <td>8</td>
      <td>637.67</td>
      <td>-55.10</td>
    </tr>
    <tr>
      <th>323</th>
      <td>ORD-100323</td>
      <td>2024-08-31</td>
      <td>CUST-1058</td>
      <td>Heather Gomez</td>
      <td>Lake Laura</td>
      <td>North</td>
      <td>Office Supplies</td>
      <td>3</td>
      <td>330.37</td>
      <td>-59.13</td>
    </tr>
    <tr>
      <th>335</th>
      <td>ORD-100335</td>
      <td>2024-01-30</td>
      <td>CUST-1590</td>
      <td>Richard Wright</td>
      <td>Gilbertmouth</td>
      <td>North</td>
      <td>Office Supplies</td>
      <td>5</td>
      <td>228.56</td>
      <td>-10.89</td>
    </tr>
    <tr>
      <th>455</th>
      <td>ORD-100455</td>
      <td>2024-10-23</td>
      <td>CUST-2193</td>
      <td>Christine Henderson</td>
      <td>Rachelfurt</td>
      <td>North</td>
      <td>Office Supplies</td>
      <td>9</td>
      <td>653.82</td>
      <td>-5.28</td>
    </tr>
    <tr>
      <th>477</th>
      <td>ORD-100477</td>
      <td>2024-01-06</td>
      <td>CUST-9702</td>
      <td>Leah Ho</td>
      <td>South Kimberlyfurt</td>
      <td>North</td>
      <td>Office Supplies</td>
      <td>2</td>
      <td>427.50</td>
      <td>-71.30</td>
    </tr>
    <tr>
      <th>482</th>
      <td>ORD-100482</td>
      <td>2025-02-28</td>
      <td>CUST-9976</td>
      <td>Richard Porter</td>
      <td>Juantown</td>
      <td>North</td>
      <td>Office Supplies</td>
      <td>8</td>
      <td>626.74</td>
      <td>-32.89</td>
    </tr>
    <tr>
      <th>496</th>
      <td>ORD-100496</td>
      <td>2025-01-15</td>
      <td>CUST-5681</td>
      <td>Steve Reynolds</td>
      <td>New Christopher</td>
      <td>North</td>
      <td>Office Supplies</td>
      <td>9</td>
      <td>450.85</td>
      <td>-21.12</td>
    </tr>
    <tr>
      <th>506</th>
      <td>ORD-100506</td>
      <td>2024-08-11</td>
      <td>CUST-7275</td>
      <td>Todd May</td>
      <td>Calderonland</td>
      <td>North</td>
      <td>Office Supplies</td>
      <td>2</td>
      <td>775.74</td>
      <td>-11.18</td>
    </tr>
    <tr>
      <th>520</th>
      <td>ORD-100520</td>
      <td>2023-12-03</td>
      <td>CUST-1320</td>
      <td>Melissa Davis</td>
      <td>Jamesfurt</td>
      <td>North</td>
      <td>Office Supplies</td>
      <td>6</td>
      <td>175.37</td>
      <td>-5.21</td>
    </tr>
    <tr>
      <th>527</th>
      <td>ORD-100527</td>
      <td>2023-12-18</td>
      <td>CUST-7865</td>
      <td>Joseph Martin</td>
      <td>Port Laura</td>
      <td>North</td>
      <td>Office Supplies</td>
      <td>3</td>
      <td>807.61</td>
      <td>-73.54</td>
    </tr>
    <tr>
      <th>559</th>
      <td>ORD-100559</td>
      <td>2024-11-16</td>
      <td>CUST-6529</td>
      <td>James Rogers</td>
      <td>Riveratown</td>
      <td>North</td>
      <td>Office Supplies</td>
      <td>2</td>
      <td>676.70</td>
      <td>-83.19</td>
    </tr>
    <tr>
      <th>570</th>
      <td>ORD-100570</td>
      <td>2024-03-12</td>
      <td>CUST-4673</td>
      <td>Gina Walker</td>
      <td>Roberttown</td>
      <td>North</td>
      <td>Office Supplies</td>
      <td>6</td>
      <td>502.20</td>
      <td>-32.58</td>
    </tr>
    <tr>
      <th>624</th>
      <td>ORD-100624</td>
      <td>2024-03-20</td>
      <td>CUST-8136</td>
      <td>Rebecca Gomez</td>
      <td>Lake Josephton</td>
      <td>North</td>
      <td>Office Supplies</td>
      <td>7</td>
      <td>386.39</td>
      <td>-38.55</td>
    </tr>
    <tr>
      <th>721</th>
      <td>ORD-100721</td>
      <td>2024-05-19</td>
      <td>CUST-3485</td>
      <td>Amanda Evans</td>
      <td>Pachecofort</td>
      <td>North</td>
      <td>Office Supplies</td>
      <td>2</td>
      <td>330.14</td>
      <td>-57.06</td>
    </tr>
    <tr>
      <th>744</th>
      <td>ORD-100744</td>
      <td>2024-06-30</td>
      <td>CUST-3847</td>
      <td>Darryl Morales</td>
      <td>Gregoryside</td>
      <td>North</td>
      <td>Office Supplies</td>
      <td>7</td>
      <td>183.55</td>
      <td>-2.08</td>
    </tr>
    <tr>
      <th>764</th>
      <td>ORD-100764</td>
      <td>2023-08-19</td>
      <td>CUST-4683</td>
      <td>Christine Richards</td>
      <td>Timborough</td>
      <td>North</td>
      <td>Office Supplies</td>
      <td>6</td>
      <td>98.23</td>
      <td>-17.28</td>
    </tr>
    <tr>
      <th>796</th>
      <td>ORD-100796</td>
      <td>2023-08-23</td>
      <td>CUST-9502</td>
      <td>Joshua Ruiz DVM</td>
      <td>East Holly</td>
      <td>North</td>
      <td>Office Supplies</td>
      <td>9</td>
      <td>90.11</td>
      <td>-1.45</td>
    </tr>
    <tr>
      <th>806</th>
      <td>ORD-100806</td>
      <td>2024-06-07</td>
      <td>CUST-6460</td>
      <td>William Watson</td>
      <td>Sydneybury</td>
      <td>North</td>
      <td>Office Supplies</td>
      <td>5</td>
      <td>923.53</td>
      <td>-127.28</td>
    </tr>
    <tr>
      <th>859</th>
      <td>ORD-100859</td>
      <td>2024-06-29</td>
      <td>CUST-7807</td>
      <td>Kathleen Ibarra</td>
      <td>Derrickhaven</td>
      <td>North</td>
      <td>Office Supplies</td>
      <td>9</td>
      <td>419.99</td>
      <td>-68.12</td>
    </tr>
    <tr>
      <th>937</th>
      <td>ORD-100937</td>
      <td>2023-08-15</td>
      <td>CUST-5906</td>
      <td>Nancy Ruiz</td>
      <td>Lopezfort</td>
      <td>North</td>
      <td>Office Supplies</td>
      <td>1</td>
      <td>771.53</td>
      <td>-149.64</td>
    </tr>
    <tr>
      <th>974</th>
      <td>ORD-100974</td>
      <td>2025-03-28</td>
      <td>CUST-8484</td>
      <td>Anna Newman</td>
      <td>New Derekville</td>
      <td>North</td>
      <td>Office Supplies</td>
      <td>3</td>
      <td>919.74</td>
      <td>-137.13</td>
    </tr>
    <tr>
      <th>997</th>
      <td>ORD-100997</td>
      <td>2025-07-04</td>
      <td>CUST-3240</td>
      <td>Paul Beck</td>
      <td>Samanthaview</td>
      <td>North</td>
      <td>Office Supplies</td>
      <td>9</td>
      <td>513.28</td>
      <td>-50.13</td>
    </tr>
  </tbody>
</table>
</div>




```python
Furniture_positive_profit = df_north_furniture.groupby("Quantity")["Profit"].sum()
sum(Furniture_positive_profit.values)
```




    3541.17




```python
Furniture_negative_profit = df_north_fur_neg.groupby("Quantity")["Profit"].sum()
sum(Furniture_negative_profit.values)
```




    -1306.1199999999997



### <b> <font color= #ABFF00> Conclusion:
- #### In the East, West, and South regions, the category with the highest sales also gave the highest profit. This shows that the current product mix in these regions is working well.
- #### In the North region, Office Supplies had the highest sales, but Furniture made more profit.
- #### Even though the sales difference between Office Supplies and Furniture was small (₹724), the profit difference was meaningful (₹705).
- #### On deeper analysis, Office Supplies in North had more negative profit orders, while Furniture mostly made positive profits.

### <b><font color= #FFFF00> Q5. Demand Prediction Case:
#### *Using historical data, identify if there is a trend or seasonal pattern in quantity sold for each product category over time.*


```python
df_trend_season = pd.DataFrame(df)
df_trend_season.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Order ID</th>
      <th>Order Date</th>
      <th>Customer ID</th>
      <th>Customer Name</th>
      <th>City</th>
      <th>Region</th>
      <th>Category</th>
      <th>Quantity</th>
      <th>Sales</th>
      <th>Profit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ORD-100000</td>
      <td>2024-08-10</td>
      <td>CUST-2824</td>
      <td>Jeffrey Jacobs</td>
      <td>Velezburgh</td>
      <td>East</td>
      <td>Furniture</td>
      <td>7</td>
      <td>238.76</td>
      <td>34.49</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ORD-100001</td>
      <td>2024-02-28</td>
      <td>CUST-1409</td>
      <td>Jacob Walker</td>
      <td>South Alyssamouth</td>
      <td>West</td>
      <td>Office Supplies</td>
      <td>4</td>
      <td>675.17</td>
      <td>159.79</td>
    </tr>
    <tr>
      <th>2</th>
      <td>ORD-100002</td>
      <td>2024-08-26</td>
      <td>CUST-5506</td>
      <td>Jennifer Baker</td>
      <td>East Anthonyburgh</td>
      <td>North</td>
      <td>Office Supplies</td>
      <td>8</td>
      <td>29.51</td>
      <td>3.49</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ORD-100003</td>
      <td>2023-11-26</td>
      <td>CUST-5012</td>
      <td>Maxwell Reed</td>
      <td>Penamouth</td>
      <td>East</td>
      <td>Technology</td>
      <td>5</td>
      <td>113.07</td>
      <td>20.42</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ORD-100004</td>
      <td>2024-11-04</td>
      <td>CUST-4657</td>
      <td>Nicole Cox</td>
      <td>Masonshire</td>
      <td>South</td>
      <td>Furniture</td>
      <td>7</td>
      <td>801.92</td>
      <td>-96.20</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_trend_season["Order Date"] = pd.to_datetime(df_trend_season["Order Date"])
```


```python
df_trend_season["Year"] = df_trend_season["Order Date"].dt.year
```


```python
df_trend_season["Month"] = df_trend_season["Order Date"].dt.month
```


```python
df_trend_season["Year-Month"] = df_trend_season["Order Date"].dt.to_period("M")
```


```python
monthly_trend = df_trend_season.groupby(['Year-Month', 'Category'])['Quantity'].sum().reset_index()
monthly_trend
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Year-Month</th>
      <th>Category</th>
      <th>Quantity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2023-07</td>
      <td>Furniture</td>
      <td>4</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2023-07</td>
      <td>Office Supplies</td>
      <td>61</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2023-07</td>
      <td>Technology</td>
      <td>78</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2023-08</td>
      <td>Furniture</td>
      <td>47</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2023-08</td>
      <td>Office Supplies</td>
      <td>147</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>70</th>
      <td>2025-06</td>
      <td>Office Supplies</td>
      <td>56</td>
    </tr>
    <tr>
      <th>71</th>
      <td>2025-06</td>
      <td>Technology</td>
      <td>100</td>
    </tr>
    <tr>
      <th>72</th>
      <td>2025-07</td>
      <td>Furniture</td>
      <td>10</td>
    </tr>
    <tr>
      <th>73</th>
      <td>2025-07</td>
      <td>Office Supplies</td>
      <td>39</td>
    </tr>
    <tr>
      <th>74</th>
      <td>2025-07</td>
      <td>Technology</td>
      <td>16</td>
    </tr>
  </tbody>
</table>
<p>75 rows × 3 columns</p>
</div>




```python
pivot_df = monthly_trend.pivot(index='Year-Month', columns='Category', values='Quantity')
pivot_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Category</th>
      <th>Furniture</th>
      <th>Office Supplies</th>
      <th>Technology</th>
    </tr>
    <tr>
      <th>Year-Month</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2023-07</th>
      <td>4</td>
      <td>61</td>
      <td>78</td>
    </tr>
    <tr>
      <th>2023-08</th>
      <td>47</td>
      <td>147</td>
      <td>81</td>
    </tr>
    <tr>
      <th>2023-09</th>
      <td>67</td>
      <td>87</td>
      <td>46</td>
    </tr>
    <tr>
      <th>2023-10</th>
      <td>65</td>
      <td>33</td>
      <td>86</td>
    </tr>
    <tr>
      <th>2023-11</th>
      <td>54</td>
      <td>44</td>
      <td>50</td>
    </tr>
    <tr>
      <th>2023-12</th>
      <td>100</td>
      <td>61</td>
      <td>67</td>
    </tr>
    <tr>
      <th>2024-01</th>
      <td>73</td>
      <td>53</td>
      <td>76</td>
    </tr>
    <tr>
      <th>2024-02</th>
      <td>71</td>
      <td>80</td>
      <td>83</td>
    </tr>
    <tr>
      <th>2024-03</th>
      <td>49</td>
      <td>86</td>
      <td>50</td>
    </tr>
    <tr>
      <th>2024-04</th>
      <td>71</td>
      <td>82</td>
      <td>58</td>
    </tr>
    <tr>
      <th>2024-05</th>
      <td>26</td>
      <td>48</td>
      <td>86</td>
    </tr>
    <tr>
      <th>2024-06</th>
      <td>72</td>
      <td>85</td>
      <td>53</td>
    </tr>
    <tr>
      <th>2024-07</th>
      <td>72</td>
      <td>71</td>
      <td>64</td>
    </tr>
    <tr>
      <th>2024-08</th>
      <td>78</td>
      <td>100</td>
      <td>79</td>
    </tr>
    <tr>
      <th>2024-09</th>
      <td>49</td>
      <td>72</td>
      <td>62</td>
    </tr>
    <tr>
      <th>2024-10</th>
      <td>53</td>
      <td>76</td>
      <td>95</td>
    </tr>
    <tr>
      <th>2024-11</th>
      <td>59</td>
      <td>52</td>
      <td>65</td>
    </tr>
    <tr>
      <th>2024-12</th>
      <td>49</td>
      <td>35</td>
      <td>48</td>
    </tr>
    <tr>
      <th>2025-01</th>
      <td>105</td>
      <td>84</td>
      <td>44</td>
    </tr>
    <tr>
      <th>2025-02</th>
      <td>59</td>
      <td>67</td>
      <td>60</td>
    </tr>
    <tr>
      <th>2025-03</th>
      <td>88</td>
      <td>41</td>
      <td>80</td>
    </tr>
    <tr>
      <th>2025-04</th>
      <td>71</td>
      <td>67</td>
      <td>94</td>
    </tr>
    <tr>
      <th>2025-05</th>
      <td>62</td>
      <td>72</td>
      <td>56</td>
    </tr>
    <tr>
      <th>2025-06</th>
      <td>68</td>
      <td>56</td>
      <td>100</td>
    </tr>
    <tr>
      <th>2025-07</th>
      <td>10</td>
      <td>39</td>
      <td>16</td>
    </tr>
  </tbody>
</table>
</div>




```python
pivot_df.plot(figsize=(12,6), marker='o')
plt.title("Monthly Quantity Sold by Category")
plt.ylabel("Total Quantity Sold")
plt.xlabel("Month")
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()
```


    
![png](output_121_0.png)
    


### <b> <font color= #ABFF00> Conclusion:
#### There is both a trend and seasonality present in the quantity sold over time:
- #### Trends: Increasing demand (especially for Office Supplies).
- #### Seasonality: Regular peaks at specific months across years.

### <b><font color= #FFFF00> Q6. Loss-Leading Product Investigation:
#### *Find products or categories that have repeatedly shown negative profit despite high sales. Should they be discontinued or repriced?*


```python
df_high_sales = pd.DataFrame(df)
df_high_sales.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Order ID</th>
      <th>Order Date</th>
      <th>Customer ID</th>
      <th>Customer Name</th>
      <th>City</th>
      <th>Region</th>
      <th>Category</th>
      <th>Quantity</th>
      <th>Sales</th>
      <th>Profit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ORD-100000</td>
      <td>2024-08-10</td>
      <td>CUST-2824</td>
      <td>Jeffrey Jacobs</td>
      <td>Velezburgh</td>
      <td>East</td>
      <td>Furniture</td>
      <td>7</td>
      <td>238.76</td>
      <td>34.49</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ORD-100001</td>
      <td>2024-02-28</td>
      <td>CUST-1409</td>
      <td>Jacob Walker</td>
      <td>South Alyssamouth</td>
      <td>West</td>
      <td>Office Supplies</td>
      <td>4</td>
      <td>675.17</td>
      <td>159.79</td>
    </tr>
    <tr>
      <th>2</th>
      <td>ORD-100002</td>
      <td>2024-08-26</td>
      <td>CUST-5506</td>
      <td>Jennifer Baker</td>
      <td>East Anthonyburgh</td>
      <td>North</td>
      <td>Office Supplies</td>
      <td>8</td>
      <td>29.51</td>
      <td>3.49</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ORD-100003</td>
      <td>2023-11-26</td>
      <td>CUST-5012</td>
      <td>Maxwell Reed</td>
      <td>Penamouth</td>
      <td>East</td>
      <td>Technology</td>
      <td>5</td>
      <td>113.07</td>
      <td>20.42</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ORD-100004</td>
      <td>2024-11-04</td>
      <td>CUST-4657</td>
      <td>Nicole Cox</td>
      <td>Masonshire</td>
      <td>South</td>
      <td>Furniture</td>
      <td>7</td>
      <td>801.92</td>
      <td>-96.20</td>
    </tr>
  </tbody>
</table>
</div>




```python
neg_profit = df_high_sales[df_high_sales["Profit"] < 0]
neg_profit
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Order ID</th>
      <th>Order Date</th>
      <th>Customer ID</th>
      <th>Customer Name</th>
      <th>City</th>
      <th>Region</th>
      <th>Category</th>
      <th>Quantity</th>
      <th>Sales</th>
      <th>Profit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4</th>
      <td>ORD-100004</td>
      <td>2024-11-04</td>
      <td>CUST-4657</td>
      <td>Nicole Cox</td>
      <td>Masonshire</td>
      <td>South</td>
      <td>Furniture</td>
      <td>7</td>
      <td>801.92</td>
      <td>-96.20</td>
    </tr>
    <tr>
      <th>6</th>
      <td>ORD-100006</td>
      <td>2024-11-28</td>
      <td>CUST-2679</td>
      <td>Jamie Nguyen</td>
      <td>East Ernest</td>
      <td>East</td>
      <td>Furniture</td>
      <td>7</td>
      <td>656.22</td>
      <td>-128.18</td>
    </tr>
    <tr>
      <th>7</th>
      <td>ORD-100007</td>
      <td>2024-07-03</td>
      <td>CUST-9935</td>
      <td>Jessica Pitts</td>
      <td>West Michelle</td>
      <td>East</td>
      <td>Furniture</td>
      <td>8</td>
      <td>245.80</td>
      <td>-18.84</td>
    </tr>
    <tr>
      <th>10</th>
      <td>ORD-100010</td>
      <td>2024-03-11</td>
      <td>CUST-1520</td>
      <td>Collin Anderson</td>
      <td>New Mandyburgh</td>
      <td>South</td>
      <td>Office Supplies</td>
      <td>8</td>
      <td>725.04</td>
      <td>-109.05</td>
    </tr>
    <tr>
      <th>13</th>
      <td>ORD-100013</td>
      <td>2025-01-08</td>
      <td>CUST-4582</td>
      <td>Taylor Owens</td>
      <td>New Heidi</td>
      <td>East</td>
      <td>Furniture</td>
      <td>6</td>
      <td>403.21</td>
      <td>-39.51</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>981</th>
      <td>ORD-100981</td>
      <td>2024-04-17</td>
      <td>CUST-7248</td>
      <td>Nicholas Cruz</td>
      <td>Fisherborough</td>
      <td>North</td>
      <td>Technology</td>
      <td>3</td>
      <td>659.60</td>
      <td>-38.76</td>
    </tr>
    <tr>
      <th>988</th>
      <td>ORD-100988</td>
      <td>2024-12-31</td>
      <td>CUST-1132</td>
      <td>Eileen Vasquez</td>
      <td>Kimberlyville</td>
      <td>South</td>
      <td>Technology</td>
      <td>3</td>
      <td>140.41</td>
      <td>-8.25</td>
    </tr>
    <tr>
      <th>989</th>
      <td>ORD-100989</td>
      <td>2024-03-18</td>
      <td>CUST-1803</td>
      <td>Nathan Evans</td>
      <td>West Daniel</td>
      <td>West</td>
      <td>Furniture</td>
      <td>2</td>
      <td>728.72</td>
      <td>-17.11</td>
    </tr>
    <tr>
      <th>997</th>
      <td>ORD-100997</td>
      <td>2025-07-04</td>
      <td>CUST-3240</td>
      <td>Paul Beck</td>
      <td>Samanthaview</td>
      <td>North</td>
      <td>Office Supplies</td>
      <td>9</td>
      <td>513.28</td>
      <td>-50.13</td>
    </tr>
    <tr>
      <th>999</th>
      <td>ORD-100999</td>
      <td>2024-10-01</td>
      <td>CUST-1645</td>
      <td>Samantha Combs</td>
      <td>Port Scottstad</td>
      <td>North</td>
      <td>Technology</td>
      <td>3</td>
      <td>569.37</td>
      <td>-60.88</td>
    </tr>
  </tbody>
</table>
<p>398 rows × 10 columns</p>
</div>




```python
neg_profit.groupby("Category")["Sales"].sum()
```




    Category
    Furniture          58127.21
    Office Supplies    71508.92
    Technology         73748.27
    Name: Sales, dtype: float64




```python
neg_profit.groupby("Category")["Profit"].sum()
```




    Category
    Furniture         -6214.81
    Office Supplies   -6843.83
    Technology        -7364.55
    Name: Profit, dtype: float64




```python
tech_sort = neg_profit[neg_profit["Category"] == "Technology"]
tech_sort
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Order ID</th>
      <th>Order Date</th>
      <th>Customer ID</th>
      <th>Customer Name</th>
      <th>City</th>
      <th>Region</th>
      <th>Category</th>
      <th>Quantity</th>
      <th>Sales</th>
      <th>Profit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>25</th>
      <td>ORD-100025</td>
      <td>2024-02-11</td>
      <td>CUST-7924</td>
      <td>Virginia Hansen</td>
      <td>Bryanville</td>
      <td>East</td>
      <td>Technology</td>
      <td>7</td>
      <td>893.74</td>
      <td>-88.27</td>
    </tr>
    <tr>
      <th>28</th>
      <td>ORD-100028</td>
      <td>2025-01-26</td>
      <td>CUST-3547</td>
      <td>Barry Alvarez</td>
      <td>West Michaelland</td>
      <td>North</td>
      <td>Technology</td>
      <td>3</td>
      <td>266.83</td>
      <td>-9.75</td>
    </tr>
    <tr>
      <th>43</th>
      <td>ORD-100043</td>
      <td>2024-05-13</td>
      <td>CUST-2291</td>
      <td>Laura Thomas</td>
      <td>Susanfurt</td>
      <td>West</td>
      <td>Technology</td>
      <td>7</td>
      <td>801.86</td>
      <td>-17.84</td>
    </tr>
    <tr>
      <th>60</th>
      <td>ORD-100060</td>
      <td>2024-10-11</td>
      <td>CUST-6820</td>
      <td>Trevor Gutierrez</td>
      <td>East Emilyborough</td>
      <td>South</td>
      <td>Technology</td>
      <td>7</td>
      <td>281.31</td>
      <td>-48.03</td>
    </tr>
    <tr>
      <th>70</th>
      <td>ORD-100070</td>
      <td>2024-01-07</td>
      <td>CUST-5422</td>
      <td>Brandon Reed</td>
      <td>Port Nicole</td>
      <td>West</td>
      <td>Technology</td>
      <td>9</td>
      <td>604.51</td>
      <td>-94.12</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>960</th>
      <td>ORD-100960</td>
      <td>2025-04-18</td>
      <td>CUST-5161</td>
      <td>Meghan Blair</td>
      <td>Westside</td>
      <td>South</td>
      <td>Technology</td>
      <td>6</td>
      <td>378.45</td>
      <td>-1.43</td>
    </tr>
    <tr>
      <th>971</th>
      <td>ORD-100971</td>
      <td>2024-07-18</td>
      <td>CUST-5901</td>
      <td>Leslie Brown</td>
      <td>South Derekside</td>
      <td>West</td>
      <td>Technology</td>
      <td>3</td>
      <td>817.79</td>
      <td>-65.26</td>
    </tr>
    <tr>
      <th>981</th>
      <td>ORD-100981</td>
      <td>2024-04-17</td>
      <td>CUST-7248</td>
      <td>Nicholas Cruz</td>
      <td>Fisherborough</td>
      <td>North</td>
      <td>Technology</td>
      <td>3</td>
      <td>659.60</td>
      <td>-38.76</td>
    </tr>
    <tr>
      <th>988</th>
      <td>ORD-100988</td>
      <td>2024-12-31</td>
      <td>CUST-1132</td>
      <td>Eileen Vasquez</td>
      <td>Kimberlyville</td>
      <td>South</td>
      <td>Technology</td>
      <td>3</td>
      <td>140.41</td>
      <td>-8.25</td>
    </tr>
    <tr>
      <th>999</th>
      <td>ORD-100999</td>
      <td>2024-10-01</td>
      <td>CUST-1645</td>
      <td>Samantha Combs</td>
      <td>Port Scottstad</td>
      <td>North</td>
      <td>Technology</td>
      <td>3</td>
      <td>569.37</td>
      <td>-60.88</td>
    </tr>
  </tbody>
</table>
<p>141 rows × 10 columns</p>
</div>




```python
tech_sort.groupby("Quantity").agg({
    "Quantity" : "count",
    "Sales" : "sum",
    "Profit" : "sum"
}).rename(columns= {"Quantity" : "Order_count"}).reset_index()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Quantity</th>
      <th>Order_count</th>
      <th>Sales</th>
      <th>Profit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>19</td>
      <td>10400.94</td>
      <td>-965.17</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>14</td>
      <td>6155.91</td>
      <td>-710.43</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>14</td>
      <td>6783.78</td>
      <td>-812.00</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>16</td>
      <td>7235.92</td>
      <td>-604.46</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>19</td>
      <td>11130.52</td>
      <td>-1283.44</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6</td>
      <td>20</td>
      <td>11033.73</td>
      <td>-960.39</td>
    </tr>
    <tr>
      <th>6</th>
      <td>7</td>
      <td>14</td>
      <td>6800.03</td>
      <td>-625.54</td>
    </tr>
    <tr>
      <th>7</th>
      <td>8</td>
      <td>11</td>
      <td>5772.87</td>
      <td>-518.92</td>
    </tr>
    <tr>
      <th>8</th>
      <td>9</td>
      <td>14</td>
      <td>8434.57</td>
      <td>-884.20</td>
    </tr>
  </tbody>
</table>
</div>




```python
tot = tech_sort.groupby(["Quantity"])["Sales"].sum()
tot_sum = sum(tot)
per = (tot.values/tot_sum)*100
```

### <b> <font color= #ABFF00> Conclusion:
- #### The `Technology` category accounts for the highest volume of negative-profit transactions, especially in quantity groups of `1`,`5`, and `6` units.
- #### These three quantity buckets alone contribute `45%` of loss transactions, indicating that these sales are frequent and significant.
- #### Since these products are selling well (high sales count), it’s more financially sound to reprice them (increase unit price, reduce discount) rather than discontinue them.

### <b><font color= #FFFF00> Q7. Regional Sales Consistency:
#### *Which region shows the most stable monthly sales performance over time? Use standard deviation or coefficient of variation to support your analysis.*


```python
df_std_cv = pd.DataFrame(df)
df_std_cv.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Order ID</th>
      <th>Order Date</th>
      <th>Customer ID</th>
      <th>Customer Name</th>
      <th>City</th>
      <th>Region</th>
      <th>Category</th>
      <th>Quantity</th>
      <th>Sales</th>
      <th>Profit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ORD-100000</td>
      <td>2024-08-10</td>
      <td>CUST-2824</td>
      <td>Jeffrey Jacobs</td>
      <td>Velezburgh</td>
      <td>East</td>
      <td>Furniture</td>
      <td>7</td>
      <td>238.76</td>
      <td>34.49</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ORD-100001</td>
      <td>2024-02-28</td>
      <td>CUST-1409</td>
      <td>Jacob Walker</td>
      <td>South Alyssamouth</td>
      <td>West</td>
      <td>Office Supplies</td>
      <td>4</td>
      <td>675.17</td>
      <td>159.79</td>
    </tr>
    <tr>
      <th>2</th>
      <td>ORD-100002</td>
      <td>2024-08-26</td>
      <td>CUST-5506</td>
      <td>Jennifer Baker</td>
      <td>East Anthonyburgh</td>
      <td>North</td>
      <td>Office Supplies</td>
      <td>8</td>
      <td>29.51</td>
      <td>3.49</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ORD-100003</td>
      <td>2023-11-26</td>
      <td>CUST-5012</td>
      <td>Maxwell Reed</td>
      <td>Penamouth</td>
      <td>East</td>
      <td>Technology</td>
      <td>5</td>
      <td>113.07</td>
      <td>20.42</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ORD-100004</td>
      <td>2024-11-04</td>
      <td>CUST-4657</td>
      <td>Nicole Cox</td>
      <td>Masonshire</td>
      <td>South</td>
      <td>Furniture</td>
      <td>7</td>
      <td>801.92</td>
      <td>-96.20</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_std_cv["Order Date"] = pd.to_datetime(df_std_cv["Order Date"])
```


```python
df_std_cv["month_year"] = df_std_cv["Order Date"].dt.to_period("M")
```


```python
monthly_sales = df_std_cv.groupby(["Region", "month_year"])["Sales"].sum().reset_index()
monthly_sales
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Region</th>
      <th>month_year</th>
      <th>Sales</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>East</td>
      <td>2023-07</td>
      <td>3158.88</td>
    </tr>
    <tr>
      <th>1</th>
      <td>East</td>
      <td>2023-08</td>
      <td>4921.11</td>
    </tr>
    <tr>
      <th>2</th>
      <td>East</td>
      <td>2023-09</td>
      <td>4621.30</td>
    </tr>
    <tr>
      <th>3</th>
      <td>East</td>
      <td>2023-10</td>
      <td>3672.32</td>
    </tr>
    <tr>
      <th>4</th>
      <td>East</td>
      <td>2023-11</td>
      <td>4340.82</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>95</th>
      <td>West</td>
      <td>2025-03</td>
      <td>5937.82</td>
    </tr>
    <tr>
      <th>96</th>
      <td>West</td>
      <td>2025-04</td>
      <td>9464.17</td>
    </tr>
    <tr>
      <th>97</th>
      <td>West</td>
      <td>2025-05</td>
      <td>6917.35</td>
    </tr>
    <tr>
      <th>98</th>
      <td>West</td>
      <td>2025-06</td>
      <td>6442.24</td>
    </tr>
    <tr>
      <th>99</th>
      <td>West</td>
      <td>2025-07</td>
      <td>96.99</td>
    </tr>
  </tbody>
</table>
<p>100 rows × 3 columns</p>
</div>




```python
region_sales = monthly_sales.groupby("Region")["Sales"].agg(["mean","std"]).reset_index()
region_sales
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Region</th>
      <th>mean</th>
      <th>std</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>East</td>
      <td>5551.1968</td>
      <td>2032.315803</td>
    </tr>
    <tr>
      <th>1</th>
      <td>North</td>
      <td>4799.5648</td>
      <td>1752.124855</td>
    </tr>
    <tr>
      <th>2</th>
      <td>South</td>
      <td>4965.0820</td>
      <td>1859.621593</td>
    </tr>
    <tr>
      <th>3</th>
      <td>West</td>
      <td>5021.1832</td>
      <td>2306.793029</td>
    </tr>
  </tbody>
</table>
</div>




```python
region_sales["cv"] = region_sales["std"] / region_sales["mean"]
```


```python
region_sales.sort_values(by= "cv")
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Region</th>
      <th>mean</th>
      <th>std</th>
      <th>cv</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>North</td>
      <td>4799.5648</td>
      <td>1752.124855</td>
      <td>0.365059</td>
    </tr>
    <tr>
      <th>0</th>
      <td>East</td>
      <td>5551.1968</td>
      <td>2032.315803</td>
      <td>0.366104</td>
    </tr>
    <tr>
      <th>2</th>
      <td>South</td>
      <td>4965.0820</td>
      <td>1859.621593</td>
      <td>0.374540</td>
    </tr>
    <tr>
      <th>3</th>
      <td>West</td>
      <td>5021.1832</td>
      <td>2306.793029</td>
      <td>0.459412</td>
    </tr>
  </tbody>
</table>
</div>



### <b> <font color= #ABFF00> Conclusion:
- #### Based on the coefficient of variation (CV) for monthly sales across regions, the North region has the most stable sales performance over time (CV = 0.36). 
- #### This indicates less fluctuation in monthly sales, making it the most consistent region in terms of sales.

### <b><font color= #FFFF00> Q8. Customer Retention Analysis:
#### *Based on Customer ID, find the number of repeats vs. one-time customers. How does their average profit and sales differ?*


```python
df_customer = pd.DataFrame(df)
```


```python
df_customer["Order Date"] = pd.to_datetime(df_customer["Order Date"])
```


```python
df_customer.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Order ID</th>
      <th>Order Date</th>
      <th>Customer ID</th>
      <th>Customer Name</th>
      <th>City</th>
      <th>Region</th>
      <th>Category</th>
      <th>Quantity</th>
      <th>Sales</th>
      <th>Profit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ORD-100000</td>
      <td>2024-08-10</td>
      <td>CUST-2824</td>
      <td>Jeffrey Jacobs</td>
      <td>Velezburgh</td>
      <td>East</td>
      <td>Furniture</td>
      <td>7</td>
      <td>238.76</td>
      <td>34.49</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ORD-100001</td>
      <td>2024-02-28</td>
      <td>CUST-1409</td>
      <td>Jacob Walker</td>
      <td>South Alyssamouth</td>
      <td>West</td>
      <td>Office Supplies</td>
      <td>4</td>
      <td>675.17</td>
      <td>159.79</td>
    </tr>
    <tr>
      <th>2</th>
      <td>ORD-100002</td>
      <td>2024-08-26</td>
      <td>CUST-5506</td>
      <td>Jennifer Baker</td>
      <td>East Anthonyburgh</td>
      <td>North</td>
      <td>Office Supplies</td>
      <td>8</td>
      <td>29.51</td>
      <td>3.49</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ORD-100003</td>
      <td>2023-11-26</td>
      <td>CUST-5012</td>
      <td>Maxwell Reed</td>
      <td>Penamouth</td>
      <td>East</td>
      <td>Technology</td>
      <td>5</td>
      <td>113.07</td>
      <td>20.42</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ORD-100004</td>
      <td>2024-11-04</td>
      <td>CUST-4657</td>
      <td>Nicole Cox</td>
      <td>Masonshire</td>
      <td>South</td>
      <td>Furniture</td>
      <td>7</td>
      <td>801.92</td>
      <td>-96.20</td>
    </tr>
  </tbody>
</table>
</div>




```python
customer_count = df_customer["Customer ID"].value_counts()
```


```python
df_customer["customer_type"] = df_customer["Customer ID"].apply(lambda x : "repeat" if customer_count[x] > 1 else "one_time")
```


```python
df_customer["customer_type"].value_counts()
```




    customer_type
    one_time    912
    repeat       88
    Name: count, dtype: int64




```python
customer_summary = df_customer.groupby("customer_type")[["Sales","Profit"]].sum()
customer_summary
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Sales</th>
      <th>Profit</th>
    </tr>
    <tr>
      <th>customer_type</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>one_time</th>
      <td>462370.09</td>
      <td>23145.79</td>
    </tr>
    <tr>
      <th>repeat</th>
      <td>46055.58</td>
      <td>2717.40</td>
    </tr>
  </tbody>
</table>
</div>




```python
customer_summary["per_sales"] = round((customer_summary["Sales"]/ customer_summary["Sales"].sum())*100, 2)
```


```python
customer_summary["per_profit"] = round((customer_summary["Profit"]/ customer_summary["Profit"].sum())*100, 2)
```


```python
customer_summary
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Sales</th>
      <th>Profit</th>
      <th>per_sales</th>
      <th>per_profit</th>
    </tr>
    <tr>
      <th>customer_type</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>one_time</th>
      <td>462370.09</td>
      <td>23145.79</td>
      <td>90.94</td>
      <td>89.49</td>
    </tr>
    <tr>
      <th>repeat</th>
      <td>46055.58</td>
      <td>2717.40</td>
      <td>9.06</td>
      <td>10.51</td>
    </tr>
  </tbody>
</table>
</div>



### <b> <font color= #ABFF00> Conclusion:
- #### Out of all customers, only a small portion are repeat customers, and they contribute ~9% of sales and ~10.5% of profit.
- #### The majority of revenue is currently driven by one-time customers, showing a potential gap in customer retention.
- #### There is no significant profitability difference between repeat and one-time buyers, indicating a possible opportunity to re-engage one-time buyers into becoming repeat customers.

### <b><font color= #FFFF00> Q9. Bulk Buying Patterns:
#### *Are their specific cities or regions where customers consistently buy in higher quantities than average? What product categories are driving this?*


```python
df_cit_reg = pd.DataFrame(df)
```


```python
region_agg = df_cit_reg.groupby("Region")[["Quantity"]].agg(["count","std","mean"]).reset_index()
```


```python
region_agg["total_avg"] = df_cit_reg["Quantity"].mean()
```


```python
region_agg
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th>Region</th>
      <th colspan="3" halign="left">Quantity</th>
      <th>total_avg</th>
    </tr>
    <tr>
      <th></th>
      <th></th>
      <th>count</th>
      <th>std</th>
      <th>mean</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>East</td>
      <td>274</td>
      <td>2.633198</td>
      <td>4.981752</td>
      <td>4.898</td>
    </tr>
    <tr>
      <th>1</th>
      <td>North</td>
      <td>241</td>
      <td>2.708019</td>
      <td>4.842324</td>
      <td>4.898</td>
    </tr>
    <tr>
      <th>2</th>
      <td>South</td>
      <td>234</td>
      <td>2.501608</td>
      <td>4.747863</td>
      <td>4.898</td>
    </tr>
    <tr>
      <th>3</th>
      <td>West</td>
      <td>251</td>
      <td>2.570603</td>
      <td>5.000000</td>
      <td>4.898</td>
    </tr>
  </tbody>
</table>
</div>




```python
a = df_cit_reg["City"].value_counts().reset_index()
a.columns = ["City", "count"]
b = a[a["count"] > 1]
```


```python
city_filter = df_cit_reg[df_cit_reg["City"].isin(b["City"])]
```


```python
city_agg = city_filter.groupby("City")[["Quantity"]].agg(["count","mean","std"]).reset_index()
```


```python
city_agg["total_avg"] = df_cit_reg["Quantity"].mean()
```

### <b> <font color= #ABFF00> Conclusion:
- #### The East and West regions show slightly above-average bulk buying behavior, with mean quantities above the overall average `(4.898)`.
- #### Standard deviation across regions is relatively consistent, indicating stable purchasing patterns.
- #### City-level analysis was inconclusive, as most cities appeared only once in the dataset and do not provide enough volume to draw meaningful conclusions.

### <b><font color= #FFFF00> Q10. Sales Efficiency Score:
#### *Create a new metric: Profit per Unit Sold. Rank cities based on this efficiency. What actionable insights can Walmart take?*


```python
df_pro_per_unit = pd.DataFrame(df)
df_pro_per_unit.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Order ID</th>
      <th>Order Date</th>
      <th>Customer ID</th>
      <th>Customer Name</th>
      <th>City</th>
      <th>Region</th>
      <th>Category</th>
      <th>Quantity</th>
      <th>Sales</th>
      <th>Profit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ORD-100000</td>
      <td>2024-08-10</td>
      <td>CUST-2824</td>
      <td>Jeffrey Jacobs</td>
      <td>Velezburgh</td>
      <td>East</td>
      <td>Furniture</td>
      <td>7</td>
      <td>238.76</td>
      <td>34.49</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ORD-100001</td>
      <td>2024-02-28</td>
      <td>CUST-1409</td>
      <td>Jacob Walker</td>
      <td>South Alyssamouth</td>
      <td>West</td>
      <td>Office Supplies</td>
      <td>4</td>
      <td>675.17</td>
      <td>159.79</td>
    </tr>
    <tr>
      <th>2</th>
      <td>ORD-100002</td>
      <td>2024-08-26</td>
      <td>CUST-5506</td>
      <td>Jennifer Baker</td>
      <td>East Anthonyburgh</td>
      <td>North</td>
      <td>Office Supplies</td>
      <td>8</td>
      <td>29.51</td>
      <td>3.49</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ORD-100003</td>
      <td>2023-11-26</td>
      <td>CUST-5012</td>
      <td>Maxwell Reed</td>
      <td>Penamouth</td>
      <td>East</td>
      <td>Technology</td>
      <td>5</td>
      <td>113.07</td>
      <td>20.42</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ORD-100004</td>
      <td>2024-11-04</td>
      <td>CUST-4657</td>
      <td>Nicole Cox</td>
      <td>Masonshire</td>
      <td>South</td>
      <td>Furniture</td>
      <td>7</td>
      <td>801.92</td>
      <td>-96.20</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_city_cal = df_pro_per_unit.groupby("City")[["Quantity", "Profit"]].sum().reset_index()
df_city_cal.columns = ["City", "Quantity", "Profit"]
df_city_cal
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>City</th>
      <th>Quantity</th>
      <th>Profit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Aaronfort</td>
      <td>8</td>
      <td>34.54</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Adammouth</td>
      <td>6</td>
      <td>53.97</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Adamshaven</td>
      <td>5</td>
      <td>84.07</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Alexanderfort</td>
      <td>3</td>
      <td>69.73</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Alexanderfurt</td>
      <td>3</td>
      <td>-58.10</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>954</th>
      <td>Williamsmouth</td>
      <td>2</td>
      <td>74.19</td>
    </tr>
    <tr>
      <th>955</th>
      <td>Wilsonfort</td>
      <td>8</td>
      <td>0.64</td>
    </tr>
    <tr>
      <th>956</th>
      <td>Wolfland</td>
      <td>3</td>
      <td>21.41</td>
    </tr>
    <tr>
      <th>957</th>
      <td>Wrightchester</td>
      <td>9</td>
      <td>23.33</td>
    </tr>
    <tr>
      <th>958</th>
      <td>Younghaven</td>
      <td>5</td>
      <td>58.60</td>
    </tr>
  </tbody>
</table>
<p>959 rows × 3 columns</p>
</div>




```python
df_city_cal["Per_unit"] = df_city_cal["Profit"] / df_city_cal["Quantity"]
df_city_cal
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>City</th>
      <th>Quantity</th>
      <th>Profit</th>
      <th>Per_unit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Aaronfort</td>
      <td>8</td>
      <td>34.54</td>
      <td>4.317500</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Adammouth</td>
      <td>6</td>
      <td>53.97</td>
      <td>8.995000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Adamshaven</td>
      <td>5</td>
      <td>84.07</td>
      <td>16.814000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Alexanderfort</td>
      <td>3</td>
      <td>69.73</td>
      <td>23.243333</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Alexanderfurt</td>
      <td>3</td>
      <td>-58.10</td>
      <td>-19.366667</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>954</th>
      <td>Williamsmouth</td>
      <td>2</td>
      <td>74.19</td>
      <td>37.095000</td>
    </tr>
    <tr>
      <th>955</th>
      <td>Wilsonfort</td>
      <td>8</td>
      <td>0.64</td>
      <td>0.080000</td>
    </tr>
    <tr>
      <th>956</th>
      <td>Wolfland</td>
      <td>3</td>
      <td>21.41</td>
      <td>7.136667</td>
    </tr>
    <tr>
      <th>957</th>
      <td>Wrightchester</td>
      <td>9</td>
      <td>23.33</td>
      <td>2.592222</td>
    </tr>
    <tr>
      <th>958</th>
      <td>Younghaven</td>
      <td>5</td>
      <td>58.60</td>
      <td>11.720000</td>
    </tr>
  </tbody>
</table>
<p>959 rows × 4 columns</p>
</div>




```python
top_10_profit = df_city_cal.sort_values(by= "Per_unit", ascending= False).head(10)
top_10_profit
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>City</th>
      <th>Quantity</th>
      <th>Profit</th>
      <th>Per_unit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>165</th>
      <td>East Stephanie</td>
      <td>1</td>
      <td>245.92</td>
      <td>245.92</td>
    </tr>
    <tr>
      <th>593</th>
      <td>North Savannah</td>
      <td>1</td>
      <td>224.53</td>
      <td>224.53</td>
    </tr>
    <tr>
      <th>325</th>
      <td>Lake Damon</td>
      <td>1</td>
      <td>219.75</td>
      <td>219.75</td>
    </tr>
    <tr>
      <th>881</th>
      <td>Wendystad</td>
      <td>1</td>
      <td>213.10</td>
      <td>213.10</td>
    </tr>
    <tr>
      <th>523</th>
      <td>New Vincentborough</td>
      <td>1</td>
      <td>204.10</td>
      <td>204.10</td>
    </tr>
    <tr>
      <th>356</th>
      <td>Lake Maryfort</td>
      <td>1</td>
      <td>202.48</td>
      <td>202.48</td>
    </tr>
    <tr>
      <th>81</th>
      <td>Cowanland</td>
      <td>1</td>
      <td>200.42</td>
      <td>200.42</td>
    </tr>
    <tr>
      <th>397</th>
      <td>Lesterview</td>
      <td>1</td>
      <td>200.25</td>
      <td>200.25</td>
    </tr>
    <tr>
      <th>82</th>
      <td>Craigport</td>
      <td>1</td>
      <td>200.23</td>
      <td>200.23</td>
    </tr>
    <tr>
      <th>202</th>
      <td>Garnerland</td>
      <td>1</td>
      <td>197.31</td>
      <td>197.31</td>
    </tr>
  </tbody>
</table>
</div>




```python
bottom_10_profit = df_city_cal.sort_values(by= "Per_unit").head(10)
bottom_10_profit
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>City</th>
      <th>Quantity</th>
      <th>Profit</th>
      <th>Per_unit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>479</th>
      <td>New Frank</td>
      <td>1</td>
      <td>-157.74</td>
      <td>-157.74</td>
    </tr>
    <tr>
      <th>185</th>
      <td>Ericksonhaven</td>
      <td>1</td>
      <td>-153.52</td>
      <td>-153.52</td>
    </tr>
    <tr>
      <th>406</th>
      <td>Lopezfort</td>
      <td>1</td>
      <td>-149.64</td>
      <td>-149.64</td>
    </tr>
    <tr>
      <th>745</th>
      <td>Rubenton</td>
      <td>1</td>
      <td>-147.55</td>
      <td>-147.55</td>
    </tr>
    <tr>
      <th>425</th>
      <td>Mckinneystad</td>
      <td>1</td>
      <td>-146.99</td>
      <td>-146.99</td>
    </tr>
    <tr>
      <th>865</th>
      <td>Trujillofurt</td>
      <td>1</td>
      <td>-125.34</td>
      <td>-125.34</td>
    </tr>
    <tr>
      <th>354</th>
      <td>Lake Mariah</td>
      <td>1</td>
      <td>-123.98</td>
      <td>-123.98</td>
    </tr>
    <tr>
      <th>786</th>
      <td>South Davidside</td>
      <td>1</td>
      <td>-117.55</td>
      <td>-117.55</td>
    </tr>
    <tr>
      <th>761</th>
      <td>Shanemouth</td>
      <td>1</td>
      <td>-116.70</td>
      <td>-116.70</td>
    </tr>
    <tr>
      <th>616</th>
      <td>Patrickfurt</td>
      <td>1</td>
      <td>-107.07</td>
      <td>-107.07</td>
    </tr>
  </tbody>
</table>
</div>



### <b> <font color= #ABFF00> Conclusion:
#### A new metric, Profit per Unit Sold, was created to evaluate the sales efficiency of each city.
- #### The top 10 cities generate high profit per unit, indicating strong product mix or premium customer segments.
- #### The bottom 10 cities show low or negative efficiency, possibly due to high volume of low-margin items or higher return rates.
#### Actionable insights for Walmart:
- #### Focus on expanding profitable categories in high-efficiency cities.
- #### In low-efficiency cities, re-evaluate pricing, optimize product assortment, or address operational inefficiencies.

### <b><font color= #FFFF00> Q11. Sales Efficiency Score:
#### *Is there a negative correlation between quantity sold and profit per unit in any region or category? What does this suggest?*


```python
df_neg_reg = pd.DataFrame(df)
```


```python
region_regg = df_neg_reg.groupby("Region")[["Quantity","Profit"]].sum().reset_index()
region_regg.columns = ["Region", "Quantity", "Profit"]
region_regg["Per_Unit"] = region_regg["Profit"] / region_regg["Quantity"]
region_regg
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Region</th>
      <th>Quantity</th>
      <th>Profit</th>
      <th>Per_Unit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>East</td>
      <td>1365</td>
      <td>6294.01</td>
      <td>4.610996</td>
    </tr>
    <tr>
      <th>1</th>
      <td>North</td>
      <td>1167</td>
      <td>7489.50</td>
      <td>6.417738</td>
    </tr>
    <tr>
      <th>2</th>
      <td>South</td>
      <td>1111</td>
      <td>5298.56</td>
      <td>4.769181</td>
    </tr>
    <tr>
      <th>3</th>
      <td>West</td>
      <td>1255</td>
      <td>6781.12</td>
      <td>5.403283</td>
    </tr>
  </tbody>
</table>
</div>




```python
a = region_regg.drop("Region", axis= 1)
```


```python
region_cor = a.corr()
sns.heatmap(region_cor,vmin= -1, vmax= 1 ,annot= True)
plt.show()
```


    
![png](output_174_0.png)
    



```python
cat_regg = df_neg_reg.groupby("Category")[["Quantity","Profit"]].sum().reset_index()
cat_regg.columns = ["Category", "Quantity", "Profit"]
cat_regg["Per_Unit"] = cat_regg["Profit"] / cat_regg["Quantity"]
cat_regg
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Category</th>
      <th>Quantity</th>
      <th>Profit</th>
      <th>Per_Unit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Furniture</td>
      <td>1522</td>
      <td>10382.24</td>
      <td>6.821445</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Office Supplies</td>
      <td>1699</td>
      <td>8460.10</td>
      <td>4.979459</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Technology</td>
      <td>1677</td>
      <td>7020.85</td>
      <td>4.186553</td>
    </tr>
  </tbody>
</table>
</div>




```python
b = cat_regg.drop("Category", axis= 1)
```


```python
cat_cor = b.corr()
sns.heatmap(cat_cor,vmin= -1, vmax= 1 ,annot= True)
plt.show()
```


    
![png](output_177_0.png)
    


### <b> <font color= #ABFF00> Conclusion:
- #### According to the region segment – Yes negative correlation between quantity sold and profit per unit is occurring and negative correlation value = `-0.36`.
- #### According to the Category segment – Yes negative correlation between quantity sold and profit per unit is occurring and Occurring negative correlation value = `-0.92`.

### <b><font color= #FFFF00> Q12. Campaign Impact Simulation:
#### *Assume Walmart ran a 10% discount campaign in August 2024. Recalculate profit for that month and evaluate how the campaign would have affected overall profitability.*


```python
df_aug_10 = pd.DataFrame(df)
df_aug_10.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Order ID</th>
      <th>Order Date</th>
      <th>Customer ID</th>
      <th>Customer Name</th>
      <th>City</th>
      <th>Region</th>
      <th>Category</th>
      <th>Quantity</th>
      <th>Sales</th>
      <th>Profit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ORD-100000</td>
      <td>2024-08-10</td>
      <td>CUST-2824</td>
      <td>Jeffrey Jacobs</td>
      <td>Velezburgh</td>
      <td>East</td>
      <td>Furniture</td>
      <td>7</td>
      <td>238.76</td>
      <td>34.49</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ORD-100001</td>
      <td>2024-02-28</td>
      <td>CUST-1409</td>
      <td>Jacob Walker</td>
      <td>South Alyssamouth</td>
      <td>West</td>
      <td>Office Supplies</td>
      <td>4</td>
      <td>675.17</td>
      <td>159.79</td>
    </tr>
    <tr>
      <th>2</th>
      <td>ORD-100002</td>
      <td>2024-08-26</td>
      <td>CUST-5506</td>
      <td>Jennifer Baker</td>
      <td>East Anthonyburgh</td>
      <td>North</td>
      <td>Office Supplies</td>
      <td>8</td>
      <td>29.51</td>
      <td>3.49</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ORD-100003</td>
      <td>2023-11-26</td>
      <td>CUST-5012</td>
      <td>Maxwell Reed</td>
      <td>Penamouth</td>
      <td>East</td>
      <td>Technology</td>
      <td>5</td>
      <td>113.07</td>
      <td>20.42</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ORD-100004</td>
      <td>2024-11-04</td>
      <td>CUST-4657</td>
      <td>Nicole Cox</td>
      <td>Masonshire</td>
      <td>South</td>
      <td>Furniture</td>
      <td>7</td>
      <td>801.92</td>
      <td>-96.20</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_extract_aug = df_aug_10[(df_aug_10["Order Date"] >= "2024-08-01") & (df_aug_10["Order Date"] <= "2024-08-31")]
df_extract_aug.reset_index().head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>index</th>
      <th>Order ID</th>
      <th>Order Date</th>
      <th>Customer ID</th>
      <th>Customer Name</th>
      <th>City</th>
      <th>Region</th>
      <th>Category</th>
      <th>Quantity</th>
      <th>Sales</th>
      <th>Profit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>ORD-100000</td>
      <td>2024-08-10</td>
      <td>CUST-2824</td>
      <td>Jeffrey Jacobs</td>
      <td>Velezburgh</td>
      <td>East</td>
      <td>Furniture</td>
      <td>7</td>
      <td>238.76</td>
      <td>34.49</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>ORD-100002</td>
      <td>2024-08-26</td>
      <td>CUST-5506</td>
      <td>Jennifer Baker</td>
      <td>East Anthonyburgh</td>
      <td>North</td>
      <td>Office Supplies</td>
      <td>8</td>
      <td>29.51</td>
      <td>3.49</td>
    </tr>
    <tr>
      <th>2</th>
      <td>21</td>
      <td>ORD-100021</td>
      <td>2024-08-28</td>
      <td>CUST-8359</td>
      <td>Carol Doyle</td>
      <td>Christopherhaven</td>
      <td>North</td>
      <td>Office Supplies</td>
      <td>6</td>
      <td>36.27</td>
      <td>-7.16</td>
    </tr>
    <tr>
      <th>3</th>
      <td>47</td>
      <td>ORD-100047</td>
      <td>2024-08-20</td>
      <td>CUST-2139</td>
      <td>Kayla Hall</td>
      <td>North David</td>
      <td>South</td>
      <td>Office Supplies</td>
      <td>4</td>
      <td>546.22</td>
      <td>22.11</td>
    </tr>
    <tr>
      <th>4</th>
      <td>65</td>
      <td>ORD-100065</td>
      <td>2024-08-03</td>
      <td>CUST-9751</td>
      <td>Calvin Martin</td>
      <td>South Vanessashire</td>
      <td>East</td>
      <td>Furniture</td>
      <td>5</td>
      <td>20.89</td>
      <td>1.22</td>
    </tr>
  </tbody>
</table>
</div>




```python
august_10_per = df_extract_aug["Profit"] - (df_extract_aug["Profit"] / 100) * 10
august_10_per.head()
```




    0     31.041
    2      3.141
    21    -6.444
    47    19.899
    65     1.098
    Name: Profit, dtype: float64




```python
seperate_aug = df_aug_10["Profit"].sum() - df_extract_aug["Profit"].sum()
seperate_aug
```




    25070.430000000004




```python
overall_10_profit = seperate_aug + august_10_per.sum()
overall_10_profit
```




    25783.914000000004




```python
without_dis = round(df_aug_10["Profit"].sum() / df_aug_10["Profit"].count(), 2)
without_dis
```




    25.86




```python
with_10_dis = round(overall_10_profit / df_aug_10["Profit"].count(), 2)
with_10_dis
```




    25.78



### <b> <font color= #ABFF00> Conclusion:
##### *Before 10% discount: ₹25.86 profit/order*
##### *After discount: ₹25.78 profit/order*
##### *Impact = only ₹0.08 difference*
- #### This indicates that Walmart can safely run such discount campaigns without significantly harming profitability.
- #### If the discount leads to even a small boost in sales volume, the overall profit may actually increase.

### <b><font color= #FFFF00> Q13. Return Risk Zones:
#### *If high-quantity orders with low profit are considered risky for returns, which region shows the highest risk exposure?*


```python
df_return_risk = pd.DataFrame(df)
```


```python
df_return_risk.groupby(["Region", "Quantity"])[["Quantity", "Profit"]].sum()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>Quantity</th>
      <th>Profit</th>
    </tr>
    <tr>
      <th>Region</th>
      <th>Quantity</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="9" valign="top">East</th>
      <th>1</th>
      <td>35</td>
      <td>1335.42</td>
    </tr>
    <tr>
      <th>2</th>
      <td>58</td>
      <td>1158.11</td>
    </tr>
    <tr>
      <th>3</th>
      <td>105</td>
      <td>606.59</td>
    </tr>
    <tr>
      <th>4</th>
      <td>76</td>
      <td>387.16</td>
    </tr>
    <tr>
      <th>5</th>
      <td>135</td>
      <td>552.92</td>
    </tr>
    <tr>
      <th>6</th>
      <td>216</td>
      <td>542.92</td>
    </tr>
    <tr>
      <th>7</th>
      <td>238</td>
      <td>1041.43</td>
    </tr>
    <tr>
      <th>8</th>
      <td>232</td>
      <td>380.41</td>
    </tr>
    <tr>
      <th>9</th>
      <td>270</td>
      <td>289.05</td>
    </tr>
    <tr>
      <th rowspan="9" valign="top">North</th>
      <th>1</th>
      <td>38</td>
      <td>1504.67</td>
    </tr>
    <tr>
      <th>2</th>
      <td>38</td>
      <td>596.46</td>
    </tr>
    <tr>
      <th>3</th>
      <td>102</td>
      <td>596.57</td>
    </tr>
    <tr>
      <th>4</th>
      <td>112</td>
      <td>2088.58</td>
    </tr>
    <tr>
      <th>5</th>
      <td>105</td>
      <td>423.76</td>
    </tr>
    <tr>
      <th>6</th>
      <td>120</td>
      <td>214.79</td>
    </tr>
    <tr>
      <th>7</th>
      <td>161</td>
      <td>790.29</td>
    </tr>
    <tr>
      <th>8</th>
      <td>248</td>
      <td>1069.09</td>
    </tr>
    <tr>
      <th>9</th>
      <td>243</td>
      <td>205.29</td>
    </tr>
    <tr>
      <th rowspan="9" valign="top">South</th>
      <th>1</th>
      <td>31</td>
      <td>1043.67</td>
    </tr>
    <tr>
      <th>2</th>
      <td>48</td>
      <td>-221.11</td>
    </tr>
    <tr>
      <th>3</th>
      <td>75</td>
      <td>355.70</td>
    </tr>
    <tr>
      <th>4</th>
      <td>112</td>
      <td>904.62</td>
    </tr>
    <tr>
      <th>5</th>
      <td>180</td>
      <td>1074.40</td>
    </tr>
    <tr>
      <th>6</th>
      <td>162</td>
      <td>-405.95</td>
    </tr>
    <tr>
      <th>7</th>
      <td>161</td>
      <td>490.97</td>
    </tr>
    <tr>
      <th>8</th>
      <td>144</td>
      <td>1104.64</td>
    </tr>
    <tr>
      <th>9</th>
      <td>198</td>
      <td>951.62</td>
    </tr>
    <tr>
      <th rowspan="9" valign="top">West</th>
      <th>1</th>
      <td>32</td>
      <td>1579.14</td>
    </tr>
    <tr>
      <th>2</th>
      <td>42</td>
      <td>519.85</td>
    </tr>
    <tr>
      <th>3</th>
      <td>84</td>
      <td>210.80</td>
    </tr>
    <tr>
      <th>4</th>
      <td>112</td>
      <td>708.67</td>
    </tr>
    <tr>
      <th>5</th>
      <td>160</td>
      <td>911.19</td>
    </tr>
    <tr>
      <th>6</th>
      <td>150</td>
      <td>472.21</td>
    </tr>
    <tr>
      <th>7</th>
      <td>196</td>
      <td>1192.76</td>
    </tr>
    <tr>
      <th>8</th>
      <td>272</td>
      <td>296.07</td>
    </tr>
    <tr>
      <th>9</th>
      <td>207</td>
      <td>890.43</td>
    </tr>
  </tbody>
</table>
</div>




```python
sep_east = df_return_risk[(df_return_risk["Region"] == "East") & (df_return_risk["Quantity"] == 9)]
q_east = sep_east["Quantity"].sum()
p_east = sep_east["Profit"].sum()
```


```python
sep_east = df_return_risk[(df_return_risk["Region"] == "South") & (df_return_risk["Quantity"] == 9)]
q_south = sep_east["Quantity"].sum()
p_south = sep_east["Profit"].sum()
```


```python
sep_east = df_return_risk[(df_return_risk["Region"] == "North") & (df_return_risk["Quantity"] == 8)]
q_north = sep_east["Quantity"].sum()
p_north = sep_east["Profit"].sum()
```


```python
sep_east = df_return_risk[(df_return_risk["Region"] == "West") & (df_return_risk["Quantity"] == 8)]
q_west = sep_east["Quantity"].sum()
p_west = sep_east["Profit"].sum()
```


```python
join = pd.DataFrame({
    "Quantity_count" : [q_east, q_south, q_north, q_west],
    "Profit_sum" : [p_east, p_south, p_north, p_west]
             }, index= ["east", "south", "north", "west"])
```


```python
join["Profit_perc"] = (join["Profit_sum"] / join["Profit_sum"].sum()) * 100
join
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Quantity_count</th>
      <th>Profit_sum</th>
      <th>Profit_perc</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>east</th>
      <td>270</td>
      <td>289.05</td>
      <td>11.092435</td>
    </tr>
    <tr>
      <th>south</th>
      <td>198</td>
      <td>951.62</td>
      <td>36.518883</td>
    </tr>
    <tr>
      <th>north</th>
      <td>248</td>
      <td>1069.09</td>
      <td>41.026851</td>
    </tr>
    <tr>
      <th>west</th>
      <td>272</td>
      <td>296.07</td>
      <td>11.361831</td>
    </tr>
  </tbody>
</table>
</div>



### <b> <font color= #ABFF00> Conclusion:
##### *Based on the Region segmentation East region has Quantity count is 270, total profit is 289.05, and percentage is 11.09% contribution.*
- #### East has high quantity but very low profit, so it is most at risk.

### <b><font color= #FFFF00> Q14. Return Risk Zones:
#### *Calculate how many days (based on order date) it took each region to cross a cumulative profit of ₹5,000. Who was fastest?*


```python
df_time = df.copy()
df_time.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Order ID</th>
      <th>Order Date</th>
      <th>Customer ID</th>
      <th>Customer Name</th>
      <th>City</th>
      <th>Region</th>
      <th>Category</th>
      <th>Quantity</th>
      <th>Sales</th>
      <th>Profit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ORD-100000</td>
      <td>2024-08-10</td>
      <td>CUST-2824</td>
      <td>Jeffrey Jacobs</td>
      <td>Velezburgh</td>
      <td>East</td>
      <td>Furniture</td>
      <td>7</td>
      <td>238.76</td>
      <td>34.49</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ORD-100001</td>
      <td>2024-02-28</td>
      <td>CUST-1409</td>
      <td>Jacob Walker</td>
      <td>South Alyssamouth</td>
      <td>West</td>
      <td>Office Supplies</td>
      <td>4</td>
      <td>675.17</td>
      <td>159.79</td>
    </tr>
    <tr>
      <th>2</th>
      <td>ORD-100002</td>
      <td>2024-08-26</td>
      <td>CUST-5506</td>
      <td>Jennifer Baker</td>
      <td>East Anthonyburgh</td>
      <td>North</td>
      <td>Office Supplies</td>
      <td>8</td>
      <td>29.51</td>
      <td>3.49</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ORD-100003</td>
      <td>2023-11-26</td>
      <td>CUST-5012</td>
      <td>Maxwell Reed</td>
      <td>Penamouth</td>
      <td>East</td>
      <td>Technology</td>
      <td>5</td>
      <td>113.07</td>
      <td>20.42</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ORD-100004</td>
      <td>2024-11-04</td>
      <td>CUST-4657</td>
      <td>Nicole Cox</td>
      <td>Masonshire</td>
      <td>South</td>
      <td>Furniture</td>
      <td>7</td>
      <td>801.92</td>
      <td>-96.20</td>
    </tr>
  </tbody>
</table>
</div>




```python
filter_1 = df_time.groupby(["Region", "Order Date"])["Profit"].sum().reset_index()
filter_1 = filter_1.sort_values(["Region", "Order Date"])
filter_1["cumsum"] = filter_1.groupby("Region")["Profit"].cumsum()
```


```python
filter_2 = filter_1[filter_1["cumsum"] > 1000]
filter_2 = filter_2.groupby("Region")["Order Date"].first().reset_index()
filter_2.columns = ["Region", "Last_day"]
filter_2
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Region</th>
      <th>Last_day</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>East</td>
      <td>2023-10-20</td>
    </tr>
    <tr>
      <th>1</th>
      <td>North</td>
      <td>2023-10-07</td>
    </tr>
    <tr>
      <th>2</th>
      <td>South</td>
      <td>2023-08-14</td>
    </tr>
    <tr>
      <th>3</th>
      <td>West</td>
      <td>2023-09-02</td>
    </tr>
  </tbody>
</table>
</div>




```python
filter_3 = df_time.groupby(["Region"])["Order Date"].min().reset_index()
filter_3.columns = ["Region", "First_day"]
filter_3
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Region</th>
      <th>First_day</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>East</td>
      <td>2023-07-13</td>
    </tr>
    <tr>
      <th>1</th>
      <td>North</td>
      <td>2023-07-12</td>
    </tr>
    <tr>
      <th>2</th>
      <td>South</td>
      <td>2023-07-11</td>
    </tr>
    <tr>
      <th>3</th>
      <td>West</td>
      <td>2023-07-14</td>
    </tr>
  </tbody>
</table>
</div>




```python
day_count = pd.merge(filter_2, filter_3, on= "Region")
day_count["Day_counts"] = day_count["Last_day"] - day_count["First_day"]
day_count
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Region</th>
      <th>Last_day</th>
      <th>First_day</th>
      <th>Day_counts</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>East</td>
      <td>2023-10-20</td>
      <td>2023-07-13</td>
      <td>99 days</td>
    </tr>
    <tr>
      <th>1</th>
      <td>North</td>
      <td>2023-10-07</td>
      <td>2023-07-12</td>
      <td>87 days</td>
    </tr>
    <tr>
      <th>2</th>
      <td>South</td>
      <td>2023-08-14</td>
      <td>2023-07-11</td>
      <td>34 days</td>
    </tr>
    <tr>
      <th>3</th>
      <td>West</td>
      <td>2023-09-02</td>
      <td>2023-07-14</td>
      <td>50 days</td>
    </tr>
  </tbody>
</table>
</div>



### <b> <font color= #ABFF00> Conclusion:
- #### Based The South Region was the fastest to reach ₹1,000 profit in just 34 days from their first order date.
- #### This shows stronger early sales momentum or better margins in that region.

### <b><font color= #FFFF00> Q15. High-Impact Customer Recovery Plan:
#### *Identify the bottom 5% of customers by profit. Suggest a personalized sales strategy for them based on their past order behaviour.*


```python
df_bottom = df.copy()
```


```python
df_bottom.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Order ID</th>
      <th>Order Date</th>
      <th>Customer ID</th>
      <th>Customer Name</th>
      <th>City</th>
      <th>Region</th>
      <th>Category</th>
      <th>Quantity</th>
      <th>Sales</th>
      <th>Profit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ORD-100000</td>
      <td>2024-08-10</td>
      <td>CUST-2824</td>
      <td>Jeffrey Jacobs</td>
      <td>Velezburgh</td>
      <td>East</td>
      <td>Furniture</td>
      <td>7</td>
      <td>238.76</td>
      <td>34.49</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ORD-100001</td>
      <td>2024-02-28</td>
      <td>CUST-1409</td>
      <td>Jacob Walker</td>
      <td>South Alyssamouth</td>
      <td>West</td>
      <td>Office Supplies</td>
      <td>4</td>
      <td>675.17</td>
      <td>159.79</td>
    </tr>
    <tr>
      <th>2</th>
      <td>ORD-100002</td>
      <td>2024-08-26</td>
      <td>CUST-5506</td>
      <td>Jennifer Baker</td>
      <td>East Anthonyburgh</td>
      <td>North</td>
      <td>Office Supplies</td>
      <td>8</td>
      <td>29.51</td>
      <td>3.49</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ORD-100003</td>
      <td>2023-11-26</td>
      <td>CUST-5012</td>
      <td>Maxwell Reed</td>
      <td>Penamouth</td>
      <td>East</td>
      <td>Technology</td>
      <td>5</td>
      <td>113.07</td>
      <td>20.42</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ORD-100004</td>
      <td>2024-11-04</td>
      <td>CUST-4657</td>
      <td>Nicole Cox</td>
      <td>Masonshire</td>
      <td>South</td>
      <td>Furniture</td>
      <td>7</td>
      <td>801.92</td>
      <td>-96.20</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_filter = df_bottom.groupby(["Customer ID"])["Profit"].sum().reset_index()
df_filter = df_fil.sort_values(by= "Profit")
```


```python
df_fill_5 = round((df_filter["Profit"].count()/100)*5)
bottom_5 = df_filter.head(df_fill_5)
```


```python
filtered_bottom_5 = df_bottom[df_bottom["Customer ID"].isin(bottom_5["Customer ID"])]
```


```python
analysis_bottom = filtered_bottom_5.groupby("Quantity")[["Quantity","Sales","Profit"]].sum()
analysis_bottom.columns = ["Quantity_counts", "Sales", "Profit"]
analysis_bottom.reset_index()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Quantity</th>
      <th>Quantity_counts</th>
      <th>Sales</th>
      <th>Profit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>8</td>
      <td>6118.52</td>
      <td>-1121.46</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>8</td>
      <td>3327.64</td>
      <td>-568.02</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>21</td>
      <td>5876.69</td>
      <td>-938.51</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>8</td>
      <td>1820.96</td>
      <td>-305.06</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>30</td>
      <td>5111.88</td>
      <td>-800.24</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6</td>
      <td>54</td>
      <td>7737.05</td>
      <td>-1322.95</td>
    </tr>
    <tr>
      <th>6</th>
      <td>7</td>
      <td>14</td>
      <td>1112.09</td>
      <td>-131.00</td>
    </tr>
    <tr>
      <th>7</th>
      <td>8</td>
      <td>32</td>
      <td>3039.40</td>
      <td>-571.50</td>
    </tr>
    <tr>
      <th>8</th>
      <td>9</td>
      <td>63</td>
      <td>5934.64</td>
      <td>-938.79</td>
    </tr>
  </tbody>
</table>
</div>




```python
analysis_bottom["Quantity_per"] = round((analysis_bottom["Quantity_counts"] / analysis_bottom["Quantity_counts"].sum())*100)
```


```python
analysis_bottom["Sales_per"] = round((analysis_bottom["Sales"] / analysis_bottom["Sales"].sum())*100)
```


```python
analysis_bottom["Profit_per"] = round((analysis_bottom["Profit"] / analysis_bottom["Profit"].sum())*100)
```


```python
analysis_bottom
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Quantity_counts</th>
      <th>Sales</th>
      <th>Profit</th>
      <th>Quantity_per</th>
      <th>Sales_per</th>
      <th>Profit_per</th>
    </tr>
    <tr>
      <th>Quantity</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>8</td>
      <td>6118.52</td>
      <td>-1121.46</td>
      <td>3.0</td>
      <td>15.0</td>
      <td>17.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>8</td>
      <td>3327.64</td>
      <td>-568.02</td>
      <td>3.0</td>
      <td>8.0</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>21</td>
      <td>5876.69</td>
      <td>-938.51</td>
      <td>9.0</td>
      <td>15.0</td>
      <td>14.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>8</td>
      <td>1820.96</td>
      <td>-305.06</td>
      <td>3.0</td>
      <td>5.0</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>30</td>
      <td>5111.88</td>
      <td>-800.24</td>
      <td>13.0</td>
      <td>13.0</td>
      <td>12.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>54</td>
      <td>7737.05</td>
      <td>-1322.95</td>
      <td>23.0</td>
      <td>19.0</td>
      <td>20.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>14</td>
      <td>1112.09</td>
      <td>-131.00</td>
      <td>6.0</td>
      <td>3.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>32</td>
      <td>3039.40</td>
      <td>-571.50</td>
      <td>13.0</td>
      <td>8.0</td>
      <td>9.0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>63</td>
      <td>5934.64</td>
      <td>-938.79</td>
      <td>26.0</td>
      <td>15.0</td>
      <td>14.0</td>
    </tr>
  </tbody>
</table>
</div>



### <b> <font color= #ABFF00> Conclusion:
#### All 48 customers in the bottom 5% are one-time customers.
#### High-quantity buyers (Qty > 4) among them are responsible for:
- #### *80% of Quantity*
- #### *60% of Sales*
- #### *56% of (Negative) Profit*
#### These may be discount-driven buyers → suggesting repricing to improve profit.


```python

```


```python

```

