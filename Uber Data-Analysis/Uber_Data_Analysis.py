import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.pyplot import subplots
from pandas import DataFrame
import matplotlib.pylab as plt
from sklearn.feature_selection import mutual_info_regression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from statistics import mean, median
from scipy import stats

df=pd.read_csv("ncr_ride_bookings.csv")
df.info()
print(df.head())
print(df.info())
print(df.describe().T)


# Organizing Categories
df_Float=df.select_dtypes(include=['float64'])
df_Object=df.select_dtypes(include=['object'])
print("-"*100)

# Checking for missing values
Missing_Values=df.isnull().sum()
if Missing_Values.sum()>0:
    print("Missing Values Detected:\n", Missing_Values)
else:  
    print("No Missing Values Detected")
print("-"*100)

df['Incomplete Rides'] = df['Incomplete Rides'].fillna(0.0)

for col in df.columns:
    if df[col].isnull().any():
        if df[col].dtype=='float64':
            median=df[col].median()
            df[col].fillna(median,inplace=True)
            print(f"Missing Values removed that be data.type 'Float64'")
        elif df[col].dtype=='object':
            mod=df[col].mode()[0]
            df[col].fillna(mod,inplace=True)
            print(f"Missing Values removed that be data.type 'Object'")
print(df.isnull().sum())
print("-"*100)

# Checking for duplicates
Duplicates=df.duplicated().sum()
if Duplicates>0:
    print("Duplicates Detected:", Duplicates)
    df=df.drop_duplicates()
    print("Duplicates Removed. New shape:", df.shape)
else:
    print("No Duplicates Detected")
print("-"*100)

# Univariate analysis for Float64;
num_plots=len(df_Float.columns)
rows=int(np.ceil(num_plots/3))

fig,axes=plt.subplots(rows,3,figsize=(20,4*rows))
axes=axes.ravel()

for index,col in enumerate(df_Float.columns):
    if index>=len(axes):
        break
    axes[index].hist(df[col],bins=10,color="steelblue",edgecolor='black',alpha=0.8)
    axes[index].set_title("Univariate Analysis Tpye Of Float64",fontsize=12,fontweight='bold',color='black')
    axes[index].set_xlabel(col)
    axes[index].set_ylabel("Frequency")

    mean_value=df[col].mean()
    median_value=df[col].median()
    axes[index].axvline(mean_value,color="darkgreen",linestyle="--",linewidth=2,label=f"Mean: {mean_value:.2f}")
    axes[index].axvline(median_value,color="darkorange",linestyle="-",linewidth=2,label=f"Median: {median_value:.2f}")
    axes[index].legend()
for i in range(num_plots,len(axes)):
    fig.delaxes(axes[i])

plt.tight_layout()
plt.savefig("Univariate Analysis.png",dpi=300,bbox_inches='tight')
plt.show()
print("-"*100)

skew=df.skew(numeric_only=True)
print("\nSkewnessOfFeatures:")
print(skew.sort_values(ascending=False))
print("-"*100)

# Univariate analysis for Object;
plt.figure(figsize=(12,6))

sns.countplot(x=df_Object['Payment Method'],color="darkblue",edgecolor="black")
plt.title("Univariate Analysis Type Of Object",fontsize=16,fontweight='bold',color='black')
plt.xlabel("Payment Method")
plt.ylabel("Frequency")
plt.xticks(rotation=45)

for p in plt.gca().patches:
    height=p.get_height()
    if height>0:
        plt.text(p.get_x()+ p.get_width() / 2., height+5,f'{int(height)}',ha='center',va='bottom',fontsize=12,fontweight='bold')

plt.tight_layout()
plt.savefig("Univariate Analysis2.png",dpi=300,bbox_inches='tight')
plt.show()
print("-"*100)

# Vehicle Type Visualization;

Vehicle_Categories=df['Vehicle Type']
Vehicle_Categories_Values=Vehicle_Categories.value_counts().sort_index()
plt.figure(figsize=(20,8))
plt.title("Vehicle Typle Details",fontsize=14,fontweight='bold')
sns.countplot(data=df,x='Vehicle Type',hue='Vehicle Type',palette="viridis",legend=False) 
plt.xlabel("Vehicle Type",fontsize=12)    
plt.ylabel("Count",fontsize=12)    
plt.xticks(rotation=45)    
plt.tight_layout()

for idx,v in enumerate(Vehicle_Categories_Values.values):
    plt.text(Vehicle_Categories_Values.index[idx],v+5,str(v),ha='center',va='bottom',fontsize=12,fontweight='bold') 

plt.savefig("Vehicle Type.png",dpi=300,bbox_inches='tight')
plt.show()

# Booking_Status Pie Plot;
status_counts=df['Booking Status'].value_counts()
values=status_counts.values
labels=status_counts.index
colors=['green','darkorange','blue','orange','red']
explode=[0.1,0,0,0,0]

plt.figure(figsize=(12,6))
plt.pie(values,labels=labels,autopct='%1.1f%%', startangle=140,shadow=True,wedgeprops={'edgecolor':'white','alpha':0.7},textprops={'fontsize':8,'fontweight':'bold'},colors=colors,explode=explode)
plt.title("Booking Status Plot")
plt.tight_layout()
plt.savefig("Booking Status Pie.png",dpi=300,bbox_inches='tight')
plt.show()
print("-"*100)

# Top 10 Pickup and Drop Location Plots;
Pickup_Location_Values=df['Pickup Location'].value_counts()
Drop_Location_Values=df['Drop Location'].value_counts()
top_10_PickupLocation=Pickup_Location_Values.head(10)
top_10_DropLocation=Drop_Location_Values.head(10)

print(top_10_PickupLocation)
print(top_10_DropLocation)

fig,axes=plt.subplots(1,2,figsize=(18,10))
axes=axes.ravel()

sns.barplot(x=top_10_PickupLocation.index,y=top_10_PickupLocation.values,ax=axes[0],hue=top_10_PickupLocation.index,palette='Reds',edgecolor='black',linewidth=1,alpha=0.7,legend=False)
axes[0].set_title("Pickup Location Plot",fontsize=12,fontweight='bold')
axes[0].set_xlabel("Top 10 Pickup Location",fontsize=10)
axes[0].set_ylabel("Values",fontsize=10)
axes[0].tick_params(axis='x',rotation=45,labelsize=8)

sns.barplot(x=top_10_DropLocation.index,y=top_10_DropLocation.values,ax=axes[1],hue=top_10_DropLocation.index,palette='rocket',edgecolor='black',linewidth=1,alpha=0.7,legend=False)
axes[1].set_title("Drop Location Plot",fontsize=12,fontweight='bold')
axes[1].set_xlabel("Top 10 Drop Location",fontsize=10)
axes[1].set_ylabel("Values",fontsize=10)
axes[1].tick_params(axis='x',rotation=45,labelsize=8)

plt.tight_layout()
plt.savefig("Top and Least 10 Pickup and Drop Location Plots.png",dpi=300,bbox_inches='tight')
plt.show()
print("-"*100)

# Driver Ratings and Customer Rating Plots;

df['Driver Ratings']=pd.to_numeric(df['Driver Ratings'],errors='coerce')
df['Customer Rating']=pd.to_numeric(df['Customer Rating'],errors='coerce')

print(df['Driver Ratings'].unique())
print(df['Customer Rating'].unique())

g=sns.jointplot(x='Driver Ratings',y='Customer Rating',data=df,kind='hex',cmap='coolwarm',gridsize=30,edgecolor='black')
g.fig.suptitle("Hexbin Plots For Driver and Customer Ratings",y=1.02,fontsize=12)
g.set_axis_labels("Driver Ratings","Customer Rating",fontsize=10)

plt.tight_layout()
plt.savefig("Driver and Customer Ratings.png",dpi=300,bbox_inches='tight')
plt.show()
print("-"*100)

# Customer Ratings with Car Selection;

plt.figure(figsize=(18,8))
sns.countplot(x=df['Vehicle Type'],hue=df['Customer Rating'],data=df,orient='h',palette="viridis",edgecolor='black',linewidth=1,alpha=0.7)
plt.title("Customer Ratings With Vehicle Types",fontsize=12,fontweight='bold')
plt.tight_layout()
plt.savefig("Customer Ratings and Vehicle Type.png",dpi=300,bbox_inches='tight')
plt.show()
print("-"*100)

# InComplete Rides Pie Plots;

df['Incomplete Rides'] = pd.to_numeric(df['Incomplete Rides'], errors='coerce')
print(df['Incomplete Rides'].unique())
incomplete_counts = df['Incomplete Rides'].value_counts()
print(incomplete_counts)

labels=incomplete_counts.index.map({0.0: 'Successful', 1.0: 'UnSuccessful'})
values=incomplete_counts.values
colors=["orange","yellow"]
explode=[0.2]+[0]*(len(values)-1)

plt.subplots(figsize=(12,6))
plt.pie(values,labels=labels,explode=explode,autopct='%1.1f%%',startangle=145,shadow=True,wedgeprops={'edgecolor':'white','alpha':0.7},textprops={'fontsize':8},colors=colors)
df['Incomplete Rides Reason']=df['Incomplete Rides Reason'].fillna("Unknown")
queue=df['Incomplete Rides Reason'].value_counts().index
queue2=df['Incomplete Rides Reason'].value_counts().values

plt.tight_layout()
plt.savefig("Successful Rate.png",dpi=300,bbox_inches='tight')
plt.show()

fig,ax=plt.subplots(figsize=(12,6))
sns.countplot(data=df,x='Incomplete Rides Reason',palette="magma",hue='Incomplete Rides Reason',edgecolor='black',linewidth=1,alpha=0.8)
ax.set_xlabel("Reasons",fontsize=8,fontweight='bold')
for v in ax.patches:
    ax.text(v.get_x()+v.get_width()/2.,v.get_height(),f'{int(v.get_height())}',ha='center',va='bottom',fontsize=10,fontweight='bold')

plt.savefig("InComplete Reasons.png",dpi=300,bbox_inches='tight')
plt.title("InComplete and Complete Rides Plots",fontsize=14,fontweight='bold')
plt.tight_layout()
plt.show()
print("-"*100)

# Reasons for cancelling by Driver or Customer;

print("Reasons For Cancelling by Driver or Customer;")
print("-"*100)

df_customer=df["Reason for cancelling by Customer"].fillna("Unkown")
df_driver=df["Driver Cancellation Reason"].fillna("Unkown")
customer_count=df_customer.value_counts()
driver_count=df_driver.value_counts()

plot_order_customer=customer_count.index
customer_line_data = customer_count.loc[plot_order_customer]

plot_order_driver=driver_count.index
driver_line_data=driver_count.loc[plot_order_driver]

fig,ax=plt.subplots(1,2,figsize=(22,10))
customer_plot=sns.countplot(x=df_customer,palette="coolwarm",edgecolor="black",alpha=0.8,linewidth=1,ax=ax[0],order=plot_order_customer)
ax[0].set_title("Reason for cancelling by Customer",fontsize=12,fontweight='bold')
ax[0].set_xlabel("Customer Reasons",fontsize=10)
ax[0].set_ylabel("Reasons Counts",fontsize=10)
ax[0].tick_params(axis='x',rotation=45,labelsize=9)
ax[0].plot(customer_line_data.index,customer_line_data.values,marker="o",color="darkred",linestyle="--")

for p in customer_plot.patches:
    customer_plot.annotate(f'{int(p.get_height())}',(p.get_x()+p.get_width()/2.,p.get_height()),ha="center",va="center",xytext=(0,10),textcoords='offset points',fontweight="bold",fontsize=9)

driver_plot=sns.countplot(x=df_driver,palette="viridis",edgecolor="black",alpha=0.8,linewidth=1,ax=ax[1],order=plot_order_driver)
ax[1].set_title("Reason for cancelling by Customer",fontsize=12,fontweight='bold')
ax[1].set_xlabel("Driver Reasons",fontsize=10)
ax[1].tick_params(axis='x',rotation=45,labelsize=9)
ax[1].plot(driver_line_data.index,driver_line_data.values,marker="s",color="navy",linestyle="--")

for p in driver_plot.patches:
    driver_plot.annotate(f'{int(p.get_height())}',(p.get_x()+p.get_width()/2.,p.get_height()),ha="center",va="center",xytext=(0,10),textcoords='offset points',fontweight="bold",fontsize=9)

plt.tight_layout()
plt.savefig("Reasons for cancelling by Driver and Customer.png",dpi=300,bbox_inches='tight')
plt.show()
print("-"*100)

print(df_customer.value_counts())
print("-"*100)
print(df_driver.value_counts())
print("-"*100)

# Driver and Customer Ratings Relationship;

ratings_df = df.dropna(subset=['Driver Ratings', 'Customer Rating', 'Ride Distance', 'Booking Value']).copy()
ratings_df['Driver Ratings'] = pd.to_numeric(ratings_df['Driver Ratings'])
ratings_df['Customer Rating'] = pd.to_numeric(ratings_df['Customer Rating'])

correlation = ratings_df['Driver Ratings'].corr(ratings_df['Customer Rating'])
print(f"Correlation: {correlation:.2f}")

plt.figure(figsize=(10,6))

sns.regplot(data=ratings_df, x='Driver Ratings', y='Customer Rating',
            scatter_kws={'alpha':0.3, 'color':"#1da6f0"},
            line_kws={'color':"#FF1900", 'linewidth':2})
plt.title('Driver Points vs Customer Points Relationship',fontsize=12,fontweight='bold')
plt.xlabel("Driver Points",fontsize=10)
plt.ylabel("Customer Points",fontsize=10)
plt.grid(True,linestyle="--",alpha=0.6)

plt.tight_layout()
plt.savefig("Driver and Customer Points Relationship.png",dpi=300,bbox_inches='tight')
plt.show()

# Points and Other Feature Relationship;

fig,axes=plt.subplots(2,2,figsize=(18, 14))
fig.suptitle('The Relationship of Scores with Other Metrics', fontsize=20, fontweight='bold')

sns.regplot(ax=axes[0,0], data=ratings_df, x='Ride Distance', y='Driver Ratings',
            scatter_kws={'alpha':0.2, 'color':'#2ecc71'}, line_kws={'color':'#2c3e50'})
axes[0,0].set_title("Driver Ratings and Ride Distance")
axes[0,0].set_xlabel("Ride Distance")
axes[0,0].set_ylabel("Driver Ratings")

sns.regplot(ax=axes[0,1], data=ratings_df, x='Ride Distance', y='Customer Rating',
            scatter_kws={'alpha':0.2,'color':'#2ecc71'}, line_kws={'color':'#2c3e50'})
axes[0,1].set_title("Customer Ratings and Ride Distance")
axes[0,1].set_xlabel("Ride Distance")
axes[0,1].set_ylabel("Customer Ratings")

sns.boxplot(ax=axes[1,0], data=ratings_df, x='Booking Status', y='Driver Ratings',palette="coolwarm")
axes[1,0].set_title("Driver Ratings and Booking Status")
axes[1,0].set_xlabel("Booking Status")
axes[1,0].set_ylabel("Driver Ratings")

sns.boxplot(ax=axes[1,1], data=ratings_df, x='Booking Status', y='Customer Rating',palette="viridis")
axes[1,1].set_title("Customer Ratings and Booking Status")
axes[1,1].set_xlabel("Booking Status")
axes[1,1].set_ylabel("Customer Ratings")

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("Points and Other Feature Relationship.png",dpi=300,bbox_inches='tight')
plt.show()
print("-"*100)

# Cancelled Riders by Customers;

df['Cancelled Rides by Customer'] = pd.to_numeric(df['Cancelled Rides by Customer'], errors='coerce')
df['Cancelled Rides by Customer'] = df['Cancelled Rides by Customer'].fillna(0.0)

df_clean=df.copy()
df_clean['Datetime']=pd.to_datetime(df_clean['Date'].astype(str)+ ' ' +df_clean['Time'].astype(str))
df_clean['Day']=df_clean["Datetime"].dt.day_name()

day_order=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
df_clean['Day']=pd.Categorical(df_clean['Day'],categories=day_order,ordered=True)

cancelled_rides_by_customers = df_clean.groupby('Day')['Cancelled Rides by Customer'].sum()
labels = cancelled_rides_by_customers.index
values = cancelled_rides_by_customers.values

fig,axes=plt.subplots(1,2,figsize=(22,10))
axes[0].plot(labels, values, marker='o', linestyle='--', color='darkblue',alpha=0.7)
axes[0].set_title("Number of Cancellations by Days", fontsize=14, fontweight='bold')
axes[0].set_xlabel("Days", fontsize=10,fontweight='bold')
axes[0].set_ylabel("Number of Cancellations", fontsize=10,fontweight='bold')
axes[0].grid(True, linestyle='--')

axes[1].pie(values, labels=labels, shadow=True,
            autopct='%1.1f%%', startangle=145,
            wedgeprops={"edgecolor": "white", "alpha": 0.7})
axes[1].set_title("Cancelled Rides by Customer Rate", fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig("Cancelled Rides by Customer.png", dpi=300, bbox_inches='tight')
plt.show()
print("-"*100)


# Cancelled Riders by Drivers;

df['Cancelled Rides by Driver'] = pd.to_numeric(df['Cancelled Rides by Driver'], errors='coerce')
df['Cancelled Rides by Driver'] = df['Cancelled Rides by Driver'].fillna(0.0)

cancelled_rides_by_drivers=df_clean.groupby('Day')['Cancelled Rides by Driver'].sum()
label=cancelled_rides_by_drivers.index
value=cancelled_rides_by_drivers.values
colors_from_palette=plt.cm.viridis(np.linspace(0,1,len(value)))

fig,axes=plt.subplots(1,2,figsize=(22,10))
axes[0].plot(label,value,marker='o',linestyle="--",color="darkgreen",alpha=0.7)
axes[0].set_title("Number Of Cancellations by Days",fontsize=14,fontweight='bold')
axes[0].set_xlabel("Days",fontsize=10,fontweight='bold')
axes[0].set_ylabel("Number of Cancellations",fontsize=10,fontweight='bold')
axes[0].grid(True,linestyle='--')

axes[1].pie(value,labels=label,shadow=True,autopct='%1.1f%%', colors=colors_from_palette,startangle=145,wedgeprops={"edgecolor":"white","alpha":0.7})
axes[1].set_title("Cancelled Rides by Driver Rate",fontsize=14,fontweight='bold')

plt.tight_layout()
plt.savefig("Cancelled Rides by Driver.png",dpi=300,bbox_inches='tight')
plt.show()
print("-"*100)

# Correlation Analysis;

numeric_columns = ["Booking Value", "Ride Distance", "Driver Ratings", "Customer Rating","Avg VTAT","Avg CTAT"]
df_numeric = df[numeric_columns].dropna()
correlation=df_numeric.corr()

plt.figure(figsize=(16,14))
mask=np.triu(np.ones_like(correlation,dtype=bool))

sns.heatmap(correlation,annot=True,fmt='.2f', 
            cmap='YlGnBu', center=0, square=True, linewidths=1,
            cbar_kws={"shrink": .8})
plt.title('Feature Correlation Matrix',fontsize=16,fontweight='bold',pad=20)
plt.tight_layout()
plt.savefig("Correlation Matrix.png",dpi=300,bbox_inches='tight')
plt.show()
print("-"*100)

# Multicollinearity Analysis;

high_corr_pairs=[]
for i in range(len(correlation.columns)):
    for j in range(i+1,len(correlation.columns)):
        if abs(correlation.iloc[i,j]>0.8):
            high_corr_pairs.append({'feature-1': correlation.columns[i], 'feature-2': correlation.columns[j], 'correlation': correlation.iloc[i, j]})

if high_corr_pairs:
    high_corr_df=pd.DataFrame(high_corr_pairs)
    print("\nHighly Correlated Feature Pairs (|correlation| > 0.8):")
    print(high_corr_df)
else:
    print("No highly correlated feature pairs found.")

print("-"*100)

# Outlier Analysis;

outlier={}
df_Numeric=df.select_dtypes(include=np.number).copy()

for col in df_Numeric.columns:
    Q1 = df_Numeric[col].quantile(0.25)
    Q3 = df_Numeric[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    outlier[col] = {
        'count': len(outliers),
        'percentage_outliers': (len(outliers) / len(df)) * 100
    }
outlier_df = pd.DataFrame(outlier).T
outlier_df = outlier_df.sort_values(by='count',ascending=False)

print("Outlier (IQR Method):")
print(outlier_df[outlier_df['count'] > 0])
print("-"*100)

# Outlier Analysis Visualization;

plt.figure(figsize=(12,6))
outlier_df[outlier_df['count'] > 0]['percentage_outliers'].plot(kind="bar",color='salmon',edgecolor='black',alpha=0.8)
plt.xlabel('Features', fontsize=12)
plt.ylabel('Percentage of Outliers', fontsize=12)
plt.title('Percentage of Outliers by Feature', fontsize=14, fontweight='bold')
plt.xticks(rotation=45,ha='right')
plt.grid(axis='y',alpha=0.75)
plt.tight_layout()
plt.savefig('Outlier_percentage.png', dpi=300, bbox_inches='tight')
plt.show()
print("-"*100)

# Summary Reports;

summary=(f"""
Data Analysis Reports;
1.Dataset Overview:
    Total Records:{df.shape[0]}
    Total Features:{df.shape[1]}
    No missing or duplicate values found.         
    Duplicate:{Duplicates}

Outliers Summary:
    - Features with %5>outliers: {len(outlier_df[outlier_df['percentage_outliers'] > 5])}
""")
with open('Data_analysis_summary.txt','w') as f:
    f.write(summary)

print("Files Saved")
print("- Univariate Analysis.png")
print("- Univariate Analysis2.png")
print("- Vehicle Type.png")
print("- Booking Status.png")
print("- Top and Least 10 Pickup and Drop Location Plots.png")
print("- Driver and Customer Ratings.png")
print("- Customer Ratings and Vehicle Type.png")
print("- Successful Rate.png")
print("- InComplete Reasons.png")
print("- Driver and Customer Points Relationship.png")
print("- Points and Other Feature Relationship.png")
print("- Cancelled Rides by Customer.png")
print("- Cancelled Rides by Driver.png")
print("- Correlation Matrix.png")
print("- Outlier_percentage.png")
print("\nReady for Modeling!")