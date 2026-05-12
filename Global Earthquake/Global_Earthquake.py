# Import necessary Libraries;
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.cm as cm
import matplotlib.colors as colors
import folium
from folium.plugins import MarkerCluster
from branca.colormap import LinearColormap
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# Load the dataset;
data = pd.read_csv('earthquake_data_tsunami.csv')

# First Look at the data;
print(data.head())
print(data.info())
print(data.describe().T)
print("-"*50)
print("Columns in dataset:", data.columns)
print("Shape of dataset:", data.shape)

# Organize the data;
data_Float64 = data.select_dtypes(include=['float64'])
data_int64 = data.select_dtypes(include=['int64'])

print("-"*50)

# Check for missing values;
if data.isnull().values.any():
    print(f"Missing values found in the dataset: {data.isnull().sum()}.")
else:
    print("No missing values in the dataset.")

# Check for duplicate rows;
if data.duplicated().any():
    print(f"Duplicate rows found: {data.duplicated().sum()}.")  
else:
    print("No duplicate rows in the dataset.")
print("-"*50)

# Check for unique values in each column;
for col in data.columns:
    unique_values = data[col].unique()
    print(f"Column '{col}' has {len(unique_values)} unique values. Number of unique values: {unique_values[:5]}...")
print("-"*50)

# Categorical Visualization;
fig,ax = plt.subplots(3,3,figsize=(12,6))
for i,col in enumerate(data_Float64):
    sns.histplot(data[col], ax=ax[i//3, i%3], kde=True)
    ax[i//3, i%3].set_title(col)
    ax[i//3, i%3].set_ylabel('Float64 Values')
plt.tight_layout()
plt.savefig('float64_distribution.png', dpi=300,bbox_inches='tight')
plt.show()

fig,ax = plt.subplots(3,3,figsize=(12,6))
for i,col in enumerate(data_int64):
    sns.histplot(data[col], ax=ax[i//3, i%3], kde=True)
    ax[i//3, i%3].set_title(col)
    ax[i//3, i%3].set_ylabel('Int64 Values')
plt.tight_layout()
plt.savefig('int64_distribution.png', dpi=300,bbox_inches='tight')
plt.show()

# Correlation Matrix;
plt.figure(figsize=(12, 8))
sns.heatmap(data.corr(),annot=True, fmt='.2f', cmap='coolwarm', square=True)
plt.title('Correlation Matrix')
plt.tight_layout()
plt.savefig('correlation_matrix.png', dpi=300,bbox_inches='tight')
plt.show()

# Do the given values have an effect on the formation of a tsunami?
Corelation_for_tsunami = data.corr()['tsunami'].sort_values(ascending=False)
print("Correlation with tsunami:",Corelation_for_tsunami)

Corelation_for_tsunami = Corelation_for_tsunami.drop('tsunami')
Corelation_for_tsunami.sort_values(ascending=False).plot(kind='bar',figsize=(10,6),color='skyblue',title='Correlation with Tsunami',ylabel='Correlation Coefficient',xlabel='Features')
plt.tight_layout()
plt.grid('y',linestyle='--')
plt.savefig('correlation_with_tsunami.png', dpi=300,bbox_inches='tight')
plt.show()

# Tsunami with Years;
data_Years = data[data['tsunami']==1]
data_Years['Year'] = pd.to_datetime(data_Years['Year'], format='%Y')
data_Years['Month'] = data_Years['Year'].dt.month
data_Years['Month'] = data_Years['Year'].dt.month_name()

fig,ax = plt.subplots(1,2,figsize=(12,8))
sns.countplot(x="Year",data=data,hue="tsunami",palette='Set3',ax=ax[0])
ax[0].set_title('Tsunami Occurrences Over the Years',fontweight='bold',fontsize=14)
ax[0].set_ylabel('All Years', fontsize=12)
ax[0].set_xticklabels(ax[0].get_xticklabels(), rotation=45)

Year_Tsunami_Mean = data.groupby('Year')['tsunami'].mean()*100
ax[1].plot(Year_Tsunami_Mean.index, Year_Tsunami_Mean.values, marker='o', color='darkred', linestyle='-', linewidth=2, markersize=6)
ax[1].set_title('Average Tsunami Occurrence Percentage Over the Years',fontweight='bold',fontsize=14)
ax[1].set_ylabel('Tsunami Occurrence (%)', fontsize=12)
ax[1].set_xlabel('Years', fontsize=12)
ax[1].axhline(Year_Tsunami_Mean.mean(),color='black',linestyle='--',linewidth=1,label=f"Overall Mean: {Year_Tsunami_Mean.mean():.2f}%")
ax[1].fill_between(Year_Tsunami_Mean.index, Year_Tsunami_Mean.values, alpha=0.15, color='black')
ax[1].grid('y',alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig('tsunami_over_years.png', dpi=300,bbox_inches='tight')
plt.show()


# Months with Magnitude of Tsunami;
Monthly_Tsunami_Size = data.groupby(['Month','tsunami']).size().unstack(fill_value=0)
Monthly_Tsunami_Size.plot(kind='line',marker='o',figsize=(10,6),title='Monthly Tsunami Size',ylabel='Tsunami Size',xlabel='Months',color=['blue','red'])
plt.grid('y',linestyle='--')
plt.xticks(ticks=range(1, 13), labels=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                            'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
plt.legend(['No Tsunami', 'Tsunami'],loc='upper right', fontsize=10)
plt.tight_layout()
plt.savefig('monthly_tsunami_size.png', dpi=300,bbox_inches='tight')
plt.show()

Monthly_Tsunami_Mean = data.groupby('Month')['tsunami'].mean()*100
plt.pie(Monthly_Tsunami_Mean.values, labels=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                            'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], autopct='%1.1f%%', startangle=140, colors=sns.color_palette('pastel'),shadow=True,explode=[0.05]*12)
plt.title('Tsunami Means by Month', fontsize=14, fontweight='bold')
plt.axis('equal')
plt.tight_layout()
plt.savefig('monthly_tsunami_mean.png', dpi=300,bbox_inches='tight')
plt.show()

# Magnitude Plot;
data['tsunami_category'] = data['tsunami'].apply(lambda x:'Tsunami' if x == 1 else 'No Tsunami')
plt.figure(figsize=(10,6))
sns.boxplot(x='tsunami_category', y='magnitude', data=data, palette=['lightcoral', 'skyblue'])
plt.title('Magnitude Distribution by Tsunami Occurrence', fontsize=14, fontweight='bold')
plt.xlabel('Tsunami Occurrence', fontsize=12)
plt.ylabel('Magnitude', fontsize=12)
plt.grid('y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('magnitude_by_tsunami.png', dpi=300,bbox_inches='tight')
plt.show()


# Magnitude Plot for tsunami;
def to_unit(lat_deg,lon_deg):
    lat,lon = np.radians(lat_deg),np.radians(lon_deg)
    x = np.cos(lat) * np.cos(lon)
    y = np.cos(lat) * np.sin(lon)
    z = np.sin(lat)
    return np.c_[x,y,z]
def unit_to_latlon(C):
    x,y,z = C[:,0],C[:,1],C[:,2]
    lat = np.degrees(np.arctan2(z, np.sqrt(x**2 + y**2)))
    lon = np.degrees(np.arctan2(y, x))
    return np.c_[lat, lon]

# -----------------

U = to_unit(data['latitude'].values, data['longitude'].values)
kmeans = KMeans(n_clusters=5, random_state=42,n_init='auto').fit(U)
data['cluster'] = kmeans.labels_
centers_location = unit_to_latlon(kmeans.cluster_centers_)

# Cluster Summary; 
cluster_summary = data.groupby('cluster').agg(
    Avg_Magnitude=('magnitude', 'mean'),
    Tsunami_Percentage=('tsunami', lambda x: x.mean() * 100),
    Count=('cluster', 'size'))


norm = plt.Normalize(data['magnitude'].min(),data['magnitude'].max())
clr_map = cm.ScalarMappable(norm=norm, cmap='coolwarm')

def mag_to_hex(m):
    return colors.rgb2hex(clr_map.to_rgba(m))


m = folium.Map(location=[0, 0], zoom_start=2, tiles='CartoDB Positron')
fg_quakes = folium.FeatureGroup(name='Magnitude of Earthquakes',show=True).add_to(m)
fg_centroids = folium.FeatureGroup(name='Cluster Centroids',show=True).add_to(m)


for _,row in data.iterrows():
    color = mag_to_hex(row['magnitude'])
    folium.CircleMarker(
        location=[row['latitude'],row['longitude']],
        radius=row['magnitude']*1.5,
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=0.7,
        popup=folium.Popup(f"Cluster: {row['cluster']} (Rate of Risk: {data[data['cluster'] == row['cluster']]['tsunami'].mean()*100:.1f}%)<br>" 
            f"Magnitude: {row['magnitude']:.2f}<br>"
            f"Depth: {row['depth']:.0f} km.", max_width=300)).add_to(fg_quakes)

for i,(lat,lon) in enumerate(centers_location):
    folium.CircleMarker(
        location=[float(lat), float(lon)],
        radius=15,
        color='black',
        weight=2.5,
        fill=True,
        fill_color="#ffd60a",
        fill_opacity=0.9,
        tooltip=f"Cluster {i} Centroid",
        popup=f"Cluster {i}: Mean Violence: {cluster_summary.loc[i, 'Avg_Magnitude']:.2f}, Rate of Tsunami: {cluster_summary.loc[i, 'Tsunami_Percentage']:.1f}%"
    ).add_to(fg_centroids)


color_list = [mag_to_hex(i) for i in np.linspace(data['magnitude'].min(), data['magnitude'].max(), 256)]
legend = LinearColormap(colors=color_list, vmin=data['magnitude'].min(), vmax=data['magnitude'].max(), caption='Earthquake Magnitude')
legend.add_to(m)
folium.LayerControl(collapsed=False).add_to(m)
m.save('earthquake_magnitude_map.html')


# Depth Plot for tsunami;
bins = [0, 70, 300, data['depth'].max() + 1]
labels = ['Shallow (0-70 km)','Intermediate (70-300 km)','Deep (>300 km)']

data['depth_category'] = pd.cut(data['depth'], bins=bins, labels=labels, right=False)
tsunami_depth = data.groupby('depth_category',observed=True)['tsunami'].mean().reset_index()
tsunami_depth['tsunami_percentage'] = tsunami_depth['tsunami'] * 100

plt.figure(figsize=(10,6))
sns.barplot(x='depth_category', y='tsunami_percentage', data=tsunami_depth, palette='viridis',errorbar=None,alpha=0.8)
plt.title('Tsunami Occurrence by Depth Category', fontsize=14, fontweight='bold')
plt.xlabel('Depth Category', fontsize=12)   
plt.ylabel('Tsunami Occurrence (%)', fontsize=12)
plt.ylim(0, tsunami_depth['tsunami_percentage'].max()*1.1)
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig('tsunami_by_depth_category.png', dpi=300,bbox_inches='tight')
plt.show()

tsunami = data[data['tsunami'] == 1]
no_tsunami = data[data['tsunami'] == 0]

fig,ax = plt.subplots(1,1,figsize=(12,6))
ax.scatter(tsunami['depth'], tsunami['magnitude'],alpha=0.6,s=60,label='Tsunami',color='darkred',linewidth=1)
ax.scatter(no_tsunami['depth'], no_tsunami['magnitude'],alpha=0.3,s=60,label='No Tsunami',color='darkblue',linewidth=1)
ax.set_title('Tsunami Occurrence by Depth Category', fontsize=14, fontweight='bold')
ax.set_xlabel('Depth (km)', fontsize=12)
ax.set_ylabel('Magnitude', fontsize=12)
ax.legend()
plt.tight_layout()
plt.savefig('tsunami_occurrence_by_depth.png', dpi=300,bbox_inches='tight')
plt.show()

# Dmin 
yearly_dmin = data.groupby('Year')['dmin'].agg(['mean', 'median', 'min', 'max']).reset_index()
data = pd.merge(data,yearly_dmin,on='Year',how='left')

fig,ax = plt.subplots(1,2,figsize=(12,6))
ax[0].plot(yearly_dmin['Year'], yearly_dmin['mean'], label='Mean dmin', color='blue',marker='o', markersize=4, linestyle='-', linewidth=1)
ax[0].plot(yearly_dmin['Year'], yearly_dmin['median'], label='Median dmin', color='orange',marker='s', markersize=4, linestyle='--', linewidth=1)
ax[1].plot(yearly_dmin['Year'], yearly_dmin['min'], label='Min dmin', color='green',marker='^', markersize=4, linestyle=':', linewidth=1)
ax[1].plot(yearly_dmin['Year'], yearly_dmin['max'], label='Max dmin', color='red',marker='v', markersize=4, linestyle='-.', linewidth=1)

ax[0].set_title('Yearly Mean and Median of dmin (Distance to Nearest Coastline)', fontsize=9, fontweight='bold')
ax[0].set_xlabel('Year',fontsize=12)
ax[0].set_ylabel('dmin Mean and Median (km)',fontsize=10)
ax[0].grid('y',linestyle='--',alpha=0.7)
ax[0].set_xticklabels(ax[0].get_xticklabels(), rotation=45)
ax[0].fill_between(yearly_dmin['Year'], yearly_dmin['mean'], yearly_dmin['median'], color='gray', alpha=0.2)

ax[1].set_title('Yearly Mean and Median of dmin (Distance to Nearest Coastline)', fontsize=9, fontweight='bold')
ax[1].set_xlabel('Year',fontsize=12)
ax[1].set_ylabel('dmin Min and Max (km)',fontsize=10)
ax[1].grid('y',linestyle='--',alpha=0.7)
ax[1].set_xticklabels(ax[1].get_xticklabels(), rotation=45)
ax[1].fill_between(yearly_dmin['Year'], yearly_dmin['min'], yearly_dmin['max'], color='gray', alpha=0.2)

plt.tight_layout()
plt.savefig('yearly_dmin.png', dpi=300,bbox_inches='tight')
plt.show()


# Model;


base_features = ['magnitude','depth','latitude','longitude','sig','dmin','gap']
X = data[base_features+['cluster']]
y = data['tsunami']

X = pd.get_dummies(X, columns=['cluster'], drop_first=True)


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)

model = RandomForestClassifier(n_estimators=150,max_depth=10, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

feature_importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
print("Feature Importances:\n", feature_importances)

