import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap  
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
# import uvicorn

df2 = pd.read_csv('/Users/tanujpant/Documents/machine_learning_poc/time_series_analysis_animal_occurence/occurence.csv')
print(df2['sex'])

#sort year in ascending order
df2 = df2.sort_values(by='year', ascending=True)
print(df2['year'])

#modify NaN values for sex: mark as 'Not specified' rather than guessing
# this preserves a third category that we can color differently
# any existing non-null entries remain unchanged
df2['sex'] = df2['sex'].fillna('Not specified')
print(df2['sex'])

# Prepare the data for linear regression
# if decimalLatitude or decimalLongitude is NaN, forward-fill using Series.ffill()
df2['decimalLatitude'] = df2['decimalLatitude'].ffill() 
df2['decimalLongitude'] = df2['decimalLongitude'].ffill()

df_clean = df2.dropna(subset=['year', 'decimalLatitude', 'decimalLongitude']).copy()

# choose a reasonable number of clusters based on data size
n_points = len(df_clean)
n_clusters = min(6, max(2, n_points // 20)) if n_points >= 10 else 1

# if too few points fall back to using historical sampling
predicted_locations = []
if n_clusters <= 1:
    # not enough data to cluster; sample historical points for future years
    future_years = np.arange(df_clean['year'].max() + 1, df_clean['year'].max() + 11)
    hist_points = df_clean[['decimalLatitude', 'decimalLongitude']]
    for year in future_years:
        lat, lon = hist_points.sample(n=1, replace=True).iloc[0]
        predicted_locations.append((year, float(lat), float(lon)))
else:
    coords = df_clean[['decimalLatitude', 'decimalLongitude']].astype(float).values
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    labels = kmeans.fit_predict(coords)
    df_clean['cluster'] = labels

    # build yearly counts per cluster
    counts = df_clean.groupby(['year', 'cluster']).size().unstack(fill_value=0)

    # fit a simple linear model per cluster to forecast counts
    cluster_models = {}
    years = counts.index.values.reshape(-1, 1)
    for cluster in counts.columns:
        y_counts = counts[cluster].values.astype(float)
        lr = LinearRegression()
        try:
            lr.fit(years, y_counts)
        except Exception:
            # fallback to mean if fit fails
            lr.coef_ = np.array([0.0])
            lr.intercept_ = float(np.mean(y_counts))
        cluster_models[cluster] = lr

    # predict cluster probabilities for each future year and sample locations accordingly
    future_years = np.arange(df_clean['year'].max() + 1, df_clean['year'].max() + 11)
    for fy in future_years:
        # predict counts for each cluster
        preds = np.array([cluster_models[c].predict(np.array([[fy]]))[0] for c in sorted(cluster_models.keys())])
        preds = np.clip(preds, 0, None)
        total = preds.sum()
        if total <= 0:
            # fallback to historical cluster frequencies
            freqs = df_clean['cluster'].value_counts().sort_index().values.astype(float)
            probs = freqs / freqs.sum()
        else:
            probs = preds / total

        # choose a cluster based on predicted probabilities
        chosen_cluster = np.random.choice(sorted(cluster_models.keys()), p=probs)
        cluster_points = df_clean[df_clean['cluster'] == chosen_cluster][['decimalLatitude', 'decimalLongitude']]
        if len(cluster_points) > 0:
            lat, lon = cluster_points.sample(n=1, replace=True).iloc[0]
        else:
            # fallback to cluster centroid
            centroid = kmeans.cluster_centers_[chosen_cluster]
            lat, lon = centroid[0], centroid[1]
        predicted_locations.append((fy, float(lat), float(lon)))

# Print predicted locations
for year, lat, lon in predicted_locations:
    print(f"Predicted for {year}: latitude={lat}, longitude={lon}")

#show the predicted occurence of animals on the map for the future years
plt.figure(figsize=(10, 8))
m = Basemap(projection='merc', llcrnrlat=6, urcrnrlat=37, llcrnrlon=68, urcrnrlon=97, resolution='i')
m.drawcoastlines()
m.drawcountries()
m.drawmapboundary(fill_color='lightblue')
m.fillcontinents(color='lightgreen', lake_color='lightblue')
# Plot historical data by sex: red for male, white for female, grey for unspecified
male = df2[df2['sex'] == 'Male']
female = df2[df2['sex'] == 'Female']
unspecified = df2[df2['sex'] == 'Not specified']
# compute counts for legend
count_male = len(male)
count_female = len(female)
count_unspec = len(unspecified)
# convert coordinates for each group
x_male, y_male = m(male['decimalLongitude'].values, male['decimalLatitude'].values)
x_female, y_female = m(female['decimalLongitude'].values, female['decimalLatitude'].values)
x_unspec, y_unspec = m(unspecified['decimalLongitude'].values, unspecified['decimalLatitude'].values)
# scatter male points in red
m.scatter(x_male, y_male, marker='o', color='red', label=f'Male ({count_male})', zorder=5)
# scatter female points in white with black edge so they are visible
m.scatter(x_female, y_female, marker='o', color='white', edgecolor='black', label=f'Female ({count_female})', zorder=5)
# scatter unspecified in grey
m.scatter(x_unspec, y_unspec, marker='o', color='grey', label=f'Unknown Sex ({count_unspec})', zorder=5)
# Plot predicted data for future years using predicted longitude and latitude
# assign a distinct color for each predicted year
cmap = plt.get_cmap('tab10')
n_pred = len(predicted_locations)
colors = [cmap(v) for v in np.linspace(0, 1, n_pred)] if n_pred > 0 else []
for idx, (year, lat, lon) in enumerate(predicted_locations):
    x_pred, y_pred = m(lon, lat)
    m.scatter(x_pred, y_pred, marker='x', color=colors[idx], label=f'Predicted Occurrence {year}', zorder=5)
plt.title('Animal Occurrence in India with Predictions')
# place legend to the right of the map
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

if __name__ == "__main__":
    # uvicorn.run("main:app", host="8000", reload=True)
    print("Data has been successfully converted from Excel to CSV format.")