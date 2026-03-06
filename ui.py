import io
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import geopandas as gpd
from flask import Flask, Response, render_template_string
import time
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans

app = Flask(__name__)
DATA_PATH = os.path.join(os.path.dirname(__file__), 'occurence.csv')

HTML_TEMPLATE = '''
<!doctype html>
<html>
<head>

<title>Panthera Tigris Occurrence Analysis</title>

<style>

body{
    font-family: Arial, sans-serif;
    margin:40px;
    background:#f5f5f5;
}

h1{
    text-align:center;
}

.section{
    margin-top:40px;
}

.map-container{
    text-align:center;
}

.map-container img{
    width:100%;
    max-width:1000px;
}

.grid{
    display:grid;
    grid-template-columns: 1fr 1fr 1fr;
    gap:20px;
    margin-top:20px;
}

.card{
    background:white;
    padding:15px;
    border-radius:8px;
    box-shadow:0px 2px 6px rgba(0,0,0,0.1);
    text-align:center;
    min-height:380px;
}

.card img{
    width:100%;
    height:320px;
    object-fit:contain;
    background:#eee;
}

</style>

</head>

<body>

<h1>Panthera Tigris Occurrence Analysis</h1>

<div class="section map-container">
    <h2>Occurrence Map</h2>
    <img src="/map.png?{{ts}}">
</div>

<div class="section">

<h2>Exploratory Analysis</h2>

<div class="grid">

<div class="card">
<h3>Monthly Occurrence</h3>
<img src="/month.png?{{ts}}" loading="lazy">
</div>

<div class="card">
<h3>Sex Distribution</h3>
<img src="/bar.png?{{ts}}" loading="lazy">
</div>

<div class="card">
<h3>Occurrence Heatmap</h3>
<img src="/heatmap.png?{{ts}}" loading="lazy">
</div>

</div>

</div>

</body>
</html>
'''

def generate_predictions(df):
    df = df.copy()
    # ensure we have a third category for missing sex values
    df['sex'] = df['sex'].fillna('Not specified')
    df['decimalLatitude'] = df['decimalLatitude'].ffill()
    df['decimalLongitude'] = df['decimalLongitude'].ffill()
    df_clean = df.dropna(subset=['year', 'decimalLatitude', 'decimalLongitude']).copy()

    n_points = len(df_clean)
    n_clusters = min(6, max(2, n_points // 20)) if n_points >= 10 else 1

    predicted_locations = []
    if n_clusters <= 1:
        future_years = np.arange(df_clean['year'].max() + 1, df_clean['year'].max() + 11)
        hist_points = df_clean[['decimalLatitude', 'decimalLongitude']]
        for year in future_years:
            lat, lon = hist_points.sample(n=1, replace=True).iloc[0]
            predicted_locations.append((int(year), float(lat), float(lon)))
    else:
        coords = df_clean[['decimalLatitude', 'decimalLongitude']].astype(float).values
        kmeans = KMeans(n_clusters=n_clusters, random_state=0)
        labels = kmeans.fit_predict(coords)
        df_clean['cluster'] = labels

        counts = df_clean.groupby(['year', 'cluster']).size().unstack(fill_value=0)

        cluster_models = {}
        years = counts.index.values.reshape(-1, 1)
        for cluster in counts.columns:
            y_counts = counts[cluster].values.astype(float)
            lr = LinearRegression()
            try:
                lr.fit(years, y_counts)
            except Exception:
                lr.coef_ = np.array([0.0])
                lr.intercept_ = float(np.mean(y_counts))
            cluster_models[cluster] = lr

        future_years = np.arange(df_clean['year'].max() + 1, df_clean['year'].max() + 11)
        for fy in future_years:
            preds = np.array([cluster_models[c].predict(np.array([[fy]]))[0] for c in sorted(cluster_models.keys())])
            preds = np.clip(preds, 0, None)
            total = preds.sum()
            if total <= 0:
                freqs = df_clean['cluster'].value_counts().sort_index().values.astype(float)
                probs = freqs / freqs.sum()
            else:
                probs = preds / total

            chosen_cluster = np.random.choice(sorted(cluster_models.keys()), p=probs)
            cluster_points = df_clean[df_clean['cluster'] == chosen_cluster][['decimalLatitude', 'decimalLongitude']]
            if len(cluster_points) > 0:
                lat, lon = cluster_points.sample(n=1, replace=True).iloc[0]
            else:
                centroid = kmeans.cluster_centers_[chosen_cluster]
                lat, lon = centroid[0], centroid[1]
            predicted_locations.append((int(fy), float(lat), float(lon)))

    return df, predicted_locations


def render_map_png(df, predicted_locations):
    fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)

    try:
        # load the shapefile for India boundary
        shapefile_path = os.path.join(os.path.dirname(__file__), 'India_State_Boundary.shp')
        india = gpd.read_file(shapefile_path)
        use_shapefile = True
    except Exception as e:
        print(f"Shapefile not available or incomplete: {e}. Falling back to Basemap.")
        use_shapefile = False

    if use_shapefile:
        # create GeoDataFrame for historical points
        gdf_hist = gpd.GeoDataFrame(
            df,
            geometry=gpd.points_from_xy(df.decimalLongitude, df.decimalLatitude),
            crs="EPSG:4326"
        )

        # plot the India boundary
        india.plot(ax=ax, color="lightgreen", edgecolor="black")

        # plot historical data by sex
        male = gdf_hist[gdf_hist['sex'] == 'Male']
        female = gdf_hist[gdf_hist['sex'] == 'Female']
        unspecified = gdf_hist[gdf_hist['sex'] == 'Not specified']
        count_male = len(male)
        count_female = len(female)
        count_unspec = len(unspecified)

        male.plot(ax=ax, color='red', marker='o', label=f'Male ({count_male})')
        female.plot(ax=ax, color='white', edgecolor='black', marker='o', label=f'Female ({count_female})')
        unspecified.plot(ax=ax, color='blue', marker='o', label=f'Unknown ({count_unspec})')

        # plot predicted locations
        if predicted_locations:
            pred_df = pd.DataFrame(predicted_locations, columns=['year', 'lat', 'lon'])
            gdf_pred = gpd.GeoDataFrame(
                pred_df,
                geometry=gpd.points_from_xy(pred_df.lon, pred_df.lat),
                crs="EPSG:4326"
            )
            cmap = plt.get_cmap('tab10')
            n_pred = len(gdf_pred)
            colors = [cmap(v) for v in np.linspace(0, 1, n_pred)] if n_pred > 0 else []
            for idx, row in gdf_pred.iterrows():
                gdf_pred.iloc[[idx]].plot(ax=ax, marker='x', color=colors[idx], label=f'Pred {int(row.year)}')

        # add legend after plotting all points, place it to the right
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
        plt.title('Panthera Tigris Occurrence in India with Predictions')
        # plt.tight_layout()
    else:
        # fallback to Basemap
        from mpl_toolkits.basemap import Basemap
        m = Basemap(projection='merc', llcrnrlat=6, urcrnrlat=37, llcrnrlon=68, urcrnrlon=97, resolution='i')
        m.drawcoastlines()
        m.drawcountries()
        m.drawmapboundary(fill_color='lightblue')
        m.fillcontinents(color='lightgreen', lake_color='lightblue')

        # historical by sex categories
        male = df[df['sex'] == 'Male']
        female = df[df['sex'] == 'Female']
        unspecified = df[df['sex'] == 'Not specified']
        count_male = len(male)
        count_female = len(female)
        count_unspec = len(unspecified)
        # convert coordinates
        x_male, y_male = m(male['decimalLongitude'].values, male['decimalLatitude'].values)
        x_female, y_female = m(female['decimalLongitude'].values, female['decimalLatitude'].values)
        x_unspec, y_unspec = m(unspecified['decimalLongitude'].values, unspecified['decimalLatitude'].values)
        m.scatter(x_male, y_male, marker='o', color='red',
                  label=f'Male ({count_male})', zorder=5)
        m.scatter(x_female, y_female, marker='o', color='white', edgecolor='black',
                  label=f'Female ({count_female})', zorder=5)
        m.scatter(x_unspec, y_unspec, marker='o', color='blue',
                  label=f'Unknown ({count_unspec})', zorder=5)

        # predicted with colors per year
        cmap = plt.get_cmap('tab10')
        n_pred = len(predicted_locations)
        colors = [cmap(v) for v in np.linspace(0, 1, n_pred)] if n_pred > 0 else []
        for idx, (year, lat, lon) in enumerate(predicted_locations):
            x_p, y_p = m(lon, lat)
            m.scatter(x_p, y_p, marker='x', color=colors[idx], label=f'Pred {year}', zorder=6)
        # add legend after plotting all points, place it to the right
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def render_bar_chart_png(df):
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    # Ensure order: Male, Female, Not specified
    order = ['Male', 'Female', 'Not specified']
    sex_counts = df['sex'].value_counts().reindex(order, fill_value=0)
    colors = ['red', 'white', 'blue']
    sex_counts.plot(kind='bar', ax=ax, color=colors, edgecolor='black')
    ax.set_title('Distribution of Animal Sex')
    ax.set_xlabel('Sex')
    ax.set_ylabel('Count')
    # plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def render_heatmap_png(df):

    fig, ax = plt.subplots(figsize=(8,6), constrained_layout=True)

    df['decimalLatitude'] = pd.to_numeric(df['decimalLatitude'], errors='coerce')
    df['decimalLongitude'] = pd.to_numeric(df['decimalLongitude'], errors='coerce')

    df = df.dropna(subset=['decimalLatitude','decimalLongitude'])

    h, xedges, yedges, image = ax.hist2d(
        df['decimalLongitude'],
        df['decimalLatitude'],
        bins=50,
        cmap='viridis'
    )

    ax.set_title("Occurrence Density Heatmap")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    fig.colorbar(image, ax=ax)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)

    buf.seek(0)
    return buf.getvalue()

def render_month_bar_png(df):

    fig, ax = plt.subplots(figsize=(8,6), constrained_layout=True)

    # Convert eventDate to datetime
    df['eventDate'] = pd.to_datetime(df['eventDate'], errors='coerce')

    # Extract month
    df['month'] = df['eventDate'].dt.month

    # Count occurrences per month
    month_counts = df['month'].value_counts().sort_index()

    # Month labels
    month_names = [
        "Jan","Feb","Mar","Apr","May","Jun",
        "Jul","Aug","Sep","Oct","Nov","Dec"
    ]

    month_counts = month_counts.reindex(range(1,13), fill_value=0)

    ax.bar(month_names, month_counts.values)

    ax.set_title("Tiger Occurrence Frequency by Month")
    ax.set_xlabel("Month")
    ax.set_ylabel("Frequency")

    # plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close(fig)

    buf.seek(0)
    return buf.getvalue()

@app.route('/month.png')
def month_png():
    df = pd.read_csv(DATA_PATH)
    img = render_month_bar_png(df)
    response = Response(img, mimetype='image/png')
    response.headers['Cache-Control'] = 'no-store'
    return response

@app.route('/bar.png')
def bar_png():
    df = pd.read_csv(DATA_PATH)
    df['sex'] = df['sex'].fillna('Not specified')
    img = render_bar_chart_png(df)
    response = Response(img, mimetype='image/png')
    response.headers['Cache-Control'] = 'no-store'
    return response


@app.route('/heatmap.png')
def heatmap_png():
    df = pd.read_csv(DATA_PATH)
    df['sex'] = df['sex'].fillna('Not specified')
    img = render_heatmap_png(df)
    response = Response(img, mimetype='image/png')
    response.headers['Cache-Control'] = 'no-store'
    return response

@app.route('/map.png')
def map_png():
    df = pd.read_csv(DATA_PATH)
    df, predicted_locations = generate_predictions(df)
    img = render_map_png(df, predicted_locations)
    response = Response(img, mimetype='image/png')
    response.headers['Cache-Control'] = 'no-store'
    return response

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE,ts=time.time())


if __name__ == '__main__':
    # run the Flask development server
    app.run(host='0.0.0.0', port=5001, debug=True)
