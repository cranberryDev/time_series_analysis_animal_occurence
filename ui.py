import io
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flask import Flask, Response, render_template_string

# Try to import Basemap; if not available we'll fallback to simple scatter
try:
    from mpl_toolkits.basemap import Basemap
    HAS_BASEMAP = True
except Exception:
    HAS_BASEMAP = False

from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans

app = Flask(__name__)
DATA_PATH = os.path.join(os.path.dirname(__file__), 'occurence.csv')

HTML_TEMPLATE = '''
<!doctype html>
<html>
  <head><title>Animal Occurrence Map</title></head>
  <body>
    <h1>Animal Occurrence Map (Predictions)</h1>
    <img src="/map.png" alt="map">
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
    fig = plt.figure(figsize=(10, 8))

    if HAS_BASEMAP:
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
        m.scatter(x_unspec, y_unspec, marker='o', color='grey',
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
    else:
        ax = fig.add_subplot(1, 1, 1)
        # historical by sex categories
        male = df[df['sex'] == 'Male']
        female = df[df['sex'] == 'Female']
        unspecified = df[df['sex'] == 'Not specified']
        count_male = len(male)
        count_female = len(female)
        count_unspec = len(unspecified)
        ax.scatter(male['decimalLongitude'], male['decimalLatitude'], c='red', s=10,
                   label=f'Male ({count_male})')
        ax.scatter(female['decimalLongitude'], female['decimalLatitude'], c='white', edgecolors='black', s=10,
                   label=f'Female ({count_female})')
        ax.scatter(unspecified['decimalLongitude'], unspecified['decimalLatitude'], c='grey', s=10,
                   label=f'Unknown ({count_unspec})')
        cmap = plt.get_cmap('tab10')
        n_pred = len(predicted_locations)
        colors = [cmap(v) for v in np.linspace(0, 1, n_pred)] if n_pred > 0 else []
        for idx, (year, lat, lon) in enumerate(predicted_locations):
            ax.scatter(lon, lat, marker='x', color=colors[idx], label=f'Pred {year}')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_title('Animal Occurrence (Historical + Predicted)')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route('/map.png')
def map_png():
    df = pd.read_csv(DATA_PATH)
    df_prepped, predicted_locations = generate_predictions(df)
    img = render_map_png(df_prepped, predicted_locations)
    return Response(img, mimetype='image/png')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
