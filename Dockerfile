FROM python:3.13-slim

# install build dependencies for basemap (GEOS, PROJ, compilers)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        libgeos-dev \
        libproj-dev \
        proj-data \
        proj-bin \
        cython3 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
# install python deps; this layer will be cached until requirements.txt changes
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:8000", "ui:app"]