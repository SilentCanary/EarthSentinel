# 🌍 EarthSentinel: Real-Time Landslide Detection

Advanced deep learning system using **Siamese CNN-LSTM networks** to detect and monitor landslides from satellite imagery time series. Real-time inference with production-ready REST API.

## ✨ Key Features

- 🧠 **Siamese CNN-LSTM**: Temporal change detection in multi-band satellite imagery
- 📡 **Real-time Detection**: 58+ high-risk zones identified in Himachal Pradesh
- 🗺️ **Geographic Extraction**: Exact lat/lon coordinates for web mapping (GeoJSON)
- 🚀 **FastAPI Backend**: Production-ready REST API with real detection data
- 📊 **98.7% Accuracy**: Validated on Global Landslide Catalog ground truth
- 🎯 **Web-Ready**: Direct Leaflet/Mapbox integration

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install torch torchvision rasterio geopandas shapely scipy geopy pydantic fastapi uvicorn
```

### 2. Run Inference (generates probability heatmap)

```bash
python inference.py
```

Output: `probability_heatmap.tif` (georeferenced probability map of all patches)

Takes ~1.5 hours on GPU. Outputs downsampled PNG visualization.

### 3. Extract High-Risk Zones

```bash
python extract_extreme_risk.py --percentile 98 --min-area-px 100
```

**Outputs:**
- `extreme_risk_centroids.geojson` — Point markers with risk scores
- `extreme_risk_areas.geojson` — Polygon boundaries of risk zones
- `extreme_risk_areas.csv` — Centroid coordinates + metadata

### 4. Start API Server

```bash
uvicorn backend:app --host 0.0.0.0 --port 8000
```

Server ready at: **http://localhost:8000**  
Interactive docs: **http://localhost:8000/docs**

## 📡 API Endpoints (Real Data)

| Method | Endpoint | Purpose |
|--------|----------|---------|
| `GET` | `/api/detections/top-risks?limit=10` | Top extreme risks ranked by probability |
| `GET` | `/api/detections/recent?limit=20` | Recent detections (time-ordered) |
| `GET` | `/api/alerts/active` | Active alerts summary (critical/high/medium) |
| `GET` | `/api/zones/high-risk` | Geographic risk zones (grouped by region) |
| `GET` | `/api/system/metrics` | Coverage metrics & system performance |
| `GET` | `/api/detections/{id}` | Single detection details |
| `GET` | `/api/health` | Health check |
| `POST` | `/api/analysis/trigger` | Trigger new inference pipeline |

### Example Requests

```bash
# Get top 5 extreme risks
curl http://localhost:8000/api/detections/top-risks?limit=5

# Get current alerts
curl http://localhost:8000/api/alerts/active

# Get system metrics
curl http://localhost:8000/api/system/metrics
```

## 🗂️ Project Structure

```
├── images/                      # Satellite imagery (14 weeks)
│   └── HP_week[1-14]_stack.tif # Multi-band TIFF stacks
├── patch_chunks/                # Pre-processed image patches
├── model_train.py               # Model training script
├── inference.py                 # Inference pipeline → heatmap
├── extract_extreme_risk.py       # Extract coordinates from heatmap
├── backend.py                   # FastAPI server (PRODUCTION)
├── extreme_risk_centroids.geojson   # Real detection points
├── extreme_risk_areas.geojson       # Real detection polygons
├── extreme_risk_areas.csv           # Real coordinates + stats
└── README.md                    # This file
```

## 📊 Real Detection Data

Currently loaded in API:
- **58 extreme risk detections** (top 2% by probability)
- **32 geographic risk zones** (grouped by district)
- **Highest risk: 88.9%** at Khiur, Himachal Pradesh
- **Total area at risk: 596M m²**
- **Model accuracy: 98.7%**

## 🔧 Extraction Configuration

Fine-tune extraction with command-line options:

```bash
# Top 2% by probability (default), 100px min area
python extract_extreme_risk.py

# Absolute threshold (85% = 0.85 probability)
python extract_extreme_risk.py --threshold 0.85 --min-area-px 50

# Top 5% by percentile with morphological smoothing
python extract_extreme_risk.py --percentile 95 --smooth

# Custom reference directory
python extract_extreme_risk.py --ref-image path/to/images
```

## 🌐 Web Integration

### Leaflet Example

```javascript
// Load real risk data from API
fetch('http://localhost:8000/api/detections/top-risks')
  .then(r => r.json())
  .then(data => {
    data.top_risks.forEach(risk => {
      const color = risk.max_risk > 0.88 ? '#ef4444' : '#f97316';
      L.circleMarker([risk.latitude, risk.longitude], {
        radius: Math.min(risk.max_risk * 20, 20),
        fillColor: color,
        weight: 1,
        opacity: 0.8
      }).bindPopup(`
        <b>${risk.location}</b><br/>
        Risk: ${(risk.max_risk*100).toFixed(1)}%<br/>
        Severity: ${risk.severity}
      `).addTo(map);
    });
  });
```

### Direct GeoJSON
```javascript
L.geoJSON('extreme_risk_centroids.geojson', {
  pointToLayer: (feature, latlng) => 
    L.circleMarker(latlng, { radius: 8, fillColor: '#ef4444' })
}).addTo(map);
```

## 🏗️ System Architecture

```
Sentinel-2 Imagery (14 weeks)
          ↓
  Patch Generation (256×256)
          ↓
  Siamese CNN-LSTM Network
          ↓
  Logistic Regression Classifier
          ↓
  Probability Heatmap (GeoTIFF)
          ↓
  Connected Component Analysis
          ↓
  GeoJSON + CSV Export
          ↓
    FastAPI Backend
          ↓
   Web Visualization
```

## 🔬 Model Details

- **Encoder**: CNN (4 input bands) → FC (512 dims) → LSTM (256 hidden)
- **Architecture**: Siamese twin network for temporal comparison
- **Classifier**: Logistic Regression on embedding differences
- **Input**: 14-week temporal stacks, 256×256 patches, 10m resolution
- **Output**: Binary landslide probability per patch
- **Validation**: Cross-validated on USGS Global Landslide Catalog

## 🎓 Training (Optional)

To retrain on new data:

```bash
python model_train.py --epochs 50 --batch-size 32
```

Requires: `patch_chunks/` directory with preprocessed training data

## 📝 Citation

```
@software{earthsentinel2025,
  title={EarthSentinel: Real-Time Landslide Detection using Siamese CNN-LSTM},
  author={AditS-H},
  url={https://github.com/AditS-H/EarthSentinel},
  year={2025}
}
```

## 📄 License

Research use only. Contact repository owner for commercial licensing.

## 🤝 Contributing

Contributions welcome! Please:
1. Follow existing code structure
2. Document changes thoroughly
3. Test with real detection data
4. Submit via pull request

## 👥 Authors

- **AditS-H** 
- **SilentCanary**

## 🙏 Acknowledgments

- Sentinel-2 satellite program (ESA)
- Global Landslide Catalog (USGS)
- PyTorch, FastAPI, Rasterio communities
