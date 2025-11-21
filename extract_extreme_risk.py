"""extract_extreme_risk.py

Extract extreme high-risk (red) areas from probability heatmap TIFF
and export as GeoJSON with coordinates for web display.

This version uses the reference image's geotransform for accurate lat/lon conversion.

Outputs:
- extreme_risk_areas.geojson (polygons in EPSG:4326)
- extreme_risk_centroids.geojson (point centroids in EPSG:4326)
- extreme_risk_areas.csv (rows with id, lon, lat, area_px, mean_prob, max_prob)

Usage:
python extract_extreme_risk.py --threshold 0.85 --min-area-px 100
python extract_extreme_risk.py --percentile 98 --min-area-px 100
"""

import argparse
import numpy as np
import rasterio
from rasterio import features
from shapely.geometry import shape
import geopandas as gpd
from scipy import ndimage
from rasterio.warp import transform
import csv
import os


def main():
    p = argparse.ArgumentParser(description="Extract extreme high-risk regions from probability heatmap")
    p.add_argument('--input', default='probability_heatmap.tif', help='Input heatmap TIFF')
    p.add_argument('--ref-image', default='images', help='Directory with reference images to get geotransform')
    p.add_argument('--threshold', type=float, default=None, help='Absolute threshold (0-1)')
    p.add_argument('--percentile', type=float, default=98, help='Use percentile (default 98 for top 2%)')
    p.add_argument('--min-area-px', type=int, default=50, help='Minimum region size in pixels')
    p.add_argument('--smooth', action='store_true', help='Apply morphological opening')
    args = p.parse_args()

    # Load heatmap
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"{args.input} not found. Run inference.py first.")
    
    with rasterio.open(args.input) as src:
        heatmap = src.read(1).astype(np.float32)
        heatmap_transform = src.transform
        heatmap_crs = src.crs
    
    print(f"Loaded heatmap: {heatmap.shape}")
    print(f"Heatmap CRS: {heatmap_crs}")
    print(f"Heatmap transform: {heatmap_transform}")
    
    # Get reference transform from first image in reference directory
    ref_dir = args.ref_image
    tif_files = sorted([f for f in os.listdir(ref_dir) if f.endswith('.tif')])
    if not tif_files:
        raise FileNotFoundError(f"No .tif files in {ref_dir}")
    
    with rasterio.open(os.path.join(ref_dir, tif_files[0])) as src:
        ref_transform = src.transform
        ref_crs = src.crs
        print(f"Reference CRS: {ref_crs}")
        print(f"Reference transform: {ref_transform}")
    
    # Determine threshold
    if args.threshold is None:
        thr = np.percentile(heatmap[~np.isnan(heatmap)], args.percentile)
        print(f"Using percentile {args.percentile} -> threshold {thr:.4f}")
    else:
        thr = args.threshold
        print(f"Using absolute threshold {thr:.4f}")
    
    # Extract high-risk regions
    mask = (heatmap >= thr).astype(np.uint8)
    
    if args.smooth:
        mask = ndimage.binary_opening(mask, structure=np.ones((3,3))).astype(np.uint8)
    
    labeled, ncomponents = ndimage.label(mask)
    print(f"Found {ncomponents} connected regions above threshold")
    
    records = []
    for lab in range(1, ncomponents + 1):
        comp = (labeled == lab)
        area_px = comp.sum()
        if area_px < args.min_area_px:
            continue
        
        # Extract geometry using rasterio.features.shapes
        shapes_list = list(features.shapes(comp.astype(np.uint8), mask=comp, transform=ref_transform))
        if not shapes_list:
            continue
        
        polys = [shape(geom) for geom, val in shapes_list]
        from shapely.ops import unary_union
        geom = unary_union(polys) if len(polys) > 1 else polys[0]
        
        # Get stats
        comp_vals = heatmap[comp]
        centroid = geom.centroid
        
        rec = {
            'id': len(records) + 1,
            'geometry': geom,
            'centroid_lon': centroid.x,
            'centroid_lat': centroid.y,
            'area_px': int(area_px),
            'mean_prob': float(np.mean(comp_vals)),
            'max_prob': float(np.max(comp_vals)),
            'min_prob': float(np.min(comp_vals))
        }
        records.append(rec)
    
    if len(records) == 0:
        print(f"No regions found with threshold {thr:.4f} and min area {args.min_area_px} px")
        print("Try lowering threshold or --percentile, or reducing --min-area-px")
        return
    
    print(f"Found {len(records)} candidate high-risk regions")
    
    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame([r for r in records], crs=ref_crs)
    
    # Save polygons
    gdf_4326 = gdf.to_crs(epsg=4326)
    gdf_4326.to_file('extreme_risk_areas.geojson', driver='GeoJSON')
    print("Saved: extreme_risk_areas.geojson")
    
    # Save centroids
    gdf_cent = gdf_4326.copy()
    gdf_cent['geometry'] = gdf_cent.geometry.centroid
    gdf_cent.to_file('extreme_risk_centroids.geojson', driver='GeoJSON')
    print("Saved: extreme_risk_centroids.geojson")
    
    # Save CSV
    with open('extreme_risk_areas.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'centroid_lon', 'centroid_lat', 'area_px', 'mean_prob', 'max_prob', 'min_prob'])
        for r in records:
            cent = r['geometry'].centroid
            writer.writerow([r['id'], cent.x, cent.y, r['area_px'], r['mean_prob'], r['max_prob'], r['min_prob']])
    print("Saved: extreme_risk_areas.csv")
    
    # Print top 10
    records_sorted = sorted(records, key=lambda r: r['max_prob'], reverse=True)
    print("\n=== Top 10 extreme risk regions ===")
    for r in records_sorted[:10]:
        cent = r['geometry'].centroid
        print(f"ID {r['id']:3d}: max_prob={r['max_prob']:.4f} "
              f"mean={r['mean_prob']:.4f} area_px={r['area_px']:6d} "
              f"lat={cent.y:.6f} lon={cent.x:.6f}")


if __name__ == '__main__':
    main()
