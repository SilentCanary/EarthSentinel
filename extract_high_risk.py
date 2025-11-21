"""extract_high_risk.py

Find extreme high-risk ("red") areas from a georeferenced probability heatmap
and export their geometries and centroids (in lat/lon) so they can be displayed
on a website.

Outputs:
- high_risk_areas.geojson  (polygons)
- high_risk_centroids.geojson  (points with metadata)
- high_risk_areas.csv (centroid lat/lon + stats)

Usage:
python extract_high_risk.py --input probability_heatmap.tif --threshold 0.8 --min-area 2500

If you prefer a percentile-based threshold (top X%), use --percentile instead of --threshold.

Dependencies: rasterio, numpy, geopandas, shapely, scipy
Install (if needed): pip install rasterio geopandas shapely scipy
"""

import argparse
import numpy as np
import rasterio
from rasterio import features
from shapely.geometry import shape, mapping
import geopandas as gpd
from scipy import ndimage
from rasterio.warp import transform
import csv
import os


def extract_regions_from_raster(arr, transform_affine, threshold, min_area_pixels=1, smooth=False):
    """Return list of (mask, bbox, stats) for connected regions above threshold.

    arr: 2D numpy array of risk probabilities
    transform_affine: rasterio transform
    threshold: value between 0-1
    min_area_pixels: minimum area in pixels to keep
    smooth: apply small morphological opening to remove speckles
    """
    mask = (arr >= threshold).astype(np.uint8)

    if smooth:
        # remove small objects / noise
        mask = ndimage.binary_opening(mask, structure=np.ones((3,3))).astype(np.uint8)

    labeled, ncomponents = ndimage.label(mask)

    results = []
    for lab in range(1, ncomponents+1):
        comp = (labeled == lab)
        area_px = comp.sum()
        if area_px < min_area_pixels:
            continue
        # compute bounding box in pixel coordinates (row, col)
        rows, cols = np.where(comp)
        rmin, rmax = rows.min(), rows.max()
        cmin, cmax = cols.min(), cols.max()

        # extract polygon using rasterio.features.shapes on the component mask
        shapes = list(features.shapes(comp.astype(np.uint8), mask=comp, transform=transform_affine))
        # shapes is list of (geom, value) where value==1; merge them into a single MultiPolygon/Polygon
        polys = [shape(geom) for geom, val in shapes]
        # union if multiple
        geom = polys[0]
        if len(polys) > 1:
            from shapely.ops import unary_union
            geom = unary_union(polys)

        # compute stats inside component
        comp_vals = arr[comp]
        stats = {
            'area_px': int(area_px),
            'mean_prob': float(np.mean(comp_vals)),
            'max_prob': float(np.max(comp_vals)),
            'min_prob': float(np.min(comp_vals))
        }

        results.append({'geometry': geom, 'stats': stats})

    return results


def pixel_to_lonlat(transform_affine, xs, ys, src_crs):
    """Convert arrays of pixel coords (col, row) to lon/lat using transform and src_crs.
    xs, ys are column and row indices (0-based). Return lon, lat arrays in EPSG:4326.
    """
    # transform: maps (col, row) to x,y in src_crs
    xs_geo = []
    ys_geo = []
    for c, r in zip(xs, ys):
        x, y = transform_affine * (c, r)
        xs_geo.append(x)
        ys_geo.append(y)
    # transform to 4326
    lon, lat = transform(src_crs, {'init': 'epsg:4326'} if isinstance(src_crs, str) else 'epsg:4326', xs_geo, ys_geo)
    return lon, lat


def main():
    p = argparse.ArgumentParser(description="Extract high-risk regions from a probability heatmap TIFF")
    p.add_argument('--input', default='probability_heatmap.tif', help='Input georeferenced heatmap TIFF')
    p.add_argument('--threshold', type=float, default=None, help='Absolute threshold between 0-1 (e.g. 0.8)')
    p.add_argument('--percentile', type=float, default=None, help='Use percentile (e.g. 98 for top 2%)')
    p.add_argument('--min-area', type=float, default=100.0, help='Minimum area in square meters to keep')
    p.add_argument('--output-prefix', default='high_risk', help='Output filename prefix')
    p.add_argument('--smooth', action='store_true', help='Apply small morphological opening to reduce speckles')
    args = p.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input file not found: {args.input}. Please run inference.py to generate the geotiff first.")

    with rasterio.open(args.input) as src:
        arr = src.read(1).astype(np.float32)
        transform_affine = src.transform
        src_crs = src.crs
        # pixel area (approx) in square meters: abs(pixel_width * pixel_height) in source CRS units
        px_w = transform_affine.a
        px_h = -transform_affine.e
        pixel_area_m2 = abs(px_w * px_h)

    # determine threshold
    if args.threshold is None and args.percentile is None:
        # default: top 2% (98th percentile)
        pct = 98.0
    elif args.percentile is not None:
        pct = args.percentile
    else:
        pct = None

    if pct is not None:
        thr = float(np.percentile(arr[~np.isnan(arr)], pct))
        print(f"Using percentile {pct} -> threshold {thr:.4f}")
    else:
        thr = args.threshold
        print(f"Using absolute threshold {thr:.4f}")

    # extract regions
    min_area_pixels = max(1, int(np.ceil(args.min_area / pixel_area_m2)))
    print(f"Pixel area (m^2): {pixel_area_m2:.2f} -> min area pixels: {min_area_pixels}")

    regions = extract_regions_from_raster(arr, transform_affine, thr,
                                         min_area_pixels=min_area_pixels,
                                         smooth=args.smooth)

    print(f"Found {len(regions)} candidate high-risk regions (before filtering).")

    # Build GeoDataFrame of polygons and centroids
    records = []
    for idx, reg in enumerate(regions, start=1):
        geom = reg['geometry']
        stats = reg['stats']
        centroid = geom.centroid
        # centroid coords in src_crs
        cx, cy = centroid.x, centroid.y
        # convert centroid to lat/lon
        lon, lat = transform(src_crs, 'EPSG:4326', [cx], [cy])
        lon = lon[0]
        lat = lat[0]
        rec = {
            'id': idx,
            'geometry': geom,
            'centroid_lon': lon,
            'centroid_lat': lat,
            'area_px': stats['area_px'],
            'area_m2': stats['area_px'] * pixel_area_m2,
            'mean_prob': stats['mean_prob'],
            'max_prob': stats['max_prob']
        }
        records.append(rec)

    if len(records) == 0:
        print("No high-risk regions found with given threshold/area. Try lowering threshold or min-area.")
        return

    gdf = gpd.GeoDataFrame(records, geometry=[r['geometry'] for r in records], crs=src_crs)

    # save polygon GeoJSON
    out_polygons = f"{args.output_prefix}_areas.geojson"
    gdf.to_crs(epsg=4326).to_file(out_polygons, driver='GeoJSON')
    print(f"Saved polygons: {out_polygons}")

    # save centroids GeoJSON
    cent_gdf = gdf.copy()
    cent_gdf['geometry'] = cent_gdf.geometry.centroid
    out_points = f"{args.output_prefix}_centroids.geojson"
    cent_gdf.to_crs(epsg=4326).to_file(out_points, driver='GeoJSON')
    print(f"Saved centroids: {out_points}")

    # save CSV with centroid lon/lat and stats
    out_csv = f"{args.output_prefix}_areas.csv"
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['id','centroid_lon','centroid_lat','area_m2','mean_prob','max_prob'])
        for rec in records:
            writer.writerow([rec['id'], rec['centroid_lon'], rec['centroid_lat'], rec['area_m2'], rec['mean_prob'], rec['max_prob']])
    print(f"Saved CSV: {out_csv}")

    # print top 10 by max_prob
    records_sorted = sorted(records, key=lambda r: r['max_prob'], reverse=True)
    print("Top regions by max_prob:")
    for r in records_sorted[:10]:
        print(f"id={r['id']} max_prob={r['max_prob']:.3f} mean={r['mean_prob']:.3f} lat={r['centroid_lat']:.6f} lon={r['centroid_lon']:.6f} area_m2={r['area_m2']:.0f}")


if __name__ == '__main__':
    main()
