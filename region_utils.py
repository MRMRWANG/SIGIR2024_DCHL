import numpy as np

def build_poi2region(pois_coos_dict, num_bins=10):
    lats = []
    lons = []
    poi_ids = []
    
    for pid, coords in pois_coos_dict.items():
        poi_ids.append(pid)
        lats.append(coords[0])
        lons.append(coords[1])
        
    min_lat, max_lat = min(lats), max(lats)
    min_lon, max_lon = min(lons), max(lons)
    
    lat_bins = np.linspace(min_lat, max_lat, num_bins + 1)
    lon_bins = np.linspace(min_lon, max_lon, num_bins + 1)
    
    poi2region = {}
    for pid, lat, lon in zip(poi_ids, lats, lons):
        lat_idx = np.digitize(lat, lat_bins) - 1
        lon_idx = np.digitize(lon, lon_bins) - 1
        
        lat_idx = max(0, min(lat_idx, num_bins - 1))
        lon_idx = max(0, min(lon_idx, num_bins - 1))
        
        # 【重要修改】+1 是为了避开 Embedding 的 padding_idx=0
        region_id = (lat_idx * num_bins + lon_idx) + 1
        poi2region[pid] = region_id
        
    num_regions = num_bins * num_bins
    
    # 自动处理数据集中约定的 Padding POI (例如 3835)
    # 让它的区域 ID 指向 0，对应模型中的 padding_idx
    all_pids = list(pois_coos_dict.keys())
    if len(all_pids) > 0:
        max_pid = max(all_pids)
        poi2region[max_pid + 1] = 0 
    
    return poi2region, num_regions