import numpy as np

def build_poi2region(pois_coos_dict, num_bins=10):
    """
    根据 POI 的经纬度，划分为 num_bins * num_bins 个网格区域
    """
    lats = []
    lons = []
    poi_ids = []
    
    # 提取所有有效 POI 的坐标
    for pid, coords in pois_coos_dict.items():
        poi_ids.append(pid)
        lats.append(coords[0])
        lons.append(coords[1])
        
    min_lat, max_lat = min(lats), max(lats)
    min_lon, max_lon = min(lons), max(lons)
    
    # 划分网格边界
    lat_bins = np.linspace(min_lat, max_lat, num_bins + 1)
    lon_bins = np.linspace(min_lon, max_lon, num_bins + 1)
    
    poi2region = {}
    for pid, lat, lon in zip(poi_ids, lats, lons):
        lat_idx = np.digitize(lat, lat_bins) - 1
        lon_idx = np.digitize(lon, lon_bins) - 1
        
        # 防止越界
        lat_idx = max(0, min(lat_idx, num_bins - 1))
        lon_idx = max(0, min(lon_idx, num_bins - 1))
        
        # 生成全局唯一的 Region ID
        region_id = lat_idx * num_bins + lon_idx
        poi2region[pid] = region_id
        
    num_regions = num_bins * num_bins
    
    # 为 Padding 的 POI_ID (例如纽约的 3835) 分配一个默认区域 0，防止索引越界
    padding_idx = max(poi_ids) + 1
    poi2region[padding_idx] = 0 
    
    return poi2region, num_regions