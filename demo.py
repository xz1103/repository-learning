import numpy as np
import pandas as pd
from ground.vgg import Vgg
from ground.plot_tools import ptr
from ground.distance import longrc
from scipy.interpolate import griddata
from ground.etopo import etp
from ground.ship import ShipGrid
import matplotlib.pyplot as plt

#定义分辨率
num_rows = 481
num_columns = 421
#经纬度参数
kwargs = {"lat_up": 20, "lat_down": 12, "lon_left": 112, "lon_right": 119}
Topography = etp.format_topography(target_size=(num_rows, num_columns), kwargs=kwargs)

# 使用 linspace 生成 300 个均匀分布的经纬度值
latitude = np.linspace(kwargs['lat_down'], kwargs['lat_up'], num=num_rows)
longitude = np.linspace(kwargs['lon_left'], kwargs['lon_right'], num=num_columns)
# 创建网格
lon_grid,lat_grid = np.meshgrid(longitude,latitude)
# 设置坐标轴刻度
y = np.linspace(kwargs['lat_down'], kwargs['lat_up'], num=5)
yy = [value.astype(str) + '°N' for value in y]
x = np.linspace(kwargs['lon_left'], kwargs['lon_right'], num=5)
xx = [f"{value.astype(str)}°E" if value > 0 else f"{(-value).astype(str)}°W" for value in x]

data = {
        "latitude": lat_grid.ravel(),
        "longitude": lon_grid.ravel(),
        "batymetry": Topography.ravel()
    }

    # 转为 DataFrame
df = pd.DataFrame(data)

# 保存 .CSV 文件
csv_file = ".%d_%d_%d_%d.csv" % (
kwargs['lon_left'], kwargs['lon_right'], kwargs['lat_down'], kwargs['lat_up'])
df.to_csv(csv_file, index=False)

print(f'转换 {csv_file}成功')