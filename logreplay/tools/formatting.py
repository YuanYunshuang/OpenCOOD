import glob, os
import numpy as np
from plyfile import PlyData
import open3d as o3d


def read_ply(filename, properties=None):
    ply = PlyData.read(filename)
    data = ply['vertex']
    properties_from_file = [p.name for p in ply.elements[0].properties]
    if properties is None:
        properties = properties_from_file
    else:
        for p in properties:
            assert p in properties_from_file, f"Property '{p}' not found."
    data_dict = {}
    for p in properties:
        data_dict[p] = np.array(data[p])

    return data_dict


def load_frame_data(cav_dir, frame):
    files = glob.glob(os.path.join(cav_dir, f'{frame:06d}.*.ply'))
    data_list = []
    for f in files:
        data = read_ply(f)
        timestamp = os.path.basename(f).split('.')[:-1]
        timestamp = int(timestamp[0]) * 0.05 + int(timestamp[1]) * 0.005
        timestamp = np.ones_like(data['x']) * timestamp
        data['time'] = timestamp.astype(np.float32)
        data_list.append(data)
    data = {k: np.concatenate([d[k] for d in data_list], axis=0) for k in data_list[0]}
    return data


if __name__=="__main__":
    data = load_frame_data("/koko/OPV2V/temporal/test/2021_08_18_19_48_05/1045",
                           68)
    pcd = o3d.geometry.PointCloud()
    points = np.stack([data[k] for k in 'xyz'], axis=-1)
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    o3d.visualization.draw_geometries([pcd])