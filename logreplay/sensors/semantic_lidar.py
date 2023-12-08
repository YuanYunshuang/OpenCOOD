"""
This is mainly used to filter out objects that is not in the sight
of cameras.
"""
import weakref

import carla
import cv2
import os
import numpy as np
from plyfile import PlyData, PlyElement
from logreplay.sensors.base_sensor import BaseSensor


class SemanticLidar(BaseSensor):
    def __init__(self, agent_id, vehicle, world, config, global_position):
        super().__init__(agent_id, vehicle, world, config, global_position)

        if vehicle is not None:
            world = vehicle.get_world()

        self.agent_id = agent_id

        blueprint = world.get_blueprint_library(). \
            find('sensor.lidar.ray_cast_semantic')
        # set attribute based on the configuration
        blueprint.set_attribute('upper_fov', str(config['upper_fov']))
        blueprint.set_attribute('lower_fov', str(config['lower_fov']))
        blueprint.set_attribute('channels', str(config['channels']))
        blueprint.set_attribute('range', str(config['range']))
        blueprint.set_attribute(
            'points_per_second', str(
                config['points_per_second']))
        blueprint.set_attribute(
            'rotation_frequency', str(
                config['rotation_frequency']))

        relative_position = config['relative_pose']
        spawn_point = self.spawn_point_estimation(relative_position,
                                                  global_position)
        self.name = 'semantic_lidar' + str(relative_position)
        self.thresh = config['thresh']

        if vehicle is not None:
            self.sensor = world.spawn_actor(
                blueprint, spawn_point, attach_to=vehicle)
        else:
            self.sensor = world.spawn_actor(blueprint, spawn_point)

        # lidar data
        self.points = None
        self.obj_idx = None
        self.obj_tag = None
        self.ring = None

        self.timestamp = None
        self.frame = 0

        # weak_self = weakref.ref(self)
        # self.sensor.listen(
        #     lambda event: SemanticLidar._on_data_event(
        #         weak_self, event))

        self.fields = {'x': 'f4', 'y': 'f4', 'z': 'f4', 'ObjIdx': 'u4', 'ObjTag': 'u4', 'ring': 'u1'}
        self.np_types = {'f4': np.float32, 'u4': np.uint32, 'u1': np.uint8}

    @staticmethod
    def _on_data_event(weak_self, event):
        """Semantic Lidar  method"""
        self = weak_self()
        if not self:
            return

        # shape:(n, 6)
        data = np.frombuffer(event.raw_data, dtype=np.dtype([
            ('x', np.float32), ('y', np.float32), ('z', np.float32),
            ('CosAngle', np.float32), ('ObjIdx', np.uint32),
            ('ObjTag', np.uint32)]))

        # (x, y, z, intensity)
        points = np.array([data['x'], data['y'], data['z']]).T
        obj_tag = np.array(data['ObjTag'])
        obj_idx = np.array(data['ObjIdx'])
        ring = []
        for i in range(event.channels):
            ring.append(np.ones(event.get_point_count(i)) * i)
        ring = np.concatenate(ring).astype(np.uint8)

        attenuation = 0.004
        noise_stddev = 0.02
        dropoff_general_rate = 0.1
        dropoff_intensity_limit = 0.7
        dropoff_zero_intensity = 0.15

        # general_drop
        samples = np.random.random(len(points))
        mask = samples > dropoff_general_rate
        points, obj_tag, obj_idx, ring = points[mask], obj_tag[mask], obj_idx[mask], ring[mask]

        # cal intensity
        dists = np.linalg.norm(points, axis=-1)
        intensity = np.exp(- attenuation * dists)

        # drop zero intensity
        drop = intensity < 0.01
        samples = np.random.random(len(points))
        drop = np.logical_and(samples <= dropoff_zero_intensity, drop)
        mask = np.logical_not(drop)
        points, obj_tag, obj_idx, ring = points[mask], obj_tag[mask], obj_idx[mask], ring[mask]
        dists = dists[mask]

        # add noise
        noise = np.random.normal(0, noise_stddev, len(points))
        azi = np.arctan2(points[:, 1], points[:, 0])
        ele = np.arctan2(points[:, 2], np.linalg.norm(points[:, :2], axis=-1))
        dists = dists + noise
        z = np.sin(ele) * dists
        d_xy = np.cos(ele) * dists
        x = np.cos(azi) * d_xy
        y = np.sin(azi) * d_xy
        # x = points[:, 0]
        # y = points[:, 1]
        # z = points[:, 2]

        self.points = np.stack([x, y, z], axis=-1)
        self.obj_tag = obj_tag
        self.obj_idx = obj_idx
        self.ring = ring

        self.data = np.stack([x, y, z, obj_tag.astype(np.float32)],
                             axis=-1).astype(np.float32)
        self.frame = event.frame
        self.timestamp = event.timestamp

    @staticmethod
    def spawn_point_estimation(relative_position, global_position):

        pitch = 0
        carla_location = carla.Location(x=0, y=0, z=0)

        if global_position is not None:
            carla_location = carla.Location(
                x=global_position[0],
                y=global_position[1],
                z=global_position[2])
            pitch = -35

        if relative_position == 'front':
            carla_location = carla.Location(x=carla_location.x + 2.5,
                                            y=carla_location.y,
                                            z=carla_location.z + 1.0)
            yaw = 0

        elif relative_position == 'right':
            carla_location = carla.Location(x=carla_location.x + 0.0,
                                            y=carla_location.y + 0.3,
                                            z=carla_location.z + 1.8)
            yaw = 100

        elif relative_position == 'left':
            carla_location = carla.Location(x=carla_location.x + 0.0,
                                            y=carla_location.y - 0.3,
                                            z=carla_location.z + 1.8)
            yaw = -100
        elif relative_position == 'back':
            carla_location = carla.Location(x=carla_location.x - 2.0,
                                            y=carla_location.y,
                                            z=carla_location.z + 1.5)
            yaw = 180
        else:
            carla_location = carla.Location(x=carla_location.x,
                                            y=carla_location.y,
                                            z=carla_location.z + 1.9)
            yaw = 0
            pitch = 0

        carla_rotation = carla.Rotation(roll=0, yaw=yaw, pitch=pitch)
        spawn_point = carla.Transform(carla_location, carla_rotation)

        return spawn_point

    def tick(self):
        while self.obj_idx is None or self.obj_tag is None or \
                self.obj_idx.shape[0] != self.obj_tag.shape[0]:
            continue

        # label 10 is the vehicle
        vehicle_idx = self.obj_idx[self.obj_tag == 10]
        # each individual instance id
        vehicle_unique_id = list(np.unique(vehicle_idx))
        vehicle_id_filter = []

        for veh_id in vehicle_unique_id:
            if vehicle_idx[vehicle_idx == veh_id].shape[0] > self.thresh:
                vehicle_id_filter.append(veh_id)

        # these are the ids that are visible
        return vehicle_id_filter

    def data_dump(self, output_root, cur_timestamp, ext='.ply'):
        # dump lidar
        output_file_name = os.path.join(output_root, cur_timestamp + ext)
        if ext == '.bin':
            data = getattr(self, 'data', None)
            # waite data to prevent data stream error because of late coming data
            while data is None:
                data = getattr(self, 'data', None)
            data.tofile(output_file_name)
        else:
            data = {
                'x': self.points[:, 0].astype(self.np_types[self.fields['x']]),
                'y': self.points[:, 1].astype(self.np_types[self.fields['y']]),
                'z': self.points[:, 2].astype(self.np_types[self.fields['z']]),
                'ObjIdx': self.obj_idx.astype(self.np_types[self.fields['ObjIdx']]),
                'ObjTag': self.obj_tag.astype(self.np_types[self.fields['ObjTag']]),
                'ring': self.ring.astype(self.np_types[self.fields['ring']])
            }
            vertex_data = list(zip(*[data[k] for k, v in self.fields.items()]))
            vertex_type = [(k, v) for k, v in self.fields.items()]
            vertex = np.array(vertex_data, dtype=vertex_type)
            el = PlyElement.describe(vertex, 'vertex')
            PlyData([el]).write(output_file_name)
        # lidar data
        self.points = None
        self.obj_idx = None
        self.obj_tag = None
        self.data= None
