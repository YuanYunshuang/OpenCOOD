import os
import random
import sys
from collections import OrderedDict

import tqdm

sys.path.append("/opt/carla-simulator/PythonAPI/carla/dist/carla-0.9.13-py3.7-linux-x86_64.egg")
import carla
import numpy as np
from scipy.spatial.transform.rotation import Rotation as R

from logreplay.assets.utils import find_town, find_blue_print
from logreplay.assets.presave_lib import bcolors
from logreplay.map.map_manager import MapManager
from logreplay.sensors.sensor_manager import SensorManager
from opencood.hypes_yaml.yaml_utils import load_yaml, save_yaml


def transformation_matrix_to_pose(tf):
    tf = np.array(tf)
    r = R.from_matrix(tf[:3, :3]).as_euler('yzx', degrees=True)
    pose = np.concatenate([tf[:3, 3], r], axis=0)
    return pose


def transformation_pose_to_matrix(pose):
    tf = np.eye(4)
    tf[:3, :3] = R.from_euler('yzx', pose[3:]).as_matrix()
    tf[:3, 3] = pose[:3]
    return tf


def interpolate(prev_pose, cur_pose, steps=10):
    to_mat = False
    if len(prev_pose) == 4:
        prev_pose = transformation_matrix_to_pose(prev_pose)
        cur_pose = transformation_matrix_to_pose(cur_pose)
        to_mat = True
    else:
        prev_pose = np.array(prev_pose)
        cur_pose = np.array(cur_pose)
    for i in [3, 4, 5]:
        if abs(cur_pose[i] - prev_pose[i]) > 180:
            if cur_pose[i] > prev_pose[i]:
                cur_pose[i] -= 360
            else:
                prev_pose[i] -= 360

    poses = np.linspace(prev_pose, cur_pose, steps + 1)
    poses[:, 3:] = poses[:, 3:] % 360
    poses[:, 3:] = poses[:, 3:] - 360 * (poses[:, 3:] > 180).astype(float)
    poses[:, 3:] = poses[:, 3:] + 360 * (poses[:, 3:] < -180).astype(float)
    # if abs((poses[0, 3:] - poses[1, 3:]).mean()) > 1:
    #     print(poses[0])
    #     print(poses[1])
    #     print('--------------')
    if to_mat:
        tfs = []
        for pose in poses:
            tfs.append(transformation_pose_to_matrix(pose).tolist())
        return tfs
    else:
        return poses.tolist()


class SceneManager:
    """
    Manager for each scene for spawning, moving and destroying.

    Parameters
    ----------
    folder : str
        The folder to the current scene.

    scene_name : str
        The scene's name.

    collection_params : dict
        The collecting protocol information.

    scenario_params : dict
        The replay params.
    """

    def __init__(self, folder, scene_name, collection_params, scenario_params):
        self.scene_name = scene_name
        self.town_name = find_town(scene_name)
        self.collection_params = collection_params
        self.scenario_params = scenario_params
        self.cav_id_list = []

        # dumping related
        self.output_root = os.path.join(scenario_params['output_dir'],
                                        scene_name)

        if 'seed' in collection_params['world']:
            np.random.seed(collection_params['world']['seed'])
            random.seed(collection_params['world']['seed'])

        # at least 1 cav should show up
        cav_list = sorted([x for x in os.listdir(folder)
                           if os.path.isdir(
                os.path.join(folder, x))])
        assert len(cav_list) > 0

        self.database = OrderedDict()
        # we want to save timestamp as the parent keys for cavs
        cav_sample = os.path.join(folder, cav_list[0])

        yaml_files = \
            sorted([os.path.join(cav_sample, x)
                    for x in os.listdir(cav_sample) if
                    x.endswith('.yaml') and 'additional' not in x and 'sparse_gt' not in x])
        self.timestamps = self.extract_timestamps(yaml_files)

        # loop over all timestamps
        for timestamp in self.timestamps:
            self.database[timestamp] = OrderedDict()
            # loop over all cavs
            for (j, cav_id) in enumerate(cav_list):
                if cav_id not in self.cav_id_list:
                    self.cav_id_list.append(str(cav_id))

                self.database[timestamp][cav_id] = OrderedDict()
                cav_path = os.path.join(folder, cav_id)

                yaml_file = os.path.join(cav_path,
                                         timestamp + '.yaml')
                self.database[timestamp][cav_id]['yaml'] = \
                    yaml_file

        # interpolate replay meta to get more fine-grained time interval for simulation
        # self.interpolate()

        # this is used to dynamically save all information of the objects
        self.veh_dict = OrderedDict()
        # used to count timestamp
        self.cur_count = 0
        # used for HDMap
        self.map_manager = None

    def interpolate(self):
        steps = 10
        self.database_interp = OrderedDict()
        self.timestamps_interp = []
        for i in tqdm.tqdm(range(len(self.timestamps))):
            frame = self.timestamps[i]
            cur_database = self.database[frame]
            database_interp = OrderedDict()
            if i == 0:
                database_interp[f'{frame}.0'] = {cav: load_yaml(cur_database[cav]['yaml']) for cav in cur_database}
            else:
                for cav in cur_database:
                    cur_cav_content = load_yaml(cur_database[cav]['yaml'])
                    prev_cav_content = load_yaml(prev_database[cav]['yaml'])
                    keys = ['camera0', 'camera1', 'camera2', 'camera3', 'lidar_pose', 'true_ego_pos', 'vehicles']
                    for k in keys:
                        out_dict = self.interpolate_one_attribute(prev_cav_content,
                                                                  cur_cav_content,
                                                                  k, steps)
                        for j in range(1, steps + 1):
                            if j < steps:
                                fk = f'{prev_frame}.{j}'
                            else:
                                fk = f'{frame}.0'

                            if fk not in database_interp:
                                database_interp[fk] = {}
                            if cav not in database_interp[fk]:
                                database_interp[fk][cav] = {}

                            # get ego speed
                            if k == 'true_ego_pos':
                                offset = np.array([out_dict[j][k][0] - out_dict[j-1][k][0],
                                                   out_dict[j][k][1] - out_dict[j-1][k][1]])
                                out_dict[j]['speed'] = float(np.linalg.norm(offset) * steps * 10 * 3.6)
                            database_interp[fk][cav].update(out_dict[j])

            prev_database = cur_database
            prev_frame = frame


            # dump yaml files
            for f, fdict in database_interp.items():
                for cav, cav_dict in fdict.items():
                    os.makedirs(os.path.join(self.output_root, cav), exist_ok=True)
                    save_yaml(cav_dict, os.path.join(self.output_root, cav, f'{f}.yaml'))

    def interpolate_one_attribute(self, prev, cur, attri, steps=10):
        out_dict = {i: {} for i in range(steps + 1)}

        if 'camera' in attri:
            cords_interp = interpolate(prev[attri]['cords'], cur[attri]['cords'], steps)
            extri_interp = interpolate(prev[attri]['extrinsic'], cur[attri]['extrinsic'], steps)
            for i in range(steps + 1):
                out_dict[i][attri] = {
                    'cords': cords_interp[i],
                    'intrinsic': prev[attri]['intrinsic'],
                    'extrinsic': extri_interp[i]
                }
        elif attri == 'vehicles':
            for vid in cur[attri]:
                if vid in prev[attri]:
                    cur_pose = cur[attri][vid]['location'] + cur[attri][vid]['angle']
                    prev_pose = prev[attri][vid]['location'] + prev[attri][vid]['angle']
                    pose_interp = interpolate(prev_pose, cur_pose, steps)
                    for i in range(steps + 1):
                        if attri not in out_dict[i]:
                            out_dict[i][attri] = {}
                        if i == steps:
                            speed = cur[attri][vid]['speed']
                        else:
                            offset = np.array([pose_interp[i+1][0] - pose_interp[i][0],
                                               pose_interp[i+1][1] - pose_interp[i][1]])
                            speed = float(np.linalg.norm(offset) * steps * 10 * 3.6)
                        out_dict[i][attri][vid] = {
                            'angle': pose_interp[i][3:],
                            'center': cur[attri][vid]['center'],
                            'extent': cur[attri][vid]['extent'],
                            'location': pose_interp[i][:3],
                            'speed': speed
                        }
                else:
                    # if a vehicle is only in the current frame, copy it to the last entry of out dict
                    if attri not in out_dict[steps]:
                        out_dict[steps][attri] = {}
                    out_dict[steps][attri][vid] = cur[attri][vid]

        else:
            interp = interpolate(prev[attri], cur[attri], steps)
            for i in range(steps + 1):
                out_dict[i][attri] = interp[i]
        return out_dict

    def start_simulator(self):
        """
        Connect to the carla simulator for log replay.
        """
        simulation_config = self.collection_params['world']

        # simulation sync mode time step
        fixed_delta_seconds = simulation_config['fixed_delta_seconds']
        weather_config = simulation_config[
            'weather'] if "weather" in simulation_config else None

        # setup the carla client
        self.client = \
            carla.Client('localhost', simulation_config['client_port'])
        self.client.set_timeout(10.0)

        # load the map
        if self.town_name != 'Culver_City':
            try:
                self.world = self.client.load_world(self.town_name)
            except RuntimeError:
                print(
                    f"{bcolors.FAIL} %s is not found in your CARLA repo! "
                    f"Please download all town maps to your CARLA "
                    f"repo!{bcolors.ENDC}" % self.town_name)
        else:
            self.world = self.client.get_world()

        if not self.world:
            sys.exit('World loading failed')

        # setup the new setting
        self.origin_settings = self.world.get_settings()
        new_settings = self.world.get_settings()

        new_settings.synchronous_mode = True
        new_settings.fixed_delta_seconds = fixed_delta_seconds

        self.world.apply_settings(new_settings)
        # set weather if needed
        if weather_config is not None:
            weather = self.set_weather(weather_config)
            self.world.set_weather(weather)
        # get map
        self.carla_map = self.world.get_map()
        # spectator
        self.spectator = self.world.get_spectator()
        # hd map manager per scene
        self.map_manager = MapManager(self.world,
                                      self.scenario_params['map'],
                                      self.output_root,
                                      self.scene_name)
        # get static vehicles
        self.static_object_dumping()

    def tick(self):
        """
        Spawn the vehicle to the correct position
        """
        if self.cur_count >= len(self.timestamps):
            return False

        cur_timestamp = self.timestamps[self.cur_count]
        cur_database = self.database[cur_timestamp]

        for i, (cav_id, cav_yml) in enumerate(cur_database.items()):
            cav_content = load_yaml(cav_yml['yaml'])
            if cav_id not in self.veh_dict:
                self.spawn_cav(cav_id, cav_content, cur_timestamp)
            else:
                self.move_vehicle(cav_id,
                                  cur_timestamp,
                                  self.structure_transform_cav(
                                      (cav_content['true_ego_pos'])))

            self.veh_dict[cav_id]['cav'] = True
            # spawn the sensor on each cav
            if 'sensor_manager' not in self.veh_dict[cav_id]:
                self.veh_dict[cav_id]['sensor_manager'] = \
                    SensorManager(cav_id, self.veh_dict[cav_id],
                                  self.world, self.scenario_params['sensor'],
                                  self.output_root)

            # set the spectator to the first cav
            if i == 0:
                transform = self.structure_transform_cav(
                    cav_content['true_ego_pos'])
                self.spectator.set_transform(
                    carla.Transform(transform.location +
                                    carla.Location(
                                        z=70),
                                    carla.Rotation(
                                        pitch=-90
                                    )))

            for bg_veh_id, bg_veh_content in cav_content['vehicles'].items():
                if str(bg_veh_id) not in self.veh_dict:
                    self.spawn_bg_vehicles(bg_veh_id,
                                           bg_veh_content,
                                           cur_timestamp)
                else:
                    self.move_vehicle(str(bg_veh_id),
                                      cur_timestamp,
                                      self.structure_transform_bg_veh(
                                          bg_veh_content['location'],
                                          bg_veh_content['angle']))
        # remove the vehicles that are not in any cav's scope
        self.destroy_vehicle(cur_timestamp)

        self.cur_count += 1
        self.world.tick()

        # we dump data after tick() so the agent can retrieve the newest info
        # self.sensor_dumping(cur_timestamp)
        # self.object_dumping()
        # self.map_dumping()

        return True

    def sub_tck(self):
        pass

    def static_object_dumping(self):
        static_vehicles = self.world.get_level_bbs(carla.CityObjectLabel.Vehicles)
        vehicles = {} # only save large 4 wheel vehicles
        cnt = 0
        for bbx in static_vehicles:
            if bbx.extent.x > 1.5:
                vehicles[f'static{cnt}'] = {
                    'angle': [bbx.rotation.roll,
                              bbx.rotation.yaw,
                              bbx.rotation.pitch],
                    'center': [0, 0, 0],
                    'extent': [bbx.extent.x, bbx.extent.y, bbx.extent.z],
                    'location': [bbx.location.x,
                                 bbx.location.y,
                                 bbx.location.z]
                }
                cnt += 1
        filename = os.path.join(self.output_root, 'static_vehicles.yaml')
        if not os.path.exists(self.output_root):
            os.makedirs(self.output_root)
        save_yaml(vehicles, filename)

    def object_dumping(self):
        """
        Dump object information into a yaml file
        """
        def tf2list(pose):
            return [
                    pose.location.x,
                    pose.location.y,
                    pose.location.z,
                    pose.rotation.roll,
                    pose.rotation.yaw,
                    pose.rotation.pitch,
                ]
        vehicles = {}
        cavs = {}
        for k, v in self.veh_dict.items():
            veh = self.world.get_actor(v['actor_id'])
            bbx = veh.bounding_box
            pose = veh.get_transform()
            vehicles[int(k)] = {
                'angle': [pose.rotation.roll,
                          pose.rotation.yaw,
                          pose.rotation.pitch],
                'center': [bbx.location.x, bbx.location.y, bbx.location.z],
                'extent': [bbx.extent.x, bbx.extent.y, bbx.extent.z],
                'location': [pose.location.x,
                             pose.location.y,
                             pose.location.z]
            }
            if v.get('cav', False):
                info = {}
                lidar_sensor = v['sensor_manager'].sensor_list[0].sensor
                lidar_pose = lidar_sensor.get_transform()
                info['lidar_pose'] = tf2list(lidar_pose)
                info['true_ego_pos'] = tf2list(pose)
                cavs[k] = info
        # save yaml files
        for cav_id, info in cavs.items():
            out_dict = {}
            out_dict.update(info)
            cur_vehicles = {k: v for k, v in vehicles.items() if k != cav_id}
            out_dict['vehicles'] = cur_vehicles
            out_file = os.path.join(
                self.output_root,
                cav_id, self.veh_dict[cav_id]['cur_count'] + '.yaml'
                )
            save_yaml(out_dict, out_file)

    def map_dumping(self):
        """
        Dump bev map related.

        Parameters
        ----------
        """
        for veh_id, veh_content in self.veh_dict.items():
            if 'cav' in veh_content:
                self.map_manager.run_step(veh_id, veh_content, self.veh_dict)

    def sensor_dumping(self, cur_timestamp):
        for veh_id, veh_content in self.veh_dict.items():
            if 'sensor_manager' in veh_content:
                veh_content['sensor_manager'].run_step(cur_timestamp)

    def spawn_cav(self, cav_id, cav_content, cur_timestamp):
        """
        Spawn the cav based on current content.

        Parameters
        ----------
        cav_id : str
            The saved cav_id.

        cav_content : dict
            The information in the cav's folder.

        cur_timestamp : str
            This is used to judge whether this vehicle has been already
            called in this timestamp.
        """

        # cav always use lincoln
        model = 'vehicle.lincoln.mkz_2017'

        # retrive the blueprint library
        blueprint_library = self.world.get_blueprint_library()
        cav_bp = blueprint_library.find(model)
        # cav is always green
        color = '0, 0, 255'
        cav_bp.set_attribute('color', color)

        cur_pose = cav_content['true_ego_pos']
        # convert to carla needed format
        cur_pose = self.structure_transform_cav(cur_pose)

        # spawn the vehicle
        vehicle = \
            self.world.try_spawn_actor(cav_bp, cur_pose)

        while not vehicle:
            cur_pose.location.z += 0.01
            vehicle = \
                self.world.try_spawn_actor(cav_bp, cur_pose)

        self.veh_dict.update({str(cav_id): {
            'cur_pose': cur_pose,
            'model': model,
            'color': color,
            'actor_id': vehicle.id,
            'actor': vehicle,
            'cur_count': cur_timestamp
        }})

    def spawn_bg_vehicles(self, bg_veh_id, bg_veh_content, cur_timestamp):
        """
        Spawn the background vehicle.

        Parameters
        ----------
        bg_veh_id : str
            The id of the bg vehicle.
        bg_veh_content : dict
            The contents of the bg vehicle
        cur_timestamp : str
            This is used to judge whether this vehicle has been already
            called in this timestamp.
        """
        # retrieve the blueprint library
        blueprint_library = self.world.get_blueprint_library()

        cur_pose = self.structure_transform_bg_veh(bg_veh_content['location'],
                                                   bg_veh_content['angle'])
        if str(bg_veh_id) in self.cav_id_list:
            model = 'vehicle.lincoln.mkz_2017'
            veh_bp = blueprint_library.find(model)
            color = '0, 0, 255'
        else:
            model = find_blue_print(bg_veh_content['extent'])
            if not model:
                print('model net found for %s' % bg_veh_id)
            veh_bp = blueprint_library.find(model)

            color = random.choice(
                veh_bp.get_attribute('color').recommended_values)

        veh_bp.set_attribute('color', color)

        # spawn the vehicle
        vehicle = \
            self.world.try_spawn_actor(veh_bp, cur_pose)

        while not vehicle:
            cur_pose.location.z += 0.01
            vehicle = \
                self.world.try_spawn_actor(veh_bp, cur_pose)

        self.veh_dict.update({str(bg_veh_id): {
            'cur_pose': cur_pose,
            'model': model,
            'color': color,
            'actor_id': vehicle.id,
            'actor': vehicle,
            'cur_count': cur_timestamp
        }})

    def move_vehicle(self, veh_id, cur_timestamp, transform):
        """
        Move the

        Parameters
        ----------
        veh_id : str
            Vehicle's id

        cur_timestamp : str
            Current timestamp

        transform : carla.Transform
            Current pose/
        """
        # this represent this vehicle is already moved in this round
        if self.veh_dict[veh_id]['cur_count'] == cur_timestamp:
            return

        self.veh_dict[veh_id]['actor'].set_transform(transform)
        self.veh_dict[veh_id]['cur_count'] = cur_timestamp
        self.veh_dict[veh_id]['cur_pose'] = transform

    def close(self):
        self.map_manager.destroy()
        self.sensor_destory()
        self.world.apply_settings(self.origin_settings)
        actor_list = self.world.get_actors()
        for actor in actor_list:
            if actor.is_alive:
                actor.destroy()


    def sensor_destory(self):
        for veh_id, veh_content in self.veh_dict.items():
            if 'sensor_manager' in veh_content:
                veh_content['sensor_manager'].destroy()

    def destroy_vehicle(self, cur_timestamp):
        """
        This is used to destroy the vehicles that are out of any cav's scope.
        """
        destroy_list = []
        for veh_id, veh_content in self.veh_dict.items():
            if veh_content['cur_count'] != cur_timestamp:
                if veh_content.get('cav', False):
                    print('d')
                veh_content['actor'].destroy()
                destroy_list.append(veh_id)

        for veh_id in destroy_list:
            self.veh_dict.pop(veh_id)

    def structure_transform_cav(self, pose):
        """
        Convert the pose saved in list to transform format.

        Parameters
        ----------
        pose : list
            x, y, z, roll, yaw, pitch

        Returns
        -------
        carla.Transform
        """
        cur_pose = carla.Transform(carla.Location(x=pose[0],
                                                  y=pose[1],
                                                  z=pose[2]),
                                   carla.Rotation(roll=pose[3],
                                                  yaw=pose[4],
                                                  pitch=pose[5]))

        return cur_pose

    @staticmethod
    def structure_transform_bg_veh(location, rotation):
        """
        Convert the location and rotation in list to carla transform format.

        Parameters
        ----------
        location : list
        rotation : list
        Returns
        -------
        carla.Transform
        """
        cur_pose = carla.Transform(carla.Location(x=location[0],
                                                  y=location[1],
                                                  z=location[2]),
                                   carla.Rotation(roll=rotation[0],
                                                  yaw=rotation[1],
                                                  pitch=rotation[2]))

        return cur_pose

    @staticmethod
    def extract_timestamps(yaml_files):
        """
        Given the list of the yaml files, extract the mocked timestamps.

        Parameters
        ----------
        yaml_files : list
            The full path of all yaml files of ego vehicle

        Returns
        -------
        timestamps : list
            The list containing timestamps only.
        """
        timestamps = []

        for file in yaml_files:
            res = file.split('/')[-1]

            timestamp = res.replace('.yaml', '')
            timestamps.append(timestamp)

        return timestamps

    @staticmethod
    def set_weather(weather_settings):
        """
        Set CARLA weather params.

        Parameters
        ----------
        weather_settings : dict
            The dictionary that contains all parameters of weather.

        Returns
        -------
        The CARLA weather setting.
        """
        weather = carla.WeatherParameters(
            sun_altitude_angle=weather_settings['sun_altitude_angle'],
            cloudiness=weather_settings['cloudiness'],
            precipitation=weather_settings['precipitation'],
            precipitation_deposits=weather_settings['precipitation_deposits'],
            wind_intensity=weather_settings['wind_intensity'],
            fog_density=weather_settings['fog_density'],
            fog_distance=weather_settings['fog_distance'],
            fog_falloff=weather_settings['fog_falloff'],
            wetness=weather_settings['wetness']
        )
        return weather
