import bz2
import functools
import numpy as np

from src.projection import CoordinateService

N_CAMERA_PARAMS = 9


class Dataset:

    def __init__(self):
        self.problem_name = None
        self.camera_params = None
        self.points_2d = None
        self.points_3d = None

        self.points_3d_indices = None
        self.camera_indices = None

        self.n_cameras = 0
        self.n_points_3d = 0
        self.n_observations_2d = 0

    def set_camera_poses(self, camera_poses):
        assert self.n_cameras == camera_poses.shape[0], "Camera_poses param should be the same dim of n_cameras"
        self.camera_params[:, 3:6] = camera_poses

    def set_camera_rotations(self, camera_rotation):
        assert self.n_cameras == camera_rotation.shape[0], "Camera_rotation param should be the same dim of n_cameras"
        self.camera_params[:, :3] = camera_rotation

    def set_camera_distortion(self, distortion_k1, distortion_k2):
        assert self.n_cameras == distortion_k1.shape[0] & self.n_cameras == distortion_k2.shape[0], "Distortions k1, k2 params should be the same dim of n_cameras"
        self.camera_params[:, 7:8] = distortion_k1
        self.camera_params[:, 8:9] = distortion_k2

    def set_camera_focal_length(self, focal_length):
        assert self.n_cameras == focal_length.shape[0], "Focal_length param should be the same dim of n_cameras"
        self.camera_params[:, 6:7] = focal_length

    def set_camera_params(self, camera_params):
        self.n_cameras = camera_params.shape[0]
        self.camera_params = camera_params


    def get_camera_params_from(self, camera_number):
        return self.camera_params[camera_number]

    def get_camera_pose_from(self, camera_number):
        return self.camera_params[camera_number,3:6]

    def read_from_file(self, file_path: str, bz2_encoding: bool = True):
        """Read BAL file contents

        Set `bz2_encoding` to `False` if the file archive is unpacked
        """
        open_file = (
            functools.partial(bz2.open, mode='rt')
            if bz2_encoding
            else functools.partial(open, mode='r')
        )
        with open_file(file_path) as file:
            self.problem_name = file_path
            n_cameras, n_points, n_observations = map(
                int, file.readline().split())

            self.n_cameras = n_cameras
            self.n_points_3d = n_points
            self.n_observations_2d = n_observations

            camera_indices = np.empty(n_observations, dtype=int)
            point_indices = np.empty(n_observations, dtype=int)
            points_2d = np.empty((n_observations, 2))

            for i in range(n_observations):
                camera_index, point_index, x, y = file.readline().split()
                camera_indices[i] = int(camera_index)
                point_indices[i] = int(point_index)
                points_2d[i] = [float(x), float(y)]

            self.camera_indices = camera_indices
            self.points_3d_indices = points_2d
            self.points_2d = points_2d

            camera_params = np.empty(n_cameras * N_CAMERA_PARAMS)
            for i in range(n_cameras * N_CAMERA_PARAMS):
                camera_params[i] = float(file.readline())
            camera_params = camera_params.reshape((n_cameras, -1))

            self.camera_params = camera_params

            points_3d = np.empty(n_points * 3)
            for i in range(n_points * 3):
                points_3d[i] = float(file.readline())
            points_3d = points_3d.reshape((n_points, -1))

            self.points_3d = points_3d

        return camera_params, points_3d, camera_indices, point_indices, points_2d, n_cameras, n_points, n_observations

    def generate(self, n_cameras: int = 1, n_points_3d: int = 10, n_observations_2d: int = 10):
        assert n_observations_2d / n_cameras < n_points_3d, "n_observations_2d param should be less than n_points_3d * n_cameras"
        self.n_cameras = n_cameras
        self.n_points_3d = n_points_3d
        self.n_observations_2d = n_observations_2d
        self.problem_name = "Random dataset: {} cameras, {} 3d points, {} 2d observations".format(self.n_cameras, self.n_points_3d, self.n_observations_2d)

        self.points_3d = np.random.rand(self.n_points_3d, 3)  # Генерация 3D точек с равномерным распределением
        self.camera_params = np.random.rand(self.n_cameras, N_CAMERA_PARAMS)
        self.camera_indices = np.random.randint(0, self.n_cameras, self.n_observations_2d, dtype=int)
        self.points_3d_indices = np.random.randint(0, self.n_points_3d, self.n_observations_2d, dtype=int)

        service = CoordinateService()
        self.points_2d = np.empty((self.n_observations_2d, 2))
        for i in range(0, self.n_observations_2d):
            camera_index = self.camera_indices[i]
            camera_param = self.camera_params[camera_index:camera_index + 1, :]
            self.points_2d[camera_index, :] = service.get_forward_projection(camera_param, self.points_3d[self.points_3d_indices[i]])

    def generate_with_default_params(self, n_cameras: int = 1):
        self.n_cameras = n_cameras
        self.set_camera_params(np.zeros((self.n_cameras, N_CAMERA_PARAMS)))

        camera_pose = np.array([[0, 0, 0]])
        self.set_camera_poses(camera_pose)

        rotation = np.array([[0, 0, 0]])
        self.set_camera_rotations(rotation)

        self.set_camera_focal_length(np.array([[1]]))
        self.set_camera_distortion(np.array([[0]]), np.array([[0]]))

        self.n_points_3d = 1
        self.n_observations_2d = 1
        self.points_3d = np.array([[0, 0, 1]])
        self.points_3d_indices = np.array([0])
        self.camera_indices = np.array([0])

        service = CoordinateService()
        self.points_2d = service.get_forward_projection(self.camera_params, self.points_3d)
        self.problem_name = "Synthetic dataset: {} cameras, {} 3d points, {} 2d observations".format(self.n_cameras, self.n_points_3d, self.n_observations_2d)
        print("here")