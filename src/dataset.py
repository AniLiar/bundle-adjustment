import bz2
import functools
import numpy as np


class Dataset:

    def __init__(self):
        self.camera_params = None
        self.points_3d = None
        self.camera_indices = None
        self.point_indices = None
        self.points_2d = None
        self.n_cameras = 0
        self.n_points = 0
        self.n_observations = 0

    def get_camera_params_from(self, camera_number):
        return self.camera_params[camera_number]

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
            n_cameras, n_points, n_observations = map(
                int, file.readline().split())

            self.n_cameras = n_cameras
            self.n_points = n_points
            self.n_observations = n_observations

            camera_indices = np.empty(n_observations, dtype=int)
            point_indices = np.empty(n_observations, dtype=int)
            points_2d = np.empty((n_observations, 2))

            for i in range(n_observations):
                camera_index, point_index, x, y = file.readline().split()
                camera_indices[i] = int(camera_index)
                point_indices[i] = int(point_index)
                points_2d[i] = [float(x), float(y)]

            self.camera_indices = camera_indices
            self.point_indices = points_2d
            self.points_2d = points_2d

            camera_params = np.empty(n_cameras * 9)
            for i in range(n_cameras * 9):
                camera_params[i] = float(file.readline())
            camera_params = camera_params.reshape((n_cameras, -1))

            self.camera_params = camera_params

            points_3d = np.empty(n_points * 3)
            for i in range(n_points * 3):
                points_3d[i] = float(file.readline())
            points_3d = points_3d.reshape((n_points, -1))

            self.points_3d = points_3d

        return camera_params, points_3d, camera_indices, point_indices, points_2d, n_cameras, n_points, n_observations