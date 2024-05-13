import numpy as np
from accessify import protected


class CoordinateService:

    def get_forward_projection(self, camera_params, x_3d):
        """Convert 3-D points to 2-D by projecting onto images.

        Get projection point in pixels from 3D- point in world coordinates
        World coordinates -> Camera coordinates -> Film coordinates -> Pixel coordinates
        """
        point_proj = self.to_camera_coords_from_world_coords(camera_params, x_3d)
        point_proj = self.to_film_coords_from_camera_coords(point_proj)
        point_proj = self.to_pixel_coords_from_film_coords(camera_params, point_proj)
        return point_proj

    @protected
    def to_film_coords_from_camera_coords(self, point_proj):
        """Convert 3-D point in camera coordinates to 2-D point in film coordinates (perspective projection):
        p = -P / P.z,
        where P is a 3-D point in camera coordinates,
        P.z is a z-coordinate of P in camera coordinates

        :param point_proj: is a 3-D point in camera coordinates
        :return: 2-D projection in film coordinates of 3-D point
        """

        points_proj_new = np.zeros((point_proj.shape[0], 2))
        mask = point_proj[:, 2] != 0
        points_proj_new[mask, :] = - point_proj[mask, :2] / point_proj[mask, 2, np.newaxis]
        return points_proj_new

    @protected
    def to_camera_coords_from_world_coords(self, camera_params, x_3d):
        """Convert 3-D point from world to camera coordinates: P  =  R * X + t"""
        camera_rotation = camera_params[:, :3]
        point_proj = self.rotate(x_3d, camera_rotation)

        camera_translation = camera_params[:, 3:6]
        point_proj += camera_translation
        return point_proj

    @protected
    def to_pixel_coords_from_film_coords(self, camera_params, point_proj):
        """Convert 2-D projection point from film (camera coords?) to pixel coordinates: p' =  f * r(p) * point_proj,
        where
        f is  a focal length,
        r(p) is function that computes a scaling factor to undo the radial distortion
        point_proj is 2-D projection which is a projection of 3-D point in film coords (camera coords?)
        """
        f = camera_params[:, 6]
        r = self.get_scaling_factor(camera_params, point_proj)
        point_proj *= (r * f)[:, np.newaxis]
        return point_proj

    @protected
    def rotate(self, points, rot_vecs):
        """Rotate points by given rotation vectors.

        Rodrigues' rotation formula is used.
        """
        theta = np.linalg.norm(rot_vecs, axis=1)
        theta = theta[:, np.newaxis]
        with np.errstate(invalid='ignore'):
            v = rot_vecs / theta
            v = np.nan_to_num(v)
        dot = np.sum(points * v, axis=1)[:, np.newaxis]
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        return cos_theta * points + sin_theta * np.cross(v, points) + dot * (1 - cos_theta) * v

    @protected
    def get_scaling_factor(self, camera_params, point_proj):
        """
        Get function r(point_proj) that computes a scaling factor to undo the radial distortion:
        r(p) = 1.0 + k1 * ||p||^2 + k2 * ||p||^4.

        :param camera_params: intrinsic parameters: two radial distortion parameters k1 and k2
        :param point_proj: 2-D point which is a projection of 3-D point
        :return: a scaling factor to undo the radial distortion
        """
        k1 = camera_params[:, 7]
        k2 = camera_params[:, 8]
        n = np.sum(point_proj ** 2, axis=1)
        r = 1 + k1 * n + k2 * n ** 2
        return r
