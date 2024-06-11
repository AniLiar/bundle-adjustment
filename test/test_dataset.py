from unittest import TestCase

from src.dataset import Dataset, N_CAMERA_PARAMS
from src.dataset_visualisation import DatasetVisualisationService


class TestDataset(TestCase):
    def test_generate(self):
        dataset = Dataset()
        dataset.generate(n_cameras=2, n_points_3d=5, n_observations_2d=20)

        self.assertEqual(dataset.n_cameras, 2)
        self.assertEqual(dataset.n_points_3d, 5)
        self.assertEqual(dataset.n_observations_2d, 20)

        self.assertEqual(dataset.points_3d.shape, (5, 3))  # Проверка размерности массива 3D точек
        self.assertEqual(dataset.camera_params.shape, (2, N_CAMERA_PARAMS))  # Проверка размерности массива параметров камеры
        self.assertEqual(dataset.points_2d.shape, (20, 2))

        visualisation_service = DatasetVisualisationService(dataset)
        visualisation_service.show_point_2d()
        visualisation_service.show_point_3d()
        visualisation_service.show_camera_position()


    def test_generate_when_default_params(self):
        dataset = Dataset()
        dataset.generate()

        self.assertEqual(dataset.n_cameras, 1)
        self.assertEqual(dataset.n_points_3d, 10)
        self.assertEqual(dataset.n_observations_2d, 10)

        self.assertEqual(dataset.points_3d.shape, (10, 3))  # Проверка размерности массива 3D точек
        self.assertEqual(dataset.camera_params.shape, (1, N_CAMERA_PARAMS))  # Проверка размерности массива параметров камеры
        self.assertEqual(dataset.points_2d.shape, (10, 2))

        visualisation_service = DatasetVisualisationService(dataset)
        visualisation_service.show_point_2d()
        visualisation_service.show_point_3d()
        visualisation_service.show_camera_position()

    def test_generate_with_default_values(self):
        dataset = Dataset()
        dataset.generate_with_default_params()
        visualisation_service = DatasetVisualisationService(dataset)
        visualisation_service.show_point_2d()
        visualisation_service.show_point_3d()
        visualisation_service.show_camera_position()
        visualisation_service.show_point_3d_and_camera_position()
        print("camera position = {}".format(dataset.get_camera_pose_from(0)))
        print("3D point = {}".format(dataset.points_3d))

