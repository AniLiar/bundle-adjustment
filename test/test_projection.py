import numpy as np
from unittest import TestCase

from src.dataset import Dataset
from src.projection import CoordinateService


class Test(TestCase):

    def test_forward_projection(self):
        ds = Dataset()
        path = "resources/ladybug/problem-49-7776-pre.txt.bz2"
        ds.read_from_file(path)
        camera_params = ds.get_camera_params_from(0)
        print('camera params #0:')
        print(camera_params)

        expected_projection = np.array([-3.326500e+02, 2.620900e+02])

        print(expected_projection)
        camera_params = np.array([1, 2, 3, 4, 5, 6])
        x_3d = np.array([0.1, 0.2, 0.3])

        coord_service = CoordinateService()
        result = CoordinateService.get_forward_projection(coord_service, camera_params, x_3d)

        self.assertEqual(len(result), 2)
        self.assertIsInstance(result, list)
        for i in range(len(result)):
            self.assertIsInstance(result[i], (int, float))
            self.assertEqual(result[i], expected_projection[i])


class TestCoordinateService(TestCase):
    def test_transform_to_intersecting_planes(self):
        coord_service = CoordinateService()

        # Генерируем случайные коэффициенты A, B, C, D
        A = np.random.uniform(-10, 10)
        B = np.random.uniform(-10, 10)
        C = np.random.uniform(-10, 10)
        D = np.random.uniform(-10, 10)

        # Генерируем случайные точки x1, y1
        x1 = np.random.uniform(-10, 10)
        y1 = np.random.uniform(-10, 10)

        # Вычисляем z по уравнению плоскости
        z1 = (-A * x1 - B * y1 - D) / C

        # Создаем точку P1
        P1 = np.array([x1, y1, z1])

        # Генерируем другие случайные точки x2, y2
        x2 = np.random.uniform(-10, 10)
        y2 = np.random.uniform(-10, 10)

        # Вычисляем z для второй точки
        z2 = (-A * x2 - B * y2 - D) / C

        # Создаем точку P2
        P2 = np.array([x2, y2, z2])
        ray_2point = np.concatenate((P1, P2), axis=0)

        # Ожидаемый результат
        # expected_result = np.array([[...], [...]])  # Вставьте ожидаемый результат

        # Вызываем метод
        result = coord_service.transform_to_intersecting_planes(ray_2point)
        self.assertEqual(result[0], A)
        self.assertEqual(result[1], B)
        self.assertEqual(result[2], C)
        self.assertEqual(result[3], D)

        # Проверяем результат
        print("here")
        # self.assertTrue(np.allclose(result, expected_result))  # Подставьте правильное условие для проверки


    def test_transform_to_plane(self):
        coord_service = CoordinateService()

        # Генерируем случайные коэффициенты A, B, C, D
        A = 1
        B = np.random.uniform(-10, 10)
        C = np.random.uniform(-10, 10)
        D = np.random.uniform(-10, 10)

        print(f"Уравнение плоскости: {A}*x + {B}*y + {C}*z + {D} = 0")

        P1 = self.generate_random_point_for_plane(A, B, C, D)
        P2 = self.generate_random_point_for_plane(A, B, C, D)
        P3 = self.generate_random_point_for_plane(A, B, C, D)

        # resultA, resultB, resultC, resultD = coord_service.to_plane(P1, P2, P3)
        result = coord_service.to_plane(P1, P2, P3)
        result.reshape(-1)
        p = result[0,0]
        print(f"Уравнение плоскости: {result[0]}*x + {result[1]}*y + {result[2]}*z + {result[3]} = 0")

        eps = 0.000000001
        self.assertTrue(result[0] - A < eps)
        self.assertTrue(result[1] - B < eps)
        self.assertTrue(result[2] - C < eps)
        self.assertTrue(result[3] - D < eps)


    def generate_random_point_for_plane(self, A, B, C, D):
        # Генерируем случайные точки x1, y1
        x = np.random.uniform(-10, 10)
        y = np.random.uniform(-10, 10)

        # Вычисляем z по уравнению плоскости: Ax + By + Cz + D = 0
        z = (-A * x - B * y - D) / C
        # Создаем точку P1
        point = np.array([x, y, z])

        eq = A*x + B*y + C*z + D
        # print("Generate point: A*x + B*y + C*z + D == {}".format(eq))
        eps = 0.000000001
        assert eq < eps, "It should be equal!"

        return point

    def test_transform_to_plane2(self):
        coord_service = CoordinateService()

        # Генерируем случайные коэффициенты A, B, C, D
        A = 1
        B = np.random.uniform(-10, 10)
        C = np.random.uniform(-10, 10)
        D = np.random.uniform(-10, 10)

        print(f"Уравнение плоскости: {A}*x + {B}*y + {C}*z + {D} = 0")

        P1 = self.generate_random_point_for_plane(A, B, C, D)
        P2 = self.generate_random_point_for_plane(A, B, C, D)
        # P3 = self.generate_random_point_for_plane(A, B, C, D)

        # resultA, resultB, resultC, resultD = coord_service.to_plane(P1, P2, P3)
        result = coord_service.to_plane2(P1, P2)
        result.reshape(-1)
        p = result[0, 0]
        print(f"Уравнение плоскости: {result[0]}*x + {result[1]}*y + {result[2]}*z + {result[3]} = 0")

        eps = 0.000000001
        self.assertTrue(result[0] - A < eps)
        self.assertTrue(result[1] - B < eps)
        self.assertTrue(result[2] - C < eps)
        self.assertTrue(result[3] - D < eps)
