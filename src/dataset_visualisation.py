import matplotlib.pyplot as plt
from random import randint

from src.dataset import Dataset


class DatasetVisualisationService:

    def __init__(self, dataset: Dataset):
        self.__dataset = dataset
        self.colors = []
        for i in range(self.__dataset.n_cameras):
            self.colors.append('#%06X' % randint(0, 0xFFFFFF))

    def show_point_2d(self):
        assert self.__dataset is not None, f"dataset should be not None"

        colors = []
        n = self.__dataset.n_cameras
        for i in range(n):
            colors.append('#%06X' % randint(0, 0xFFFFFF))

        fig = plt.figure()
        ax = fig.add_subplot(111)

        for i in range(len(self.__dataset.camera_indices)):
            point = self.__dataset.points_2d[i][:]
            print(point)
            ax.scatter(point[0], point[1], color=colors[self.__dataset.camera_indices[i]])

        plt.title("Problem {}: \n2D points of all images (cameras)".format(self.__dataset.problem_name))
        plt.xlabel("x of 2D point")
        plt.ylabel("y of 2D point")
        plt.show()

    def show_camera_position(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        camera_poses = self.__dataset.camera_params[:, 3:6]
        camera_index = 0
        for pose in camera_poses:
            ax.scatter(*pose, color=self.colors[camera_index])
            camera_index += 1
        plt.title("Problem {}: \nposes of images (cameras)".format(self.__dataset.problem_name))
        plt.xlabel("x of 2D point")
        plt.ylabel("y of 2D point")
        plt.show()

    def show_point_3d(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        for point_3d in self.__dataset.points_3d:
            ax.scatter(*point_3d)

        plt.title("Problem {}: \n3D points of all images (cameras)".format(self.__dataset.problem_name))
        plt.xlabel("x of 3D point")
        plt.ylabel("y of 3D point")
        # plt.zlabel("z of 3D point")
        plt.show()