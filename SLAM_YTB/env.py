import math
import pygame
class buildEnvironment:
    def __init__(self, MapDimensions):
        pygame.init()
        self.pointCloud = []
        self.trajectory = []
        self.position = pygame.mouse.get_pos()
        self.externalMap = pygame.image.load('map1.png')
        self.maph, self.mapw = MapDimensions
        self.MapWindowName = 'RRT path planning'
        pygame.display.set_caption(self.MapWindowName)
        self.map = pygame.display.set_mode((self.mapw, self.maph))
        self.map.blit(self.externalMap,(0,0))

        # colors
        self.black = (0, 0, 0)
        self.grey = (70, 70, 70)
        self.Blue = (0, 0, 255)
        self.Green = (0, 255, 0)
        self.Red = (255, 0, 0)
        
    def AD2pos(self, distance, angle, robotPosition):
        x = distance * math.cos(angle) + robotPosition[0]
        y = -distance * math.sin(angle) + robotPosition[1]
        return (int(x), int(y))

    def dataStorage(self, data):
        # print(len(self.pointCloud))
        if data != False:
            for element in data:
                point = self.AD2pos(element[0], element[1], element[2])
                if point not in self.pointCloud:
                    self.pointCloud.append(point)
                if element[2] not in self.trajectory:
                    self.trajectory.append(element[2])
        else:
            print("Robot is not inside the MAP")

    def show_sensorData(self):
        self.infomap = self.map.copy()
        # print(self.pointCloud)
        for point in self.pointCloud:
            self.infomap.set_at((int(point[0]), int(point[1])), (255, 0, 0))
        for point in self.trajectory:
            self.infomap.set_at((int(point[0]), int(point[1])), (0, 200, 255))
            print("Robot position", point)

    # def show_robot_trajectory(self):
    #     self.infomap = self.map.copy()
    #     for point in self.pointCloud:
    #         self.infomap.set_at((int(point[0]), int(point[1])), (0, 0, 255))