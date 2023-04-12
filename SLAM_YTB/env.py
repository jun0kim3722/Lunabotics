import math
import pygame
import Particle_filter

class buildEnvironment:
    def __init__(self, MapDimensions):
        pygame.init()
        self.pointCloud = []
        self.true_trajectory = []
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

    def dataStorage(self, data, position, particle):
        if data != False:
            for element in data:
                point = self.AD2pos(element[0], element[1], element[2])

                if point not in self.pointCloud:
                    self.pointCloud.append(point)
                    particle.creating_particle.Ct = True
                else:
                    particle.creating_particle.Ct = False

        else:
            print("No lazer data")

        if position not in self.true_trajectory:
            self.true_trajectory.append(position)

    def show_sensorData(self):
        self.infomap = self.map.copy()
        # print(self.pointCloud)
        for point in self.pointCloud:
            self.infomap.set_at((int(point[0]), int(point[1])), (255, 255, 255))
        for point in self.true_trajectory:
            self.infomap.set_at((int(point[0]), int(point[1])), (0, 200, 255))
            print("Robot position", point)
