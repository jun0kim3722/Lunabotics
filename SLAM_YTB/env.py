import math
import pygame

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
                    particle.Ct.append([True, 0])
                else:
                    idx = self.pointCloud.index(point)
                    particle.Ct.append([False, idx])

        else:
            print("No lazer data")

        if position[0:2] not in self.true_trajectory:
            self.true_trajectory.append(position)

    def show_sensorData(self, particle_set, particle_bar):
        self.infomap = self.map.copy()

        for point in self.pointCloud:
            self.infomap.set_at((int(point[0]), int(point[1])), (255, 255, 255))

        for point in self.true_trajectory:
            self.infomap.set_at((int(point[0]), int(point[1])), (0, 200, 255))
        print("Robot's actual position", self.true_trajectory[-1])
        
        self.infomap.set_at((int(particle_bar[0]), int(particle_bar[1])), (50, 255, 0))
        # print("Robot's estimated position", particle_bar)
        
        for point in particle_set:
            self.infomap.set_at((int(point[0]), int(point[1])), (255, 0, 0))
