import env
import sensors
import robot_drive
import pygame
import Particle_filter
import math

environment = env.buildEnvironment((600, 1200))
environment.originalMap = environment.map.copy()
laser = sensors.Laserensor(300, environment.originalMap, uncertainty=(0.5, 0.01))
robot = robot_drive.Robot([200, 200], 10)
environment.map.fill((0,0,0))
environment.infomap = environment.map.copy()
running = True
particle = Particle_filter.particle_filter([0.01,0.01, 0.01], [0.5,0.5,0.01], 10, [600, 1200])

dt = 0
lasttime = pygame.time.get_ticks()

while running:
    dt = (pygame.time.get_ticks()-lasttime) / 1000
    sensorON = True
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            robot.control(event, dt)
            x = ((robot.vl + robot.vr) / 2) * math.cos(robot.theta)
            y = ((robot.vl + robot.vr) / 2) * math.sin(robot.theta)
            theta = (robot.vr - robot.vl) / robot.w * dt

            position = [robot.x, robot.y, robot.theta]
            laser.position = position
            sensor_data = laser.sense_obstacles()
            environment.dataStorage(sensor_data, position, particle)

            dis_ang = [i[0:2] for i in sensor_data]
            particle_set, particle_bar = particle.creating_particles([x, y, theta], dis_ang)
            environment.show_sensorData(particle_set, particle_bar)
        environment.map.blit(environment.infomap, (0,0))
        pygame.display.update()


