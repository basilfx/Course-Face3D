"""
Copyright (c) 2012 Bas Stottelaar, Jeroen Senden
See the file LICENSE for copying permission.
"""

from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D

class Face(object):

    def __init__(self, abs_file):
        self.abs_file = abs_file
        self.compressed = False
        self.features = False
        self.key_points = {}
        self.reset()

    def reset(self):
        self.set_view(0, 0, self.abs_file.width, self.abs_file.height)

    def set_view(self, x_min, y_min, x_max, y_max):
        self.x_min = x_min
        self.y_min = y_min
        self.x_max = x_max
        self.y_max = y_max

    def center_at(self, x, y, delta_x, delta_y):
        # Translate and sanitize input
        x = int(x) + self.x_min
        y = int(y) + self.y_min
        delta_x = int(delta_x)
        delta_y = int(delta_y) 

        # Save difference
        old_x_min = self.x_min
        old_y_min = self.y_min

        # Check values
        if delta_x * 2 > self.abs_file.width:
            raise ValueError("Delta x out of range")

        if delta_y * 2 > self.abs_file.height:
            raise ValueError("Delta y out of range")

        # X axis
        if x + delta_x > self.abs_file.width:
            self.x_min = self.abs_file.width - delta_x - delta_x
            self.x_max = self.abs_file.width
        elif x - delta_x < 0:
            self.x_min = 0
            self.x_max = delta_x + delta_x
        else:
            self.x_min = x - delta_x
            self.x_max = x + delta_x

        # Y axis
        if y + delta_y > self.abs_file.height:
            self.y_min = self.abs_file.height - delta_y - delta_y
            self.y_max = self.abs_file.height
        elif y - delta_y < 0:
            self.y_min = 0
            self.y_max = delta_y + delta_y
        else:
            self.y_min = y - delta_y
            self.y_max = y + delta_y

        # Translate each point
        for name, point in self.key_points.iteritems():
            x,  y = point
            self.key_points[name] = (x + (old_x_min - self.x_min), y + (old_y_min - self.y_min))

    def add_key_point(name, x, y):
        self.key_points[name] = (x + self.x_min, y + self.y_min)

    @property
    def X(self):
        return self.abs_file.data['X'][range(self.y_min, self.y_max), :][:, range(self.x_min, self.x_max)]

    @property
    def Y(self):
        return self.abs_file.data['Y'][range(self.y_min, self.y_max), :][:, range(self.x_min, self.x_max)]

    @property
    def Z(self):
        return self.abs_file.data['Z'][range(self.y_min, self.y_max), :][:, range(self.x_min, self.x_max)]

    @property
    def width(self):
        return self.x_max - self.x_min 

    @property
    def height(self):
        return self.y_max - self.y_min

    def plot_3d(self):
        figure = pyplot.figure()

        # Draw surface
        axis = Axes3D(figure)
        axis.plot_surface(X=self.X, Y=self.Y, Z=self.Z)

        return figure

    def compress(self):
        self.abs_file.data['X'] = self.X
        self.abs_file.data['Y'] = self.Y
        self.abs_file.data['Z'] = self.Z

        self.abs_file.col_size = self.width
        self.abs_file.row_size = self.height

        self.compressed = True
        self.reset()

