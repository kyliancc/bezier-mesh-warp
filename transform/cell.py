import numpy as np
import math


class TransformCell:
    @staticmethod
    def _cubic_bezier(t, points):
        ret = np.zeros((*t.shape[:-1], 2), dtype=np.float32)
        for i in range(4):
            ret += math.comb(3, i) * (1 - t) ** (3 - i) * t ** i * points[i]
        return ret

    def __init__(self, x_range=(0.0, 1.0), y_range=(0.0, 1.0)):

        self.x_min = x_range[0]
        self.x_max = x_range[1]
        self.y_min = y_range[0]
        self.y_max = y_range[1]

        self.corners = np.array([
            [[self.y_min, self.x_min], [self.y_min, self.x_max]],
            [[self.y_max, self.x_min], [self.y_max, self.x_max]]
        ], dtype=np.float32)

        self.orient_bottom = np.array([
            [self.y_min, self.x_min + 1/3 * (self.x_max - self.x_min)],
            [self.y_min, self.x_min + 2/3 * (self.x_max - self.x_min)]
        ], dtype=np.float32)

        self.orient_top = np.array([
            [self.y_max, self.x_min + 1/3 * (self.x_max - self.x_min)],
            [self.y_max, self.x_min + 2/3 * (self.x_max - self.x_min)]
        ], dtype=np.float32)

        self.orient_left = np.array([
            [self.y_min + 1/3 * (self.y_max - self.y_min), self.x_min],
            [self.y_min + 2/3 * (self.y_max - self.y_min), self.x_min]
        ], dtype=np.float32)

        self.orient_right = np.array([
            [self.y_min + 1/3 * (self.y_max - self.y_min), self.x_max],
            [self.y_min + 2/3 * (self.y_max - self.y_min), self.x_max]
        ], dtype=np.float32)

    def bottom_contour(self, t):
        return self._cubic_bezier(t, [
            self.corners[0,0],
            self.orient_bottom[0],
            self.orient_bottom[1],
            self.corners[0,1]
        ])

    def top_contour(self, t):
        return self._cubic_bezier(t, [
            self.corners[1,0],
            self.orient_top[0],
            self.orient_top[1],
            self.corners[1,1]
        ])

    def left_contour(self, t):
        return self._cubic_bezier(t, [
            self.corners[0,0],
            self.orient_left[0],
            self.orient_left[1],
            self.corners[1,0]
        ])

    def right_contour(self, t):
        return self._cubic_bezier(t, [
            self.corners[0,1],
            self.orient_right[0],
            self.orient_right[1],
            self.corners[1,1]
        ])

    def map(self, uv):
        v, u = np.split(uv, 2, axis=-1)

        l1 = (1 - v) * self.bottom_contour(u) + v * self.top_contour(u)
        l2 = (1 - u) * self.left_contour(v) + u * self.right_contour(v)
        b = ((1 - u) * (1 - v) * self.corners[0,0] +
             u * (1 - v) * self.corners[0,1] +
             (1 - u) * v * self.corners[1,0] +
             u * v * self.corners[1,1])
        coons = l1 + l2 - b

        return coons

    def map_grid(self, u, v):
        # developping...
        pass
