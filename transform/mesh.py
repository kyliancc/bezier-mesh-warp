import numpy as np

from transform.cell import TransformCell


class TransformMesh:
    def __init__(self, rows=4, cols=4,
                 x_range=(0.0, 1.0), y_range=(0.0, 1.0),
                 x_mid_bound=None, y_mid_bound=None):

        self.rows = rows
        self.cols = cols

        if x_mid_bound is not None:
            self.x_bound = (0.0, *x_mid_bound, 1.0)
        else:
            x_bound_list = []
            for i in range(cols+1):
                x_bound_list.append(i / cols)
            self.x_bound = tuple(x_bound_list)

        if y_mid_bound is not None:
            self.y_bound = (0.0, *y_mid_bound, 1.0)
        else:
            y_bound_list = []
            for i in range(rows+1):
                y_bound_list.append(i / rows)
            self.y_bound = tuple(y_bound_list)

        self.x_min = x_range[0]
        self.x_max = x_range[1]
        self.y_min = y_range[0]
        self.y_max = y_range[1]

        self.cells = []
        for j in range(rows):
            self.cells.append([])
            for i in range(cols):
                self.cells[-1].append(TransformCell(
                    x_range=(self.x_min + self.x_bound[i] * (self.x_max - self.x_min), self.x_min + self.x_bound[i+1] * (self.x_max - self.x_min)),
                    y_range=(self.y_min + self.y_bound[j] * (self.y_max - self.y_min), self.y_min + self.y_bound[j+1] * (self.y_max - self.y_min))
                ))

    def map(self, uv):
        uv_shape = uv.shape
        uv = uv.reshape(-1, 2)
        ret = np.zeros_like(uv, dtype=np.float32)
        for i in range(ret.shape[0]):
            x_idx = 0
            y_idx = 0

            for j in range(self.cols-1):
                if uv[i,1] > self.x_bound[j+1]:
                    x_idx += 1
                else:
                    break

            for j in range(self.rows-1):
                if uv[i,0] > self.y_bound[j+1]:
                    y_idx += 1
                else:
                    break

            cell = self.cells[y_idx][x_idx]

            cell_uv = ((uv[i] - np.array([self.y_bound[y_idx], self.x_bound[x_idx]])) /
                       np.array([self.y_bound[y_idx+1] - self.y_bound[y_idx],
                                 self.x_bound[x_idx+1] - self.x_bound[x_idx]]))

            ret[i] = cell.map(cell_uv)

        return ret.reshape(*uv_shape)

    def map_grid(self, rows, cols):
        x_lin = np.linspace(0.0, 1.0, cols)
        y_lin = np.linspace(0.0, 1.0, rows)

        x_idx = np.zeros_like(x_lin, dtype=np.uint8)
        y_idx = np.zeros_like(y_lin, dtype=np.uint8)

        curr_idx = 0
        for i in range(x_lin.shape[0]):
            while True:
                if x_lin[i] > self.x_bound[curr_idx+1]:
                    curr_idx += 1
                else:
                    break

            x_idx = curr_idx

        curr_idx = 0
        for i in range(y_lin.shape[0]):
            while True:
                if y_lin[i] > self.y_bound[curr_idx + 1]:
                    curr_idx += 1
                else:
                    break

            y_idx = curr_idx

        # developing...





mesh = TransformMesh(x_range=(300, 800), y_range=(100, 200))
uv = np.array([
    [0.2, 0.8],
])
print(mesh.map(uv))
