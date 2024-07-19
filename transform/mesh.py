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

    def _to_cell_u(self, u, cell_x_idx):
        return (u - self.x_bound[cell_x_idx]) / (self.x_bound[cell_x_idx + 1] - self.x_bound[cell_x_idx])

    def _to_cell_v(self, v, cell_y_idx):
        return (v - self.y_bound[cell_y_idx]) / (self.y_bound[cell_y_idx + 1] - self.y_bound[cell_y_idx])

    def _to_cell_uv(self, uv, cell_x_idx, cell_y_idx):
        return ((uv - np.array([self.y_bound[cell_y_idx], self.x_bound[cell_x_idx]])) /
                   np.array([self.y_bound[cell_y_idx + 1] - self.y_bound[cell_y_idx],
                             self.x_bound[cell_x_idx + 1] - self.x_bound[cell_x_idx]]))

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

            cell_uv = self._to_cell_uv(uv[i], x_idx, y_idx)

            ret[i] = cell.map(cell_uv)

        return ret.reshape(*uv_shape)

    def map_grid(self, rows, cols):
        u_lin = np.linspace(0.0, 1.0, cols)
        v_lin = np.linspace(0.0, 1.0, rows)

        u_grid_bound = np.digitize(u_lin, self.x_bound)
        u_grid_bound[-1] = cols
        v_grid_bound = np.digitize(v_lin, self.y_bound)
        v_grid_bound[-1] = cols

        for j in range(self.rows):
            for i in range(self.cols):
                inter_u_lin = u_lin[u_grid_bound[i]:u_grid_bound[i + 1]]
                inter_v_lin = v_lin[v_grid_bound[j]:v_grid_bound[j + 1]]

                cell = self.cells[j][i]

                cell_u = self._to_cell_u(inter_u_lin, i)
                cell_v = self._to_cell_v(inter_v_lin, j)

                cell.map_grid(cell_u, cell_v)

                # developping...





mesh = TransformMesh(x_range=(300, 800), y_range=(100, 200))
uv = np.array([
    [0.2, 0.8],
])
print(mesh.map(uv))
