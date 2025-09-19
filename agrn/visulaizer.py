import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D
import math

class GRNVisualizer:
    def __init__(self, grn,
                 interval=500,
                 max_per_col=10,
                 col_width=5,
                 y_spacing=3.0,
                 x_input=0.0,
                 x_reg_start=5,
                 x_output=20,
                 random_seed=None):
        """
        grn: object with attributes
            - nin, nreg, size
            - enh_affinity_matrix, inh_affinity_matrix
            - concentrations (1D np.array)
            optional: nout, step() method
        """
        if random_seed is not None:
            np.random.seed(random_seed)

        self.grn = grn
        self.interval = interval
        self.max_per_col = max_per_col
        self.col_width = col_width
        self.y_spacing = y_spacing
        self.x_input = x_input
        self.x_reg_start = x_reg_start
        self.x_output = x_output

        # be robust if grn doesn't have nout attribute
        self.nin = getattr(self.grn, "nin", 0)
        self.nreg = getattr(self.grn, "nreg", 0)
        self.size = getattr(self.grn, "size", (self.nin + self.nreg))
        self.nout = getattr(self.grn, "nout", self.size - self.nin - self.nreg)

        # ensure concentrations exist
        if not hasattr(self.grn, "concentrations") or self.grn.concentrations is None:
            self.grn.concentrations = np.ones(self.size) / max(1, self.size)

        # ensure affinity matrices exist (fallback zeros)
        if not hasattr(self.grn, "enh_affinity_matrix"):
            self.grn.enh_affinity_matrix = np.zeros((self.size, self.size))
        if not hasattr(self.grn, "inh_affinity_matrix"):
            self.grn.inh_affinity_matrix = np.zeros((self.size, self.size))

        self.fig, self.ax = plt.subplots(figsize=(10, 7))
        self.pos = {}

        # ---------- compute regulator columns (wrap at max_per_col) ----------
        columns = []
        for idx, node in enumerate(range(self.nin, self.nin + self.nreg)):
            col_idx = idx // self.max_per_col
            if len(columns) <= col_idx:
                columns.append([])
            columns[col_idx].append(node)

        # place nodes in each column, centered vertically per column
        for col_idx, nodes_in_col in enumerate(columns):
            col_x = self.x_reg_start + col_idx * self.col_width
            n_in_col = len(nodes_in_col)
            # center vertically: top_y = (n_in_col - 1) * y_spacing / 2
            top_y = (n_in_col - 1) * self.y_spacing / 2.0
            for r, node in enumerate(nodes_in_col):
                ry = top_y - r * self.y_spacing
                self.pos[node] = (col_x, ry)

        # ---------- inputs: place vertically centered if multiple ----------
        if self.nin > 0:
            top_y_in = (self.nin - 1) * self.y_spacing / 2.0
            for idx, node in enumerate(range(0, self.nin)):
                ry = top_y_in - idx * self.y_spacing
                self.pos[node] = (self.x_input, ry)

        # ---------- outputs: place vertically centered ----------
        if self.nout > 0:
            top_y_out = (self.nout - 1) * self.y_spacing / 2.0
            for idx, node in enumerate(range(self.nin + self.nreg, self.size)):
                ry = top_y_out - idx * self.y_spacing
                self.pos[node] = (self.x_output, ry)

        # ---------- node colors ----------
        self.node_colors = []
        for i in range(self.size):
            if i < self.nin:
                self.node_colors.append('skyblue')
            elif i < self.nin + self.nreg:
                self.node_colors.append('lightgreen')
            else:
                self.node_colors.append('orange')

        # ---------- initial node sizes (based on concentrations) ----------
        sizes = 100 + 2000 * np.array(self.grn.concentrations)

        # draw nodes
        xs = [self.pos[n][0] for n in range(self.size)]
        ys = [self.pos[n][1] for n in range(self.size)]
        self.node_scatter = self.ax.scatter(xs, ys, s=sizes,
                                           c=self.node_colors, edgecolors='black', zorder=3)

        # ---------- draw edges (alpha scaled with affinity strength) ----------
        # compute normalization to get visible alpha mapping
        enh_max = np.max(self.grn.enh_affinity_matrix) if np.any(self.grn.enh_affinity_matrix) else 1.0
        inh_max = np.max(self.grn.inh_affinity_matrix) if np.any(self.grn.inh_affinity_matrix) else 1.0
        # small floor to avoid invisible lines
        alpha_floor = 0.05

        for i in range(self.size):
            for j in range(self.size):
                w_enh = float(self.grn.enh_affinity_matrix[i, j])
                w_inh = float(self.grn.inh_affinity_matrix[i, j])

                if w_enh > 1e-6:
                    alpha = float(np.clip(w_enh / max(enh_max, 1e-12), alpha_floor, 1.0))
                    arrow = FancyArrowPatch(self.pos[i], self.pos[j],
                                            connectionstyle="arc3,rad=0.18",
                                            arrowstyle='-|>',
                                            color="green", alpha=alpha, lw=1, zorder=1)
                    self.ax.add_patch(arrow)
                if w_inh > 1e-6:
                    alpha = float(np.clip(w_inh / max(inh_max, 1e-12), alpha_floor, 1.0))
                    arrow = FancyArrowPatch(self.pos[i], self.pos[j],
                                            connectionstyle="arc3,rad=-0.18",
                                            arrowstyle='-|>',
                                            color="red", alpha=alpha, lw=1, zorder=1)
                    self.ax.add_patch(arrow)

        # ---------- labels ----------
        for i, (x, y) in self.pos.items():
            if i < self.nin:
                label = f"I{i}"
            elif i < self.nin + self.nreg:
                label = f"R{i - self.nin}"
            else:
                label = f"O{i - self.nin - self.nreg}"
            self.ax.text(x, y + 0.12, label, ha='center', va='bottom', fontsize=9, zorder=4)

        # legend
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', label='Input', markerfacecolor='skyblue', markersize=10),
            Line2D([0], [0], marker='o', color='w', label='Regulator', markerfacecolor='lightgreen', markersize=10),
            Line2D([0], [0], marker='o', color='w', label='Output', markerfacecolor='orange', markersize=10),
            Line2D([0], [0], color='green', lw=2, label='Enhancing'),
            Line2D([0], [0], color='red', lw=2, label='Inhibiting'),
        ]
        self.ax.legend(handles=legend_elements, loc='upper right')

        # ---------- make sure everything is visible ----------
        min_x = min(xs) - 0.8
        max_x = max(xs) + 0.8
        min_y = min(ys) - 0.8
        max_y = max(ys) + 0.8
        self.ax.set_xlim(min_x, max_x)
        self.ax.set_ylim(max_y, min_y)  # flip y so top is top (optional)
        self.ax.set_aspect('equal', adjustable='box')
        self.ax.axis('off')

        # create animation (disable frame caching to avoid the warning)
        self.ani = FuncAnimation(self.fig, self.update,
                                 interval=self.interval,
                                 blit=False,
                                 cache_frame_data=False)

    def update(self, frame):
        """Refresh visualization after GRN state changes.
           If grn has a step() method we call it here (so no background thread needed)."""
        if hasattr(self.grn, "step") and callable(self.grn.step):
            self.grn.step()

        # ensure concentrations length matches size
        if len(self.grn.concentrations) != self.size:
            # try to fix by resizing/normalizing
            c = np.array(self.grn.concentrations)
            if c.size < self.size:
                c = np.pad(c, (0, self.size - c.size), mode='constant', constant_values=0.0)
            else:
                c = c[:self.size]
            if c.sum() > 0:
                c = c / c.sum()
            self.grn.concentrations = c

        sizes = 100 + 2000 * np.array(self.grn.concentrations)
        self.node_scatter.set_sizes(sizes)
        # return artists (blit=False so not strictly necessary)
        return (self.node_scatter,)

    def show(self):
        plt.show()


# ----------------- minimal FakeGRN for testing -----------------
if __name__ == "__main__":
    class FakeGRN:
        def __init__(self, nin=1, nreg=23, nout=1):
            self.nin = nin
            self.nreg = nreg
            self.nout = nout
            self.size = nin + nreg + nout
            self.concentrations = np.random.rand(self.size)
            self.concentrations /= self.concentrations.sum()
            # affinities in [0,1)
            self.enh_affinity_matrix = np.random.rand(self.size, self.size) * 0.8
            self.inh_affinity_matrix = np.random.rand(self.size, self.size) * 0.6

        def step(self):
            # small random walk in concentration space
            self.concentrations += 0.03 * (np.random.rand(self.size) - 0.5)
            self.concentrations = np.clip(self.concentrations, 0, None)
            s = self.concentrations.sum()
            if s > 0:
                self.concentrations /= s

    grn = FakeGRN(nin=2, nreg=23, nout=2)   # test many regulators -> multiple columns
    viz = GRNVisualizer(grn, interval=250, max_per_col=10, col_width=0.9, random_seed=42)
    viz.show()
