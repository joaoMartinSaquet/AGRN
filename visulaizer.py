import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from matplotlib.patches import FancyArrowPatch

class GRNVisualizer:
    def __init__(self, grn):
        self.grn = grn
        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        self.pos = {}

        # Column layout
        y_spacing = 1.0
        x_input, x_reg, x_output = 0.0, 1.5, 3.0

        # Regulators
        for idx, i in enumerate(range(self.grn.nin, self.grn.nin + self.grn.nreg)):
            self.pos[i] = (x_reg, -idx * y_spacing)

        # Center inputs/outputs
        if self.grn.nreg > 0:
            y_center = -((self.grn.nreg - 1) * y_spacing) / 2
        else:
            y_center = 0

        for i in range(self.grn.nin):
            self.pos[i] = (x_input, y_center)
        for i in range(self.grn.nin + self.grn.nreg, self.grn.size):
            self.pos[i] = (x_output, y_center)

        # Node colors
        self.node_colors = []
        for i in range(self.grn.size):
            if i < self.grn.nin:
                self.node_colors.append('skyblue')
            elif i < self.grn.nin + self.grn.nreg:
                self.node_colors.append('lightgreen')
            else:
                self.node_colors.append('orange')

        # Draw nodes
        self.node_scatter = self.ax.scatter(
            [self.pos[n][0] for n in range(self.grn.size)],
            [self.pos[n][1] for n in range(self.grn.size)],
            s=[300 for _ in range(self.grn.size)],
            c=self.node_colors, edgecolors='black'
        )

        # Draw edges with FancyArrowPatch (curved arcs)
        for i in range(self.grn.size):
            for j in range(self.grn.size):
                if self.grn.enh_affinity_matrix[i, j] > 0.05:
                    arrow = FancyArrowPatch(self.pos[i], self.pos[j],
                                            connectionstyle="arc3,rad=0.2",
                                            arrowstyle='-|>',
                                            color="green", alpha=0.5,
                                            lw=1)
                    self.ax.add_patch(arrow)
                if self.grn.inh_affinity_matrix[i, j] > 0.05:
                    arrow = FancyArrowPatch(self.pos[i], self.pos[j],
                                            connectionstyle="arc3,rad=-0.2",
                                            arrowstyle='-|>',
                                            color="red", alpha=0.5,
                                            lw=1)
                    self.ax.add_patch(arrow)

        self.ax.axis("off")
        plt.ion()
        plt.show()

    def update(self):
        """Refresh visualization after GRN state changes."""
        concentrations = self.grn.concentrations
        sizes = 100 + 2000 * concentrations
        self.node_scatter.set_sizes(sizes)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
