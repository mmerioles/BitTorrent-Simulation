import matplotlib.pyplot as plt
from matplotlib.widgets import Button

from bittorrent_sim import BitTorrentSimulation, SimulationConfig
from bittorrent_sim.visualization import build_layout, refresh_axis


simulation = BitTorrentSimulation(SimulationConfig())
pos = build_layout(simulation)
fig = None
graph_ax = None


def redraw() -> None:
    global fig, graph_ax, pos
    if fig is None or graph_ax is None:
        return

    pos = refresh_axis(simulation, ax=graph_ax, pos=pos)
    fig.canvas.draw_idle()


def update(event=None):
    simulation.update()
    redraw()


def add_node(event):
    new_node_id = simulation.add_node()
    print("Node", new_node_id, "added!")
    redraw()


def delete_node(event):
    removed_node_id = simulation.delete_node()
    if removed_node_id is not None:
        print("Node", removed_node_id, "removed!")
    redraw()


def print_total_nodes(event):
    print("+---------------------------------------------------------------------------------------------------------------------------------------------------------+")
    for row in simulation.node_rows():
        print(row)
    print("+---------------------------------------------------------------------------------------------------------------------------------------------------------+")
    print("G Nodes:", list(simulation.graph.nodes))
    print("+---------------------------------------------------------------------------------------------------------------------------------------------------------+")


fig, graph_ax = plt.subplots(figsize=(10, 8))
plt.subplots_adjust(bottom=0.2)

ax_add_node = plt.axes([0.6, 0.05, 0.1, 0.075])
ax_delete_node = plt.axes([0.8, 0.05, 0.1, 0.075])
ax_print = plt.axes([0.2, 0.05, 0.1, 0.075])
ax_update = plt.axes([0.4, 0.05, 0.1, 0.075])

btn_add_node = Button(ax_add_node, "Add Node")
btn_delete_node = Button(ax_delete_node, "Delete Node")
btn_print = Button(ax_print, "Print Nodes")
btn_update = Button(ax_update, "Update")

btn_add_node.on_clicked(add_node)
btn_delete_node.on_clicked(delete_node)
btn_print.on_clicked(print_total_nodes)
btn_update.on_clicked(update)

redraw()
plt.show()
