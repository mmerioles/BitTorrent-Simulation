import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button
import random
from  collections import OrderedDict

SIZE_OF_FILE = 30
INITIAL_NUMBER_OF_NODES = 100
PEER_OPTIMISTIC_SELECTION_COUNT = 2
TOTAL_NUMBER_OF_CONNECTIONS = 4
OPTIMISTICALLY_UNCHOKE_CONSTANT = 4

class Node:
    def __init__(self, node_number, is_added_chunk=False):
        self.node_number = node_number
        self.connected_to = []
        
        if is_added_chunk:
            self.chunks = []    
        else:
            self.chunks = self.generate_random_chunks()
            
        self.upload_rate = random.randint(1, 5)

    def add_connection(self, other_node_number):
        if other_node_number not in self.connected_to:
            self.connected_to.append(other_node_number)

    def remove_connection(self, other_node_number):
        if other_node_number in self.connected_to:
            self.connected_to.remove(other_node_number)

    def generate_random_chunks(self):
        # Each node starts off with 2 or 3 chunks (if not added)
        num_chunks = random.randint(2, 3)  
        return random.sample(range(1, SIZE_OF_FILE+1), num_chunks)
    
    def identify_rarest_chunk_among_best_connections(self, total_nodes):
        chunk_frequency = {}
        peer_chunk_source = {}
        own_chunks = set(self.chunks)
        
        highest_rate_peers = self.identify_highest_rate_peers()
        
        # Finding chunk frequency
        for node_number in highest_rate_peers.keys():
            connected_node_chunks = total_nodes[node_number].chunks
            for chunk in connected_node_chunks:
                if chunk not in own_chunks:
                    chunk_frequency[chunk] = chunk_frequency.get(chunk, 0) + 1
                    peer_chunk_source[chunk] = node_number

        # Find the rarest chunk
        if chunk_frequency:
            rarest_chunk = min(chunk_frequency, key=chunk_frequency.get)
            return (rarest_chunk, peer_chunk_source[rarest_chunk])
        else:
            return (None, None)
        
    def identify_highest_rate_peers(self):
        highest_rate_peers = {}
        for node_number in self.connected_to:
            if len(highest_rate_peers) < PEER_OPTIMISTIC_SELECTION_COUNT:
                highest_rate_peers[node_number] = total_nodes[node_number].upload_rate
            else:
                lowest_value_node = min(highest_rate_peers, key=highest_rate_peers.get)
                if highest_rate_peers[lowest_value_node] < total_nodes[node_number].upload_rate:
                    highest_rate_peers.pop(lowest_value_node)
                    highest_rate_peers[node_number] = total_nodes[node_number].upload_rate
        return highest_rate_peers
    
    def exchange_data(self):
        for _ in range(self.upload_rate):
            rarest_chunk, provider_node = self.identify_rarest_chunk_among_best_connections(total_nodes)
            if rarest_chunk is not None and len(self.chunks) < SIZE_OF_FILE:
                # Node grabs a rarest chunk among the best peers
                self.chunks.append(rarest_chunk)
                if self.chunks:  
                    # In return, node sends one of its own chunks to the peer that provided the chunk
                    chunk_to_send = random.choice(self.chunks)
                    if chunk_to_send not in total_nodes[provider_node].chunks:
                        total_nodes[provider_node].chunks.append(chunk_to_send)
        
    def get_data(self):
        connected_to_formatted = ', '.join(f"{num:2d}" for num in sorted(self.connected_to))
        chunks_formatted = ', '.join(f"{chunk:2d}" for chunk in sorted(self.chunks))
        rarest_chunk, _ = self.identify_rarest_chunk_among_best_connections(total_nodes)
        rarest_chunk_formatted = f"{rarest_chunk:2d}" if rarest_chunk is not None else "None"

        max_connected_width = 50  
        max_chunks_width = 40     

        return (f"Node: {self.node_number:2d} | Connected to: [{connected_to_formatted:<{max_connected_width}}] | "
                f"Chunks: [{chunks_formatted:<{max_chunks_width}}] | RAREST CHUNK: {rarest_chunk_formatted} | DOWNLOAD COMPLETION PERCENTAGE: {str(round(len(self.chunks)/SIZE_OF_FILE* 100, 2))}% | UPLOAD RATE: {str(self.upload_rate)} | BEST PEERS: {str(self.identify_highest_rate_peers())}")
    
def optimisically_unchoke_peers():
    global total_nodes, G

    # Clear all current connections
    for node in total_nodes.values():
        node.connected_to.clear()
    G.clear_edges()

    # Re-establish connections randomly
    for node_number, node in total_nodes.items():
        while len(node.connected_to) < TOTAL_NUMBER_OF_CONNECTIONS:
            possible_connections = [num for num in total_nodes if num != node_number and num not in node.connected_to]
            if len(possible_connections) < TOTAL_NUMBER_OF_CONNECTIONS:
                break  # Avoid infinite loop if not enough nodes to connect
            connect_to = random.sample(possible_connections, 1)[0]
            node.add_connection(connect_to)
            total_nodes[connect_to].add_connection(node_number)
        G.add_edges_from((node_number, connect) for connect in node.connected_to)

    print("Connections have been reshuffled.")

def draw_graph():
    global time_text
    node_colors = ['green' if len(total_nodes[node].chunks) == SIZE_OF_FILE else 'skyblue' for node in G.nodes()]    

    # Draw the nodes and edges
    nx.draw_networkx_nodes(G, pos, ax=graph_ax, node_size=500, node_color=node_colors)
    nx.draw_networkx_edges(G, pos, ax=graph_ax)
    nx.draw_networkx_labels(G, pos, font_size=10, horizontalalignment='center', verticalalignment='center', ax=graph_ax)
    graph_ax.set_title("BitTorrent Simulation")
    
    time_text = graph_ax.text(-0.1, 1.05, f'Time Count: {time_counter}', transform=graph_ax.transAxes, ha='left', fontsize=10)

    plt.draw()
    plt.pause(0.001)

def update(event=None):
    global total_nodes, time_counter, time_text
    time_counter += 1
    graph_ax.clear()
    plt.axis('off')

    if time_counter % OPTIMISTICALLY_UNCHOKE_CONSTANT == 0:
        optimisically_unchoke_peers()

    for node in total_nodes.values():
        node.exchange_data()

    draw_graph()
    time_text.set_text(f'Time Count: {time_counter}')
    plt.draw()
    
def add_node(event):
      global node_number_tracker
      global total_nodes
      new_node = Node(node_number_tracker, False)
      total_nodes[node_number_tracker] = new_node

      G.add_node(new_node.node_number)
      pos[new_node.node_number] = np.random.rand(2) * 2 - 1  # Random position

      possible_connections = list(total_nodes.keys())[:-1]  # Exclude the new node itself
      connections = random.sample(possible_connections, min(TOTAL_NUMBER_OF_CONNECTIONS, len(possible_connections)))
      for connect_to in connections:
        new_node.add_connection(connect_to)
        total_nodes[connect_to].add_connection(node_number_tracker)
        G.add_edge(new_node.node_number, connect_to)
            
      print("Node ", node_number_tracker, " added!")
      node_number_tracker +=1
      
      update()

def delete_node(event):
    global total_nodes
    if G.nodes:
       # Picking a random node to remove
       node_to_remove = np.random.choice(list(G.nodes())) 
       total_nodes.pop(node_to_remove)

       # Churn behavior to remove connections of removed node
       for node in total_nodes.values():
          node.remove_connection(node_to_remove)
        
       G.remove_node(node_to_remove)
       pos.pop(node_to_remove, None)
       print("Node ", node_to_remove, " removed!")
    
       update()

def print_total_nodes(event):
    global total_nodes
    print("+---------------------------------------------------------------------------------------------------------------------------------------------------------+")
    for node in total_nodes.values():
        print(node.get_data())   
    print("+---------------------------------------------------------------------------------------------------------------------------------------------------------+")
    print("G Nodes: ", list(G.nodes))
    print("+---------------------------------------------------------------------------------------------------------------------------------------------------------+")

G = nx.Graph()
node_number_tracker = 1
time_text = None
time_counter = 0
total_nodes = OrderedDict()

for i in range(1, INITIAL_NUMBER_OF_NODES+1):
    total_nodes[node_number_tracker] = Node(node_number_tracker)
    node_number_tracker += 1   

optimisically_unchoke_peers()

pos = nx.circular_layout(G)
for key, value in pos.items():
    pos[key] = [2 * x for x in value]

fig, ax = plt.subplots(figsize=(10, 8))
plt.subplots_adjust(bottom=0.2)
graph_ax = plt.subplot(111)
plt.axis('off')

# # Buttons
ax_add_node = plt.axes([0.6, 0.05, 0.1, 0.075])
ax_delete_node = plt.axes([0.8, 0.05, 0.1, 0.075])
ax_print = plt.axes([0.2, 0.05, 0.1, 0.075])
ax_update = plt.axes([0.4, 0.05, 0.1, 0.075])

btn_add_node = Button(ax_add_node, 'Add Node')
btn_delete_node = Button(ax_delete_node, 'Delete Node')
btn_print = Button(ax_print, 'Print Nodes')
btn_update = Button(ax_update, 'Update')

btn_add_node.on_clicked(add_node)
btn_delete_node.on_clicked(delete_node)
btn_print.on_clicked(print_total_nodes)
btn_update.on_clicked(update)

draw_graph()
plt.show()
