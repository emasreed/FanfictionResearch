import pandas as pd

from operator import itemgetter
import networkx as nx
from networkx.algorithms import community
import csv

import numpy as np

import plotly.graph_objects as go

import argparse


# Creating a Visualization
# I created the visualization by first reading in my nodes and edges files using the Program Historian's code. Then, I used Plotly's network graph documentation to create the graph.

# The main line of code I added is three cells below creating a "pos" (positions) column that Plotly can then use to map onto the graph. Positon means the literal x, y position of a node. NetworkX has different layouts to choose from for positioning your nodes.

with open('bts_nodes.csv', 'r') as nodecsv:
    nodereader = csv.reader(nodecsv)
    nodes = [n for n in nodereader][1:]

# Get a list of just the node names (the first item in each row)
node_names = [n[0] for n in nodes]

# Read in the edgelist file
with open('bts_edges.csv', 'r') as edgecsv:
    edgereader = csv.reader(edgecsv)
    edges = [tuple(e) for e in edgereader][1:]
print(len(node_names))
print(len(edges))

G = nx.Graph()  # Initialize a Graph object
G.add_nodes_from(node_names)  # Add nodes to the Graph
G.add_weighted_edges_from(edges)  # Add edges to the Graph
print(nx.info(G))  # Print information about the Graph

density = nx.density(G)
print("Network density:", density)
nx.draw(G, with_labels=True)

nx.write_gexf(G, 'bts_network.gexf')

# Adding positions in my data

pos = nx.spring_layout(G, k=0.6, iterations=50)
for n, p in pos.items():
    G.nodes[n]['pos'] = p
edge_x = []
edge_y = []
for edge in G.edges():
    x0, y0 = G.nodes[edge[0]]['pos']
    x1, y1 = G.nodes[edge[1]]['pos']
    edge_x.append(x0)
    edge_x.append(x1)
    edge_x.append(None)
    edge_y.append(y0)
    edge_y.append(y1)
    edge_y.append(None)

edge_trace = go.Scatter(
    x=edge_x, y=edge_y,
    line=dict(width=0.9, color='#888'),
    hoverinfo='none',
    mode='lines')

node_x = []
node_y = []
node_label = []
for node in G.nodes():
    x, y = G.nodes[node]['pos']
    node_x.append(x)
    node_y.append(y)
    node_label.append(node)

node_trace = go.Scatter(
    x=node_x, y=node_y,
    mode='markers',
    text=node_label, textposition='top center',
    hoverinfo='text',
    marker=dict(
        showscale=True,
        # colorscale options
        #'Greys' | 'YlGnBu' | 'Greens' | 'YlOrRd' | 'Bluered' | 'RdBu' |
        #'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' |
        #'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis' |
        colorscale='purd',
        reversescale=True,
        color=[],
        size=10,
        colorbar=dict(
            thickness=20,
            title='Node Connections',
            xanchor='left',
            titleside='right'
        ),
        line_width=2))
node_adjacencies = []
# node_text = []
for node, adjacencies in enumerate(G.adjacency()):
    node_adjacencies.append(len(adjacencies[1]))
#     node_text.append('Tag:' +str(node))

node_trace.marker.color = node_adjacencies
# node_trace.text = node_text
fig = go.Figure(data=[edge_trace, node_trace],
                layout=go.Layout(
    #                 title='The Legend of data Fanfiction additional_tags',
                titlefont_size=16,
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20, l=5, r=5, t=40),
                annotations=[dict(
                    text="Python code: <a href='https://plotly.com/ipython-notebooks/network-graphs/'> https://plotly.com/ipython-notebooks/network-graphs/</a>",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.005, y=-0.002)],
                xaxis=dict(showgrid=False, zeroline=False,
                           showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                )
fig.show()

fig.write_html("graph.html")
