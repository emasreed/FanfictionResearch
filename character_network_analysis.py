import pandas as pd

from operator import itemgetter
import networkx as nx
from networkx.algorithms import community
import csv

import numpy as np

import plotly.graph_objects as go

import argparse

import time

import scipy

from gensim import corpora
from gensim.summarization import bm25

parser = argparse.ArgumentParser(
    description='Extract metadata from a fic csv')
parser.add_argument(
    'csv', metavar='csv',
    help='the name of the csv with the original data')
parser.add_argument(
    'out_html', metavar='out_html',
    help="The name of the html containing the model"
)

args = parser.parse_args()
csv_name = args.csv
out_html = args.out_html

def modified_bm25(total_occurences_in_all_fics, total_number_of_tags, avg_number_of_tags):
    idf = np.log(1 + (total_occurences_in_all_fics - 1)/1.5)
    return pow(idf * 2.2/(1 + 1.2 * (1-5 + 5 * total_number_of_tags/avg_number_of_tags)), 3)

start_time = time.perf_counter()
data_all = pd.read_csv(csv_name)
data_all.drop_duplicates()
end_time = time.perf_counter()
print(f"Data loaded in {end_time- start_time:0.4f} seconds")

start_time = time.perf_counter()

data_all["additional_tags"] = data_all["additional_tags"].str.lower().str.split(",")
data_all["relationship"] = data_all["relationship"].str.lower().str.split(",")
data_all["character"] = data_all["character"].str.lower().str.split(",")

# exploding and counting additional tags
data_additional_tags = data_all[['additional_tags', 'work_id']].rename(
    columns={"additional_tags": "tags"}).explode("tags")
print(data_additional_tags[:5])
additional_tags_value_counts = data_additional_tags["tags"].value_counts().to_frame(
).reset_index().rename(columns={"index": "tags", "tags": "count"})
additional_tags_value_counts.to_csv("additional_tags_tag_counts.csv")
additional_tag_length_fic = data_additional_tags["work_id"].value_counts().to_frame(
).reset_index().rename(columns={"index": "work_id", "work_id": "tag_count"})

# exploding and counting relationship tags
data_relationship = data_all[['relationship', 'work_id']].rename(
    columns={"relationship": "tags"}).explode("tags")
print(data_relationship[:5])
relationship_value_counts = data_relationship["tags"].value_counts().to_frame(
).reset_index().rename(columns={"index": "tags", "tags": "count"})
relationship_value_counts.to_csv("relationship_tag_counts.csv")
relationship_tag_length_fic = data_relationship["work_id"].value_counts().to_frame(
).reset_index().rename(columns={"index": "work_id", "work_id": "tag_count"})

# exploding and counting character tags
data_character = data_all[['character', 'work_id']].rename(
    columns={"character": "tags"})
data_character = data_character.explode("tags")
print(data_character.shape)
character_value_counts = data_character["tags"].value_counts().to_frame(
).reset_index().rename(columns={"index": "tags", "tags": "count"})
character_value_counts.to_csv("character_tag_counts.csv")
character_tag_length_fic = data_character["work_id"].value_counts().to_frame(
).reset_index().rename(columns={"index": "work_id", "work_id": "tag_count"})
character_tag_length_fic.to_csv("character_tag_length.csv")


hp = data_character[data_character.tags.str.contains("luna lovegood", na=False)]
print(hp.shape)
filtered_works = hp['work_id'].tolist

filtered_data_relationship = data_relationship[data_relationship['work_id'].isin(
    hp['work_id'].array)]
filtered_data_character = data_character[data_character['work_id'].isin(
    hp['work_id'].array)]
filtered_data_additional_tags = data_additional_tags[data_additional_tags['work_id'].isin(
    hp['work_id'].array)]


data_all = pd.concat([filtered_data_additional_tags, filtered_data_relationship], ignore_index=True)
dataDFexploded = data_all.drop_duplicates()
dataDFexploded = dataDFexploded.reset_index(drop=True)
tag_value_counts = dataDFexploded["tags"].value_counts().to_frame(
).reset_index().rename(columns={"index": "tags", "tags": "tag_count"})
tag_value_counts.to_csv('tag_value_counts.csv')
work_tag_counts = dataDFexploded["work_id"].value_counts().to_frame(
).reset_index().rename(columns={"index": "work_id", "work_id": "total_tag_count"})
work_tag_counts.to_csv("work_tag_counts.csv")

dataDFexploded = dataDFexploded.merge(tag_value_counts, how='outer', on="tags")
dataDFexploded = dataDFexploded.merge(work_tag_counts, how='outer', on='work_id')
dataDFexploded = dataDFexploded[dataDFexploded["tag_count"] > 1]
avg_tags = dataDFexploded['total_tag_count'].mean()
print(f"Average number of tags:{avg_tags}")
print(f"Median number of tags: {dataDFexploded['total_tag_count'].median()}")
dataDFexploded["count"] = modified_bm25(dataDFexploded.tag_count, dataDFexploded.total_tag_count, avg_tags)
dataDFexploded["count"] = 1
dataDFexploded.to_csv("data_exploded.csv")
end_time = time.perf_counter()
print(f"Data processed in {end_time- start_time:0.4f} seconds")

start_time = time.perf_counter()
dataWorkID = dataDFexploded.pivot(
    index="work_id", columns="tags", values="count").fillna(0)

dataWorkID.to_csv("dataWorkId.csv")
end_time = time.perf_counter()
print(f"Data pivoted in {end_time- start_time:0.4f} seconds")

# #removing column metadata
# dataWorkID.columns.name = None

# #renaming "Nan" into "none" so it actually appears instead of being a null value
# dataWorkID.columns = dataWorkID.columns.fillna('none')
# dataWorkID = dataWorkID.drop(columns='none')

dataWorkID.to_csv("dataWorkId.csv")
print("generated dataWorkId.csv")

start_time = time.perf_counter()
dataMatrix = dataWorkID.T.dot(dataWorkID).astype(float)
end_time = time.perf_counter()
print(f"made matrix in {end_time- start_time:0.4f} seconds")

# start_time = time.perf_counter()
# numpy_array = dataWorkID.to_numpy()
# dataMatrix = numpy_array.transpose().dot(numpy_array)
# end_time = time.perf_counter()
# print(
#     f"made matrix with numpy in {end_time - start_time:0.4f} seconds")

# pd.DataFrame(dataMatrix, columns=dataDFexploded["tags"], index=dataDFexploded["tags"]).to_csv("matrix.csv")
dataMatrix.to_csv("matrix.csv")
print("generated matrix.csv")

dataMatrix = pd.read_csv("matrix.csv")

G = nx.from_pandas_adjacency(dataMatrix.set_index("tags"))
G = nx.Graph([(u, v, d) for u, v, d in G.edges(data=True) if d
              ['weight'] > 10])
G.name = "Additional Tag from pandas adjacency"


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
        # 'Greys' | 'YlGnBu' | 'Greens' | 'YlOrRd' | 'Bluered' | 'RdBu' |
        # 'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' |
        # 'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis' |
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

fig.write_html(f"models/{time.time()}.html")
