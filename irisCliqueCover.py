
import numpy as np
from tqdm import tqdm
from typing import List
import networkx as nx
from pydrake.all import (
    RigidTransform, 
    RotationMatrix, 
    Sphere, 
    Cylinder, 
    Rgba,
    HPolyhedron
)

import random
import colorsys


def plot_point(point, meshcat_instance, name,
               color=Rgba(0.06, 0.0, 0, 1), radius=0.01, visible=False):
    meshcat_instance.SetObject(name,
                               Sphere(radius),
                               color
                               )
    meshcat_instance.SetTransform(name, RigidTransform(
        RotationMatrix(), point))
    meshcat_instance.SetProperty(name, "visible", visible)

def plot_points(meshcat, points, name, size = 0.05, color = Rgba(0.06, 0.0, 0, 1), visible=False):
    for i, pt in enumerate(points):
        n_i = name+f"/pt{i}"
        plot_point(pt, meshcat, n_i, color = color, radius=size, visible=visible)
        
        
        
#visibility graphs
def compute_rotation_matrix(a, b):
    # Normalize the points
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    
    # Calculate the rotation axis
    rotation_axis = np.cross(a, b)
    rotation_axis /= np.linalg.norm(rotation_axis)
    
    # Calculate the rotation angle
    dot_product = np.dot(a, b)
    rotation_angle = np.arccos(np.clip(dot_product, -1.0, 1.0))
    
    # Construct the rotation matrix using Rodrigues' formula
    skew_matrix = np.array([[0, -rotation_axis[2], rotation_axis[1]],
                            [rotation_axis[2], 0, -rotation_axis[0]],
                            [-rotation_axis[1], rotation_axis[0], 0]])
    rotation_matrix = np.eye(3) + np.sin(rotation_angle) * skew_matrix + (1 - np.cos(rotation_angle)) * np.dot(skew_matrix, skew_matrix)
    
    return rotation_matrix

def plot_edge(meshcat, pt1, pt2, name, color, translation = np.zeros(3), size = 0.01, visible=False):
    meshcat.SetObject(name,
                        Cylinder(size, np.linalg.norm(pt1-pt2)),
                        color)
    assert len(pt1) == len(pt2)
    # pt1 = (stretch_array_to_3d(pt1)).squeeze()
    # pt2 = (stretch_array_to_3d(pt2)).squeeze()
    assert len(pt1) == len(pt2) ==3
    dir = pt2-pt1
    rot = compute_rotation_matrix(np.array([0,0,1]), dir )
    offs = rot@np.array([0,0,np.linalg.norm(pt1-pt2)/2])
    meshcat.SetTransform(name, 
                        RigidTransform(
                        RotationMatrix(rot), 
                        np.array(pt1)+offs+translation))
    meshcat.SetProperty(name, "visible", visible)
    

def plot_edges(meshcat, edges, name, color = Rgba(0,0,0,1), size = 0.01, translation= np.zeros(3), visible=False):
    for i, e in enumerate(edges):
         plot_edge(meshcat, 
                   e[0], 
                   e[1], 
                   name + f"/e_{i}", 
                   color= color, 
                   size=size, 
                   translation=translation,
                   visible=visible)

def plot_visibility_graph(meshcat, 
                          points, 
                          ad_mat,
                          name,
                          color = Rgba(0,0,0,1), 
                          size = 0.01,
                          translation = np.zeros(3)):
    edges = []
    N = ad_mat.shape[0]
    for i in range(N):
        for j in range(i+1, N):
            if ad_mat[i,j]:
                edges.append([points[i,:], points[j,:]])
    plot_edges(meshcat, edges, name, color, size, translation)


def plot_edges_clique(meshcat,
                      clique, 
                      points, 
                      name,
                      color = Rgba(0,0,0,1), 
                      size = 0.01,
                      translation = np.zeros(3)):
    edges = get_edges_clique(clique, points)
    plot_edges(meshcat, edges, name, color, size, translation)


def get_edges_clique(clique, points, downsampling):
    edges = []
    for i,c1 in enumerate(clique[:-1]):
        for c2 in clique[i+1:]:
            if np.linalg.norm(points[c1, :]- points[c2, :])> 0.001:
                edges.append([points[c1, :], points[c2,:]])
    if len(edges)>400:
        edges = edges[::downsampling]
    return edges

def generate_maximally_different_colors(n):
    """
    Generate n maximally different random colors for matplotlib.

    Parameters:
        n (int): Number of colors to generate.

    Returns:
        List of RGB tuples representing the random colors.
    """
    if n <= 0:
        raise ValueError("Number of colors (n) must be greater than zero.")

    # Define a list to store the generated colors
    colors = []

    # Generate n random hues, ensuring maximally different colors
    hues = [i / n for i in range(n)]

    # Shuffle the hues to get random order of colors
    random.shuffle(hues)
   
    # Convert each hue to RGB
    for hue in hues:
        # We keep saturation and value fixed at 0.9 and 0.8 respectively
        saturation = 0.9
        value = 0.8
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        colors.append(rgb)

    return colors

def plot_cliques(meshcat,
                 cliques,
                 points,
                 name,
                 size = 0.01,
                 translation= np.zeros(3),
                 downsampling=10,
                 colors = None,
                 visible = False
                 ):
    cl_edges = []
    for cl in cliques:
        cl_edges.append(get_edges_clique(cl, points, downsampling))

    if colors is None:
        colors = [Rgba(c[0], c[1], c[2], 1.) for c in generate_maximally_different_colors(len(cliques))]
    for i, cl_e in enumerate(cl_edges):
        plot_edges(meshcat, 
                   cl_e, 
                   name+f"/cl_e{i}", 
                   color=colors[i], 
                   size = size, 
                   translation=translation,
                   visible = visible)
        
        
def point_in_regions(pt, regions: List[HPolyhedron]):
    for r in regions:
        if r.PointInSet(pt.reshape(-1,1)):
            return True
    return False

def check_edge(point, other, obstacles: List[HPolyhedron], n_checks = 500):
    tval = np.linspace(0, 1, n_checks)
    for t in tval:
        pt = (1-t)*point + t* other
        if point_in_regions(pt, obstacles):
            return False
    return True

def vgraph(points, obstacles: List[HPolyhedron]):
    n = len(points)
    adj_mat = np.zeros((n,n))
    for i in tqdm(range(n)):
        point = points[i, :]
        point_name = f"point_{i}"
        #meshcat.SetObject(point_name, Sphere(0.1), Rgba(0, 1, 0, 1))  # Green points
        #meshcat.SetTransform(point_name, RigidTransform(points[i]))
        for j in range(len(points[:i])):
            other = points[j]
            if check_edge(point, other, obstacles):
                adj_mat[i,j] = adj_mat[j,i] = 1
                #line_name = f"line_{i}_{j}"
                #line_points = np.vstack([points[i], points[j]])
                #line_points = np.asfortranarray(line_points.T)
                #meshcat.SetLine(line_name, line_points, line_width=0.02, rgba=Rgba(0, 0, 1, 1))  # Blue lines
                
    return adj_mat

def compute_minimal_clique_partition_nx(adj_mat):
    n = len(adj_mat)

    adj_compl = 1- adj_mat
    np.fill_diagonal(adj_compl, 0)
    graph = nx.Graph(adj_compl)
    sol = nx.greedy_color(graph, strategy='largest_first', interchange=True)

    colors= [sol[i] for i in range(n)]
    unique_colors = list(set(colors))
    cliques = []
    nr_cliques = len(unique_colors)
    for col in unique_colors:
        cliques.append(np.where(np.array(colors) == col)[0])
    return cliques