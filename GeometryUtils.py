
import pydrake.geometry as pyGeo
import pydrake.math as pyMath
import numpy as np

from scipy.spatial import ConvexHull

def get_obstacleID_geometries(plant):
    
    obstacle_bodies = plant.GetBodiesWeldedTo(plant.GetBodyByName("ground"))
    
    # only get the bodies that are obstacles
    obstacle_bodies = [body for body in obstacle_bodies if "obs" in body.name()]
    
    obstacle_geometries = []
    
    for body in obstacle_bodies:
        for geom in plant.GetCollisionGeometriesForBody(body):
            obstacle_geometries.append(geom)
    
    return obstacle_geometries
    
def get_obstacle_vertices(plant, query_object, inspector ):
    
    geometry_ids = get_obstacleID_geometries(plant)
    
    obstacle_vertices = []
           
    for geometry_id in geometry_ids:
        shape = inspector.GetShape(geometry_id) ## should all be boxes    
        if not isinstance(shape, pyGeo.Box):
            raise ValueError("Unsupported geometry type")
        
        X_WB = query_object.GetPoseInWorld(geometry_id)
        box_vert = get_box_vertices(shape, X_WB)
        obstacle_vertices.append( box_vert )        
    
    return obstacle_vertices
    
def get_box_vertices(box, X_WB):
    # Return axis-aligned bounding-box vertices
    # Order:                +y
    #       3------2          |
    #      /|     /|          |
    #     / 4----/-5          ------  +x
    #    0------1 /          /
    #    |/     |/          /
    #    7------6        +z

    
    half_width = box.width() / 2
    half_height = box.height() / 2
    half_depth = box.depth() / 2
    
    # vertices in box frame
    vertices_box_frame = [
        [-half_width, half_depth, half_height],
        [half_width, half_depth , half_height ],
        [half_width, half_depth,  -half_height],
        [-half_width, half_depth, -half_height],
        [-half_width, -half_depth,  -half_height],
        [half_width, -half_depth, -half_height],
        [half_width, -half_depth , half_height],
        [-half_width, -half_depth , half_height],
    ]
    
    vertices_box_frame = np.array(vertices_box_frame).T # shape: (3, 8)
    
    # print("vertices_box_frame: ", vertices_box_frame)
    # print("vertices_box_frame.shape: ", vertices_box_frame.shape)
    # print()
    
    transformed_points = X_WB.multiply(vertices_box_frame)
    
    # print("transformed_points: ", transformed_points)
    # print("transformed_points.shape: ", transformed_points.shape)
    
    
    
    # print all the vertices
    # for i in range(transformed_points.shape[0]):
    #     print(f"Vertex {i}: {transformed_points[i]}")
        
    
    return transformed_points



def get_triangular_mesh(vertices):
    # vertices: list of vertices of the convex hull
    # returns: list of 3D faces of the convex hull
    
    # ensure that the shape is (N, 3)
    if vertices.shape[1] != 3:
        vertices = vertices.T

    hull = ConvexHull(vertices)    
    faces = hull.simplices

    return faces

def get_2Dfaces_of_convex_hull():
    # returns: list of 2D faces of the convex hull
    return np.array([[0, 1, 2], [0, 2, 3]]).T



def visualizeToMesh_3D_convex_hull(meshcat, vertices, label="convex_hull", visible=False, fill=True, color=[0.3, 0.3, 0.3, 1]):
    # vertices: list of vertices of the convex hull
    # returns: None
    
    faces_3D = get_triangular_mesh(vertices)

    # ensure that the shape is (3, N)
    if vertices.shape[0] != 3:
        vertices = vertices.T
    
    if faces_3D.shape[0] != 3:
        faces_3D = faces_3D.T
    
    meshcat.SetTriangleMesh(
        path=label,
        vertices=vertices,
        faces=faces_3D,
        rgba=pyGeo.Rgba(*color), # grey
        wireframe= not fill,                # Fill in the mesh (not just lines)
        wireframe_line_width=2.0,
        # side=self.meshcat.SideOfFaceToRender.kBackSide
        )
    
    meshcat.SetProperty(label, "visible", visible)
    return

def set_intensity(meshcat, intensity, label):
    meshcat.SetProperty(label, "intensity", intensity)
    
    return

def visualize_point(meshcat, point, label="trajetory/point", radius=0.05, color=[0, 0, 1, 1], visible=True):
    
    meshcat.SetObject(path=label,
                      shape=pyGeo.Sphere(radius), 
                      rgba=pyGeo.Rgba(*color)
                      )

    meshcat.SetTransform(path=label,
                         X_ParentPath=pyMath.RigidTransform(point))

    meshcat.SetProperty(label, "visible", visible)
    
    return

