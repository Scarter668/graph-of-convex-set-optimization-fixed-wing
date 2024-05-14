
import pydrake.geometry as pyGeo
import pydrake.math as pyMath
import numpy as np

from scipy.spatial import ConvexHull

from underactuated.meshcat_utils import AddMeshcatTriad


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



def visualizeToMesh_3D_convex_hull(meshcat, vertices, label="convex_hull", visible=False, fill=True, color=pyGeo.Rgba(0.3, 0.3, 0.3, 1)):
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
        rgba= color, # grey
        wireframe= not fill,                # Fill in the mesh (not just lines)
        wireframe_line_width=2.0,
        # side=self.meshcat.SideOfFaceToRender.kBackSide
        )
    
    meshcat.SetProperty(label, "visible", visible)
    return

def set_intensity(meshcat, intensity, label):
    meshcat.SetProperty(label, "intensity", intensity)
    
    return

def visualize_point(meshcat, point, label="trajetory/point", radius=0.05, color=pyGeo.Rgba(0, 0, 1, 1), visible=True):
    
    meshcat.SetObject(path=label,
                      shape=pyGeo.Sphere(radius), 
                      rgba=color
                      )

    meshcat.SetTransform(path=label,
                         X_ParentPath=pyMath.RigidTransform(point))

    meshcat.SetProperty(label, "visible", visible)
    
    return



def visualize_frame(meshcat, name, X_WF, length=0.15, radius=0.003):
    """
    visualize imaginary frame that are not attached to existing bodies

    Input:
        name: the name of the frame (str)
        X_WF: a RigidTransform from frame F to world.

    Frames whose names already exist will be overwritten by the new frame
    """
    AddMeshcatTriad(
        meshcat, "traj_source/" + name, length=length, radius=radius, X_PT=X_WF
    )

    # print("Visualized frame: ", X_WF)

## Visualization of key frames:
def visualize_key_frames(meshcat, frame_poses):
    for i, pose in enumerate(frame_poses):
        visualize_frame(meshcat,"frame_{}".format(i), pose, length=0.3)
        
        