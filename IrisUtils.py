import pydrake.geometry.optimization as pyOpt
import numpy as np
import GeometryUtils as geoUtils
import time 
import pickle

from pydrake.common import RandomGenerator

REGION_FILE_PATH = "iris_regions.pkl"

class IrisWrapper:

    iris_obstacles = None
    domain = None
    
    lower_bound = None
    upper_bound = None
    
    iris_regions = []
        
    def __init__(self):
        pass
    
    def determine_and_set_domain(self, center_ground_xy, 
                                 ground_width, ground_length,
                                 obst_height):

        lbx = center_ground_xy[0] - ground_width/2.0
        ubx = center_ground_xy[0] + ground_width/2.0
        lby = center_ground_xy[1] - ground_length/2.0
        uby = center_ground_xy[1] + ground_length/2.0
        
        self.lower_bound = np.array([lbx , lby, 0])             
        self.upper_bound = np.array([ubx, uby, obst_height+1])      
        
        self.domain = pyOpt.HPolyhedron.MakeBox(self.lower_bound, self.upper_bound)
        
        return self.domain 
    
    
    def retrieve_iris_obstacles_from_query(self, query_object, scene_graph):
        
        self.iris_obstacles = pyOpt.MakeIrisObstacles(query_object=query_object,
                                                      reference_frame=scene_graph.world_frame_id())

        return self.iris_obstacles
    
    def solveIRIS_with_num_regions(self, num_regions=15,  obstacle_scale_factor=1.05, seed=0):
        
        iris_obstacles_scaled = [obstacle.Scale(obstacle_scale_factor) for obstacle in self.iris_obstacles]
        
        if self.iris_obstacles is None:
            raise ValueError("No obstacles to solve IRIS")
        if self.domain is None:
            raise ValueError("No domain to solve IRIS")
        
        self.randomGen = RandomGenerator(seed)
        options = pyOpt.IrisOptions()
        options.require_sample_point_is_contained = True
        

        start_time = time.perf_counter()
        temp_time = start_time
        
        prev_sample = None
        for _ in range(num_regions):
            sample = self.find_samplepoint_rand(prev_sample)
            region = pyOpt.Iris(iris_obstacles_scaled, sample ,self.domain, options)
            self.iris_regions.append(region)
            
            
            t = time.perf_counter()
            print(f"Region {len(self.iris_regions): .1f} computed in {t - temp_time} seconds\n")
            temp_time = t
            
        
        end_time = time.perf_counter()
        print(f"IRIS computation took {end_time - start_time: .1f} seconds")
        
        
        return self.iris_regions
    
    def add_meshVisualization_iris_obstacles(self, meshcat, is_visible=False):
        if meshcat is None:
            return
        
        i = 0
        for hpolyhydron in self.iris_obstacles:
            vertices = pyOpt.VPolytope(hpolyhydron).vertices()
            geoUtils.visualizeToMesh_3D_convex_hull(meshcat, vertices, label=f"iris/obstacles/obstacle_{i}", visible=is_visible, fill=True, color=[0.3, 0.3, 0.3, .5])
            geoUtils.set_intensity(meshcat, 0.5, f"iris/obstacles/obstacle_{i}")
            i += 1

        return
    
    def add_meshVisualization_iris_domain(self, meshcat, is_visible=False):
        
        if meshcat is None:
            return
        
        vertices = pyOpt.VPolytope(self.domain).vertices()
        geoUtils.visualizeToMesh_3D_convex_hull(meshcat, vertices, label="iris/domain", visible=is_visible, fill=False)
        
        return
    
    def add_meshVisualization_iris_regions(self, meshcat, is_visible=False):
        if meshcat is None:
            return
        
        i=0
        for region in self.iris_regions:
            vertices = pyOpt.VPolytope(region).vertices()
            geoUtils.visualizeToMesh_3D_convex_hull(meshcat, 
                                                    vertices, 
                                                    label=f"iris/regions/region_{i}", 
                                                    visible=is_visible, 
                                                    fill=True,
                                                    color=[0, 0.1, 0, .2])
            geoUtils.set_intensity(meshcat, 0.2, f"iris/regions/region_{i}")
            i += 1
        return
    
    
    def find_best_samplepoint(self, delta_step=[0.3, 0.3, 0.3]):
        delta_step = np.array(delta_step)   
        vp_obstacles = convert_and_minimize(self.iris_obstacles) 
        vp_regions = convert_and_minimize(self.iris_regions)  
        
        print(vp_regions)
        
        furthest_point, max_distance = explore_domain(self.lower_bound, self.upper_bound, delta_step, vp_obstacles, vp_regions)
        print(f"Furthest point: {furthest_point}, Distance: {max_distance}")

        return furthest_point
    
    def find_samplepoint_rand(self, prev_sample, max_iter=100):
        
        def in_union_of_regions_obstacles(point):
            vp_obstacles = convert_and_minimize(self.iris_obstacles) 
            vp_regions = convert_and_minimize(self.iris_regions)  
            return any(vp.PointInSet(point) for vp in vp_obstacles + vp_regions)
        
        
        iter = 0
        sample = prev_sample
        if prev_sample is not None:
            
            while in_union_of_regions_obstacles(sample) and iter<=max_iter:
                sample = self.domain.UniformSample(self.randomGen,previous_sample=sample)
                iter += 1

            if iter>max_iter:
                print("Max iterations reached")
            print(f"Chosen point: {sample}")
            return sample
        
        else:
            sample = self.domain.UniformSample(self.randomGen)
            while in_union_of_regions_obstacles(sample) and iter<=max_iter:
                sample = self.domain.UniformSample(self.randomGen)
                iter += 1
                
            if iter>max_iter:
                print("Max iterations reached")
                
            print(f"Chosen point: {sample}")
            return sample

    def save_regions_to_file(self, filename=REGION_FILE_PATH):

        with open(filename, 'wb') as f:
            pickle.dump(self.iris_regions, f)
        
        print(f"Regions saved to {filename}")
        return self.iris_regions
    
    def load_regions_from_file(self, filename=REGION_FILE_PATH):
        
        reg = None
        
        # if file exist 
        
        try:
            
            with open(filename, 'rb') as f:
                reg = pickle.load(f)
                
            print(f"Regions loaded from {filename}")
            
            
        except FileNotFoundError:
            print(f"Regions not loaded from {filename}")
            return None
        
           
        if reg is None or reg == {}:
            print(f"Regions not loaded from {filename}")
            return None

        self.iris_regions = reg
                
        return reg
        


def convert_and_minimize(hpoly_list):
        return [pyOpt.VPolytope(poly).GetMinimalRepresentation() for poly in hpoly_list]

def compute_distance_to_mesh(point, vertices, triangles):
    point = np.array(point)
    min_distance = np.inf
    
    if vertices.shape[1] != 3:
        vertices = vertices.T
        
    for triangle in triangles:
        tri_points = vertices[triangle]
        a, b, c = tri_points[0], tri_points[1], tri_points[2]
        # Compute the normal vector of the plane
        normal_vector = np.cross(b - a, c - a)
        # Normalize the normal vector
        normal_unit_vector = normal_vector / np.linalg.norm(normal_vector)
        # Compute the distance from the point to the plane using the unit normal vector
        distance = np.abs(np.dot(point - a, normal_unit_vector))
        min_distance = min(min_distance, distance)
    return min_distance

def explore_domain(lower_bound, upper_bound, delta_step, vp_obstacles, vp_regions):
        num_steps = np.ceil((upper_bound - lower_bound) / delta_step).astype(int) + 1
        grid = [np.linspace(lower, upper, num) for lower, upper, num in zip(lower_bound, upper_bound, num_steps)]
        grid_points = np.meshgrid(*grid, indexing='ij')
        grid_points = np.vstack(list(map(np.ravel, grid_points))).T

        max_distance = -np.inf
        furthest_point = None

        all_vertices = []
        all_triangles = []

        for vp in vp_obstacles + vp_regions:
            vertices = vp.vertices()
            triangles = geoUtils.get_triangular_mesh(vertices)
            all_vertices.append(vertices)
            all_triangles.append(triangles)

        for point in grid_points:
            if not any(vp.PointInSet(point) for vp in vp_obstacles + vp_regions):
                current_distance = min(compute_distance_to_mesh(point, vertices, triangles)
                                    for vertices, triangles in zip(all_vertices, all_triangles))
                if current_distance > max_distance:
                    max_distance = current_distance
                    furthest_point = point

        return furthest_point, max_distance


