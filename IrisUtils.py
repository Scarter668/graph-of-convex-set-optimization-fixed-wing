import pydrake.geometry.optimization as pyOpt
import pydrake.geometry as pyGeo
import numpy as np
import GeometryUtils as geoUtils
import time 
import pickle

from pydrake.common import RandomGenerator
import irisCliqueCover as isc

REGION_FILE_PATH = "iris_regions"

class IrisWrapperOptions:
    use_CliqueCover = False
    num_regions = 15
    obstacle_offset_factor = 1.05
    seed = 0
    region_file_path = REGION_FILE_PATH
    
    clique_num_points = 500
    
    def __init__(self):
        return
    
    


class IrisWrapper:

    iris_obstacles = None
    domain = None
    
    lower_bound = None
    upper_bound = None
    
    iris_regions = []
    
    use_CliqueCover = False
    
    samples = None 
    cliques = None
    
    
    
        
    def __init__(self, options=IrisWrapperOptions()):

        self.use_CliqueCover = options.use_CliqueCover
        self.num_regions = options.num_regions
        self.obstacle_scale_factor = options.obstacle_offset_factor
        self.region_file_path = options.region_file_path
        self.randomGen = RandomGenerator(options.seed)
        self.num_points = options.clique_num_points
        return
    
    def determine_and_set_domain(self, center_ground_xy, 
                                 ground_width, ground_length,
                                 obst_height):

        lbx = center_ground_xy[0] - ground_width/2.0
        ubx = center_ground_xy[0] + ground_width/2.0
        lby = center_ground_xy[1] - ground_length/2.0
        uby = center_ground_xy[1] + ground_length/2.0
        
        self.lower_bound = np.array([lbx , lby, 0])             
        self.upper_bound = np.array([ubx, uby, obst_height])      
        
        self.domain = pyOpt.HPolyhedron.MakeBox(self.lower_bound, self.upper_bound)
        
        return self.domain 
    
    
    def retrieve_iris_obstacles_from_query(self, query_object, scene_graph):
        
        self.iris_obstacles = pyOpt.MakeIrisObstacles(query_object=query_object,
                                                      reference_frame=scene_graph.world_frame_id())

        return self.iris_obstacles
    
    def solveIRIS(self):
        
        iris_obstacles_scaled = [pyOpt.HPolyhedron(o.A(), o.b()+self.obstacle_scale_factor) for o in self.iris_obstacles]
        
        if self.iris_obstacles is None:
            raise ValueError("No obstacles to solve IRIS")
        if self.domain is None:
            raise ValueError("No domain to solve IRIS")
        
        
        
        
        
        options = pyOpt.IrisOptions()
        options.require_sample_point_is_contained = True
        
        start_time = time.perf_counter()
        temp_time = start_time

        
        if self.use_CliqueCover:
            
            self.samples = self.find_NSamplePoints(self.num_points)
            print(f"\nComputing minimal clique partition with {self.num_points} points\n")
            
            adj_mat = isc.vgraph(self.samples, self.iris_obstacles)
            self.cliques = isc.compute_minimal_clique_partition_nx(adj_mat)
            
            print("\nCalculating ellipsoids for cliques\n")
            cliquepts = [self.samples[cl] for cl in self.cliques]
            ells = []
            for clpt in cliquepts:
                try:
                    ells.append(pyOpt.Hyperellipsoid.MinimumVolumeCircumscribedEllipsoid(clpt.T))
                
                except Exception as e:
                    print(f"Failed to compute ellipsoid for clique {clpt}")
                    print(f"Error: {e}")
                    continue    
            
            print("\nSolving IRIS regions\n")
            self.iris_regions = []
            for e in ells:
                seed = e.center()
                options.require_sample_point_is_contained = True
                options.starting_ellipse = e
                # options.random_seed = seed
                
                r = pyOpt.Iris(iris_obstacles_scaled, seed, self.domain, options)
                self.iris_regions.append(r)
    
        else:       
            
            prev_sample = None
            for _ in range(self.num_regions):
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
            geoUtils.visualizeToMesh_3D_convex_hull(meshcat, vertices, label=f"iris/obstacles/obstacle_{i}", visible=is_visible, fill=True, color=pyGeo.Rgba(0.3, 0.3, 0.3, .5))
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
        
        if self.use_CliqueCover:
            if (self.samples is None) or (self.cliques is None):
                raise ValueError("None sample or clique type")

            isc.plot_points(meshcat, self.samples, "sample_points", visible=False)
            
            isc.plot_cliques(meshcat, self.cliques, self.samples, "cliques", visible=False)
        
        
        i=0
        colors = [pyGeo.Rgba(c[0], c[1], c[2], .2) for c in isc.generate_maximally_different_colors(len(self.iris_regions))]
        for region in self.iris_regions:
            vertices = pyOpt.VPolytope(region).vertices()
            geoUtils.visualizeToMesh_3D_convex_hull(meshcat, 
                                                    vertices, 
                                                    label=f"iris/regions/region_{i}", 
                                                    visible=is_visible, 
                                                    fill=True,
                                                    color=colors[i])
            # geoUtils.set_intensity(meshcat, 0.2, f"iris/regions/region_{i}")
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

    def find_NSamplePoints(self, N, max_iter=100):
        vp_obst = convert_and_minimize(self.iris_obstacles)
        
        def in_collision(point):
            return any(vp.PointInSet(point) for vp in vp_obst)

        samples = []
        prevSample = None
        for i in range(N):
            sample = None
            if prevSample:
                sample = self.domain.UniformSample(self.randomGen,previous_sample=sample)
            else:
                sample = self.domain.UniformSample(self.randomGen)

            iter = 0
            while in_collision(sample) and iter<=max_iter:
                sample = self.domain.UniformSample(self.randomGen,previous_sample=sample)
                iter += 1
            
            samples.append(sample)
            
            if iter>max_iter:
                print("Max iterations reached")
                
            print(f"Chosen point_({i}/{N}): {sample}")
            
        return np.array(samples)
    
    
    
    def save_regions_to_file(self):
        
        if self.use_CliqueCover:
            
            try:
                path = self.region_file_path+"_clique.pkl"
                with open(path, 'wb') as f:
                    pickle.dump((self.iris_regions, self.samples, self.cliques), f)
                
                print(f"Regions saved to {path}")
                return self.iris_regions, self.samples, self.cliques
            except Exception as e:
                print(f"Failed to save regions to {self.region_file_path}")
                print(f"Error: {e}")
                return None
        else:
            
            try:
                path = self.region_file_path+".pkl"
                with open(path, 'wb') as f:
                    pickle.dump(self.iris_regions, f)
                
                print(f"Regions saved to {path}")
                return self.iris_regions
            except Exception as e:
                print(f"Failed to save regions to {path}")
                print(f"Error: {e}")
                return None
            

    
    def load_regions_from_file(self):
        
        reg = None
        
        if self.use_CliqueCover:
            samples = None
            cliques = None
            try:
                path = self.region_file_path+"_clique.pkl"
                with open(path, 'rb') as f:
                    reg, samples, cliques = pickle.load(f)
            
                self.iris_regions = reg
                self.samples = samples
                self.cliques = cliques
                print(f"Regions loaded from {path}")
                return reg, samples, cliques
            except Exception as e:
                print(f"Regions not loaded from {path}")
                print(f"Error: {e}")
                return None
        else:
            try:
                path = self.region_file_path+".pkl"
                with open(path, 'rb') as f:
                    reg = pickle.load(f)
                    
                print(f"Regions loaded from {path}")
                self.iris_regions = reg
                return reg
                
            except Exception as e:
                print(f"Regions not loaded from {path}")
                print(f"Error: {e}")
                return None
            



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


