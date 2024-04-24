import pydrake.planning as pyPlan
import pydrake.geometry.optimization as pyOpt
import pydrake.geometry as pyGeo
import numpy as np
import GeometryUtils as geoUtils
import time

from pydrake.solvers import MosekSolver

# remember to export pythonpath: export PYTHONPATH=$PYTHONPATH:/home/steve/drake_ws/underactuated/gcs-science-robotics
import gcs.bezier as gcb
import pickle



TRAJECTORY_FILE_PATH = "trajectory.pkl"

class GCSTrajectoryOptions():
    
    regions = None
    Bspline_order = None
    path_continuity_order = None
    edges = None
    hdot_min = 1e-6 
    full_dim_overlap = True
    
    use_BezierGCS = False
    traj_file_path = TRAJECTORY_FILE_PATH
    derivative_regularization = 1e-3
    
    def __init__(self):
        return
    
    def is_defined(self):
        return (self.regions is not None) and \
               (self.Bspline_order is not None) and   \
               (self.path_continuity_order is not None)  
               # \ edges are aloud to be None
    
   

class GCSTrajectory():  
    USE_BEZIER = False
    
    kdimentions = 3
    gcsTrajOpt = None
    source = None
    target = None
    regions = None
    
    # edges
    start_to_region = None
    region_to_goal = None
    
    trajectory = None
    
    
    def __init__(self, kdimentions = 3, options = GCSTrajectoryOptions()):
        self.kdixmentions = kdimentions
        
        self.USE_BEZIER = options.use_BezierGCS
        self.traj_file_path = options.traj_file_path
        
        self.path_continuity_order = options.path_continuity_order
        self.Bspline_order = options.Bspline_order
        self.regions = options.regions
        self.derivative_regularization = options.derivative_regularization
        
        if not options.is_defined():
            raise ValueError("Options are not defined when using Bezier/GCS")
        
        if self.USE_BEZIER:
            self.gcsTrajOpt = gcb.BezierGCS(regions=options.regions, 
                                            order=options.Bspline_order, 
                                            continuity=options.path_continuity_order,  # path continuity constraints
                                            edges=options.edges,
                                            hdot_min=options.hdot_min,
                                            full_dim_overlap=options.full_dim_overlap)
          
        else:
            
            self.gcsTrajOpt = pyPlan.GcsTrajectoryOptimization(self.kdimentions)
            self.cont_order_constraint_set = False
            self.add_regions()
        return
    
    def add_start_goal_and_viz(self, start, goal, meshcat, is_visible=True, velocity=None, zero_deriv_boundary=1):
        
        if self.USE_BEZIER:
            self.gcsTrajOpt.addSourceTarget(start, goal, velocity=None , zero_deriv_boundary=1)
        else:
            self.source = self.gcsTrajOpt.AddRegions([pyOpt.Point(start)], order=0) # 1 control point
            self.target = self.gcsTrajOpt.AddRegions([pyOpt.Point(goal)], order=0)
        
        self.visualize_start_goal(start, goal, meshcat, is_visible)
        return
    
    def visualize_start_goal(self, start, goal, meshcat, is_visible=True):
        if meshcat is None:
            return
            
        geoUtils.visualize_point(meshcat,
                              start,
                              label="Gcs/source",
                              radius=0.05, 
                              color=pyGeo.Rgba(0, 1, 0, 1), #Green
                              visible=is_visible)
        
        
        geoUtils.visualize_point(meshcat,
                              goal,
                              label="Gcs/goal",
                              radius=0.1, 
                              color=pyGeo.Rgba(1, 0, 0, 1), #red
                              visible=is_visible)
        
        return
        
        
    
    def add_regions(self):
        if not self.USE_BEZIER:
            # regions: list of HPolyhedron's
            self.regions = self.gcsTrajOpt.AddRegions(self.regions, self.Bspline_order, h_min=1e-3)
            return
        raise ValueError("add_regions are not supported for Bezier GCS")
    
    def connect_graph(self):
        if not self.USE_BEZIER:
            if self.source is None:
                raise ValueError("No source")
            if self.target is None:
                raise ValueError("No target")
            if self.regions is None:
                raise ValueError("No regions")
            
            self.start_to_region = self.gcsTrajOpt.AddEdges(self.source, self.regions)
            self.region_to_goal = self.gcsTrajOpt.AddEdges(self.regions, self.target)
            return
        raise ValueError("connect_graph is not supported for Bezier GCS")
    
    def add_pathLengthCost(self, weight=1):
        if self.USE_BEZIER:
            self.gcsTrajOpt.addPathLengthCost(weight=weight)
        else:
            self.gcsTrajOpt.AddPathLengthCost(weight=weight)
        return
    
    def add_timeCost(self, weight=1):
        if self.USE_BEZIER:
            self.gcsTrajOpt.addTimeCost(weight=weight)
        else:
            self.gcsTrajOpt.AddTimeCost(weight=weight)
        return
    
    def add_velocityBounds(self, lb, ub):
        lb = np.array(lb)
        ub = np.array(ub)
        if lb.shape[0] != self.kdimentions:
            raise ValueError("Lower bound has wrong shape")
        if ub.shape[0] != self.kdimentions:
            raise ValueError("Upper bound has wrong shape")
        
        if self.USE_BEZIER:
            self.gcsTrajOpt.addVelocityLimits(lb, ub)
        else:
            self.gcsTrajOpt.AddVelocityBounds(lb, ub)
        return
    
    def add_continuityContraint(self):
        if self.USE_BEZIER:
            self.gcsTrajOpt.addDerivativeRegularization(self.derivative_regularization, self.derivative_regularization, self.path_continuity_order)
            return        
        
        if self.path_continuity_order < 1:
            raise ValueError("Continuety order must be greater than 0")
        
        for o in range(1, self.path_continuity_order + 1):
            print(f"adding C{o} constraints")
            self.gcsTrajOpt.AddPathContinuityConstraints(continuity_order=o)
    
        self.cont_order_constraint_set = True
        # Note that the constraints are on the control points of the derivatives of r(s) 
        # and not q(t). This may result in discontinuities of the trajectory return by 
        # SolvePath() since the r(s) will get rescaled by the duration h to yield q(t). 
        # NormalizeSegmentTimes() will return r(s) with valid continuity.
        return
    
    def add_derivativeRegularization_r_h(self, weight, order):
        if self.USE_BEZIER:
            
            return
        
        raise ValueError("Derivative regularization is not supported for non-Bezier GCS")
    
    def add_pathEnergyCost(self, weight):
        if self.USE_BEZIER:
            self.gcsTrajOpt.addPathEnergyCost(weight)
            return
        raise ValueError("Path energy cost is not supported for non-Bezier GCS")

    def add_pathLengtIntegralCost(self, weight):
        if self.USE_BEZIER:
            self.gcsTrajOpt.addPathLengthIntegralCost(weight)
            return
        raise ValueError("Path length integral cost is not supported for non-Bezier GCS")
    
    
    def solve(self, preprocessing=True):
        
        if not self.USE_BEZIER:
            self.connect_graph()
        
        start_time = time.perf_counter()
        
        
        trajectory = None
        success = False
        if self.USE_BEZIER:
            self.gcsTrajOpt.setPaperSolverOptions()
            self.gcsTrajOpt.setSolver(MosekSolver())
            trajectory = self.gcsTrajOpt.SolvePath(rounding=True, verbose=False, preprocessing=preprocessing)[0]
            
            success = True
        else:
            options = pyOpt.GraphOfConvexSetsOptions()
            options.max_rounding_trials = 10000
            options.max_rounded_paths = 5000
            options.solver = MosekSolver()
            options.convex_relaxation = True
            trajectory, result = self.gcsTrajOpt.SolvePath(self.source, self.target, options=options)
            success = result.is_success()
        
        end_time = time.perf_counter()
        
        time_elapsed = end_time - start_time
        if not success:
            raise ValueError(f"Failed to find a solution after {time_elapsed: .1f} seconds")
        else:
            print(f"Solution found after {time_elapsed: .1f} seconds")
        
        
        
        if not self.USE_BEZIER and self.cont_order_constraint_set:
            trajectory = self.gcsTrajOpt.NormalizeSegmentTimes(trajectory)
        
        self.trajectory = trajectory
        
        return trajectory
    
    def visualize_trajectory(self, meshcat, num_points=2000):
        
        if meshcat is None:
            return
        
        start_time = self.trajectory.start_time()
        end_time = self.trajectory.end_time()
        time = np.linspace(start_time, end_time, num_points)
        
        for t in time:
            point = self.trajectory.value(t)
            geoUtils.visualize_point(meshcat,
                                     point,
                                     label=f"Gcs/trajetory/point_{t: .1f}")
        return
         
    def load_trajectory_from_file(self):
        

        traj = None
        
        if self.USE_BEZIER:
            try:
                path = self.traj_file_path+"_BezierGCS.pkl"
                with open(path, 'rb') as f:
                    traj = pickle.load(f)
            
                print(f"Trajectory loaded from {path}")
                
            except Exception as e:
                print(f"Regions not loaded from {path}")
                print(f"Error: {e}")
                return None
        else:

            try:
                path = self.traj_file_path+".pkl"
                with open(path, 'rb') as f:
                    traj = pickle.load(f)
            
                print(f"Trajectory loaded from {path}")
                
            except Exception as e:
                print(f"Regions not loaded from {path}")
                print(f"Error: {e}")
                return None            

        self.trajectory = traj
        return traj
    
    def save_trajectory_to_file(self):
        
        print("\n\nSaving trajectory to file")
        if self.trajectory is None:
            print("Trajectory is not defined")
            return False

        if self.USE_BEZIER:
            try:
                path = self.traj_file_path+"_BezierGCS.pkl"
                with open(path, 'wb') as f:
                    pickle.dump((self.trajectory), f)
                print(f"Trajectory saved to {path}")
                return True
        
            except Exception as e:
                ## print error message:
                print(f"Failed to save trajectory to {path}")
                print(f"Error: {e}")
            return False
        
        else:
            print("Saving non Bezier trajectory to file")
            path = self.traj_file_path+".pkl"   
            try:
                with open(path, 'wb') as f:
                    pickle.dump((self.trajectory), f)
                print(f"Trajectory saved to {path}")
                return True
        
            except Exception as e:
                ## print error message:
                print(f"Failed to save trajectory to {path}")
                print(f"Error: {e}")
            return False
         
