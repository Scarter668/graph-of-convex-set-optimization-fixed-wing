import Simulation as sim
import IrisUtils as irisUtils
import GcsTrajectoryUtils as gcsUtils

import numpy as np

from pydrake.solvers import MosekSolver

import gcs.bezier

import GeometryUtils as geoUtils


PATH_TO_OBSTACLE_ENV = "models/obstacles_house_formation.urdf"

# width of ground, lenght of ground,height of highest obstavle + 1. Look in urdf file
center_ground_xy = np.array([0, 5]) 
ground_width = 7
ground_length = 15
obst_height = 2

obstacle_ScaleFactor=1.3

# trajectory
kdimentions = 3
vel_lower_bound = np.ones(kdimentions) * -1
vel_upper_bound =  np.ones(kdimentions) * 1 #np.array([100000, 100000, 100000])






if __name__ == "__main__":
    
    
    
    
    print("################ Building environment model ################\n")
    simEnv = sim.SimulationEnvironment(obstacle_load_path=PATH_TO_OBSTACLE_ENV)
    # simEnv.connect_meshcat()
    simEnv.build_model()
    simEnv.save_and_display_diagram()
    sim.print_model_instances(simEnv.plant)
    print("################ FINISHED building environment model ################\n\n")

    # Solve IRIS

    print("################ Starting IRIS region computation ################\n")
    irisWrapper = irisUtils.IrisWrapper()    
    simEnv.compute_obstacles(irisWrapper)
    irisWrapper.add_meshVisualization_iris_obstacles(simEnv.meshcat, is_visible=True)
    irisWrapper.determine_and_set_domain(center_ground_xy, ground_width, ground_length, obst_height)    
    irisWrapper.add_meshVisualization_iris_domain(simEnv.meshcat, is_visible=True)
    
    
    if irisWrapper.load_regions_from_file() is None:
            
        num_regions = 25 
        irisWrapper.solveIRIS_with_num_regions(num_regions, obstacle_scale_factor=obstacle_ScaleFactor)
        irisWrapper.save_regions_to_file()
    
    irisWrapper.add_meshVisualization_iris_regions(simEnv.meshcat, is_visible=False) 
    
    
    print("################ FINISHED IRIS computation ################\n\n")
    
    # Save the regions to a file
    
    
    
    print("################ Solving GCS trajectory ################\n")
    options = gcsUtils.GCSTrajectoryOptions()
    options.use_BezierGCS = True
    options.regions = irisWrapper.iris_regions
    options.path_continuity_order = 3 # 3 works
    options.Bspline_order = 4   # 4 works
    # options.edges = None
    options.hdot_min = 1e-3
    options.full_dim_overlap = True
    
    gcsTraj = gcsUtils.GCSTrajectory(3, options)
    start = [0, 0, 1]
    goal = [0.5, 11.5, 1.5]
    # goal = [0, 10.5, 1.5]
    
    gcsTraj.add_start_goal_and_viz(start, goal, simEnv.meshcat, zero_deriv_boundary=1)
    gcsTraj.add_pathLengthCost(1)
    gcsTraj.add_timeCost(0.001)
    gcsTraj.add_velocityBounds(vel_lower_bound, vel_upper_bound)
    regularization = 1e-3
    gcsTraj.add_derivativeRegularization_r_h(weight=regularization, order=2) # add nth derivative as a cost in costfunction
    
    # gcsTraj.add_regions(irisWrapper.iris_regions, order=7)
    # gcsTraj.add_continuityContraint(3)
    
    if gcsTraj.load_trajectory_from_file() is None:
        gcsTraj.solve(preprocessing=False)   
        gcsTraj.save_trajectory_to_file()
        
    num_points = 2000
    gcsTraj.visualize_trajectory(simEnv.meshcat, num_points)

    print("################ FINISHED solving GCS tracjectory  ################\n\n")
    
    
    
    print("################ Startting Simulation ################\n\n")
    
    simulator = sim.SimulationEnvironment(PATH_TO_OBSTACLE_ENV)
    simulator.connect_meshcat()
    simulator.add_fixed_wing()
    simulator.add_controller(gcsTraj.trajectory.value(6))
    simulator.build_model()
    simulator.save_and_display_diagram()
    
    
    irisWrapper.add_meshVisualization_iris_regions(simulator.meshcat, is_visible=False) 
    gcsTraj.visualize_start_goal(start, goal, simulator.meshcat)
    gcsTraj.visualize_trajectory(simulator.meshcat, num_points)

    
    simulator.simulate(5)

    print("################ FINISHED Simulation ################\n\n")
    
    
    
    input("Press Enter to quit...")