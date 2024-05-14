import Simulation as sim
import IrisUtils as irisUtils
import GcsTrajectoryUtils as gcsUtils

import numpy as np


import time


scenery = 2

if scenery == 0:
    PATH_TO_OBSTACLE_ENV = "models/obstacles_house_formation.urdf"

    # width of ground, lenght of ground,height of highest obstavle + 1. Look in urdf file
    CENTER_GROUND_XY = np.array([0, 5]) 
    GROUND_WIDTH = 7
    GROUND_LENGTH = 15
    OBST_HEIGHT = 3
elif scenery == 1: # Simple environment
    
    PATH_TO_OBSTACLE_ENV = "models/obstacles_house_formation_Large.urdf"
    # width of ground, lenght of ground,height of highest obstavle + 1. Look in urdf file
    CENTER_GROUND_XY = np.array([0, 9]) 
    GROUND_WIDTH = 11
    GROUND_LENGTH = 20
    OBST_HEIGHT = 2

elif scenery == 2: #Complex environment
    PATH_TO_OBSTACLE_ENV = "models/obstacles.urdf"
    CENTER_GROUND_XY = np.array([0, 9]) 
    GROUND_WIDTH = 11
    GROUND_LENGTH = 20
    OBST_HEIGHT = 2    

PLANE_WIDTH = 1

# trajectory
KDIMENTIONS = 3
VEL_LOWER_BOUND = np.ones(KDIMENTIONS) * -5
VEL_UPPER_BOUND =  np.ones(KDIMENTIONS) * 5 #np.array([100000, 100000, 100000])

START_VELOCITY_LB = np.ones(KDIMENTIONS)* 0.5
# START_VELOCITY_LB = VEL_LOWER_BOUND
GOAL_VELOCITY_LB = VEL_LOWER_BOUND #np.array([-0.1, 0.2, -0.1])





if __name__ == "__main__":
    
    
    print("################ Building environment model ################\n")
    simEnv = sim.SimulationEnvironment(obstacle_load_path=PATH_TO_OBSTACLE_ENV)
    simEnv.connect_meshcat()
    simEnv.build_model()
    simEnv.save_and_display_diagram()
    sim.print_model_instances(simEnv.plant)
    print("################ FINISHED building environment model ################\n\n")

    # input("Press Enter to continue...")
    

    print("################ Starting IRIS region computation ################\n")
    irisOptions = irisUtils.IrisWrapperOptions()
    irisOptions.use_CliqueCover = True
    irisOptions.clique_num_points = 100
    irisOptions.obstacle_offset_factor = 0.2 # PLANE_WIDTH/2.0
    irisOptions.seed = 0
    irisOptions.num_regions = 15
    if irisOptions.use_CliqueCover:
        # irisOptions.region_file_path = f" region_files/iris_regions_Large"
        # irisOptions.region_file_path = f"region_files/iris_regions_Large_p{irisOptions.clique_num_points}"
        irisOptions.region_file_path = f"region_files/cliques/regions_scenery{scenery}_p{irisOptions.clique_num_points}"
        # irisOptions.region_file_path = "region_files/iris_regions_clique"   
    
    else:
        irisOptions.region_file_path = f"region_files/random/regions_scenery{scenery}"
    irisWrapper = irisUtils.IrisWrapper(irisOptions)    
    simEnv.compute_obstacles(irisWrapper)
    irisWrapper.determine_and_set_domain(CENTER_GROUND_XY, GROUND_WIDTH, GROUND_LENGTH, OBST_HEIGHT)    
    
    irisWrapper.add_meshVisualization_iris_obstacles(simEnv.meshcat, is_visible=True)
    irisWrapper.add_meshVisualization_iris_domain(simEnv.meshcat, is_visible=True)
    
    regions = irisWrapper.load_regions_from_file() 
    # regions = None
    if regions is None: 
        irisWrapper.solveIRIS()
        irisWrapper.save_regions_to_file()
    
    irisWrapper.add_meshVisualization_iris_regions(simEnv.meshcat, is_visible=True) 
    
    
    print("################ FINISHED IRIS computation ################\n\n")
    
  
    

    
    input("Press Enter to continue...\n")
    
    print("################ Solving GCS trajectory ################\n")
    options = gcsUtils.GCSTrajectoryOptions()
    options.use_BezierGCS = False
    options.regions = irisWrapper.iris_regions
    options.path_continuity_order = 4 # 3 works for Bezier
    options.Bspline_order = 7   # 4 works for Bezier
    # options.edges = None

    options.start_velocity_lb = START_VELOCITY_LB
    options.goal_velocity_lb = GOAL_VELOCITY_LB




    options.hdot_min = 1e-3
    options.full_dim_overlap = True
    if irisOptions.use_CliqueCover:
        options.traj_file_path = f'trajectory_files/cliques/traj_scenery{scenery}_p{irisOptions.clique_num_points}_numRegs{irisOptions.num_regions}'
    else:
        options.traj_file_path = f'trajectory_files/random/traj_scenery{scenery}_numRegs{irisOptions.num_regions}'
        

    options.derivative_regularization = 1e-3 
    
    gcsTraj = gcsUtils.GCSTrajectory(KDIMENTIONS, options)
    start = [0, 0, 1]
    goal1 = [0.5, 11.5, 1.5]
    goal2 = [0, 10.5, 1.5]
    goal3 = [0, 17.5, 1.2]
    goal4 = [-2, 6.5, 1.5]
    goal5 = [-3, 12, 0.5]
    goal6 = [-3, 17.5, 1.2]
    goal7 = [4.5, 17.5, .5]
    
    goal8 = [3.5, 18.5, .5]
    goal =  goal6
    
        
    gcsTraj.add_start_goal_and_viz(start, goal, simEnv.meshcat, zero_deriv_boundary=None)
    gcsTraj.add_pathLengthCost(10)
    gcsTraj.add_timeCost(1000)
    gcsTraj.add_velocityBounds(VEL_LOWER_BOUND, VEL_UPPER_BOUND)
    gcsTraj.add_continuityContraint()
    
    
    input("Press Enter to continue...")
    
    # input("Press Enter to solve the trajectory...")
    if gcsTraj.load_trajectory_from_file() is None:
        gcsTraj.solve(preprocessing=True)   
        gcsTraj.save_trajectory_to_file()
        
    num_points = 2000
    gcsTraj.visualize_trajectory(simEnv.meshcat, num_points)

    print("################ FINISHED solving GCS tracjectory  ################\n\n")
    
    
    
    print("################ Starting Simulation ################\n\n")
    
    simulator = sim.SimulationEnvironment(PATH_TO_OBSTACLE_ENV)
    simulator.connect_meshcat()
    simulator.add_fixed_wing()
    # simulator.add_controller(gcsTraj.trajectory.value(6))
    simulator.add_constant_inputSource([0,0,0,0])
    simulator.build_model()
    simulator.save_and_display_diagram()
    
    
    sim.print_model_instances(simulator.plant)

    # Visualization
    irisWrapper.add_meshVisualization_iris_regions(simulator.meshcat, is_visible=True) 
    gcsTraj.visualize_start_goal(start, goal, simulator.meshcat)
    gcsTraj.visualize_trajectory(simulator.meshcat, num_points)
    irisWrapper.add_meshVisualization_iris_obstacles(simulator.meshcat, is_visible=False)
    irisWrapper.add_meshVisualization_iris_domain(simulator.meshcat, is_visible=True)

    simulator.simulate(5, gcsTraj.trajectory)
    
    
    # s0 = pp.GliderState(np.zeros(7))
    # s0.x = -3.5
    # s0.z = 0.1
    # s0.xdot = 7.0
    
    # pp.draw_glider(s0[:], simulator.meshcat )
    
    print("################ FINISHED Simulation ################\n\n")
    
    
    
    input("Press Enter to quit...")
    
    