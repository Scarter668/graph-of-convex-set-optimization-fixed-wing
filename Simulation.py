import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Ellipse
from matplotlib.transforms import Affine2D
from scipy.spatial import HalfspaceIntersection

from matplotlib.animation import FuncAnimation

from pydrake.all import (
    MathematicalProgram, Solve, eq, le, ge,

)
import pydrake.geometry.optimization as opt

from underactuated import running_as_notebook, ManipulatorDynamics

# Might be handy for calculating dynamics
# from underactuated import ManipulatorDynamics
# notebook 1 from underactuated
# https://deepnote.com/workspace/underactuated-87a9-0a169961-cde7-4b84-9a8c-cb2238edaf1d/project/01-Fully-actuated-vs-Underactuated-Systems-Priv-cfba75c0-ab8d-4d8e-ad91-7cd690cd39b1/notebook/intro-3deea25e98654b67906829e2765be9a7



## used 
from pydrake.multibody.plant import AddMultibodyPlantSceneGraph
from pydrake.multibody.parsing import Parser

from pydrake.systems.framework import DiagramBuilder
from pydrake.systems.primitives import ConstantVectorSource
from pydrake.systems.controllers import FiniteHorizonLinearQuadraticRegulatorOptions, MakeFiniteHorizonLinearQuadraticRegulator
from pydrake.systems.sensors import CameraInfo, RgbdSensor

# from underactuated import ConfigureParser

from pydrake.examples import QuadrotorPlant, QuadrotorGeometry, StabilizingLQRController

import pydrake.geometry as pyGeo

import IrisUtils as irisUtils

from pydrake.all import (
    Simulator,
    BodyIndex, 
    ModelInstanceIndex,
    LogVectorOutput,
    RigidTransform
)


from matplotlib import pyplot as plt

from IPython.display import SVG, display
import pydot
import cairosvg ##To save the svg file

import PerchingPlane as pp
import FixedWing as fw
import FlatnessInverter as fi
import GeometryUtils as geoUtils
import gcs
import gcs.bezier




FIXED_WING = 0
QUADROTOR = 1
PERCHING_PLANE = 2

PLANT_TYPE = FIXED_WING


class SimulationEnvironment():
    
    builder = None
    plant = None
    scene_graph = None
    meshcat = None
    diagram = None
    # obstacles_vertices = []

    
        
    def __init__(self, obstacle_load_path=""):
        
        
        ## assign the builder, plant and scene_graph
        self.builder = DiagramBuilder()
        self.plant, self.scene_graph = AddMultibodyPlantSceneGraph(self.builder, time_step = 0.0)
        
        self.plant.set_name("obstacle_UAV_plant")
        
        if obstacle_load_path:
            ## load the obstacles
            with open(obstacle_load_path, 'r') as file:
                urdf_obstacle = file.read()
            
            obstacle_instance = Parser(self.plant).AddModelsFromString(urdf_obstacle, "urdf")[0] # returns a list with length 1
            
            self.plant.WeldFrames(self.plant.world_frame(), 
                              self.plant.GetFrameByName("ground", obstacle_instance)
                              )
            print("Obstacles loaded")
            
        else:
            print("No obstacles loaded")
            
        self.plant.Finalize()
            
        return
    
    
    def add_fixed_wing(self):
        if PLANT_TYPE == QUADROTOR:
            m = 0.775
            arm_length = 0.15
            inertia = np.diag([0.0015, 0.0025, 0.0035])
            k_f = 1.0
            k_m = 0.0245   
            self.quad_plant = self.builder.AddSystem(QuadrotorPlant(m, arm_length, inertia, k_f ,k_m))
            self.quad_plant.set_name("quadrotor")
            QuadrotorGeometry.AddToBuilder(self.builder, self.quad_plant.get_output_port(0), self.scene_graph)
            return
        
        elif PLANT_TYPE == PERCHING_PLANE:
            
            self.perching_plane = self.builder.AddSystem(pp.GliderPlant())
            pp.GliderGeometry.AddToBuilder(self.builder, self.perching_plane.GetOutputPort("state"), self.scene_graph)
            self.perching_plane.set_name("perching_plane")
            return
            
        elif PLANT_TYPE == FIXED_WING:
            self.fixed_plane = self.builder.AddSystem(fw.FixedWingPlant())
            fw.FixedWingGeometry.AddToBuilder(self.builder, self.fixed_plane.GetOutputPort("full_state"), self.scene_graph)
            self.fixed_plane.set_name("fixed_wing")
            
            self.logger_state = LogVectorOutput(self.fixed_plane.GetOutputPort("full_state"), self.builder)
            self.logger_force = LogVectorOutput(self.fixed_plane.GetOutputPort("spatial_force"), self.builder)

            return
        
        raise ValueError("Invalid plant type")
    
    def AddRgbdSensor(self, plant, builder, X_PC,
                      depth_camera=None,
                      renderer=None,
                      parent_frame_id=None):
        
        # if sys.platform == "linux" and os.getenv("DISPLAY") is None:
        #     from pyvirtualdisplay import Display

        #     virtual_display = Display(visible=0, size=(1400, 900))
        #     virtual_display.staz()

        if not renderer:
            renderer = "my_renderer"

        if not parent_frame_id:
            parent_frame_id = self.scene_graph.world_frame_id()
            print("Parent frame id used for camera: World_frame_id -", parent_frame_id)

        if not self.scene_graph.HasRenderer(renderer):
            self.scene_graph.AddRenderer(
                renderer, pyGeo.MakeRenderEngineVtk(pyGeo.RenderEngineVtkParams())
            )

        if not depth_camera:
            depth_camera = pyGeo.DepthRenderCamera(
                pyGeo.RenderCameraCore(
                    renderer,
                    CameraInfo(width=640, height=480, fov_y=np.pi / 4.0),
                    pyGeo.ClippingRange(near=0.1, far=10.0),
                    RigidTransform(),
                ),
                pyGeo.DepthRange(0.1, 10.0),
            )

        rgbd = builder.AddSystem(
            RgbdSensor(
                parent_id=parent_frame_id,
                X_PB=X_PC,
                depth_camera=depth_camera,
                show_window=False,
            )
        )
            
    
    
    def add_constant_inputSource(self, value):
        if PLANT_TYPE == FIXED_WING:
            vsource = self.builder.AddSystem(ConstantVectorSource(value))
            self.builder.Connect(vsource.get_output_port(), self.fixed_plane.get_input_port(0))
            
    
    def add_controller(self, x_traj=None, u_traj=None, Q=None, R=None,Qf=None, point=None):
        
        if PLANT_TYPE == QUADROTOR:
            if point is None:
                point = [0, 0, 1]
            controller = self.builder.AddSystem(StabilizingLQRController(self.quad_plant, point))
            self.builder.Connect(controller.get_output_port(0), self.quad_plant.get_input_port(0))
            self.builder.Connect(self.quad_plant.get_output_port(0), controller.get_input_port(0))
            return
        
        elif PLANT_TYPE == PERCHING_PLANE:
            
            self.x_traj , self.u_traj = pp.dircol_perching()
            
            Q = np.diag([10, 10, 10, 1, 1, 1, 1])
            R = [0.1]
            options = FiniteHorizonLinearQuadraticRegulatorOptions()
            options.Qf = np.diag(
                [
                    (1 / 0.05) ** 2,
                    (1 / 0.05) ** 2,
                    (1 / 3.0) ** 2,
                    (1 / 3.0) ** 2,
                    1,
                    1,
                    (1 / 3.0) ** 2,
                ]
            )
            # options.use_square_root_method = True  # Pending drake PR #16812
            options.x0 = self.x_traj
            options.u0 = self.u_traj

            controller = self.builder.AddSystem(
                MakeFiniteHorizonLinearQuadraticRegulator(
                    system=self.perching_plane,
                    context=self.perching_plane.CreateDefaultContext(),
                    t0=self.x_traj.start_time(),
                    tf=self.x_traj.end_time(),
                    Q=Q,
                    R=R,
                    options=options,
                )
            )
            self.builder.Connect(controller.get_output_port(), self.perching_plane.get_input_port())
            self.builder.Connect(self.perching_plane.GetOutputPort("state"), controller.get_input_port())
            
            return
            
        elif PLANT_TYPE == FIXED_WING:
            if x_traj==None or u_traj==None:
                raise ValueError("Missing x or u trajectory")
            
            options = FiniteHorizonLinearQuadraticRegulatorOptions()
            options.Qf = Qf
            # options.use_square_root_method = True  # Pending drake PR #16812
            options.x0 = x_traj
            options.u0 = u_traj

            controller = self.builder.AddSystem(
                MakeFiniteHorizonLinearQuadraticRegulator(
                    system=self.fixed_plane,
                    context=self.fixed_plane.CreateDefaultContext(),
                    t0=x_traj.start_time(),
                    tf=x_traj.end_time(),
                    Q=Q,
                    R=R,
                    options=options,
                )
            )
            self.builder.Connect(controller.get_output_port(), self.fixed_plane.get_input_port())
            self.builder.Connect(self.fixed_plane.GetOutputPort("full_state"), controller.get_input_port())
            
            
            
            return
        
        
        
        
        raise ValueError("Invalid plant type")
        
        
        
    
    def connect_meshcat(self):
        self.meshcat = pyGeo.StartMeshcat()
        self.visualizer = pyGeo.MeshcatVisualizer.AddToBuilder(self.builder, self.scene_graph, self.meshcat)
        self.visualizer.set_name("obstacle_UAV_visualizer")

        return
    
    def build_model(self):
        self.diagram = self.builder.Build()
        self.diagram.set_name("Root_Diagram")
        return 
    
    def save_and_display_diagram(self, save=True):
        svg = pydot.graph_from_dot_data(
                    self.diagram.GetGraphvizString(
                        max_depth=3))[0].create_svg()
        svgdisplay = SVG(svg)
        
        svg_filename = "figures/Root_diagram.svg"
        image_filename = "figures/Root_diagram.png"
        
        
        if save:
            # Write the SVG content to a file
            with open(svg_filename, "wb") as svg_file:
                svg_file.write(svg)

        cairosvg.svg2png(url=svg_filename, write_to=image_filename)
                
        display(svgdisplay)
        
        
        ## force publish to meshcat
        c = self.diagram.CreateDefaultContext()
        self.diagram.ForcedPublish(c)
        return 
    
    # def view_model(self, urdf_obstacle):
    #     visualizer = ModelVisualizer(meshcat=self.meshcat)
    #     visualizer.parser().AddModelsFromString(urdf_obstacle, "urdf")
    #     visualizer.Run(loop_once=False)
    #     return
    
    def compute_obstacles(self, irisWrapper):
        
        root_context = self.diagram.CreateDefaultContext()
        scenegraph_context = self.scene_graph.GetMyContextFromRoot(root_context)
        
        query_object = self.scene_graph.get_query_output_port().Eval(scenegraph_context)
        
        # inspector = self.scene_graph.model_inspector()
        # self.obstacles_vertices = geoUtils.get_obstacle_vertices(self.plant, query_object, inspector)
        # print("Obstacle vertices: ", self.obstacles_vertices)
        
        # ## loop through obslist and print
        # print("Obstacle list: ", obslist)
        # for obs in obslist:
        #     print("Obstacle: ", obs)
        
        

        return irisWrapper.retrieve_iris_obstacles_from_query(query_object, self.scene_graph)
        
    
     
    def simulate(self, time=5, trajectory=None, POV_ENABLED=True):
        simulator = Simulator(self.diagram)
        simulator.set_target_realtime_rate(1.0)
        simulator_context = simulator.get_mutable_context()
        
        
        if PLANT_TYPE == QUADROTOR:
            quadrotor_context = self.diagram.GetMutableSubsystemContext(self.quad_plant, simulator_context)
        
            # Simulate
            for _ in range(5):
                simulator_context.SetTime(0.0)
                quadrotor_context.SetContinuousState(
                    0.5
                    * np.random.randn(
                        12,
                    )
                )
                simulator.Initialize()
                
                self.visualizer.StartRecording()
                simulator.AdvanceTo(time)
                self.visualizer.StopRecording()
                self.visualizer.PublishRecording()
                return
            
        elif PLANT_TYPE == PERCHING_PLANE:
            
            perching_plane_context = self.diagram.GetMutableSubsystemContext(self.perching_plane, simulator_context)
            rng = np.random.default_rng(123)
            
            for _ in range(2):
                simulator_context.SetTime(self.x_traj.start_time())
                initial_state = pp.GliderState(self.x_traj.value(self.x_traj.start_time()))
                initial_state.z += 0.04 * rng.standard_normal()
                perching_plane_context.SetContinuousState(initial_state[:])

                simulator.Initialize()
                self.visualizer.StartRecording()
                simulator.AdvanceTo(self.x_traj.end_time())
                self.visualizer.StopRecording()
                self.visualizer.PublishRecording()
                
                return
            
        elif PLANT_TYPE == FIXED_WING:
            perching_plane_context = self.diagram.GetMutableSubsystemContext(self.fixed_plane, simulator_context)
            if trajectory is None:
                raise ValueError("Trajectory is required for fixed wing simulation")
            
            
            num_timesteps = 1500
            num_dofs = 3
            p_numeric = np.empty((num_timesteps, num_dofs))
            dp_numeric = np.empty((num_timesteps, num_dofs))
            ddp_numeric = np.empty((num_timesteps, num_dofs))
            dddp_numeric = np.empty((num_timesteps, num_dofs))
            ddddp_numeric = np.empty((num_timesteps, num_dofs))
            sample_times_s = np.linspace(
                trajectory.start_time(), trajectory.end_time(), num=num_timesteps, endpoint=True
            )

            # Calculating the velocity magnitudes
            velocity_magnitudes = np.empty(num_timesteps)

            for i, t in enumerate(sample_times_s):
                
                p_numeric[i] = trajectory.value(t).flatten()
                dp_numeric[i] = trajectory.EvalDerivative(t, derivative_order=1).flatten()
                ddp_numeric[i] = trajectory.EvalDerivative(t, derivative_order=2).flatten()
                velocity_magnitudes[i] = np.sqrt(np.sum(dp_numeric[i]**2))
                
                if type(trajectory) != gcs.bezier.BezierTrajectory:    
                    dddp_numeric[i] = trajectory.EvalDerivative(t, derivative_order=3).flatten()
                    ddddp_numeric[i] = trajectory.EvalDerivative(t, derivative_order=4).flatten()

            # Find indices where the velocity is not zero
            non_zero_velocity_indices = np.where(velocity_magnitudes > 0)[0]

            if len(non_zero_velocity_indices) == 0:
                raise ValueError("All velocities are zero, which is unexpected in a valid trajectory.")

            # Clipping trajectories to non-zero velocities
            start_index = non_zero_velocity_indices[0]
            end_index = non_zero_velocity_indices[-1]

            p_numeric = p_numeric[start_index:end_index+1]
            dp_numeric = dp_numeric[start_index:end_index+1]
            ddp_numeric = ddp_numeric[start_index:end_index+1]
            dddp_numeric = dddp_numeric[start_index:end_index+1]
            ddddp_numeric = ddddp_numeric[start_index:end_index+1]
            sample_times_s = sample_times_s[start_index:end_index+1]

            m = 1.54
            g = 9.80665
            
            
            keep_looping = True
            
            
            def muFromfullStateAhead(n, ahead=15):
                if (n+ahead) >= len(p_numeric):
                    step = -1
                else:
                    step = n+ahead 
                
                p = np.array(p_numeric[step, :])
                dp = np.array(dp_numeric[step, :])
                ddp = np.array(ddp_numeric[step, :])
                dddp = np.array(dddp_numeric[step, :])
                ddddp = np.array(ddddp_numeric[step, :])
                
               
                stateNED_ahead = fi.UnflattenFixedWingStatesNED(p, dp, ddp, dddp, ddddp, m, g)
                
                return fi.FixedWingStatesNED(stateNED_ahead).mu
            
            
            while keep_looping:
            
                simulator_context.SetTime(sample_times_s[0])
                simulator.Initialize()
                self.meshcat.StartRecording()

                trajectory_frames = []
                for i, (p, dp, ddp, dddp, ddddp, t) in enumerate(zip(
                    p_numeric,
                    dp_numeric,
                    ddp_numeric,
                    dddp_numeric,
                    ddddp_numeric,
                    sample_times_s
                )):
                    NUM_FULL_STATE = 16
                    fullState = fw.FullState(np.zeros(NUM_FULL_STATE))
                    stateNED = fi.UnflattenFixedWingStatesNED(p, dp, ddp, dddp, ddddp, m, g)
                    fullState[:12] = stateNED
                    
                    X_DrB = fi.ExtractTransformation(stateNED)
                    
                    
                    if POV_ENABLED:
                        R_DrB = X_DrB.rotation()
                        pos_B = X_DrB.translation()
                        camera_pos_B = pos_B + R_DrB @ np.array([-1, 0, -.2])
                        self.meshcat.SetCameraPose(camera_pos_B, pos_B)

                    
                    mu = muFromfullStateAhead(i, ahead=30)
                    # Set some values for simulation
                    fullState.delta_le = np.clip(mu, -0.8, 0.8)
                    fullState.delta_re = np.clip(-mu, -0.8, 0.8)
                    fullState.delta_lm = 1.5*t
                    fullState.delta_rm = 1.5*t
                    
                    
                    trajectory_frames.append(X_DrB)                

                    simulator_context.SetContinuousState(fullState[:])
                    simulator.AdvanceTo(t)

                geoUtils.visualize_key_frames(self.meshcat, trajectory_frames)

                self.meshcat.StopRecording()
                self.meshcat.PublishRecording()
                
                if POV_ENABLED and replay_requested_from_user():
                    keep_looping = True
                    simulator.set_target_realtime_rate(1.0)
                    
                else:                 
                    keep_looping = False
            
            return
            
            
        
        raise ValueError("Invalid plant type")



def replay_requested_from_user():
    #ask user if we want to loop again
    while True:
        ans = input("Do you want to loop again? (y/n): ")
        if ans.lower() == 'y':
            return True
        elif ans.lower() == 'n':
            return False
        else:
            print("Invalid input")
            continue


######################## PLANT UTILS ############################
    
def print_model_instances(plant):
    
    print("Model Instances and Bodies:\n")
    # Iterate over all model instances in the plant
    for i in range(plant.num_model_instances()):
        model_instance = ModelInstanceIndex(i)
        model_instance_name = plant.GetModelInstanceName(model_instance)
        print(f"Model Instance: {model_instance_name}")

        # Get all body indices for the current model instance
        body_indices = plant.GetBodyIndices(model_instance)

        # Iterate over all body indices and print their names
        for body_index in body_indices:
            body = plant.get_body(BodyIndex(body_index))
            print(f"  Body Name: {body.name()}")
    
    print("Number of model instances: ", plant.num_model_instances())
    print("Number of bodies: ", plant.num_bodies())
    print()
    

######################## direcol_perching UTILS ############################


        
#################################################################
        
## once simulatin : 
#  if running_as_notebook:
#         simulator.set_target_realtime_rate(1.0)
   
   

 
if __name__ == "__main__":
    

    path_to_urdf_obstacle = "models/obstacles_house_formation.urdf"

    simEnv = SimulationEnvironment(obstacle_load_path=path_to_urdf_obstacle)
    simEnv.connect_meshcat()
        
    simEnv.build_model()
    
    simEnv.save_and_display_diagram()

    
    print_model_instances(simEnv.plant)

    
    
    # print("Obstacle geometries:")
    # print(geoUtils.get_obstacleID_geometries(simEnv.plant))
    # print()
    
    irisWrapper = irisUtils.IrisWrapper()
    
    simEnv.compute_obstacles(irisWrapper)
    
    
    

    
    # simEnv.test_mescat()
    

 
    ## to look at the examples
    # models = [
    #     "example_models/obstacles_corridors.urdf",
    #     "example_models/obstacles_forrest.urdf",
    #     "example_models/obstacles_groups.urdf",
    #     "example_models/obstacles_simple.urdf",
    #     "example_models/obstacles_walls.urdf",
    #     "example_models/obstacles.urdf",
    # ]
     
    # for model in models:
    #     simEnv = SimulationEnvironment(model, visualize=True)
    #     simEnv.build_model()
    
        
    
    
    
    # read input from the user
    
    input("Press Enter to quit...")
       
        
    


