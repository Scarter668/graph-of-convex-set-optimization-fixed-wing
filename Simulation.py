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
from pydrake.systems.framework import DiagramBuilder
from pydrake.multibody.parsing import Parser
# from underactuated import ConfigureParser

from pydrake.examples import QuadrotorPlant, QuadrotorGeometry, StabilizingLQRController

import pydrake.geometry as pyGeo

import IrisUtils as irisUtils

from pydrake.all import (
    Simulator,
    BodyIndex, 
    ModelInstanceIndex,
    LinearQuadraticRegulator   
)


from IPython.display import SVG, display
import pydot
import cairosvg ##To save the svg file


class SimulationEnvironment():
    
    builder = None
    plant = None
    scene_graph = None
    meshcat = None
    diagram = None
    # obstacles_vertices = []

    
        
    def __init__(self, obstacle_load_path):
        
        
        ## assign the builder, plant and scene_graph
        self.builder = DiagramBuilder()
        self.plant, self.scene_graph = AddMultibodyPlantSceneGraph(self.builder, time_step = 0.0)
        
        self.plant.set_name("obstacle_UAV_plant")
        
        
        ## load the obstacles
        with open(obstacle_load_path, 'r') as file:
            urdf_obstacle = file.read()
        
        obstacle_instance = Parser(self.plant).AddModelsFromString(urdf_obstacle, "urdf")[0] # returns a list with length 1
        

        self.plant.WeldFrames(self.plant.world_frame(), 
                              self.plant.GetFrameByName("ground", obstacle_instance)
                              )
        self.plant.Finalize()
            
        return
    
    
    def add_fixed_wing(self):
        m = 0.775
        arm_length = 0.15
        inertia = np.diag([0.0015, 0.0025, 0.0035])
        k_f = 1.0
        k_m = 0.0245   
        self.quad_plant = self.builder.AddSystem(QuadrotorPlant(m, arm_length, inertia, k_f ,k_m))
        self.quad_plant.set_name("quadrotor")
        QuadrotorGeometry.AddToBuilder(self.builder, self.quad_plant.get_output_port(0), self.scene_graph)
    
    def add_controller(self, point):
        controller = self.builder.AddSystem(StabilizingLQRController(self.quad_plant, point))
        self.builder.Connect(controller.get_output_port(0), self.quad_plant.get_input_port(0))
        self.builder.Connect(self.quad_plant.get_output_port(0), controller.get_input_port(0))
        
        
    
    def connect_meshcat(self):
        self.meshcat = pyGeo.StartMeshcat()
        meshvis  = pyGeo.MeshcatVisualizer.AddToBuilder(self.builder, self.scene_graph, self.meshcat)
        meshvis.set_name("obstacle_UAV_visualizer")
        return
    
    def build_model(self):
        self.diagram = self.builder.Build()
        self.diagram.set_name("Root_Diagram")
        return 
    
    def save_and_display_diagram(self):
        svg = pydot.graph_from_dot_data(
                    self.diagram.GetGraphvizString(
                        max_depth=3))[0].create_svg()
        svgdisplay = SVG(svg)
        
        svg_filename = "figures/Root_diagram.svg"
        image_filename = "figures/Root_diagram.png"
        
        # Write the SVG content to a file
        with open(svg_filename, "wb") as svg_file:
            svg_file.write(svg)

        cairosvg.svg2png(url=svg_filename, write_to=image_filename)
                
        display(svgdisplay)
        
        
        ## force publish to meshcat
        self.diagram.ForcedPublish(self.diagram.CreateDefaultContext())
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
        
    
     
    def simulate(self, time=5):
        simulator = Simulator(self.diagram)
        simulator.set_target_realtime_rate(1.0)
        simulator_context = simulator.get_mutable_context()

        fixed_wing_context = self.diagram.GetMutableSubsystemContext(self.quad_plant, simulator_context)
        
        
        # Simulate
        for i in range(5):
            simulator_context.SetTime(0.0)
            fixed_wing_context.SetContinuousState(
                0.5
                * np.random.randn(
                    12,
                )
            )
            simulator.Initialize()
            simulator.AdvanceTo(time)
    


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
    

######################## IRIS UTILS ############################


        
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
       
        
    


