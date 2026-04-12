# planner thing which will compute the trajectories, and reachability of them

# expected behaviors:
# input: grasping brick geo, placing brick geo, environmental mesh (converted from depth image)
# output: a collision free trajectory with minimum joint movements, return "PLANNING FAILED DUE TO REACHABILITY" if all trials failed.

# The idea: 
# Have the planner attempt a few possible trajectories given a couple of grasp possibilities
# Check for collisions by comparing to either brick volume list or mesh 
#  --> Don't have to do collision avoidance for now 

# Brick volume "database":
# If the brick center is 0,0,0 and its pose is aligned with the world XYZ frame,
# BRICK_X is the length of the brick
# BRICK_Y is the width of the brick
# BRICK_Z is the height of the brick
BRICK_X = 0.05 # dummy value for now
BRICK_Y = 0.05 # another dummy value
BRICK_Z = 0.02 # another dummy value

brick_pos_list = [] # store list of bricks positions here
brick_quat_list = [] # store list of brick quaternions here 

# This function just checks if a single point (x,y,z) is within a volume of a placed brick. 
# LATER we want to check if the gripper / arm moves through any volumes that are already placed. 
def check_brick(pos): 
    x = pos[0]
    y = pos[1] 
    z = pos[2] 

    for b in brick_pos_list:
        # check if the volumes
        collision_detected = False # todo 
        if collision_detected: 
            return False
   # if all bricks are cleared
    return True 
    

# Grasping geometry: 
#    |
#    ^
# [     ]
# 
# - 1)  position: offset -x / center (0) /offset +x 
# - 2)  rotation: -45 deg / 0 deg / +45 deg
# output: gripping pose, gripping quat.  
def grasping(grasp_pos: int, 
             grasp_rot: int):

            
    return True

def _make_joint_constraint(
    joint_name: str,
    position: float,
    tolerance_above: float = 1e-3,
    tolerance_below: float = 1e-3,
    weight: float = 1.0,
) -> JointConstraint:
    jc = JointConstraint()
    jc.joint_name = joint_name
    jc.position = float(position)
    jc.tolerance_above = float(tolerance_above)
    jc.tolerance_below = float(tolerance_below)
    jc.weight = float(weight)
    return jc


# check the depth mesh and make sure that the trajectories of the sequence won't collide
