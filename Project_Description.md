Project Guidelines (from final project document): 

The goal of the final project is to develop an end-to-end robotic assembly workflow that integrates perception, planning, and execution. Students will begin with a structured pick-and-place pipeline using known object configurations, and progressively incorporate perception, reasoning, and human–robot interaction to enable adaptive assembly processes. While initial tasks focus on assembling tower-like structures, the final project may extend beyond fixed designs to explore generative, data-driven, or interactive fabrication workflows.

Direction: 

Overarching objective: 
Behavior cloning - learning brick stacking rules by observing. 

Key components: 
	Perception module: given the current camera observation (depth or RGB), compare it with the previous observation and identify the brick added to the scene, return the brick’s pose, and then store the current observation as the previous. 
#input: new observation, previous observation
#output: pose of the brick added

	Scene mesh updater: given a brick pose, reconstruct the brick in the current scene so the scene mesh is up-to-date. 
#input: brick pose to add
#output: updated scene mesh

	Trajectory planner module: given the target brick geometries for grasping and placing, explore possible frames of grasping (from different sides of the brick and with different angles of grasping), cross-matching the frames between two brick geometries, and try to generate a collision-free trajectory. Return failure if no valid trajectory from all matches. 
#input: grasping brick geo, placing brick geo, scene mesh
#output: an executable trajectory if it succeeds, a failure message if planning fails

	Behavior cloning module: A trained agent that guesses the next brick pose to add to the current scene/structure based on the past observations. The output space can be continuous or discretized (this might be simpler to train). 
#input: past 8~12 brick poses
#output: next brick pose to add to the scene
	


Training pipeline: 
	Data collection: Collect brick-by-brick scene observations of humans completing a complex brick structure. Identify the brick poses using the perception module. For each brick pose, use a trajectory planner module to determine whether it is robotically placeable. Placeable brick poses and correlated scene observations will be included in the training dataset. 
Data collection alternative solution: instead of using noisy real-world sensor data. Let’s use generated brick poses verified through simulation, meaning given a series of human-defined brick poses that form a complex structure, we test it in simulation to check buildability and if the structure stays stable after adding each brick, we take the brick poses as if we got them from real-world-sensed, and check robot reachability, then inclusion for training. 
Training: Use the collected data to train a behavior cloning agent to predict the next placeable brick based on scene observations. Additionally, we can try to encode past observations to make the representation more physically descriptive. 
Runtime: Geometric-fabric-guided policy filtering - enforced joints, environmental, and grasping constraints as a safety layer and to improve the accuracy required for successful stacking. Learned stacking policy - predicting next brick location and pose based on observation. 







Division of work: 
Geometric fabric component [both data screening for training and run time][fix today]
Constraints for the policy on the robot in terms of joint constraints 
Making specific poses to constrain how the gripper grips the block
Checking to see if a block is placed, can the robot reach it?
Brick detection component [for both training phase and run time (for detecting feedstock only)][JG, PG]
Depth image from a depth camera → translate to brick pose 
Stacking policy component - Demo data collection and training



Core References: 

Singh, R., Allshire, A., Handa, A., Ratliff, N., Wyk, K.V., n.d. DextrAH-RGB: Visuomotor Policies to Grasp Anything with Dexterous Hands.