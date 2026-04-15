# as-built mesh construction from depth image and brick pose detection by comparing meshes/depth images.

# function that takes two depth images (2d & 3d) and retrieve a single brick pose.

# from depth point cloud images --> line up both frames and find the point cloud of the new block
# then fit the block to the point cloud (icp) --> say that one side will always be flat
# then extract pose and position from that block


# iterate to see if the block is reachable

# use joint constaints

# returns this all to the sequencer

# remembering function -- cases for recognizing it if the original structure is rotated + translated

# use depth image to see what is there --> turn it into a mesh
