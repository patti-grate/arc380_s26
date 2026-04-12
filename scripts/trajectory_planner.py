# planner thing which will compute the trajectories, and reachability of them

# expected behaviors:
# input: grasping brick geo, placing brick geo, environmental mesh (converted from depth image)
# output: a collision free trajectory with minimum joint movements, return "PLANNING FAILED DUE TO REACHABILITY" if all trials failed.


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
