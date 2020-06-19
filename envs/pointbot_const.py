"""
Constants associated with the PointBot env.
"""

START_POS = [-50, 0]
END_POS = [0, 0]
START_STATE = [START_POS[0], 0, START_POS[1], 0]
GOAL_STATE = [END_POS[0], 0, END_POS[1], 0]
GOAL_THRESH = 1.

MAX_FORCE = 1
HORIZON = 50
HARD_MODE = True
NOISE_SCALE = 0.05


