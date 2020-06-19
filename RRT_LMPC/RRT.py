import numpy as np
import casadi as ca
from copy import deepcopy
import sys
sys.path.append('../envs')
from pointbot import *
import matplotlib.pyplot as plt
import scipy.linalg as la

class Node():
	"""RRT nodes"""
	def __init__(self, state=None):
		self.state = state
		self.prev_action = []
		self.parent = None

	def _is_goal(self, goal, d_thresh=0.5):
		print(np.linalg.norm(self.state[:3:2] - goal[:3:2]))
		return (np.linalg.norm(self.state[:3:2] - goal[:3:2]) <= d_thresh)

class RRT(Node):
	"""Class for Rapid Random Tree methods"""
	def __init__(self, env, mode):
		super(RRT, self).__init__()
		self.env = env
		self.mode = mode
		self.node_arr = []

	def sample_random(self):
		# randomly sample a point in the observation space
		rand_node = Node()
		state = self.env.observation_space.sample() #TODO: check feasibility of state
		rand_node.state = state
		return rand_node

	def find_nearest_node(self, sample_node):
		dist = [self.get_dist(sample_node, node) for node in self.node_arr]
		near_node_ind = dist.index(min(dist))
		return near_node_ind
	
	def get_dist(self, node1, node2):
		dist = np.linalg.norm(node1.state - node2.state)
		return np.round(dist, decimals=3)	

	def steer(self, from_node, to_node):
		if self.mode == 'mpc':
			# steer tree to desired node (using 1. MPC 2. LMPC 3. Feedback policy)
			horizon = 5
			temp_node = Node(self.node_arr[from_node].state)
			dist = self.get_dist(temp_node, to_node)
			n = self.env.observation_space.shape[0]
			d = self.env.action_space.shape[0]
			# while(dist >= 0.1):
			# define variables
			x = ca.SX.sym('x', (horizon+1) * n)
			u = ca.SX.sym('u', horizon * d)
			# define cost
			cost = 0
			for i in range(horizon):
				cost += ca.norm_2(x[(i+1)*n:(i+2)*n] - to_node.state)
			# define constraints
			current_state = temp_node.state
			constraints = []
			for j in range(horizon):
				next_state = (self.env.A @ (current_state)) + (self.env.B @ (u[j*d:(j+1)*d]))
				constraints = ca.vertcat(constraints, x[(j+1)*n + 0] == next_state[0])
				constraints = ca.vertcat(constraints, x[(j+1)*n + 1] == next_state[1])
				constraints = ca.vertcat(constraints, x[(j+1)*n + 2] == next_state[2])
				constraints = ca.vertcat(constraints, x[(j+1)*n + 3] == next_state[3])
			lbg = [0] * (horizon) * n
			ubg = [0] * (horizon) * n
			# solve
			opts = {'verbose':False, 'ipopt.print_level':0, 'print_time':0}
			nlp = {'x':ca.vertcat(x,u), 'f':cost, 'g':constraints}
			solver = ca.nlpsol('solver', 'ipopt', nlp, opts)
			lbx = current_state.tolist() + [-100]*n*horizon + [-1]*d*horizon
			ubx = current_state.tolist() + [100]*n*horizon + [1]*d*horizon

			sol = solver(lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
			sol_val = np.array(sol['x'])
			u_star = sol_val[(horizon+1)*n:(horizon+1)*n+d]

			new_node = Node()
			new_node.state = self.env._next_state(current_state, u_star.reshape(-1))
			new_node.parent = self.node_arr[from_node]

		if self.mode == 'lqr':
			n = self.env.observation_space.shape[0]
			d = self.env.action_space.shape[0]
			
			# get lqr gains
			Q=np.eye(n)
			R=10*np.eye(d)
			A=self.env.A
			B=self.env.B
			P = la.solve_discrete_are(A, B, Q, R)
			Ks = -np.linalg.inv(R + B.T.dot(P).dot(B)).dot(B.T).dot(P).dot(A)

			# obtain optimal control action
			start_state = self.node_arr[from_node].state
			goal_state = to_node.state
			u_star = np.dot(Ks, (start_state - goal_state))

			# obtain new node
			new_node = Node()
			new_node.state = self.env._next_state(start_state, u_star)
			new_node.parent = self.node_arr[from_node]
 
		return new_node		

	def check_collision(self, from_node, end_node):
		return False

	def build_tree(self, start_node, goal_node=None):
		# initialize tree with starting node
		self.node_arr.append(start_node)
		print(self.node_arr[-1].state)
		# while goal not reached or fixed number of nodes
		while not self.node_arr[-1]._is_goal(goal_node.state):
			# randomly sample point in the observation space
			rnd_sample = self.sample_random()
			# find nearest point in the tree
			nearest_node_index = self.find_nearest_node(rnd_sample)
			nearest_node = self.node_arr[nearest_node_index]
			# plan to the sampled node and add each node to the tree
			new_node = self.steer(from_node=nearest_node_index, to_node=rnd_sample)
			self.node_arr.append(new_node)
			plt.cla()
			for node in self.node_arr:
				if node.parent == None:
					continue
				x = [node.state[0], node.parent.state[0]]
				y = [node.state[2], node.parent.state[2]]
				plt.plot(x,y, color='blue')
			plt.pause(0.01)

	def plot_final_traj(self):
		x = []
		y = []
		current_node = self.node_arr[-1]
		x.append(current_node.state[0])
		y.append(current_node.state[2])

		while not current_node.parent == None:
			x.append(current_node.parent.state[0])
			y.append(current_node.parent.state[2])
			current_node = current_node.parent

		plt.cla()
		plt.plot(x,y, color='blue')
		plt.show()

	def rewire(self):
		pass

if __name__ == '__main__':
	env = PointBot()
	tree = RRT(env=env, mode='lqr')
	start_config = Node(state = np.array([-10,0,0,0]))
	goal_config = Node(state = np.array([0,0,0,0]))
	tree.build_tree(start_node=start_config, goal_node=goal_config)
	tree.plot_final_traj()