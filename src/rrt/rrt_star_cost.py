from .rrt_star import RRTStar
from .heuristics import path_cost
import heapq
import random
import numpy as np
from ..utilities.geometry import steer

class RRTStarCost(RRTStar):
    def __init__(self, X, Q, x_init, goal_cost, max_samples, r, prc=0.01, rewire_count=None):

        x_goal = None
        super(RRTStarCost, self).__init__(X, Q, x_init, x_goal, max_samples, r, prc=prc, rewire_count=rewire_count)
        self.goal_cost = goal_cost  # inputs: node, outputs: raw cost, and satisfy or not

        self.cost_heap = []


    def rrt_star_cost(self):
        self.add_vertex(0, self.x_init)
        self.add_edge(0, self.x_init, None)

        while True:
            for q in self.Q:  # iterate over different edge lengths
                for i in range(q[1]):  # iterate over number of edges of given length to add

                    # sample based on cost with prc
                    if self.prc and random.random() < self.prc and len(self.cost_heap) > 0:
                        _, _, _, x_cur_best = self.cost_heap[0]
                        x_new, x_nearest = self.new_and_near_point(0, q, x_cur_best)
                    else:
                        x_new, x_nearest = self.new_and_near(0, q)
                    if x_new is None:
                        continue

                    # get nearby vertices and cost-to-come
                    L_near = self.get_nearby_vertices(0, self.x_init, x_new)

                    # check nearby vertices for total cost and connect shortest valid edge
                    self.connect_shortest_valid(0, x_new, L_near)

                    if x_new in self.trees[0].E:
                        # rewire tree
                        self.rewire(0, x_new, L_near)

                        """
                        solution = self.check_solution()
                        if solution[0]:
                            return solution[1]
                        """
                        # TODO: check why x_new might not be in E
                        # TODO: with prc, use goal to sample

                        cost_g, goal_sat = self.goal_cost(x_new)
                        cost_p = path_cost(self.trees[0].E, self.x_init, x_new)
                        if goal_sat:
                            sat_cost = 0
                        else:
                            sat_cost = 1  # reverse to be top of the heap

                        heapq.heappush(self.cost_heap, (sat_cost, cost_g, cost_p, x_new))

                    # when budget, return best
                    if self.samples_taken >= self.max_samples:
                        sat_cost, cost_g, cost_p, x_best = heapq.heappop(self.cost_heap)

                        if sat_cost == 0:
                            # reach goal
                            return cost_p, self.reconstruct_path(0, self.x_init, x_best)
                        else:
                            return None

    def connect_shortest_valid(self, tree, x_new, L_near):
        # check nearby vertices for total cost and connect shortest valid edge
        for c_near, x_near in L_near:
            if c_near + self.goal_cost(x_near)[0] < self.c_best and self.connect_to_point(tree, x_near, x_new):
                break

    def new_and_near_point(self, tree, q, point):
        v_len_sq = q[0] * q[0]
        side_len = np.sqrt(v_len_sq / self.X.dimensions)

        x_rand = self.X.sample_point_free(point, side_len)
        x_nearest = self.get_nearest(tree, x_rand)
        x_new = self.bound_point(steer(x_nearest, x_rand, q[0]))
        # check if new point is in X_free and not already in V
        if not self.trees[0].V.count(x_new) == 0 or not self.X.obstacle_free(x_new):
            return None, None
        self.samples_taken += 1
        return x_new, x_nearest