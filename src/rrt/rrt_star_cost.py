from src.rrt.rrt_star import RRTStar
from src.rrt.heuristics import path_cost
import heapq

class RRTStarCost(RRTStar):
    def __init__(self, X, Q, x_init, goal_cost, max_samples, r, prc=0.01, rewire_count=None):

        x_goal = None
        super(RRTStarCost, self).__init__(X, Q, x_init, x_goal, max_samples, r, prc=prc, rewire_count=rewire_count)
        self.goal_cost = goal_cost  # inputs: node, outputs: raw cost, and satisfy or not

    def rrt_star_cost(self):
        self.add_vertex(0, self.x_init)
        self.add_edge(0, self.x_init, None)

        cost_heap = []

        while True:
            for q in self.Q:  # iterate over different edge lengths
                for i in range(q[1]):  # iterate over number of edges of given length to add
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

                    cost_g, goal_sat = self.goal_cost(x_new)
                    cost_p = path_cost(self.trees[0].E, self.x_init, x_new)
                    if goal_sat:
                        sat_cost = 0
                    else:
                        sat_cost = 1  # reverse to be top of the heap

                    heapq.heappush(cost_heap, (sat_cost, cost_g + cost_p, x_new))

                    # when budget, return best
                    if self.samples_taken >= self.max_samples:
                        sat_cost, total_cost, x_best = heapq.heappop(cost_heap)

                        if sat_cost == 0:
                            # reach goal
                            return total_cost, self.reconstruct_path(0, self.x_init, x_best)
                        else:
                            return None

    def connect_shortest_valid(self, tree, x_new, L_near):
        # check nearby vertices for total cost and connect shortest valid edge
        for c_near, x_near in L_near:
            if c_near + self.goal_cost(x_near) < self.c_best and self.connect_to_point(tree, x_near, x_new):
                break