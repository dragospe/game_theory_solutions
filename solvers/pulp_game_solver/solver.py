"""Package to solve finite, zero-sum, two person games using linear programming
as in Ferguson's "Game Theory," page 38."""


from pulp import LpVariable, LpProblem, LpStatus, lpSum, LpMaximize
import numpy as np

class strategic_form_game():
  def __init__(self,
               payoff_matrix=None,
               game_title=None):

    self.payoff_matrix = payoff_matrix
    self.game_title = game_title

    self.player_I_optimal_strategy = None
    self.player_II_optimal_strategy = None
    self.value = None
    self.problem = None

  def lp_solve(self, write=False):
    offset = self.payoff_matrix.min() 
    

    problem = LpProblem(self.game_title, LpMaximize)

    # Problem should maximize game value (for player I)
    value = LpVariable("value")
    problem += value

    # Initialize array of player I's strategies
    player_I_strategies = np.empty(self.payoff_matrix.shape[1],
                                   dtype = np.dtype(object))
    for i in range(player_I_strategies.shape[0]):
      player_I_strategies[i] = \
        LpVariable("Row " + str(i+1) + " probability", 0, 1)


    # Add constraints 
    for i in enumerate(np.matmul(player_I_strategies, 
                                 self.payoff_matrix - offset)):
      problem += value <= i[1], \
                 "Player II plays column " + str(i[0]+1)

    problem += np.sum(player_I_strategies) == 1, \
               "Probabilities sum to 1"

    if write == True:
      problem.writeLP(game_title + ".lp")
    
    problem.solve()

    print("Status:", LpStatus[problem.status])
    if problem.status == 1:
      self.value = value.varValue + offset

    for v in problem.variables()[:-1]:
      print(v.name, "=", v.varValue)
    
    print("Value =", self.value)
    
    self.problem = problem

    return self.problem
