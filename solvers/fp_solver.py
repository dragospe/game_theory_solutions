import numpy as np
from tabulate import tabulate
import threading 
import queue

# Testing array
A = np.array([[2,-1,6],[0,1,-1],[-2,2,1]])

class naive_fp():

    def __init__(self, 
                 payoff_matrix, 
                 initial_i=0,
                 print_head=5,
                 print_tail=5,
                 print_indexing_at_1 = False):
        """Variables named according to Ferguson's 'Game Theory',
        part II, chapter 4, page 46.
        
        Note: to ease indexing, the strategies begin at 0; i.e.,
        a 3x3 matrix will have strategies 0, 1, and 2 rather than
        1, 2, and 3. This also means that the index for the `round`
        starts at k = 0 (meaning only the inital play has occured).
        """
        
        # Set how many lines to pretty-print
        self.print_head, self.print_tail = print_head, print_tail 
        # Set whether the indexing for printing starts a 0 (default) or 1
        # (makes it easier to see if the values match the values in texts that
        # start at k = 1)
        self.print_indexing_at_1 = print_indexing_at_1
        
        
        self.payoff_matrix = payoff_matrix
        
        # Round index
        self.k = 0
        
        # Pure strategy selections for player I
        self.i = np.array([initial_i], dtype = int)
        # Incremental payoffs for player II
        self.s = np.array([self.payoff_matrix[self.i[self.k]]])
     
        # Pure strategy selection for player II
        self.j = np.array([np.argmin(self.s[self.k])], dtype = int)
        # Incremental payoffs for player I
        self.t = np.array([self.payoff_matrix[:, self.j[self.k]]])
        
        # Select best-response for player I; choose i_{k+1}
        self.i = np.append(self.i, np.argmax(self.t[self.k]))
        
        # Game value lists
        self.v_lower = np.array([self.s[self.k][self.j[self.k]]], dtype = np.float64)
        self.v_upper = np.array([self.t[self.k][self.i[self.k + 1]]], dtype = np.float64)
        
        
        # supremum of v_upper
        self.sup_v_upper = self.v_upper
        # infimum of v_lower
        self.inf_v_lower = self.v_lower
        
        ## Best strategies on round k
        # Player I's initial strategy is the first row with probability 1
        self.pI_strategy = np.zeros([payoff_matrix.shape[0]])
        self.pI_strategy[0] = 1
        # Player II's initial strategy is j[0] = argmin(s[0]) with probability 1
        self.pII_strategy = np.zeros([payoff_matrix.shape[1]])
        self.pII_strategy[self.j[0]] = 1
        
        
    def next_rounds(self, rounds=1):
        """The rounds are calculated as follows:
        At the end of each round, all values of i[k], s[k], j[k], t[k], v_upper[k] and v_lower[k]
        have been updated.
        
        During each round, they are updated in that order.
        """

        
        for _ in range(rounds):   
            # Increment round
            self.k += 1
            
            # Increment payoffs for player II; calculate s_k
            self.s = np.append(self.s, [self.s[self.k - 1] + self.payoff_matrix[self.i[self.k]]], axis=0)
            # Select best-response for player II; choose j_k
            self.j = np.append(self.j, np.argmin(self.s[self.k]))
        
            # Increment payoffs for player I: calculate t_k
            self.t = np.append(self.t, [self.t[self.k - 1] + self.payoff_matrix[:, self.j[self.k]]], axis = 0)
            # Select best-response for player I; choose i_k
            self.i = np.append(self.i, np.argmax(self.t[self.k]))
       
    
            ## Change value bounds
            # Calculate v_upper_k
            self.v_upper = np.append(self.v_upper, 1 / (self.k+1) * self.t[self.k][self.i[self.k+1]])
            # Calculate v_lower_k
            self.v_lower = np.append(self.v_lower, 1 / (self.k + 1) * self.s[self.k][self.j[self.k]])
            
            
        
        # Compute supremum and infimum
        self.sup_v_upper = {'value' : min(self.v_upper), 'index' : np.argmin(self.v_upper)}
        self.inf_v_lower = {'value' : max(self.v_lower), 'index' : np.argmax(self.v_lower)}
      
        ## Update strategies    
        # Zero out the strategies
        self.pI_strategy = np.zeros(self.payoff_matrix.shape[0])
        # Get the number of times each pure strategy was played up to the inf
        strats_I, counts_I = np.unique(self.i[: self.inf_v_lower['index'] + 1], return_counts = True)
        # Play each strategy equally likely
        for i,j in zip(strats_I, counts_I):
            self.pI_strategy[i] = j / (self.inf_v_lower['index']+1)
    
        # Same for player II
        self.pII_strategy = np.zeros(self.payoff_matrix.shape[1])
        # Get the number of times each pure strategy was played up to the sup
        strats_II, counts_II = np.unique(self.j[: self.sup_v_upper['index'] + 1], return_counts = True)
        for i,j in zip(strats_II, counts_II):
            self.pII_strategy[i] = j / (self.sup_v_upper['index']+1)   
    
    
    def __str__(self):
        ph = self.print_head
        pt = self.print_tail
        
        headers = ('k', 'i_k', 's_k', 'v_lower_k', 'j_k', 't_k', 'v_upper_k')
        
        bounds_str = "\n\nUpper game value: " + str(self.sup_v_upper) + "\n" \
            "Lower game value: " + str(self.inf_v_lower) + "\n\n"
        
        strategies_str = "Player I optimal strategy: \n\t" + str(list(self.pI_strategy)) + \
                         "\nPlayer II optimal strategy: \n\t" + str(list(self.pII_strategy)) +\
                         "\n\n"
        
        # Build up a return value as we go
        rv = bounds_str + strategies_str
        
        if ph + pt >= self.k+1:
            # If the total number of lines to print is less than or equal to the
            # number of rounds, we don't need to print a line of '[...]'.
            
            # Return table with inidces starting at 1
            if self.print_indexing_at_1 == True:
                data = zip(
                    list(range(1, self.k + 1)), 
                    self.i + 1, 
                    self.s, 
                    self.v_lower, 
                    self.j + 1, 
                    self.t, 
                    self.v_upper)
                rv += tabulate(data, headers = headers)
                return rv
                
            # Return table with indices starting at 0
            data = zip(
                list(range(self.k + 1)), 
                self.i, 
                self.s, 
                self.v_lower, 
                self.j, 
                self.t, 
                self.v_upper)
            rv += tabulate(data, headers = headers)
            return rv
        
        # Total number of rounds exceeds ph+pt; print a line break
        line_break = "\n\n[...]\n" + str(self.k - ph - pt) + " lines skipped... \n" + "[...]\n\n"
        
        if self.print_indexing_at_1 == True:
                data_head = list(zip(
                    list(range(1, ph + 1)), 
                    self.i[:ph] + 1, 
                    self.s[:ph], 
                    self.v_lower[:ph], 
                    self.j[:ph] + 1, 
                    self.t[:ph], 
                    self.v_upper[:ph]))
                 
                rv += tabulate(data_head, headers = headers)
                rv += line_break
                
                data_tail = list(zip(
                    list(range(self.k - pt + 1, self.k + 2)), 
                    self.i[self.k - pt:self.k] + 1, 
                    self.s[self.k - pt:self.k], 
                    self.v_lower[self.k - pt:self.k], 
                    self.j[self.k - pt:self.k] + 1, 
                    self.t[self.k - pt:self.k], 
                    self.v_upper[self.k - pt:self.k]))
                
                rv += tabulate(data_tail, headers = headers)
                
                return rv
            
        # Return table with indices starting at 0
        data_head = list(zip(
                    list(range(ph)), 
                    self.i[:ph], 
                    self.s[:ph], 
                    self.v_lower[:ph], 
                    self.j[:ph], 
                    self.t[:ph], 
                    self.v_upper[:ph]))
        
        rv += tabulate(data_head, headers = headers)            
        rv += line_break
        
        data_tail = list(zip(
                    list(range(self.k  - pt, self.k + 2)), 
                    self.i[self.k - pt:self.k], 
                    self.s[self.k - pt:self.k], 
                    self.v_lower[self.k - pt:self.k], 
                    self.j[self.k - pt:self.k], 
                    self.t[self.k - pt:self.k], 
                    self.v_upper[self.k - pt:self.k]))
        
        rv += tabulate(data_tail, headers = headers)
        return rv

class serial_fp():
    def __init__(self, payoff_matrix, initial_i = 0):
        self.payoff_matrix = payoff_matrix
        
        # Round index
        self.k = 0
        
        # Create counts
        self.i_counts = np.zeros(payoff_matrix.shape[0])      
        self.j_counts = np.zeros(payoff_matrix.shape[1])

        ## Initial play
        # Initial i
        self.i_counts[initial_i] += 1
        
        # Recursively generated payoffs for player I
        self.s = self.payoff_matrix[initial_i]
        
        # Initialize strategy values for player II
        j = np.argmin(self.s)
        
        # Recursively generated payoffs for player II
        self.t = self.payoff_matrix[:,j]
        
        # The infimum of the v_lower calculated so far
        self.inf_v_lower = float("-inf")
        # The supremum of the v_upper calculated so far
        self.sup_v_upper = float("inf")
        
        # Optimal strategies at round k
        self.player_I_optimal_strategy_with_divisor = [None, None]
        self.player_II_optimal_strategy_with_divisor = [None, None]
        
    def _calculate_player_I_strategy(self, i_counts, sj, k):                
        if sj/(k + 1) > self.inf_v_lower:
            self.inf_v_lower = sj/(k + 1)
            self.player_I_optimal_strategy_with_divisor = [i_counts, k + 1]

    def _calculate_player_II_strategy(self, j_counts, ti, k):
        if ti/(k) < self.sup_v_upper:
            self.sup_v_upper = ti/(k)
            self.player_II_optimal_strategy_with_divisor = [j_counts, k]
            
    def next_rounds(self, rounds = 1):
        for _ in range(rounds):
            # Increment round
            self.k += 1
            
            # Calculate the best-response and update the count for player I
            i = np.argmax(self.t)
            self.i_counts[i] += 1
            
            # Recursively generate incremental payoffs for player I
            self.s = self.s + self.payoff_matrix[np.argmax(self.t)]
            
            # Calculate the best-response and update the count for player II
            j = np.argmin(self.s)
            self.j_counts[j] += 1
            
            self._calculate_player_I_strategy(self.i_counts.copy(), self.s[j], self.k)
            self._calculate_player_II_strategy(self.j_counts.copy(), self.t[i], self.k)
                        
            self.t = self.t + self.payoff_matrix[:, j]   

import threading
import queue

class threaded_fp():
    def __init__(self, payoff_matrix, initial_i = 0):
        self.payoff_matrix = payoff_matrix
        
        # Round index
        self.k = 0
        
        # Create a FIFO queues and counts
        self.i_counts = np.zeros(payoff_matrix.shape[0])
        self.i_counts[initial_i] += 1
        
        self.j_counts_ti_k_queue = queue.Queue()
        
        self.j_counts = np.zeros(payoff_matrix.shape[1])
        self.i_counts_sj_k_queue = queue.Queue()
        
        
        # Initialize queue values
        self.j_counts_ti_k_queue.put((0, float("inf"), 1))
        
        # Recursively generated payoffs for player I
        self.s = self.payoff_matrix[initial_i]
        
        # Initialize strategy values for player II
        j = np.argmin(self.s)
        self.i_counts_sj_k_queue.put((j, self.s[j], 1))
        
        # Recursively generated payoffs for player II
        self.t = self.payoff_matrix[:,j]
        
        # The infimum of the v_lower calculated so far
        self.inf_v_lower = float("-inf")
        # The supremum of the v_upper calculated so far
        self.sup_v_upper = float("inf")
        
        # Optimal strategies at round k
        self.player_I_optimal_strategy_with_divisor = [None, None]
        self.player_II_optimal_strategy_with_divisor = [None, None]
        
    def _calculate_player_I_strategy(self):
        while True:
            i_counts, sj, k = self.i_counts_sj_k_queue.get()                
            if sj/(k + 1) > self.inf_v_lower:
                self.inf_v_lower = sj/(k + 1)
                self.player_I_optimal_strategy_with_divisor = [i_counts, k + 1]

    def _calculate_player_II_strategy(self):
        while True:
            j_counts, ti, k = self.j_counts_ti_k_queue.get()
            if ti/(k) < self.sup_v_upper:
                self.sup_v_upper = ti/(k)
                self.player_II_optimal_strategy_with_divisor = [j_counts, k]
            
    def next_rounds(self, rounds = 1):
        """At the start of each round, we've calculated:
        i_k, s_k, j_k, t_k"""
        self.pI_strategies = threading.Thread(
                                group = None,
                                target = self._calculate_player_I_strategy,
                                name = 'pI_strategies')
        self.pI_strategies.daemon = True
        self.pI_strategies.start()
            
        self.pII_strategies = threading.Thread(
                                group = None,
                                target = self._calculate_player_II_strategy,
                                name = 'pII_strategies')
        self.pII_strategies.daemon = True
        self.pII_strategies.start()
        
        for _ in range(rounds):
            
            # Increment round
            self.k += 1
            
            i = np.argmax(self.t)
            self.i_counts[i] += 1
            
            
            # Recursively generated payoffs for player I
            self.s = self.s + self.payoff_matrix[np.argmax(self.t)]
            
            j = np.argmin(self.s)
            self.j_counts[j] += 1
            
            self.j_counts_ti_k_queue.put((self.j_counts.copy(), self.t[i], self.k))
            self.i_counts_sj_k_queue.put((self.i_counts.copy(), self.s[j], self.k))
                        
            self.t = self.t + self.payoff_matrix[:, j]      

