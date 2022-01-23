#!/usr/bin/python3

from which_pyqt import PYQT_VER
if PYQT_VER == 'PYQT5':
	from PyQt5.QtCore import QLineF, QPointF
elif PYQT_VER == 'PYQT4':
	from PyQt4.QtCore import QLineF, QPointF
else:
	raise Exception('Unsupported Version of PyQt: {}'.format(PYQT_VER))




import time
import numpy as np
from TSPClasses import *
import heapq
import itertools
import random
import math
import heapq
import operator, pandas as pd, matplotlib.pyplot as plt



class TSPSolver:
	def __init__( self, gui_view ):
		self._scenario = None

	def setupWithScenario( self, scenario ):
		self._scenario = scenario


	''' <summary>
		This is the entry point for the default solver
		which just finds a valid random tour.  Note this could be used to find your
		initial BSSF.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of solution, 
		time spent to find solution, number of permutations tried during search, the 
		solution found, and three null values for fields not used for this 
		algorithm</returns> 
	'''
	
	def defaultRandomTour( self, time_allowance=60.0 ):
		results = {}
		cities = self._scenario.getCities()
		ncities = len(cities)
		foundTour = False
		count = 0
		bssf = None
		start_time = time.time()
		while not foundTour and time.time()-start_time < time_allowance:
			# create a random permutation
			perm = np.random.permutation( ncities )
			route = []
			# Now build the route using the random permutation
			for i in range( ncities ):
				route.append( cities[ perm[i] ] )
			bssf = TSPSolution(route)
			count += 1
			if bssf.cost < np.inf:
				# Found a valid route
				foundTour = True
		end_time = time.time()
		results['cost'] = bssf.cost if foundTour else math.inf
		results['time'] = end_time - start_time
		results['count'] = count
		results['soln'] = bssf
		results['max'] = None
		results['total'] = None
		results['pruned'] = None
		return results


	''' <summary>
		This is the entry point for the greedy solver, which you must implement for 
		the group project (but it is probably a good idea to just do it for the branch-and
		bound project as a way to get your feet wet).  Note this could be used to find your
		initial BSSF.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution, 
		time spent to find best solution, total number of solutions found, the best
		solution found, and three null values for fields not used for this 
		algorithm</returns> 
	'''

	def greedy( self,time_allowance=60.0 ):
		results = {}
		cities = self._scenario.getCities()
		ncities = len(cities)
		foundTour = False
		count = 0
		bssf = None
		start_time = time.time()
		while not foundTour and time.time()-start_time < time_allowance:
			rand = random.randint(0, ncities - 1)
			startCity = cities[rand]
			route = []
			route.append(startCity)
			neighbor_added = True
			#IF YOU DON'T ADD A NEW CITY THAN STOP
			while neighbor_added:
				neighbor_added = False
				min_neighbor_cost = np.inf
				min_neighbor = None
				for i in range( ncities ):
					#CHECK THE COST TO ALL OF THE NEIGHBORS OF THE LAST CITY ADDED TO route THAT ARE NOT ALREADY IN route
					if cities[i] not in route and route[-1].costTo(cities[i]) < min_neighbor_cost:
						min_neighbor_cost = route[-1].costTo(cities[i])
						min_neighbor = cities[i]
				#IF THE LAST CITY ADDED HAS A NEIGHBOR THAN ADD IT TO THE ROUTE
				if min_neighbor != None:
					route.append(min_neighbor)
					neighbor_added = True
			#INCREMENT THE NUMBER OF ATTEMPTED SOLUTIONS
			count += 1
			if len(route) == ncities:
				bssf = TSPSolution(route)
				if bssf.cost < np.inf:
					# Found a valid route
					foundTour = True
		end_time = time.time()
		results['cost'] = bssf.cost if foundTour else math.inf
		results['time'] = end_time - start_time
		results['count'] = count
		results['soln'] = bssf
		results['max'] = None
		results['total'] = None
		results['pruned'] = None
		return results

#-----------------------------START BRANCH AND BOUND------------------------------------#	
	
	''' <summary>
		This is the entry point for the branch-and-bound algorithm that you will implement
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution, 
		time spent to find best solution, total number solutions found during search (does
		not include the initial BSSF), the best solution found, and three more ints: 
		max queue size, total number of states created, and number of pruned states.</returns> 
	'''
	# BranchAndBound(Po):
	# S <— {Po}
	# BSSF = Infinity
	# while S is not empty do
	# 	P <— S.eject()
	# 	if lowerbound(P) < BSSF then
	# 		T <— expand(P)
	# 		for each Pi in T do
	# 			if test(Pi) < BSSF then
	# 				BSSF = test(Pi)
	# 			else if lowerbound(Pi) < BSSF then
	# 				S <— S U {Pi}
	# return BSSF	
	def branchAndBound( self, time_allowance=60.0 ):
		results = {}
		Prune = 0
		cities = self._scenario.getCities()
		ncities = len(cities)
		count = 0
		maxQsize = 0
		totalStates = 0
		start_time = time.time()
		bssf = None
		# ----------INITIALIZE THE ROOT NODE------------------#
		initial_matrix = np.matrix(np.ones((ncities,ncities)) * np.inf)
		for i in range(0,ncities):
			for j in range(0,ncities):
				initial_matrix[i,j] = cities[i].costTo(cities[j])
		root_node = Node(initial_matrix)
		totalStates += 1
		root_node.ReduceMatrix()
		# ----------FINISHED INITIALIZING THE ROOT NODE------------------#
		priorityQ = [root_node]
		maxQsize = len(priorityQ)
		# INITIALIZE BSSF TO A GREEDY SOLUTION IF POSSIBLE 
		greedyBSSF = self.greedy(1)
		if greedyBSSF != np.inf:
			bssf = greedyBSSF['soln']
		# ELSE INITIALIZE BSSF TO A RANDOM TOUR
		else:
			bssf = self.defaultRandomTour(1)['soln']
		# WHILE THE PRIORITYQ IS NOT EMPTY AND TIME ALLOWANCE IS NOT EXPIRED
		while len(priorityQ) > 0 and time.time()-start_time < time_allowance:
			# BEFORE POPPING OFF THE Q, CHEKC IF ITS SIZE IS GREATER THAN MAXQSIZE. IF SO, UPDATE MAXQSIZE
			if len(priorityQ) > maxQsize:
				maxQsize = len(priorityQ)
			P = heapq.heappop(priorityQ)
			# IF THE CURRENT NODE'S (P's) LOWERBOUND IS LESS THAN THE CURRENT BSSF:
			if P.lower_bound < bssf.cost:
				children = P.SpawnChildren()
				totalStates += len(children)
				# FOR EACH OF P's CHILDREN:
				for child in children:
					# IF THE CITIES VISITED IS EQUAL TO THE NUMBER OF CITIES:
					if len(child.partial_path) == ncities:
						list_of_cities = []
						for index in child.partial_path:
							list_of_cities.append(cities[index])
						bssf = TSPSolution(list_of_cities) 
						count += 1
					elif child.lower_bound < bssf.cost:
						heapq.heappush(priorityQ, child)
						#PRUNE
						Prune += 1
			else:
				#PRUNE
				Prune += 1
		end_time = time.time()
		results['cost'] = bssf.cost
		results['time'] = end_time - start_time
		results['count'] = count
		results['soln'] = bssf
		results['max'] = maxQsize
		results['total'] = totalStates
		results['pruned'] = Prune
		return results		

class Node:
	def __init__(self, parent_matrix, lower_bound = 0, partial_path = [0]):
		self.RCM = parent_matrix # numpy matrix
		self.lower_bound = lower_bound
		self.partial_path = partial_path # the list of visited cities / path taken to the current city

#-----------REDUCE MATRIX-------------------#
	def ReduceMatrix(self):
		rowReduced = self.ReduceRows(self.RCM)
		row_colReduced = self.ReduceCols(rowReduced)
		self.RCM = row_colReduced

	def ReduceRows(self, matrix):
		rowMins = self.getRCmin(matrix, 1)
		# matrix.shape[0] returns the number of columns in a row
		for i in range(matrix.shape[0]):
			# matrix.shape[1] returns the number of rows in a column
			for j in range(matrix.shape[1]):
				# if matrix element [i,j] is not infinity and the minimum value in the row is not infinity:
				if matrix[i,j] != np.inf and rowMins[i,0] != np.inf:
					# subtract the minimum value in the row from matrix element [i,j]
					matrix[i,j] = matrix[i,j] - rowMins[i,0]
			# if the minimum value in row i is not infinity:
			if rowMins[i,0] != np.inf:
				# add the minimum value in row i to self.lower_bound
				self.lower_bound += rowMins[i,0]
		return matrix

	def ReduceCols(self, matrix):
		colMins = self.getRCmin(matrix, 0)
		# matrix.shape[1] returns the number of rows in a column
		for i in range(matrix.shape[1]):
			# matrix.shape[0] returns the number of columns in a row
			for j in range(matrix.shape[0]):
				# if matrix element [j,i] is not infinity and the minimum value in the column is not infinity:
				if  matrix[j,i] != np.inf and colMins[0,i] != np.inf:
					# subtract the minimum value in column i from matrix element [j,i]
					matrix[j,i] = matrix[j,i] - colMins[0,i]
			# if the minimum value in column i is not infinity:
			if colMins[0,i] != np.inf:
				# add the minimum value in column i to self.lower_bound
				self.lower_bound += colMins[0,i]
		return matrix

	def getRCmin(self, matrix, row_1_col_0):
		# np.amin() returns the minimum value in a matrix row if row_1_col_0 = 1 or the minimum value in a column if row_1_col_0 = 0
		return np.amin(matrix, axis=row_1_col_0)
#-----------END REDUCE MATRIX---------------#

#-----------SPAWN CHILDREN-------------------#
	def SpawnChildren(self):
		#LOOP OVER THE COLUMNS
		children = []
		for col in range(self.RCM.shape[1]): 
			row = self.partial_path[-1]
			#IF THE COLUMN NUMBER IS NOT IN partial_path AND THE ELEMENT AT ROW[LAST ELEMENT IN partial_path] COLUMN[col] IS NOT INFINITY
			if col not in self.partial_path and self.RCM[row,col] != np.inf:
				child = Node(self.RCM.copy(), self.lower_bound.copy(), self.partial_path.copy())
				child.InfinityRowsCols(row,col)
				child.ReduceMatrix()
				children.append(child)
		return children

	def InfinityRowsCols(self, row, col):
		self.partial_path.append(col)
		#CHANGE ALL ELEMENTS IN COLUMN col TO INFINITY
		self.lower_bound += self.RCM[row,col]
		for i in range(self.RCM.shape[0]):
			self.RCM[i, col] = np.inf
		#CHANGE ALL ELEMENTS IN ROW row TO INFINITY
		for j in range(self.RCM.shape[1]):
			self.RCM[row, j] = np.inf
		#CHANGE THE ELEMENT AT [COL, ROW] TO BE INFINITY
		self.RCM[col,row] = np.inf

#-----------END SPAWN CHILDREN---------------#

#------------------Heuristic methods for priority queue--------------------

	# ------------- METHOD 1--------------------
	# def __lt__(self, other):
	# 	#RCM IS AN nxn MATRIX SO ITS SHAPE IS THE PROBLEM SIZE
	# 	problem_size = self.RCM.shape[0]
	# 	self_remainingDepth = problem_size - len(self.partial_path)
	# 	other_remainingDepth = problem_size - len(other.partial_path)
	# 	#IF THE CELING OF THE (PROBLEM SIZE)/2 < SELF'S CURRENT DEPTH AND OTHER'S CURRENT DEPTH
	# 	if self_remainingDepth < other_remainingDepth and self.lower_bound * 0.9 < other.lower_bound:
	# 		return True
	# 	elif other_remainingDepth < self_remainingDepth and other.lower_bound * 0.9 < self.lower_bound:
	# 		return False
	# 	else:
	# 		sqrt_probSize = math.ceil(math.sqrt(problem_size))
	# 		#IF THE CELING OF THE SQRT OF THE PROBLEM SIZE < SELF'S CURRENT DEPTH AND OTHER'S CURRENT DEPTH
	# 		if sqrt_probSize < self_remainingDepth and sqrt_probSize < other_remainingDepth:
	# 			#FORCE DOWN THE ONE WITH SMALLER lower_bound
	# 			if self.lower_bound < other.lower_bound:
	# 				return True
	# 		#IF THE CEILING OF THE SQRT OF THE PROBLEM SIZE < SELF'S CURRENT DEPTH BUT >= OTHER'S CURRENT DEPTH
	# 		elif sqrt_probSize < self_remainingDepth and sqrt_probSize >= other_remainingDepth:
	# 			return True
	# 		else:
	# 			#FORCE DOWN THE ONE WITH SMALLER lower_bound
	# 			if self.lower_bound < other.lower_bound:
	# 				return True
	# 		return False
# ------------- METHOD 2--------------------
	def __lt__(self, other):
		#RCM IS AN nxn MATRIX SO ITS SHAPE IS THE PROBLEM SIZE
		problem_size = self.RCM.shape[0]
		self_remainingDepth = problem_size - len(self.partial_path)
		other_remainingDepth = problem_size - len(other.partial_path)
		#IF THE CELING OF THE (PROBLEM SIZE)/2 < SELF'S CURRENT DEPTH AND OTHER'S CURRENT DEPTH
		diff = self_remainingDepth - other_remainingDepth
		# BECAUSE self_remainingDepth IS DEEPER/LARGER, self_factor WILL BE SMALLER THAN other_factor
		self_factor = self_remainingDepth/problem_size
		other_factor = other_remainingDepth/problem_size
		# self_remainingDepth IS DEEPER
		if diff > 0:
			if self.lower_bound*self_factor  < other.lower_bound*other_factor:
				return True
			else:
				return True
		# other_remainingDepth IS DEEPER
		elif diff < 0:
			if other.lower_bound*other_factor < self.lower_bound*self_factor:
				return False
			else:
				return True
		else:
			sqrt_probSize = math.ceil(math.sqrt(problem_size))
			#IF THE CELING OF THE SQRT OF THE PROBLEM SIZE < SELF'S CURRENT DEPTH AND OTHER'S CURRENT DEPTH
			if sqrt_probSize < self_remainingDepth and sqrt_probSize < other_remainingDepth:
				#FORCE DOWN THE ONE WITH SMALLER lower_bound
				if self.lower_bound < other.lower_bound:
					return True
			#IF THE CEILING OF THE SQRT OF THE PROBLEM SIZE < SELF'S CURRENT DEPTH BUT >= OTHER'S CURRENT DEPTH
			elif sqrt_probSize < self_remainingDepth and sqrt_probSize >= other_remainingDepth:
				return True
			else:
				#FORCE DOWN THE ONE WITH SMALLER lower_bound
				if self.lower_bound < other.lower_bound:
					return True
			return False

	# def __lt__(self, other):
	# 	#RCM IS AN nxn MATRIX SO ITS SHAPE IS THE PROBLEM SIZE
	# 	self_remainingDepth = self.RCM.shape[0] - len(self.partial_path)
	# 	other_remainingDepth = other.RCM.shape[0] - len(other.partial_path)
	# 	#IF THE CELING OF THE SQRT OF THE PROBLEM SIZE < SELF'S CURRENT DEPTH AND OTHER'S CURRENT DEPTH
	# 	if math.ceil(math.sqrt(self.RCM.shape[0])) < self_remainingDepth and math.ceil(math.sqrt(other.RCM.shape[0])) < other_remainingDepth:
	# 		#FORCE DOWN THE ONE WITH SMALLER lower_bound
	# 		if self.lower_bound < other.lower_bound:
	# 			return True
	# 	#IF THE CEILING OF THE SQRT OF THE PROBLEM SIZE < SELF'S CURRENT DEPTH BUT >= OTHER'S CURRENT DEPTH
	# 	elif math.ceil(math.sqrt(self.RCM.shape[0])) < self_remainingDepth and math.ceil(math.sqrt(other.RCM.shape[0])) >= other_remainingDepth:
	# 		return True
	# 	else:
	# 		#FORCE DOWN THE ONE WITH SMALLER lower_bound
	# 		if self.lower_bound < other.lower_bound:
	# 			return True
	# 	return False

#methods of node class
#reduce matrix
#spawn children - children will be the columns not in the partial path
#take path will update cost matrix based on path taken, and update partial path, call reduce matrix
#less than __LT__ - this will automatically be used as the key for priority Q
#heuristic: prioritize lower bound cost unless ceiling sqrt(problem)

#numpy amin(): https://thispointer.com/numpy-amin-find-minimum-value-in-numpy-array-and-its-index/

	# def __lt__(self, other):
	# 	#RCM IS AN nxn MATRIX SO ITS SHAPE IS THE PROBLEM SIZE
	# 	self_remainingDepth = self.RCM.shape[0] - len(self.partial_path)
	# 	other_remainingDepth = other.RCM.shape[0] - len(other.partial_path)
	# 	#IF THE CELING OF THE SQRT OF THE PROBLEM SIZE < SELF'S CURRENT DEPTH AND OTHER'S CURRENT DEPTH
	# 	if math.ceil(math.sqrt(self.RCM[0])) < self_remainingDepth and math.ceil(math.sqrt(other.RCM[0])) < other_remainingDepth:
	# 		#FORCE DOWN THE ONE WITH SMALLER lower_bound
	# 		if self.lower_bound < other.lower_bound:
	# 			return True
	# 		else:
	# 			return False
	# 	#IF THE CEILING OF THE SQRT OF THE PROBLEM SIZE < SELF'S CURRENT DEPTH BUT >= OTHER'S CURRENT DEPTH
	# 	elif math.ceil(math.sqrt(self.RCM[0])) < self_remainingDepth and math.ceil(math.sqrt(other.RCM[0])) >= other_remainingDepth:
	# 		return True
	# 	#IF THE CEILING OF THE SQRT OF THE PROBLEM SIZE >= SELF'S CURRENT DEPTH BUT < OTHER'S CURRENT DEPTH
	# 	elif math.ceil(math.sqrt(self.RCM[0])) >= self_remainingDepth and math.ceil(math.sqrt(other.RCM[0])) < other_remainingDepth:
	# 		return False
	# 	else:
	# 		#FORCE DOWN THE ONE WITH SMALLER lower_bound
	# 		if self.lower_bound < other.lower_bound:
	# 			return True
	# 		else:
	# 			return False
	
	
#-----------------------------END BRANCH AND BOUND------------------------------------#

# ----------------------------Genetic Algorithm---------------------------------------#
''' <summary>
       This is the entry point for the algorithm you'll write for your group project.
       </summary>
       <returns>results dictionary for GUI that contains three ints: cost of best solution,
       time spent to find best solution, total number of solutions found during search, the
       best solution found.  You may use the other three field however you like.
       algorithm</returns>
   '''


def get_inf_count(self, route, cities): # time: O(n)
   count = 0
   for i in range(len(route)):
      if route[i].costTo(route[(i + 1) % len(route)]) == float('+inf'):
         count += 1
   return count

def insert(self, q, route, cities): # time: O(n)
   tspSol = TSPSolution(route)
   inf_count = self.get_inf_count(route, cities)
   cost = tspSol.cost + random.random() if tspSol.cost < 999999 else 999999 - random.random() + inf_count
   heapq.heappush(q, (cost, route))

def updateGraph(self, q, previous_time, x, y, start_time): # time: O(n)
   tspSol = TSPSolution(q[0][1])
   bssf = tspSol
   if time.time() - previous_time > 0.5:
      x.append(time.time() - start_time)
      y.append(tspSol.cost)
      previous_time = time.time()
   return bssf, q, x, y, previous_time

def insert_to_q(self, q, route, previous_time, x, y, start_time, cities): # time: O(n)
   self.insert(q, route, cities)
   return self.updateGraph(q, previous_time, x, y, start_time)


# # GA(Fitness()):
# #   population = InitializePopulation()
# #   while not done do:
# #       parents = Select(population, Fitness())
# #       children = Crossover(parents)
# #       children = Mutate(children)
# #       population = Survive(population, children, Fitness())
# #   return HighestFitness(population)
def fancy(self, time_allowance=60.0):
   population = []
   results = {}
   cities = self._scenario.getCities()
   ncities = len(cities)
   start_time = time.time()
   bssf = self.defaultRandomTour(2)
   x = []
   y = []
   previous_time = time.time()

   # # ----------- Tuning Variables ----------
   cut_size = math.ceil(.4 * ncities)
   num_children = 9
   init_pop_size = 10
   num_mutations = math.ceil(.2 * ncities)
   make_new_child_on_mutation = False
   percent_to_replace = .2
   # ------------------------------------------

   a = 6
   b = 0.5
   c = 1
   x1 = np.linspace(0, 10, init_pop_size, endpoint=True)
   distribution = (a * np.exp(-b * x1)) + c

   # time: O(n)
   for i in range(init_pop_size): # time: O(1)
      route = self.greedy(2)["soln"].route # time: O(n^3)
      bssf, q, x, y, previous_time = self.insert_to_q(population, route, previous_time, x, y, start_time, cities) # time: O(n)

   # total O(n^2) one n is the num cities and the other is the num generations
   while time.time() - start_time < time_allowance: # time: O(n)
      # Select
      parent1, parent2 = self.getParents(population, distribution) # time: O(1)
      children = []
      # Crossover
      for _ in range(num_children): # time: O(1)
         child = self.Crossover(parent1, parent2, cut_size) # time: O(n)
         self.insert(children, child, cities)
      # Mutate
      mutated_children = []
      for c in children: # time: O(1)
         mutated_child = self.Mutate(c[1]) # time: O(1)
         for i in range(num_mutations): # time: O(1)
            mutated_child = self.Mutate(mutated_child) # time: O(1)
         self.insert(mutated_children, mutated_child, cities) # time: O(n)
      # Survive
      num_surviving = len(population) - len(mutated_children) # time: O(1)
      population = heapq.nsmallest(num_surviving, population) # time: O(n)
      population = list(heapq.merge(population, mutated_children)) # time: O(n)
      bssf, q, x, y, previous_time = self.updateGraph(population, previous_time, x, y, start_time)

   x = np.array(x)
   y = np.array(y)
   plt.clf()
   plt.plot(x, y)  # Plot the chart
   plt.show()  # display

   end_time = time.time()
   results['cost'] = bssf.cost
   results['time'] = end_time - start_time
   results['count'] = None
   results['soln'] = bssf
   results['max'] = None
   results['total'] = None
   results['pruned'] = None
   return results

# time: O(log(n))
def getParents(self, population, y):
   selected_tuples = random.choices(population, weights=y, k=2)
   return selected_tuples[0][1], selected_tuples[1][1]

# time: O(n)
# space: O(n)
def Crossover(self, parent1, parent2, cutSize=3):
   numCities = len(parent1)
   endplace = random.randint(cutSize, numCities)
   start = endplace - cutSize
   end = endplace
   subArray = parent1[start:end] # time: O(n)
   remainingCities = [item for item in parent2 if item not in subArray] # time: O(n)
   child = subArray + remainingCities
   return child

# time: O(1)
def Mutate(self, r):
   p1 = random.randint(0, len(r) - 1)
   p2 = p1
   while p2 == p1:
      p2 = random.randint(0, len(r) - 1)
   r[p1], r[p2] = r[p2], r[p1]
   return r


# -------------------------------------End Genetic Algorithm-----------------------------------------------		
