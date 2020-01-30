
import util
import mdp
import os
import numpy as np




if __name__ == "__main__":
	prob = mdp.StudentMRP()
	solver = util.BellmanMatrix()
	V = solver.solve(prob)
	print(V)