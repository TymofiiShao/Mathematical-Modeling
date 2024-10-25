#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np

class BlockThomasSolver:
 
    # Initialization of grid boundaries, steps, and coefficient functions. 
    
    def __init__(self, x_config, y_config, C, G, internal_conditions=[]):
        self.x_min, self.x_max = x_config['range']
        self.y_min, self.y_max = y_config['range']
        self.hx = (self.x_max - self.x_min) / (x_config['number_ofgrid_points'] - 1)
        self.hy = (self.y_max - self.y_min) / (y_config['number_ofgrid_points'] - 1)
        self.lattice_size = (y_config['number_ofgrid_points'], x_config['number_ofgrid_points'])

        rows, cols = self.lattice_size
        self.x = self.x_min + self.hx * np.arange(cols)
        self.y = self.y_min + self.hy * np.arange(rows)
        self.X, self.Y = np.meshgrid(self.x, self.y)

        self.A_func = x_config['func']
        self.B_func = y_config['func']
        self.C_func = C
        self.G_func = G
        self.internal_conditions = internal_conditions

        self._initialize_coefficients()
        self._apply_boundary_conditions(x_config['boundary_conditions'], y_config['boundary_conditions'])
        self._apply_internal_conditions()
        self._initialize_block_coefficients()
        
    
    # Performing the forward sweep of the  algorithm

    def forwards(self):
        n, m = self.lattice_size
        forward_matrices = np.zeros((n, m, m))
        solution_vectors = np.zeros((n, m, 1))

        forward_matrices[0] = -np.linalg.inv(self.Bb[0]) @ self.Cb[0]
        solution_vectors[0] = np.linalg.inv(self.Bb[0]) @ self.Db[0]
        for i in range(1, n):
            forward_matrices[i] = -np.linalg.inv(self.Bb[i] + self.Ab[i] @ forward_matrices[i - 1]) @ self.Cb[i]
            solution_vectors[i] = (np.linalg.inv(self.Bb[i] + self.Ab[i] @ forward_matrices[i - 1]) @
                                   (self.Db[i] - self.Ab[i] @ solution_vectors[i - 1]))
        
        return forward_matrices, solution_vectors  
    
    
   #  Performing the backward sweep of the  algorithm.

    def backwards(self, forward_matrices, solution_vectors):
        # Зворотний хід
        n, m = self.lattice_size
        result = np.zeros((n, m, 1))

        # Initialize the final step
        result[n - 1] = solution_vectors[n - 1].copy()
        for i in range(n - 2, -1, -1):
            result[i] = forward_matrices[i] @ result[i + 1] + solution_vectors[i]
        
        return result
    
    # Combining the forward and backward sweeps to obtain the complete solution
    
    
    def solution(self):
        forward_matrices, solution_vectors = self.forwards()
        result = self.backwards(forward_matrices, solution_vectors)
        return result.squeeze()

  
   # Initializing the coefficients for the grid based on the input functions A(x, y), B(x, y), C(x, y), and G(x, y)

    def _initialize_coefficients(self):
        self.AX = self.A_func(self.X, self.Y) / self.hx**2
        self.CX = self.A_func(self.X, self.Y) / self.hx**2
        self.AY = self.B_func(self.X, self.Y) / self.hy**2
        self.CY = self.B_func(self.X, self.Y) / self.hy**2
        self.B = self.C_func(self.X, self.Y) - 2 * self.AX - 2 * self.AY
        self.D = self.G_func(self.X, self.Y)
        
        
    # Applies the specified boundary conditions to the grid

    def _apply_boundary_conditions(self, x_boundary_conditions, y_boundary_conditions):
        self.left_cond = np.array([(x_boundary_conditions[0][0], x_boundary_conditions[0][1], x_boundary_conditions[0][2](y)) for y in self.y]).reshape(-1, 3)
        self.right_cond = np.array([(x_boundary_conditions[1][0], x_boundary_conditions[1][1], x_boundary_conditions[1][2](y)) for y in self.y]).reshape(-1, 3)
        self.top_cond = np.array([(y_boundary_conditions[0][0], y_boundary_conditions[0][1], y_boundary_conditions[0][2](x)) for x in self.x]).reshape(-1, 3)
        self.bottom_cond = np.array([(y_boundary_conditions[1][0], y_boundary_conditions[1][1], y_boundary_conditions[1][2](x)) for x in self.x]).reshape(-1, 3)

        n, m = self.lattice_size 

       
    # left boundary
    
        a_left_0, a_left_1, f_left = self.left_cond[:, 0], self.left_cond[:, 1], self.left_cond[:, 2]
        self.AX[:, 0] = np.zeros(n)
        self.B[:, 0] = a_left_0 - a_left_1 / self.hx
        self.CX[:, 0] = a_left_1 / self.hx
        self.D[:, 0] = f_left

        self.AY[:, 0] = np.zeros(n)
        self.CY[:, 0] = np.zeros(n)
        
    # Right boundary
        
        b_right_0, b_right_1, f_right = self.right_cond[:, 0], self.right_cond[:, 1], self.right_cond[:, 2]
        self.AX[:, m - 1] = -b_right_1 / self.hx
        self.B[:, m - 1] = b_right_0 + b_right_1 / self.hx
        self.CX[:, m - 1] = np.zeros(n)
        self.D[:, m - 1] = f_right

        self.AY[:, m - 1] = np.zeros(n)
        self.CY[:, m - 1] = np.zeros(n)
        
    # Calculation of boundary conditions for the upper bound    

        a_top_0, a_top_1, f_top = self.top_cond[:, 0], self.top_cond[:, 1], self.top_cond[:, 2]
        self.AY[0, :] = np.zeros(m)
        self.B[0, :] = a_top_0 - a_top_1 / self.hy
        self.CY[0, :] = a_top_1 / self.hy
        self.D[0, :] = f_top

        self.AX[0, :] = np.zeros(n)
        self.CX[0, :] = np.zeros(n)

    # Calculation of boundary conditions for the lower bound  
    
        b_bottom_0, b_bottom_1, f_bottom = self.bottom_cond[:, 0], self.bottom_cond[:, 1], self.bottom_cond[:, 2]
        self.AY[n - 1, :] = -b_bottom_1 / self.hy
        self.B[n - 1, :] = b_bottom_0 + b_bottom_1 / self.hy
        self.CY[n - 1, :] = np.zeros(m)
        self.D[n - 1, :] = f_bottom

        self.AX[n - 1, :] = np.zeros(n)
        self.CX[n - 1, :] = np.zeros(n)
        
    # Applies internal conditions to specific regions of the grid

    def _apply_internal_conditions(self):
        for condition in self.internal_conditions:
            (x1, x2), (y1, y2) = condition['area']
            mask = ((self.X >= x1) & (self.X <= x2) & (self.Y >= y1) & (self.Y <= y2))

            self.AX[mask] = 0
            self.CX[mask] = 0
            self.AY[mask] = 0
            self.CY[mask] = 0

            self.B[mask] = 1
            self.D[mask] = condition['cond_value']
            
            
    # Initializes the block coefficients for the Block Thomas algorithm

    def _initialize_block_coefficients(self):
        # Ініціалізація блочних коефіцієнтів
        n, m = self.lattice_size
        self.Ab = np.zeros((n, m, m))
        self.Bb = np.zeros((n, m, m))
        self.Cb = np.zeros((n, m, m))
        self.Db = np.zeros((n, m, 1))

        for i in range(n):
            self.Ab[i] = np.diag(self.AY[i, :], k=0)
            self.Cb[i] = np.diag(self.CY[i, :], k=0)
            self.Bb[i] = (np.diag(self.AX[i, 1:], k=-1) + 
                          np.diag(self.B[i, :], k=0) + 
                          np.diag(self.CX[i, :-1], k=1))
            self.Db[i] = self.D[i].reshape(-1, 1)


# In[ ]:




