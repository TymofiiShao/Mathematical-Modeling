# Lab1
Solving numerical analysis problems for elliptic equations using ADI, block tridiagonal solver, and relaxation method

# Tasks

# Task 1: 
![image](https://github.com/user-attachments/assets/cfbf948e-1cd3-4883-a45b-bf0022bba1e5)

# Task 2:
![image](https://github.com/user-attachments/assets/c331d6f2-c465-4705-a4b2-b9bb86909f4c)

# Parameters, used in this lab

1) x_config and y_config : dict
- 'func' - functions A(x, y) or B(x, y)
- 'range' - (minimum, maximum) value for axes
- 'boundary_conditions' - boundary conditions for the left/upper and right/lower edges
- 'number_ofgrid_points' - number of points on the grid
2) C : callable Function C(x, y)
3) G : callable Function G(x, y) ## 4) internal_conditions : dict
- 'area' - boundaries for the area of ​​internal conditions
- 'cond_value' - value of U in this area

## Additionaly for ThomasAlgo
- left_boundary_cond/right_boundary_cond - left/right boundary conditions
- left_boundary_cond = (alpha_0, alpha1_, A)
- right_boundary_cond = (beta_0, beta_1, B)
