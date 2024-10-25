#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import time

class ThomasAlgo:
    
    def __init__(self, a, b, c, d, left_boundary_cond, right_boundary_cond, step_size):
        # Ініціалізація коефіцієнтів
        self.coeff_a = a
        self.coeff_b = b
        self.coeff_c = c
        self.coeff_d = d

        # Зберігання результатів та проміжних значень
        self.result_y = np.zeros_like(a)
        self.temp_p = np.zeros_like(a)
        self.temp_q = np.zeros_like(a)

        # Граничні умови та розмір кроку
        self.left_cond = left_boundary_cond
        self.right_cond = right_boundary_cond
        self.step = step_size

    def forwards(self):
        # Прямий прохід для обчислення проміжних коефіцієнтів
        n = len(self.result_y) - 1
        alpha_0, alpha_1, A = self.left_cond
        h = self.step

        # Ініціалізація початкових значень
        self.temp_p[0] = alpha_1 / (alpha_1 - alpha_0 * h)
        self.temp_q[0] = -A * h / (alpha_1 - alpha_0 * h)
        for i in range(1, n):
            # Запобігання діленню на нуль
            small_value = 1e-12
            denominator = self.coeff_b[i] - self.coeff_a[i] * self.temp_p[i - 1] + small_value
            if denominator > 1.05 * small_value:
                self.temp_p[i] = self.coeff_c[i] / denominator
                self.temp_q[i] = (self.coeff_a[i] * self.temp_q[i - 1] - self.coeff_d[i]) / denominator
            else:
                # Обробка випадку малого значення
                self.temp_p[i] = 0
                self.temp_q[i] = self.coeff_d[i]

    def backwards(self):
        # Зворотний прохід для визначення результатів
        n = len(self.result_y) - 1
        alpha_0, alpha_1, A = self.left_cond
        beta_0, beta_1, B = self.right_cond
        h = self.step

        # Обчислення останнього значення
        self.result_y[n] = (h * B + beta_1 * self.temp_q[n - 1]) / (h * beta_0 - beta_1 * self.temp_p[n - 1] + beta_1)
        for i in range(n - 1, 0, -1):
            self.result_y[i] = self.temp_p[i] * self.result_y[i + 1] + self.temp_q[i]
        # Обчислення першого значення
        self.result_y[0] = (A * h - alpha_1 * self.result_y[1]) / (alpha_0 * h - alpha_1)
        
    
    def solve(self):
        self.forwards()
        self.backwards()
        return self.result_y
        
        
        
         ##   Вирішуємо спочатку рівняння звичайним алгоритмом прогонки ThomasAlgo, для цього достатньо визначити такі змінні:
    
   ## left_boundary_cond/right_boundary_cond - ліві/праві граничні умови
  ##  left_boundary_cond = (alpha_0, alpha1_, A)
  ##  right_boundary_cond = (beta_0, beta_1, B)


class ADIMethod:

    def __init__(self, x_config, y_config, C, G, internal_conditions=[]):
        # Ініціалізація розмірів сітки та кроків
        self.x_min, self.x_max = x_config['range']
        self.y_min, self.y_max = y_config['range']
        self.hx = (self.x_max - self.x_min) / (x_config['number_ofgrid_points'] - 1)
        self.hy = (self.y_max - self.y_min) / (y_config['number_ofgrid_points'] - 1)
        self.grid_size = (y_config['number_ofgrid_points'], x_config['number_ofgrid_points'])
        self.error_history = []

        self.x_coords = self.x_min + self.hx * np.arange(self.grid_size[1])
        self.y_coords = self.y_min + self.hy * np.arange(self.grid_size[0])
        self.X, self.Y = np.meshgrid(self.x_coords, self.y_coords)

        self.A_func = x_config['func']
        self.B_func = y_config['func']
        self.C_func = C
        self.G_func = G
        self.internal_conditions = internal_conditions
        self._initialize_coefficients()
        self._set_boundary_conditions(x_config['boundary_conditions'], y_config['boundary_conditions'])
        self._apply_internal_conditions()

    def solve(self, max_iterations, accuracy_threshold=1e-7):
        # Ініціалізація початкового розв'язку
        U_current = np.zeros(self.grid_size)
        U_next = U_current.copy()
        last_check_time = time.time()

        for iteration in range(max_iterations):
            # Обчислення по стовпцях
            for j in range(1, self.grid_size[0] - 1):
                a = self.AY[:, j]
                c = self.CY[:, j]
                b = self.B[:, j]
                d = self.D[:, j] - self.AX[:, j] * U_next[:, j - 1] - self.CX[:, j] * U_next[:, j + 1]
                left_boundary = self.top_bound[j]
                right_boundary = self.bottom_bound[j]

                column_solver = ThomasAlgo(a, b, c, d, left_boundary, right_boundary, self.hy)
                U_next[:, j] = column_solver.solve()

            # Обчислення по рядках
            for i in range(1, self.grid_size[1] - 1):
                a = self.AX[i, :]
                c = self.CX[i, :]
                b = self.B[i, :]
                d = self.D[i, :] - self.AY[i, :] * U_next[i - 1, :] - self.CY[i, :] * U_next[i + 1, :]
                left_boundary = self.left_bound[i]
                right_boundary = self.right_bound[i]

                row_solver = ThomasAlgo(a, b, c, d, left_boundary, right_boundary, self.hx)
                U_next[i, :] = row_solver.solve()
            
            delta_U = np.mean(np.abs(U_current - U_next) / (np.abs(U_next) + 1e-12))
            if delta_U < accuracy_threshold:
                return U_current
            if time.time() - last_check_time > 0.5:
                self.error_history.append(delta_U)
                last_check_time = time.time()
            U_current = U_next.copy()

        return U_current
    
    def _initialize_coefficients(self):
        # Ініціалізація коефіцієнтів сітки
        self.AX = self.A_func(self.X, self.Y) / self.hx**2
        self.CX = self.A_func(self.X, self.Y) / self.hx**2
        self.AY = self.B_func(self.X, self.Y) / self.hy**2
        self.CY = self.B_func(self.X, self.Y) / self.hy**2
        self.B = 2 * self.AX + 2 * self.AY - self.C_func(self.X, self.Y)
        self.D = self.G_func(self.X, self.Y)

    def _set_boundary_conditions(self, x_bound, y_bound):
        # Ініціалізація граничних умов
        self.left_bound = [(x_bound[0][0], x_bound[0][1], x_bound[0][2](y_val)) for y_val in self.y_coords]
        self.right_bound = [(x_bound[1][0], x_bound[1][1], x_bound[1][2](y_val)) for y_val in self.y_coords]
        self.top_bound = [(y_bound[0][0], y_bound[0][1], y_bound[0][2](x_val)) for x_val in self.x_coords]
        self.bottom_bound = [(y_bound[1][0], y_bound[1][1], y_bound[1][2](x_val)) for x_val in self.x_coords]

    def _apply_internal_conditions(self):
        # Застосування внутрішніх умов
        for condition in self.internal_conditions:
            (x1, x2), (y1, y2) = condition['area']
            mask = ((self.X >= x1) & (self.X <= x2) & (self.Y >= y1) & (self.Y <= y2))

            self.AX[mask] = 0
            self.CX[mask] = 0
            self.AY[mask] = 0
            self.CY[mask] = 0

            self.B[mask] = -1
            self.D[mask] = condition['cond_value']


# In[ ]:




