# Julia-Optimization-Code
This is code that I wrote for assignments in my Linear and Nonlinear Optimization course at NYU. Most of the code was written by the professor, and students were tasked with modifying the code for homework assignments. 

This document will summarize the task that the code was written for. Furthermore, I will also state which areas are my work, and which areas are the professor's work.

hw25a.jl
My work starts at line 58. The task was to try various values for alpha, which is the learning rate of the steepest descent algorithm.

hw25b.jl
I wrote the entirety of the script. I calculated the eigenvalues of the matrix to explain why the algorithm diverges for certain alphas. 

hw25c.jl
I altered the script at lines 55,56,57,69, and 70. Here, I used the optimal step length that can be directly calculated from quadratic functions and evaluate its performance. 

hw25d.jl
I modified the code from line 107 onward. Here, I implemented Armijo's rule and examined how different slopes altered the performance of the steepest descent algorithm. 

hw25e.jl
I modified the code at line 68. Here, I implemented the momentum method and evaluated the algorithm's performance. 

hw26a.jl
I modified the code around line 45 and past line 63. Here, I modified the steepest descent algorithm to use momentum, and tried different learning rates to evaluate its performance.

hw26b.jl
Here, I modified the code so that the step length was determined using Armijo's rule. I evaluate the performance of the algorithm at different slopes.

hw26c.jl
Here, I implemented the momentum method and found parameters alpha and beta that lead to convergence/divergence.

6b.jl
I altered the script at lines 51,59-63,199-210. Here, the task was to find conditions and parameters for both the steepest descent with constant step size and steepest descent without Armijo's rule. I evaluated the algorithm's performance by noting the number of iterations required for convergence. 

6c.jl
I altered the script at lines 51, 59-63, and 149-159. The task was using Newton's method for minization. I found initial conditions where the algorithm converges to a point found previously, to a point not found previously, and where it diverges. I noted the number of iterations required for convergence.

6e.jl
Here, I took 6c.jl and altered the code at lines 145-154. The task was to show that the matrix is not always positive definite to explain why the method does not always result in convergence.

6f.jl
Here, I took 6e.jl and altered the code at lines 132 to 137. Here, I used a hessian modification so that the matrix remains positive definite. I used the same script to evaluate the performance of Newton's method with backtracking implemented.
