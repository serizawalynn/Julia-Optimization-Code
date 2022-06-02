# Implementation of steepest descent algorithm, with 
# -- constant step size (SteepestDescent), and
# -- step size chosen via Armijo's rule (SteepestDescentArmijo)
#
# The built-in minimization problem is
#           min   0.5*x'*Q*x - b'*x + 1
#
# Run this as (for example):
# >> include("steep-hw2.jl");
# >> x0 = [10;10];
# >> alpha = 0.1;
# >> x = SteepestDescent(x0,alpha);
# >> c1 = 1e-3;
# >> y = SteepestDescentArmijo(x0, c1);
#

using LinearAlgebra

using Printf

# matrix and vector used in quadratic form. 
# defined here, because they are used in both F(x), and DF(x)
Q = [5 -1; -1 2];
b = [1;1];

# set parameters here, for all gradient descent algorithms
tol = 1e-10;     # tolerance on norm of gradient
MaxIter = 10000;  # maximum number of iterations of gradient descent

# define function
function F(x)
   val = 0.5*x'*Q*x - b'*x + 1

   return val
end

# define gradient
function DF(x)
     grad = Q*x-b;
   return grad
end

#
# steepest descent algorithm, with constant step size
# input: 
#    x0 = initial point, a 2-vector (e.g. x0=[1;2])
#    alpha = step size. Constant, in this algorithm.
# output: 
#    x = final point
#
function SteepestDescent(x0)
   big_eig=eigmax(Q)
   small_eig=eigmin(Q)
   root_big_eig=sqrt(big_eig)
   root_small_eig=sqrt(small_eig)
   beta=((root_big_eig-root_small_eig)/(root_small_eig+root_big_eig))^2
   alpha=(2/(root_big_eig+root_small_eig))^2

   # setup for steepest descent
   x = x0;
   alpha=alpha
   successflag = false;

   # perform steepest descent iterations
   for iter = 1:MaxIter
       Fval = F(x);
       #Lynn: I altered this function to use momentum
       Fgrad = DF(x)+beta*DF(x);
       if sqrt(Fgrad'*Fgrad) < tol
          @printf("Converged after %d iterations, function value %f\n", iter, Fval)
          successflag = true;
          break;
       end
       # perform steepest descent step
       x = x - alpha*Fgrad;

   end
   if successflag == false
       @printf("Failed to converge after %d iterations, function value %f\n", MaxIter, F(x))
   end
   return x,alpha,beta;
end

print(SteepestDescent(x_null))