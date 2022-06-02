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
# steepest descent algorithm, with Armijo's rule for backtracking
# input: 
#    x0 = initial point, a 2-vector (e.g. x0=[1;2])
#    c1 = slope, in Armijo's rule.
# output: 
#    x = final point
#
function SteepestDescentArmijo(x0, c1)

   # parameters for Armijo's rule
   alpha0 = 5.0;    # initial value of alpha, to try in backtracking
   eta = 0.5;       # factor with which to scale alpha, each time you backtrack
   MaxBacktrack = 20;  # maximum number of backtracking steps

   # setup for steepest descent
   x = x0;
   successflag = false;   

   # perform steepest descent iterations
   for iter = 1:MaxIter

      alpha = alpha0;
      Fval = F(x);
      Fgrad = DF(x);

      # check if norm of gradient is small enough
      if sqrt(Fgrad'*Fgrad) < tol
         @printf("Converged after %d iterations, function value %f\n", iter, Fval)
         successflag = true;
         break;
      end

      # perform line search
      for k = 1:MaxBacktrack
         x_try = x - alpha*Fgrad;
         Fval_try = F(x_try);
         if (Fval_try > Fval - c1*alpha *Fgrad'Fgrad)
            alpha = alpha * eta;
         else
            Fval = Fval_try;
            x = x_try;
            break;
         end
      end

      # print how we're doing, every 10 iterations
      #Lynn: I am going to mute this
      #if (iter%10==0)
      #   @printf("iter: %d: alpha: %f, %f, %f, %f\n", iter, alpha, x[1], x[2], Fval)
      #end
   end

   if successflag == false
       @printf("Failed to converge after %d iterations, function value %f\n", MaxIter, F(x))
   end

   return x;
end

x_null=[2;2]

slopes=[(10^-3) (10^-2) (10^-1) (0.2)]
for i in slopes
   print("For slope "*string(i))
   print(SteepestDescentArmijo(x_null,i))
end