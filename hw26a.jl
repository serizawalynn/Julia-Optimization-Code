using LinearAlgebra

using Printf


# matrix and vector used in quadratic form. 
# defined here, because they are used in both F(x), and DF(x)
Q = [5 -1; -1 2];
b = [1;1];

# set parameters here, for all gradient descent algorithms
tol = 1e-10;     # tolerance on norm of gradient
MaxIter = 1000000;  # maximum number of iterations of gradient descent

# define function
function F(x)
   val = (100*((x[2,1]-(x[1,1]^2))^2))+((1-x[1,1])^2)
   return val
end

# define gradient
function DF(x)
   grad = [((400*(x[1,1]^3))-(400*x[1,1]*x[2,1])+(2*x[1,1])-2);((-200*(x[1,1]^2))+(200*x[2,1]))]
   return grad
end

# steepest descent algorithm, with constant step size
# input: 
#    x0 = initial point, a 2-vector (e.g. x0=[1;2])
#    alpha = step size. Constant, in this algorithm.
# output: 
#    x = final point
#
function SteepestDescent(x0,alpha)
   # setup for steepest descent
   x = x0;
   alpha=alpha
   successflag = false;

   # perform steepest descent iterations
   for iter = 1:MaxIter
       Fval = F(x);
       #Lynn: I altered this function to use momentum
       #Fgrad = DF(x)+beta*DF(x);
       Fgrad=DF(x)
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
   return x,alpha;
end

x_null=[2;2]

#Lynn: Note: Some lines are commented out because they may conflict with code set up for other problems

#5a
alphas=[.0001 .001 .002 .003 .004 .005 .006 .007 .008 .009 .01 .02 .03 .04 .05 .1 .5]
for i in alphas
   print("For alpha="*string(i))
   SteepestDescent(x_null,i)
end
