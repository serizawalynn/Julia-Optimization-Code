# Implementation of Newton's method. 
#
# Also includes Steepest Descent algorithm (constant step size, & Armijo's rule), and 
# Momentum Descent
#
# This version: for HW3
#

using Printf
using LinearAlgebra
using Plots

# set parameters here, for all optimization algorithms
tol = 1e-6;     # tolerance on norm of gradient
MaxIter = 10000;  # maximum number of iterations of gradient descent
MaxIterNewton = 500;  # maximum number of iterations for Newton-type methods. 


# load data
include("data-hw3.jl");
z = data[:,1];
y = data[:,2];
n = length(z);   # number of data points


# sigmoidal function, and its derivatives
function s(x)
   return 1/(1+exp(-x));
end

function sp(x)
    return exp(-x)/(1+exp(-x))^2;
end

function spp(x)
    return 2*exp(-2*x)/(1+exp(-x))^3 - exp(-x)/(1+exp(-x))^2;
end


# model function: y = m(z,x), where x are parameters
function m(z,x)
    a = x[1];
    b = x[2];
    return s(a*z+b);
end

# gradient of model function
function Dm(z,x)
   a = x[1]; b = x[2]; 
   g1 = z*sp(a*z+b);
   g2 = sp(a*z+b);   ### YOU INSERT ### 
   return [g1;g2];
end

# Hessian of model function
function Dm2(z,x)
   a = x[1]; b = x[2]; 
   H = zeros(2,2);
   ### YOU INSERT ###
   H[1,1] = spp(a*z+b)*(z^2);
   H[1,2] = z*spp(a*z+b);
   H[2,1] = z*spp(a*z+b);
   H[2,2] = spp(a*z+b);
   ### END INSERT ###
   return H;
end



# Loss function
function F(x)
   L = 0;
   for i=1:n
      L = L + 0.5*(y[i] - m(z[i],x))^2;
   end
   return L;
end

# gradient of Loss function
function DF(x)
   g = zeros(2);
   for i=1:n
      g = g-(y[i]-m(z[i],x))*Dm(z[i],x);
   end
   return g;
end

# Hessian of Loss function
function DF2(x)
    H = zeros(2,2);
    for i=1:n
      H = H - ( (y[i]-m(z[i],x))*Dm2(z[i],x) - Dm(z[i],x)*Dm(z[i],x)');
   end
   return H;
end




# Newton's algorithm
# input: 
#    x0 = initial point, a 2-vector (e.g. x0=[1;2])
# output: 
#    xsave = list of points
#
function Newton(x0)

   # setup 
   x = x0;
   successflag = false;
   xsave = zeros(length(x0),MaxIter+1);
   xsave[:,1] = x0;

   # iterate
   for iter = 1:MaxIter
      
       # compute gradient
       Fgrad = DF(x);

       #@printf("x = %11.10f, %11.10f, F(x) = %10.8f, |grad F| = %10.8f \n", x[1],x[2],F(x),sqrt(Fgrad'*Fgrad));

       # check whether gradient is small enough
       if sqrt(Fgrad'*Fgrad) < tol
          @printf("\nConverged after %d iterations, F(x) = %f\n", iter, F(x));
          println("x = ", x');
          successflag = true;
          xsave = xsave[:,1:iter];
          break;
       end

       # compute Hessian
       H = DF2(x);

       # check if hessian is positive definite
       ### YOU INSERT ###
      
       # Newton step
       x = x - inv(H)*Fgrad;    # normally you don't actually compute a matrix inverse
       
       # save point
       xsave[:,iter+1] = x;
   end
   if successflag == false
       @printf("Failed to converge after %d iterations, function value %F\n", MaxIter, F(x))
   end
   return 1 #xsave;
end

#6c
x_1s=5*randn(1)
x_2s=5*randn(1)
for i in x_1s
   for j in x_2s
      vec=[i,j]
      print("For vector "*string(vec)*":\n")
      print(Newton(vec))
      print("\n")
   end
end