# ORF 535 HW3
# Assignment 3: Root finding and automated testing
# Author: Chase Perlen

This package implements Newton's method for finding roots of single and multi-dimensional functions. 

newton.py: Creates and returns a new solver object to find the roots of f(x) = 0.
  Arguments include:
        tol:     tolerance for iteration (iterate until |f(x)| < tol)
        maxiter: maximum number of iterations to perform
  Optional:
        dx:      step size for computing approximate Jacobian
        Df:      Analytic Jacobian.  If not included, approximate Jacobian will be used
        r:       radius.  Only permits solutions that are within radius r of initial guess x0
                 ie execption thrown if ||x-x0|| > r.  Initial value of infinity prevents this from occuring

  Throws:
        Exceptions are thrown if the Jacobian is noninvertible, the test did not find a root ie reach within tol 
        of zero, or if the initial guess does not capture the solution within a certain radius
        
functions.py: Contains Analytic and Approximate Jacobian methods, as well as a class Polynomial to initialize callable polynomial object.
  Polynomial of the form \sum_{i=0}^n a_ix^i can be obtained via Polynomial([a_n,a_{n-1},...,a_0]) 


testNewton.py: Contains tests for newton.py
    testLinear: tests convergence for linear functions

    testLinearTol:  tests correctly recognizes when solution not within tolerence in linear case

    testNoRoots: tests correctly recognizes when there is no root

    testQuadraticFail: tests correctly recognizes when solution not within tolerence in quadratic case

    testQuadraticConv: tests correctly computes for  quadratic case

    testStep: tests step works appropriately

    testStep2: tests step works appropriately

    testInvertibility: tests correctly handles singular Df_x

    testQuadratic2RootsLow: tests converges to nearest root

    testQuadratic2RootsHigh: tests converges to nearest root

    testMultiDimLinearConv: tests properly handles 2d linear problems 

    testMultiDimConv: tests properly handles more general 2d  problems 

    testMultiDimConv2: tests properly handles more general 2d  problems 

    testMultiDimNoRoots: tests catches when function has no roots in multidimensions

    testAnalytic1D: tests analytic derivative properly implemented

    testAnalyticNoRoot: tests analytic derivative recognizes no roots

    testAnalyticSingular: tests analytic derivative recognizes singular derivative

    testAnalyticStep: tests analytic derivative had proper step size

    testAnalyticMultiDim:tests analytic derivative had proper step size in multiple dimensions

    testRadius1D: tests using a radius does not affect solution when guess is suitably close

    testRadius1DFails: tests recognizes when guess is suitably close

    testRadiusMultiDim: tests using a radius does not affect solution when guess is suitable close in multidim sense

    testRadiusMultiDimFails: ests recognizes when guess is suitably close in multidim sense


testFunctions: Contains tests for functions.py

    testPolynomial: Test the Polynomial class.
      
    testApproxJacobian1: tests approximate Jacobian correctly handles linear functions
      
    testApproxJacobian2: tests approximate Jacobian correctly handles multidimensional linear function

    testApproxJacobian3: tests approximate Jacobian correctly handles polynomial functions
  
    testApproxJacobian4: tests approximate Jacobian correctly handles multidim nonlinear functions
  
    testApproxJacobian5: tests approximate Jacobian correctly handles multidim nonlinear functions (nonseparable variables)
  
    testApproxJacobian6: tests approximate Jacobian correctly handles function composition
  
    testAnalytic1d: tests analytic jacobian correctly computed and differs from approximate 1d
  
    testAnalyticMulti:tests analytic jacobian correctly computed
