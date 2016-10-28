#!/usr/bin/env python

import functions as F
import numpy as N
import unittest

class TestFunctions(unittest.TestCase):

    def testApproxJacobian1(self):
        slope = 3.0
        def f(x):
            return slope * x + 5.0
        x0 = 2.0
        dx = 1.e-3
        Df_x = F.ApproximateJacobian(f, x0, dx)
        self.assertEqual(Df_x.shape, (1,1))
        self.assertAlmostEqual(Df_x, slope)  
        

    def testApproxJacobian2(self):
        A = N.matrix("1. 2.; 3. 4.")
        def f(x):
            return A * x
        x0 = N.matrix("5; 6")
        dx = 1.e-6
        Df_x = F.ApproximateJacobian(f, x0, dx)
        self.assertEqual(Df_x.shape, (2,2))
        N.testing.assert_array_almost_equal(Df_x, A)

    def testPolynomial(self):
        # p(x) = x^2 + 2x + 3
        p = F.Polynomial([1, 2, 3])
        for x in N.linspace(-2,2,11):
            self.assertEqual(p(x), x**2 + 2*x + 3)
    
    '''
    tests Jacobian correct for polynomial (ie nonlinear) 1-dim function'''
    def testApproxJacobian3(self):
        p = F.Polynomial([1, 2, 3])
        dp_dx = F.Polynomial([2,2]) 
        dx = 1.e-6
        for x in N.linspace(-2,2,11):
            Df_x = F.ApproximateJacobian(p, x, dx)
            self.assertEqual(Df_x.shape, (1,1))
            N.testing.assert_array_almost_equal(Df_x, dp_dx(x))

    ''' 
    tests Jacobian correctly computes deriv of multidim nonlinear functions
    where each function depends on only one variable'''
    def testApproxJacobian4(self):
        def f(x):
            p = F.Polynomial([2,-3, 0, 8])
            return N.matrix([[N.cos(x.item(0))],[p(x.item(1))]])
            
        def df_dx(x):
            dp_dx = F.Polynomial([6,-6,0])
            return N.matrix([[-N.sin(x.item(0)),0],[0,dp_dx(x.item(1))]]) 
            
        dx = 1.e-6
        for x in [[1.,1.],[0.,3.],[3.,0.],[2.,-3.],[0.,0.]]: 
            x = N.transpose(N.matrix(x))
            Df_x = F.ApproximateJacobian(f, x, dx)
            self.assertEqual(Df_x.shape, (2,2))
            N.testing.assert_array_almost_equal(Df_x, df_dx(x), decimal=4)
    
    ''' 
    tests Jacobian correctly computes deriv of multidim function where fct 
    depend on multiple variables'''
    
    def testApproxJacobian5(self):
        def f(x):
            p = F.Polynomial([2,-3, 0, 8])
            return N.matrix([[x.item(1)*N.cos(x.item(0))],
                              [x.item(0)*p(x.item(1))]])
            
        def df_dx(x):
            p = F.Polynomial([2,-3, 0, 8])
            dp_dx = F.Polynomial([6,-6,0])
            return N.matrix([[-x.item(1)*N.sin(x.item(0)),N.cos(x.item(0))],
                              [p(x.item(1)),x.item(0)*dp_dx(x.item(1))]]) 
            
        dx = 1.e-6
        for x in [[1.,1.],[0.,3.],[3.,0.],[2.,-3.],[0.,0.]]: 
            x = N.transpose(N.matrix(x))           
            Df_x = F.ApproximateJacobian(f, x, dx)
            self.assertEqual(Df_x.shape, (2,2))
            N.testing.assert_array_almost_equal(Df_x, df_dx(x), decimal=4)

    ''' 
    tests Jacobian correctly handles function composition'''
    def testApproxJacobian6(self):
        def f(x):
            return N.sin(x*x) + x*N.cos(x)
        def df_dx(x):
            return 2*x * N.cos(x*x) + N.cos(x) - x * N.sin(x)
        dx = 1.e-6
        for x in N.linspace(-2,2,11): 
            Df_x = F.ApproximateJacobian(f, x, dx)
            self.assertEqual(Df_x.shape, (1,1))
            self.assertAlmostEqual(Df_x, df_dx(x),places = 4)

    
if __name__ == '__main__':
    unittest.main()



