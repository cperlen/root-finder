#!/usr/bin/env python

import newton
import unittest
import numpy as N

class TestNewton(unittest.TestCase):
    def testLinear(self):
        f = lambda x : 3.0 * x + 6.0
        solver = newton.Newton(f, tol = 1.e-15, maxiter=2)
        #x = solver.solve(2.0)
        #self.assertEqual(x, -2.0) fails for prescribed tol
        with self.assertRaises(Exception):
            solver.solve(2.0)
            
    def testNoRoots(self):
        f = F.Polynomial([1,0,1]) #x^2 + 1 has no roots
        solver = newton.Newton(f, tol=1.e-15, maxiter=1000)
        with self.assertRaises(Exception):
            solver.solve(0.0)
    
    def testQuadraticFail(self):
        f = F.Polynomial([1,2,1])
        solver = newton.Newton(f, tol = 1.e-15, maxiter = 10)
        with self.assertRaises(Exception):
            solver.solve(1.0)
    
    def testQuadraticConv(self): 
        f = F.Polynomial([1,2,1])
        solver = newton.Newton(f, tol = 1.e-15, maxiter = 200)
        x = solver.solve(1.0)        
        self.assertAlmostEqual(x, -1.0)
    
    def testStep(self):
        f = lambda x : x*x
        x0 = 3.0
        solver = newton.Newton(f, tol=1.e-15, maxiter=1)
        x=solver.step(x0)
        self.assertAlmostEqual(x-x0,- 9/6.0, places = 4)

    def testStep2(self):
        f = lambda x : N.sin(x)
        x0 = 0.0
        solver = newton.Newton(f, tol=1.e-15, maxiter=1)
        x = solver.step(x0)
        self.assertAlmostEqual(x - x0,0.0)

    def testInvertibility(self):
        f = lambda x : N.cos(x)
        solver = newton.Newton(f, tol=1.e-15, maxiter=100)
        with self.assertRaises(Exception):
            solver.solve(0.0)

    def testQuadratic2RootsLow(self):
        f = F.Polynomial([1,-8,12]) #roots are 6 and 2
        solver = newton.Newton(f, tol=1.e-14, maxiter=100)
        x = solver.solve(3.9)
        self.assertAlmostEqual(x, 2.0)
    
    def testQuadratic2RootsHigh(self):
        f = F.Polynomial([1,-8,12]) #roots are 6 and 2
        solver = newton.Newton(f, tol=1.e-14, maxiter=100)
        x = solver.solve(4.1)
        self.assertAlmostEqual(x, 6.0)
    
    def testMultiDimLinearConv(self): 
        A = N.matrix("1. 2.; 3. 4.")
        def f(x):
            return A * x
        solver = newton.Newton(f, tol=1.e-14, maxiter=100)
        x0 = N.matrix("5; 6")
        x = solver.solve(x0)
        N.testing.assert_array_almost_equal(x, N.matrix([[0.],[0.]]))
    
    def testMultiDimConv(self):
        def f(x):
            p = F.Polynomial([1, -8, 12])
            return N.matrix([[N.sin(x.item(0))],[p(x.item(1))]])
        solver = newton.Newton(f, tol=1.e-14, maxiter=1000)
        x0 = N.matrix("1; 5")
        x = solver.solve(x0)
        N.testing.assert_array_almost_equal(x, N.matrix([[0.],[6.]]))

    def testMultiDimNoRoots(self):
        def f(x):
            p = F.Polynomial([1, 0, 1])
            return N.matrix([[N.sin(x.item(0))],[p(x.item(1))]])
        solver = newton.Newton(f, tol=1.e-15, maxiter=1000)
        with self.assertRaises(Exception):
            solver.solve(2.0)   
            
    def testMultiDimConv2(self): 
        def f(x):
            p = F.Polynomial([1, 0, 12])
            return N.matrix([[(x.item(1)+1)*N.cos(x.item(0))],
                              [(x.item(0)-3)*p(x.item(1))]])
            
        solver = newton.Newton(f, tol=1.e-14, maxiter=1000)
        x0 = N.matrix("1; 1")
        x = solver.solve(x0)
        N.testing.assert_array_almost_equal(x, N.matrix([[3.],[-1.]]))
    
    '''tests including analytic solver'''
    
    def testAnalytic1D(self):
        f = F.Polynomial([1,2,1])
        df = F.Polynomial([2,2])
        solver = newton.Newton(f, tol = 1.e-15, maxiter = 200, Df = df)
        x = solver.solve(1.0)        
        self.assertAlmostEqual(x, -1.0)   
    
    def testAnalyticNoRoot(self):
        f = F.Polynomial([1,0,1])
        df = F.Polynomial([2,0])
        solver = newton.Newton(f, tol = 1.e-15, maxiter = 200, Df = df)
        with self.assertRaises(Exception):
            solver.solve(2.0)  
    
    def testAnalyticSingular(self):
        f = lambda x : F.cos(x)
        df = lambda x : F.sin(x)
        solver = newton.Newton(f, tol = 1.e-15, maxiter = 200, Df = df)
        with self.assertRaises(Exception):
            solver.solve(0.0)  
        
    def testAnalyticStep(self):
        f = lambda x : x*x
        df = F.Polynomial([2,0])
        x0 = 3.0
        solver = newton.Newton(f, tol=1.e-15, maxiter=1, Df = df)
        approx_solver = newton.Newton(f, tol=1.e-15, maxiter=1)
        x = solver.step(x0)
        xa = approx_solver.step(x0) 
        self.assertNotEqual(x,xa)   
        self.assertEqual(x, 1.5)

    def testAnalyticMultiDim(self):
        def f(x):
            p = F.Polynomial([1, -8, 12]) 
            return N.matrix([[N.sin(x.item(0))],[p(x.item(1))]])
        
        def df(x):
            dp_dx = F.Polynomial([2,-8])
            return N.matrix([[N.cos(x.item(0)),0],
                              [0,dp_dx(x.item(1))]])     
        solver = newton.Newton(f, tol=1.e-14, maxiter=1000, Df = df)
        x0 = N.matrix("1; 5")
        x = solver.solve(x0)
        N.testing.assert_array_almost_equal(x, N.matrix([[0.],[6.]]))

    ''' test radius feature'''
    def testRadius1D(self):
        f = F.Polynomial([1,2,1])
        df = F.Polynomial([2,2])
        solver = newton.Newton(f, tol = 1.e-15, maxiter = 200, Df = df, r = 4.0)
        x = solver.solve(1.0)        
        self.assertAlmostEqual(x, -1.0)   
    
    def testRadius1DFails(self): 
        f = F.Polynomial([1,2,1])
        df = F.Polynomial([2,2])
        solver = newton.Newton(f, tol = 1.e-15, maxiter = 200, Df = df, r = 1.0)
        with self.assertRaises(Exception):
            solver.solve(2.0)  

    def testRadiusMultiDim(self):
        def f(x):
            p = F.Polynomial([1, -8, 12]) 
            return N.matrix([[N.sin(x.item(0))],[p(x.item(1))]])
        
        def df(x):
            dp_dx = F.Polynomial([2,-8])
            return N.matrix([[N.cos(x.item(0)),0],
                              [0,dp_dx(x.item(1))]])     
        solver = newton.Newton(f, tol=1.e-14, maxiter=1000, Df = df, r = 10)
        x0 = N.matrix("1; 5")
        x = solver.solve(x0)
        N.testing.assert_array_almost_equal(x, N.matrix([[0.],[6.]]))

    
    def testRadiusMultiDimFails(self):
        def f(x):
            p = F.Polynomial([1, -8, 12]) 
            return N.matrix([[N.sin(x.item(0))],[p(x.item(1))]])
        
        def df(x):
            dp_dx = F.Polynomial([2,-8])
            return N.matrix([[N.cos(x.item(0)),0],
                              [0,dp_dx(x.item(1))]])     
        solver = newton.Newton(f, tol=1.e-14, maxiter=1000, Df = df, r = .05)
        x0 = N.matrix("1; 5")
        with self.assertRaises(Exception):
            x = solver.solve(x0)
    
    
if __name__ == "__main__":
    unittest.main()
