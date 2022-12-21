# created 18.12.2022
# Dawid Bucki

# Library with tools for analysis of N-body problems

import numpy as np
import sympy as sp
from scipy.integrate import odeint
from scipy.optimize import root_scalar,root


def iterate(fun, x0, tol):
    x1 = fun(x0)
    while abs(x1-x0)>tol:
        x0 = x1
        x1 = fun(x0)
        
    return x1
    
def L1(mu,tol=.1**10):
#return the location of L1 libration point
    def f1(xi):
    #fixed point of this function will give us the libration point
        a = mu*xi**2-2*mu*xi+mu
        b = xi**2 - (3-mu)*xi - (2*mu-3)
        return (a/b)**(1./3)
            
    fixed_point = iterate(f1,0,tol)
    return 1-mu-fixed_point
        
def L2(mu,tol=.1**10):
#return the location of L2 libration point 
    def f2(xi):
    #fixed point of this function will give us the libration point 
        a = mu*(1+xi)**2
        b = xi**2 + xi*(3-mu) + (3-2*mu)
        return (a/b)**(1./3)
            
    fixed_point = iterate(f2,0,tol)
    return 1-mu+fixed_point 
        
def L3(mu,tol=.1**10):
#return the location of L3 libration point
    def f3(xi):
    #fixed point of this function will give us the libration point
        a = mu-1-xi*(2-2*mu)-xi**2*(1-mu)
        b = -xi**2+xi*(-mu-2)+(-2*mu-1)
        return (a/b)**(1./3)
           
    fixed_point = iterate(f3,1,tol)
    return -mu-fixed_point 
    

def dM(mu,x,y): #distance to Mars
    return np.sqrt((x-1+mu)**2+y**2)

def dS(mu,x,y): #distance to Sun
    return np.sqrt((x+mu)**2+y**2)

def V(mu,x,y): #potential
    return (x**2+y**2)/2 + (1-mu)/dS(mu,x,y) + mu/dM(mu,x,y)

def Vx(mu,x,y): #x-derivative of the potential
    return mu*(-mu-x+1)/dM(mu,x,y)**3 + x + (1-mu)*(-mu-x)/dS(mu,x,y)**3

def Vy(mu,x,y): #y-derivative of the potential
    return -mu*y/dM(mu,x,y)**3 - y*(1-mu)/dS(mu,x,y)**3 + y

def X(q,t,mu): #vector field governing the motion of the asteroid
    x,y,vx,vy = q[0],q[1],q[2],q[3]
    return np.array([vx,vy,2*vy+Vx(mu,x,y),-2*vx+Vy(mu,x,y)])
    
def C(q,mu):
    x,y,vx,vy = tuple(q)
    return .5*(vx**2+vy**2) - V(x=x,y=y,mu=mu)
    
    
def eigen_L1(mu):
    x_e = L1(mu) #location of the L1 point
    mu_bar = mu*abs(x_e-1+mu)**(-3) + (1-mu)*abs(x_e+mu)**(-3)
    a = 2*mu_bar+1
    b = mu_bar-1

    A = np.matrix([[0,0,1,0],
                   [0,0,0,1],
                   [a,0,0,2],
                   [0,-b,-2,0]])
                   
    eig = np.linalg.eig(A)
    eig_vals = eig[0]
    eig_vects = np.array(eig[1].T)
    
    return eig_vals,eig_vects
    
    
def X_lin_L1(q,t,mu):
    x_e = L1(mu) #location of the L1 point
    mu_bar = mu*abs(x_e-1+mu)**(-3) + (1-mu)*abs(x_e+mu)**(-3)
    a = 2*mu_bar+1
    b = mu_bar-1

    A = np.matrix([[0,0,1,0],
                   [0,0,0,1],
                   [a,0,0,2],
                   [0,-b,-2,0]])
                   
    return np.array(np.dot(A,q.reshape(-1,1)).T)[0]
    
    
def periodic_orbit_L1_initial_condition(mu,h):
#find periodic orbit starting at x_e+h using Newton's method
#where x_e is a location of L1 libration point
    eig_vals,eig_vects = eigen_L1(mu)
    x_e = L1(mu)
    
    period = 2*np.pi/np.imag(eig_vals[2])
    p = np.real(eig_vects[2]+eig_vects[3])
    a = h/p[0]
    
    def F(point):
        vy, time = point[0],point[1]
        q0 = np.array([x_e+h,0,0,vy])
        T = np.array([0,time])
        orbit = odeint(X, q0, T, args=(mu,))
        point = orbit[-1]
        return np.array([point[1],point[2]])
    
    if a<.005:
        initial_point = np.array([x_e,0,0,0]) + a*p
        initial_guess = np.array([initial_point[3],period/2])
    else:
        q,t = periodic_orbit_L1_initial_condition(mu,h-.0005)
        initial_guess = np.array([q[3],t])
        
    sol = root(F, initial_guess)
        
    vy,time = tuple(sol.x)
        
    return np.array([x_e+h,0,0,vy]), time


def periodic_orbit_L1(mu,h):
    pass