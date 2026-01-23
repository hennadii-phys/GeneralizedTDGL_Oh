# %%
import inspect
import itertools
import logging
import math
import numbers
import os
from datetime import datetime
import typing
import cmath 

import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
import json
import csv
from matplotlib import cm

import matplotlib.animation as animation
from pylab import *

# %%
import matplotlib.pyplot as plt
import matplotlib.colors as col
import seaborn as sns
import hsluv # install via pip
import scipy.special # just for the example function
from scipy.sparse.linalg import splu

# %%
##### generate custom colormaps for the phase plots
def make_segmented_cmap(): 
    white = '#ffffff'
    black = '#000000'
    red = '#ff0000'
    blue = '#0000ff'
    anglemap = col.LinearSegmentedColormap.from_list(
        'anglemap', [black, red, white, blue, black], N=256, gamma=1)
    return anglemap

def make_anglemap( N = 256, use_hpl = True ):
    h = np.ones(N) # hue
    h[:N//2] = 11.6 # red 
    h[N//2:] = 258.6 # blue
    s = 100 # saturation
    l = np.linspace(0, 100, N//2) # luminosity
    l = np.hstack( (l,l[::-1] ) )

    colorlist = np.zeros((N,3))
    for ii in range(N):
        if use_hpl:
            colorlist[ii,:] = hsluv.hpluv_to_rgb( (h[ii], s, l[ii]) )
        else:
            colorlist[ii,:] = hsluv.hsluv_to_rgb( (h[ii], s, l[ii]) )
    colorlist[colorlist > 1] = 1 # correct numeric errors
    colorlist[colorlist < 0] = 0 
    return col.ListedColormap( colorlist )

N = 256
segmented_cmap = make_segmented_cmap()
flat_huslmap = col.ListedColormap(sns.color_palette('husl',N))
hsluv_anglemap = make_anglemap( use_hpl = False )
hpluv_anglemap = make_anglemap( use_hpl = True )

##### generate data grid
x = np.linspace(-2,2,N)
y = np.linspace(-2,2,N)
z = np.zeros((len(y),len(x))) # make cartesian grid
for ii in range(len(y)): 
    z[ii] = np.arctan2(y[ii],x) # simple angular function
    z[ii] = np.angle(scipy.special.gamma(x+1j*y[ii])) # some complex function

##### plot with different colormaps
fig = plt.figure(1)
fig.clf()
colormapnames = ['segmented map', 'hue-HUSL', 'lum-HSLUV', 'lum-HPLUV']
colormaps = [segmented_cmap, flat_huslmap, hsluv_anglemap, hpluv_anglemap]
for ii, cm in enumerate(colormaps):
    ax = fig.add_subplot(2, 2, ii+1)
    pmesh = ax.pcolormesh(x, y, z/np.pi, 
        cmap = cm, vmin=-1, vmax=1)
    plt.axis([x.min(), x.max(), y.min(), y.max()])
    cbar = fig.colorbar(pmesh)
    cbar.ax.set_ylabel('Phase [pi]')
    ax.set_title( colormapnames[ii] )
plt.show()

# %%
# Make a grid on the square film
a=10;b=10;xl=-a;xr=a;yb=-b;yt=b
Nx=50;Ny=50;delx=(xr-xl)/(Nx);dely=(yt-yb)/(Ny);
xgrid=np.linspace(xl,xr,Nx+1)
ygrid=np.linspace(yb,yt,Ny+1)
# Parameters of the time-grid
Gs=1;Nt=10000;tau=1/Gs;dt=tau/Nt;
xg, yg = np.meshgrid(xgrid,ygrid)

# %%
#Choose type of the magnetic field or light beam:

# %%
# Vector potential of uniform magnetic field:
B=0.6
def applied_vector_potential_x_fun(x,y,t):
    return 0
t=0;w=1;#Ax=0.5;
xg, yg = np.meshgrid(xgrid,ygrid)
#xg, yg = np.meshgrid(xgrid[10:20],ygrid[10:20])
Ax=np.array(applied_vector_potential_x_fun(xg,yg,t))
def applied_vector_potential_y_fun(x,y,t):
    return B*x
#Ax=np.zeros((Nx+1,Ny+1))
Ay=np.transpose(np.asarray(applied_vector_potential_y_fun(xg,yg,t)))
#Ay=np.asarray(applied_vector_potential_y_fun(xg,yg,t))
Ax=np.zeros(np.shape(xg))
mu=0

# %%
# Vector potential of uniform plane-wave, x-polirized:
def applied_vector_potential_x_fun(x,y,t):
    return A0x*np.sin(w*t)+0*x
def applied_vector_potential_y_fun(x,y,t):
    return 0.0+0.0*x#*np.exp(-(x**2+y**2)/w0**2)
t=0;w=1;A0x=0.5;
xg, yg = np.meshgrid(xgrid,ygrid)
Ax=np.transpose(np.asarray(applied_vector_potential_x_fun(xg,yg,t)))
Ay=np.zeros((Nx+1,Ny+1));

# %%
# Vector potential of finite-sized Gaussian beam, x-polirized:
A0x=0.5;w0=4
t=0;w=1;
xg, yg = np.meshgrid(xgrid,ygrid)
#xg, yg = np.meshgrid(xgrid[10:20],ygrid[10:20])
#Ax=np.array(applied_vector_potential_x_fun(xg,yg,t))
def applied_vector_potential_x_fun(x,y,t):
    return A0x*np.sin(w*t)*np.exp(-(x**2+y**2)/w0**2)
def applied_vector_potential_y_fun(x,y,t):
    return 0.0+0.0*x#*np.exp(-(x**2+y**2)/w0**2)
#Ax=np.zeros((Nx+1,Ny+1))
Ax=np.transpose(np.asarray(applied_vector_potential_x_fun(xg,yg,t)))
#Ay=np.asarray(applied_vector_potential_y_fun(xg,yg,t))
Ay=np.zeros(np.shape(xg))
mu=0
def Bz_fun(x,y,t):
    return dy(Ax)-dx(Ay)

# %%
# Vector potential of finite-sized Gaussian beam, circularly polirized:
A0x=0.5;A0y=0.5;w0=4
t=0;w=1;
xg, yg = np.meshgrid(xgrid,ygrid)
#xg, yg = np.meshgrid(xgrid[10:20],ygrid[10:20])
#Ax=np.array(applied_vector_potential_x_fun(xg,yg,t))
def applied_vector_potential_x_fun(x,y,t):
    return A0x*np.sin(w*t)*np.exp(-(x**2+y**2)/w0**2)
def applied_vector_potential_y_fun(x,y,t):
    return A0y*np.cos(w*t)*np.exp(-(x**2+y**2)/w0**2)
#Ax=np.zeros((Nx+1,Ny+1))
Ax=np.transpose(np.asarray(applied_vector_potential_x_fun(xg,yg,t)))
#Ay=np.asarray(applied_vector_potential_y_fun(xg,yg,t))
Ay=np.transpose(np.asarray(applied_vector_potential_y_fun(xg,yg,t)))
mu=0
def Bz_fun(x,y,t):
    return dy(Ax)-dx(Ay)

# %%
# Vector potential uniform planewave, circularly polirized:
A0x=0.5;A0y=0.5
t=0;w=1;
xg, yg = np.meshgrid(xgrid,ygrid)
#xg, yg = np.meshgrid(xgrid[10:20],ygrid[10:20])
#Ax=np.array(applied_vector_potential_x_fun(xg,yg,t))
def applied_vector_potential_x_fun(x,y,t):
    return A0x*np.sin(w*t)+0*x
def applied_vector_potential_y_fun(x,y,t):
    return A0y*np.cos(w*t)+0*x
#Ax=np.zeros((Nx+1,Ny+1))
Ax=np.transpose(np.asarray(applied_vector_potential_x_fun(xg,yg,t)))
#Ay=np.asarray(applied_vector_potential_y_fun(xg,yg,t))
Ay=np.transpose(np.asarray(applied_vector_potential_y_fun(xg,yg,t)))
mu=0
def Bz_fun(x,y,t):
    return dy(Ax)-dx(Ay)

# %%
#Supercurrent
def Js(psi,Px,Py,Pz,D1,D2,G1,G2,G3,D3,D4,D5):
    JA1gx=np.imag(np.conjugate(psi)*dx(psi) - 1j*Ax*psi*np.conjugate(psi))

    JA1gy=np.imag(np.conjugate(psi)*dy(psi) - 1j*Ay*psi*np.conjugate(psi))



    JA1gGx=np.imag(1/2.*aA1gEg*np.conjugate(psi)*Dx(D1) + 1/2.*aA1gEg*np.conjugate(D1)*Dx(psi) - 1/2.*aA1gEg*np.conjugate(psi)*Dx(D2)  - 1/2.*aA1gEg*np.conjugate(D2)*Dx(psi)     - aA1gT1g*np.conjugate(psi)*Dy(G1)  + aA1gT1g*np.conjugate(G1)*Dy(psi)     + aA1gT2g*np.conjugate(psi)*Dy(D3)  + aA1gT2g*np.conjugate(D3)*Dy(psi))

    JA1gGy=np.imag(-1/2.*aA1gEg*np.conjugate(psi)*Dy(D1)  -1/2.*aA1gEg*np.conjugate(D1)*Dy(psi) - 1/2.*aA1gEg*np.conjugate(psi)*Dy(D2)  - 1/2.*aA1gEg*np.conjugate(D2)*Dy(psi)    + aA1gT1g*np.conjugate(psi)*Dx(G1)  - aA1gT1g*np.conjugate(G1)*Dx(psi)     + aA1gT2g*np.conjugate(psi)*Dx(D3)  + aA1gT2g*np.conjugate(D3)*Dx(psi))



    JEgGx=np.imag((aEg1-aEg2)*np.conjugate(D1)*Dx(D1)    + (aEg1+aEg2)*np.conjugate(D2)*Dx(D2)   + sqrt(3)*aEg2*np.conjugate(D1)*Dx(D2)    + sqrt(3)*aEg2*np.conjugate(D2)*Dx(D1)   -1/3.*(aEgT1g1+aEgT1g2)*np.conjugate(D1)*Dy(G1)  -1/3.*(aEgT1g1+aEgT1g2)*np.conjugate(G1)*Dy(D1)     + (aEgT1g1-aEgT1g2)*np.conjugate(D2)*Dy(G1)      + (-aEgT1g1+aEgT1g2)*np.conjugate(G1)*Dy(D2)   +1/3.*(aEgT2g1-aEgT2g2)*np.conjugate(D1)*Dy(D3)  -1/3.*(aEgT2g1-aEgT2g2)*np.conjugate(D3)*Dy(D1)     - (aEgT2g1+aEgT2g2)*np.conjugate(D2)*Dy(D3)      + (-aEgT2g1-aEgT2g2)*np.conjugate(D3)*Dy(D2) )

    JEgGy=np.imag((aEg1-aEg2)*np.conjugate(D2)*Dy(D1)   + (aEg1+aEg2)*np.conjugate(D2)*Dy(D2)   -sqrt(3)*aEg2*np.conjugate(D1)*Dy(D2)     -sqrt(3)*aEg2*np.conjugate(D2)*Dy(D1)    -1/3.*(aEgT1g1+aEgT1g2)*np.conjugate(G1)*Dx(D1)   -1/3.*(aEgT1g1+aEgT1g2)*np.conjugate(D1)*Dx(G1)     + (aEgT1g1-aEgT1g2)*np.conjugate(G1)*Dx(D2)      + (-aEgT1g1+aEgT1g2)*np.conjugate(D2)*Dx(G1)   +1/3.*(aEgT2g1-aEgT2g2)*np.conjugate(D3)*Dx(D1)   -1/3.*(aEgT2g1-aEgT2g2)*np.conjugate(D1)*Dx(D3)     - (aEgT2g1+aEgT2g2)*np.conjugate(D3)*Dx(D2)     + (-aEgT2g1-aEgT2g2)*np.conjugate(D2)*Dx(D3) )


    JT1gGx=np.imag(aT1g1*np.conjugate(G1)*Dx(G1)  + aT1g2*np.conjugate(G2)*Dx(G2)  + aT1g2*np.conjugate(G3)*Dx(G3)   + aT1g3*np.conjugate(G1)*Dy(G2)  + aT1g4*np.conjugate(G2)*Dy(G1)  - aT1gT2g1*np.conjugate(G1)*Dx(D3)  - aT1gT2g1*np.conjugate(D3)*Dx(G1)    - aT1gT2g1*np.conjugate(G2)*Dx(D4)  - aT1gT2g1*np.conjugate(D4)*Dx(G2)      - aT1gT2g2*np.conjugate(G2)*Dy(D5)   - aT1gT2g3*np.conjugate(D5)*Dy(G2)    - aT1gT2g3*np.conjugate(G3)*Dy(D4) )

    JT1gGy=np.imag(aT1g1*np.conjugate(G2)*Dy(G2)  + aT1g2*np.conjugate(G1)*Dy(G1)  + aT1g2*np.conjugate(G3)*Dy(G3)   + aT1g3*np.conjugate(G2)*Dx(G1)  + aT1g4*np.conjugate(G1)*Dx(G2)  + aT1gT2g1*np.conjugate(G1)*Dy(D3)  + aT1gT2g1*np.conjugate(D3)*Dy(G1)    - aT1gT2g1*np.conjugate(G3)*Dy(D5)  - aT1gT2g1*np.conjugate(D5)*Dy(G3)      - aT1gT2g2*np.conjugate(D5)*Dx(G2)   - aT1gT2g3*np.conjugate(G2)*Dx(D5)    - aT1gT2g3*np.conjugate(D4)*Dx(G3) )



    JT2gx=np.imag(aT2g1*np.conjugate(D3)*Dx(D3)  + aT2g2*np.conjugate(D4)*Dx(D4)  + aT2g2*np.conjugate(D5)*Dx(D5)   + aT2g3*np.conjugate(D3)*Dy(D4)  + aT2g4*np.conjugate(D4)*Dy(D3) )

    JT2gy=np.imag(aT2g1*np.conjugate(D4)*Dy(D4)  + aT2g2*np.conjugate(D3)*Dy(D3)  + aT2g2*np.conjugate(D5)*Dy(D5)   + aT2g3*np.conjugate(D4)*Dx(D3)  + aT2g4*np.conjugate(D3)*Dx(D4) )



    JT1uGx=np.imag(aT1u1*np.conjugate(Px)*Dx(Px)  + aT1u2*np.conjugate(Py)*Dx(Py)  + aT1u2*np.conjugate(Pz)*Dx(Pz)   + aT1u3*np.conjugate(Px)*Dy(Py)  + aT1u4*np.conjugate(Py)*Dy(Px)      - 2*(aA1gT1u*np.conjugate(psi)*Px  +1/2.*aT1uEg*np.conjugate(D1)*Px  -1/2.*aT1uEg*np.conjugate(D2)*Px    - aT1uT1g*np.conjugate(G1)*Py  - aT1uT1g*np.conjugate(G2)*Pz     - aT1uT2g*np.conjugate(G1)*Py   - aT1uT2g*np.conjugate(G2)*Pz  )  )

    JT1uGy=np.imag(aT1u1*np.conjugate(Py)*Dy(Py)  + aT1u2*np.conjugate(Px)*Dy(Px)  + aT1u2*np.conjugate(Pz)*Dy(Pz)   + aT1u3*np.conjugate(Py)*Dx(Px)  + aT1u4*np.conjugate(Px)*Dx(Py)      - 2*(aA1gT1u*np.conjugate(psi)*Py  -1/2.*aT1uEg*np.conjugate(D1)*Py  -1/2.*aT1uEg*np.conjugate(D2)*Py    + aT1uT1g*np.conjugate(G1)*Px  - aT1uT1g*np.conjugate(G3)*Pz     - aT1uT2g*np.conjugate(G1)*Px   - aT1uT2g*np.conjugate(G3)*Pz  )  )

    Jsx=JA1gx+JA1gGx+JEgGx+JT1gGx+JT2gx  +JT1uGx - (dx(np.transpose(np.asarray(applied_vector_potential_x_fun(xg,yg,t+dt))))-dx(Ax))/dt
    #print(np.asarray(applied_vector_potential_y_fun(xg,yg,t+dt))-Ay)
    Jsy=JA1gy+JA1gGy+JEgGy+JT1gGy+JT2gy  +JT1uGy  - (dy(np.transpose(np.asarray(applied_vector_potential_y_fun(xg,yg,t+dt))))-dy(Ay))/dt
    return (Jsx+1j*Jsy)

# %%
#Divergence of the supercurrent
def div_Js(psi,Px,Py,Pz,D1,D2,G1,G2,G3,D3,D4,D5):
    JA1gx=np.imag(np.conjugate(psi)*dx(psi) - 1j*Ax*psi*np.conjugate(psi))

    JA1gy=np.imag(np.conjugate(psi)*dy(psi) - 1j*Ay*psi*np.conjugate(psi))



    JA1gGx=np.imag(1/2.*aA1gEg*np.conjugate(psi)*Dx(D1) + 1/2.*aA1gEg*np.conjugate(D1)*Dx(psi) - 1/2.*aA1gEg*np.conjugate(psi)*Dx(D2)  - 1/2.*aA1gEg*np.conjugate(D2)*Dx(psi)     - aA1gT1g*np.conjugate(psi)*Dy(G1)  + aA1gT1g*np.conjugate(G1)*Dy(psi)     + aA1gT2g*np.conjugate(psi)*Dy(D3)  + aA1gT2g*np.conjugate(D3)*Dy(psi))

    JA1gGy=np.imag(-1/2.*aA1gEg*np.conjugate(psi)*Dy(D1)  -1/2.*aA1gEg*np.conjugate(D1)*Dy(psi) - 1/2.*aA1gEg*np.conjugate(psi)*Dy(D2)  - 1/2.*aA1gEg*np.conjugate(D2)*Dy(psi)    + aA1gT1g*np.conjugate(psi)*Dx(G1)  - aA1gT1g*np.conjugate(G1)*Dx(psi)     + aA1gT2g*np.conjugate(psi)*Dx(D3)  + aA1gT2g*np.conjugate(D3)*Dx(psi))



    JEgGx=np.imag((aEg1-aEg2)*np.conjugate(D1)*Dx(D1)    + (aEg1+aEg2)*np.conjugate(D2)*Dx(D2)   + sqrt(3)*aEg2*np.conjugate(D1)*Dx(D2)    + sqrt(3)*aEg2*np.conjugate(D2)*Dx(D1)   -1/3.*(aEgT1g1+aEgT1g2)*np.conjugate(D1)*Dy(G1)  -1/3.*(aEgT1g1+aEgT1g2)*np.conjugate(G1)*Dy(D1)     + (aEgT1g1-aEgT1g2)*np.conjugate(D2)*Dy(G1)      + (-aEgT1g1+aEgT1g2)*np.conjugate(G1)*Dy(D2)   +1/3.*(aEgT2g1-aEgT2g2)*np.conjugate(D1)*Dy(D3)  -1/3.*(aEgT2g1-aEgT2g2)*np.conjugate(D3)*Dy(D1)     - (aEgT2g1+aEgT2g2)*np.conjugate(D2)*Dy(D3)      + (-aEgT2g1-aEgT2g2)*np.conjugate(D3)*Dy(D2) )

    JEgGy=np.imag((aEg1-aEg2)*np.conjugate(D2)*Dy(D1)   + (aEg1+aEg2)*np.conjugate(D2)*Dy(D2)   -sqrt(3)*aEg2*np.conjugate(D1)*Dy(D2)     -sqrt(3)*aEg2*np.conjugate(D2)*Dy(D1)    -1/3.*(aEgT1g1+aEgT1g2)*np.conjugate(G1)*Dx(D1)   -1/3.*(aEgT1g1+aEgT1g2)*np.conjugate(D1)*Dx(G1)     + (aEgT1g1-aEgT1g2)*np.conjugate(G1)*Dx(D2)      + (-aEgT1g1+aEgT1g2)*np.conjugate(D2)*Dx(G1)   +1/3.*(aEgT2g1-aEgT2g2)*np.conjugate(D3)*Dx(D1)   -1/3.*(aEgT2g1-aEgT2g2)*np.conjugate(D1)*Dx(D3)     - (aEgT2g1+aEgT2g2)*np.conjugate(D3)*Dx(D2)     + (-aEgT2g1-aEgT2g2)*np.conjugate(D2)*Dx(D3) )


    JT1gGx=np.imag(aT1g1*np.conjugate(G1)*Dx(G1)  + aT1g2*np.conjugate(G2)*Dx(G2)  + aT1g2*np.conjugate(G3)*Dx(G3)   + aT1g3*np.conjugate(G1)*Dy(G2)  + aT1g4*np.conjugate(G2)*Dy(G1)  - aT1gT2g1*np.conjugate(G1)*Dx(D3)  - aT1gT2g1*np.conjugate(D3)*Dx(G1)    - aT1gT2g1*np.conjugate(G2)*Dx(D4)  - aT1gT2g1*np.conjugate(D4)*Dx(G2)      - aT1gT2g2*np.conjugate(G2)*Dy(D5)   - aT1gT2g3*np.conjugate(D5)*Dy(G2)    - aT1gT2g3*np.conjugate(G3)*Dy(D4) )

    JT1gGy=np.imag(aT1g1*np.conjugate(G2)*Dy(G2)  + aT1g2*np.conjugate(G1)*Dy(G1)  + aT1g2*np.conjugate(G3)*Dy(G3)   + aT1g3*np.conjugate(G2)*Dx(G1)  + aT1g4*np.conjugate(G1)*Dx(G2)  + aT1gT2g1*np.conjugate(G1)*Dy(D3)  + aT1gT2g1*np.conjugate(D3)*Dy(G1)    - aT1gT2g1*np.conjugate(G3)*Dy(D5)  - aT1gT2g1*np.conjugate(D5)*Dy(G3)      - aT1gT2g2*np.conjugate(D5)*Dx(G2)   - aT1gT2g3*np.conjugate(G2)*Dx(D5)    - aT1gT2g3*np.conjugate(D4)*Dx(G3) )



    JT2gx=np.imag(aT2g1*np.conjugate(D3)*Dx(D3)  + aT2g2*np.conjugate(D4)*Dx(D4)  + aT2g2*np.conjugate(D5)*Dx(D5)   + aT2g3*np.conjugate(D3)*Dy(D4)  + aT2g4*np.conjugate(D4)*Dy(D3) )

    JT2gy=np.imag(aT2g1*np.conjugate(D4)*Dy(D4)  + aT2g2*np.conjugate(D3)*Dy(D3)  + aT2g2*np.conjugate(D5)*Dy(D5)   + aT2g3*np.conjugate(D4)*Dx(D3)  + aT2g4*np.conjugate(D3)*Dx(D4) )



    JT1uGx=np.imag(aT1u1*np.conjugate(Px)*Dx(Px)  + aT1u2*np.conjugate(Py)*Dx(Py)  + aT1u2*np.conjugate(Pz)*Dx(Pz)   + aT1u3*np.conjugate(Px)*Dy(Py)  + aT1u4*np.conjugate(Py)*Dy(Px)      - 2*(aA1gT1u*np.conjugate(psi)*Px  +1/2.*aT1uEg*np.conjugate(D1)*Px  -1/2.*aT1uEg*np.conjugate(D2)*Px    - aT1uT1g*np.conjugate(G1)*Py  - aT1uT1g*np.conjugate(G2)*Pz     - aT1uT2g*np.conjugate(G1)*Py   - aT1uT2g*np.conjugate(G2)*Pz  )  )

    JT1uGy=np.imag(aT1u1*np.conjugate(Py)*Dy(Py)  + aT1u2*np.conjugate(Px)*Dy(Px)  + aT1u2*np.conjugate(Pz)*Dy(Pz)   + aT1u3*np.conjugate(Py)*Dx(Px)  + aT1u4*np.conjugate(Px)*Dx(Py)      - 2*(aA1gT1u*np.conjugate(psi)*Py  -1/2.*aT1uEg*np.conjugate(D1)*Py  -1/2.*aT1uEg*np.conjugate(D2)*Py    + aT1uT1g*np.conjugate(G1)*Px  - aT1uT1g*np.conjugate(G3)*Pz     - aT1uT2g*np.conjugate(G1)*Px   - aT1uT2g*np.conjugate(G3)*Pz  )  )

    Jsx=JA1gx+JA1gGx+JEgGx+JT1gGx+JT2gx  +JT1uGx - (dx(np.transpose(np.asarray(applied_vector_potential_x_fun(xg,yg,t+dt))))-dx(Ax))/dt
    #print(np.asarray(applied_vector_potential_y_fun(xg,yg,t+dt))-Ay)
    Jsy=JA1gy+JA1gGy+JEgGy+JT1gGy+JT2gy  +JT1uGy  - (dy(np.transpose(np.asarray(applied_vector_potential_y_fun(xg,yg,t+dt))))-dy(Ay))/dt
    divJs= dx(Jsx) + dy(Jsy)
    return divJs

# %%
#Define derivatives
def dx(f: np.ndarray):
    fplusx=np.roll(f,-1,axis=0)
    fminusx=np.roll(f,1,axis=0)
    return (fplusx-fminusx)/(2*delx)

def dy(f: np.ndarray):
    fplusy=np.roll(f,-1,axis=1)
    fminusy=np.roll(f,1,axis=1)
    return (fplusy-fminusy)/(2*dely)

def dxx(f: np.ndarray):
    fplusx=np.roll(f,-1,axis=0)
    fminusx=np.roll(f,1,axis=0)
    return (fplusx-2*f+fminusx)/delx**2

def dyy(f: np.ndarray):
    fplusy=np.roll(f,-1,axis=1)
    fminusy=np.roll(f,1,axis=1)
    return (fplusy-2*f+fminusy)/dely**2

def dxy(f: np.ndarray):
    fplusxplusy=np.roll(np.roll(f,-1,axis=0),-1,axis=1)
    fplusxminusy=np.roll(np.roll(f,-1,axis=0),1,axis=1)
    fminusxplusy=np.roll(np.roll(f,1,axis=0),-1,axis=1)
    fminusxminusy=np.roll(np.roll(f,1,axis=0),1,axis=1)
    return (fplusxplusy-fplusxminusy-fminusxplusy+fminusxminusy)/(4*delx*dely)

def Dxx(f: np.ndarray):
    return dxx(f)-1j*dx(Ax*f)-1j*Ax*dx(f)-(Ax**2)*f

def Dyy(f: np.ndarray):
    return dyy(f)-1j*dy(Ay*f)-1j*Ay*dy(f)-(Ay**2)*f

def Laplacian(f: np.ndarray):
    return Dxx(f)+Dyy(f)

def Dxy(f: np.ndarray):
    return dxy(f)-1j*dx(Ay*f)-1j*Ax*dy(f)-(Ax*Ay)*f

def Dyx(f: np.ndarray):
    return dxy(f)-1j*dy(Ax*f)-1j*Ay*dx(f)-(Ax*Ay)*f

def Dx(f: np.ndarray):
    return dx(f)-1j*(Ax*f)

def Dy(f: np.ndarray):
    return dy(f)-1j*(Ay*f)

# %%
#def Poissons_eq_sol(rhs): Matrix for the Poisson equation for the Forward difference derivatives used in the boundary conditions
N2=(Nx-1)*(Ny-1)
A=np.zeros((N2,N2))
## Diagonal            
for i in range (1,Nx-2):
    for j in range (1,Ny-2):           
        A[(Ny-1)*i+j,(Ny-1)*i+j]=-4

# LOWER DIAGONAL        
for i in range (1,Nx-2):
    for j in range (1,Ny-2):           
        A[(Ny-1)*i+j,(Ny-1)*i+j-1]=1   
# UPPPER DIAGONAL        
for i in range (1,Nx-2):
    for j in range (1,Ny-2):           
        A[(Ny-1)*i+j,(Ny-1)*i+j+1]=1   


# LOWER IDENTITY MATRIX
for i in range (1,Nx-2):
    for j in range (1,Ny-2):           
        A[(Ny-1)*i+j,(Ny-1)*(i-1)+j]=1        
        
# UPPER IDENTITY MATRIX
for i in range (1,Nx-2):
    for j in range (1,Ny-2):           
        A[(Ny-1)*i+j,(Ny-1)*(i+1)+j]=1

#FOR POINTS NEXT TO THE LEFT BOUNDARY
for j in range (1,Ny-2):           
        A[(Ny-1)*0+j,(Ny-1)*(1)+j]=1
        A[(Ny-1)*0+j,(Ny-1)*(0)+j+1]=1
        A[(Ny-1)*0+j,(Ny-1)*(0)+j-1]=1 
        A[(Ny-1)*0+j,(Ny-1)*(0)+j]=-3

#FOR POINTS NEXT TO THE RIGHT BOUNDARY
for j in range (1,Ny-2):           
        A[(Ny-1)*(Nx-2)+j,(Ny-1)*(Nx-3)+j]=1
        A[(Ny-1)*(Nx-2)+j,(Ny-1)*(Nx-2)+j+1]=1  
        A[(Ny-1)*(Nx-2)+j,(Ny-1)*(Nx-2)+j-1]=1    
        A[(Ny-1)*(Nx-2)+j,(Ny-1)*(Nx-2)+j]=-3

#FOR POINTS NEXT TO THE BOTTOM BOUNDARY
for i in range (1,Nx-2):           
        A[(Ny-1)*i+0,(Ny-1)*(i+1)+0]=1   
        A[(Ny-1)*i+0,(Ny-1)*(i-1)+0]=1   
        A[(Ny-1)*i+0,(Ny-1)*(i)+1]=1   
        A[(Ny-1)*i+0,(Ny-1)*(i)+0]=-3

#FOR POINTS NEXT TO THE TOP BOUNDARY
for i in range (1,Nx-2):           
        A[(Ny-1)*i+Ny-2,(Ny-1)*(i+1)+Ny-2]=1  
        A[(Ny-1)*i+Ny-2,(Ny-1)*(i-1)+Ny-2]=1    
        A[(Ny-1)*i+Ny-2,(Ny-1)*(i)+Ny-3]=1    
        A[(Ny-1)*i+Ny-2,(Ny-1)*(i)+Ny-2]=-3

#FOR CORNER POINTS
A[(Ny-1)*0+0,(Ny-1)*(1)+0]=1
A[(Ny-1)*0+0,(Ny-1)*(0)+1]=1
A[(Ny-1)*0+0,(Ny-1)*(0)+0]=-2

A[(Ny-1)*0+Ny-2,(Ny-1)*(1)+Ny-2]=1
A[(Ny-1)*0+Ny-2,(Ny-1)*(0)+Ny-3]=1
A[(Ny-1)*0+Ny-2,(Ny-1)*(0)+Ny-2]=-2

A[(Ny-1)*(Nx-2)+0,(Ny-1)*(Nx-3)+0]=1
A[(Ny-1)*(Nx-2)+0,(Ny-1)*(Nx-2)+1]=1
A[(Ny-1)*(Nx-2)+0,(Ny-1)*(Nx-2)+0]=-2

#This conditions are for the value of /mu at the top right to be not set
A[(Ny-1)*(Nx-2)+Ny-2,(Ny-1)*(Nx-3)+Ny-2]=1
A[(Ny-1)*(Nx-2)+Ny-2,(Ny-1)*(Nx-2)+Ny-3]=1
A[(Ny-1)*(Nx-2)+Ny-2,(Ny-1)*(Nx-2)+Ny-2]=-2

#This conditions are for the value of /mu at the top right corner set to zero
#A[(Ny-1)*(Nx-2)+Ny-2,(Ny-1)*(Nx-3)+Ny-2]=1
#A[(Ny-1)*(Nx-2)+Ny-2,(Ny-1)*(Nx-2)+Ny-3]=0
#A[(Ny-1)*(Nx-2)+Ny-2,(Ny-1)*(Nx-2)+Ny-2]=0

# %%
#def Poissons_eq_sol(rhs): Matrix for the Poisson equation for the Central difference derivatives used in the boundary conditions
N2=(Nx-1)*(Ny-1)
A=np.zeros((N2,N2))
## Diagonal            
for i in range (1,Nx-2):
    for j in range (1,Ny-2):           
        A[(Ny-1)*i+j,(Ny-1)*i+j]=-4

# LOWER DIAGONAL        
for i in range (1,Nx-2):
    for j in range (1,Ny-2):           
        A[(Ny-1)*i+j,(Ny-1)*i+j-1]=1   
# UPPPER DIAGONAL        
for i in range (1,Nx-2):
    for j in range (1,Ny-2):           
        A[(Ny-1)*i+j,(Ny-1)*i+j+1]=1   


# LOWER IDENTITY MATRIX
for i in range (1,Nx-2):
    for j in range (1,Ny-2):           
        A[(Ny-1)*i+j,(Ny-1)*(i-1)+j]=1        
        
# UPPER IDENTITY MATRIX
for i in range (1,Nx-2):
    for j in range (1,Ny-2):           
        A[(Ny-1)*i+j,(Ny-1)*(i+1)+j]=1

#FOR POINTS NEXT TO THE LEFT BOUNDARY
for j in range (1,Ny-2):           
        A[(Ny-1)*0+j,(Ny-1)*(1)+j]=2
        A[(Ny-1)*0+j,(Ny-1)*(0)+j+1]=1
        A[(Ny-1)*0+j,(Ny-1)*(0)+j-1]=1 
        A[(Ny-1)*0+j,(Ny-1)*(0)+j]=-4

#FOR POINTS NEXT TO THE RIGHT BOUNDARY
for j in range (1,Ny-2):           
        A[(Ny-1)*(Nx-2)+j,(Ny-1)*(Nx-3)+j]=2
        A[(Ny-1)*(Nx-2)+j,(Ny-1)*(Nx-2)+j+1]=1  
        A[(Ny-1)*(Nx-2)+j,(Ny-1)*(Nx-2)+j-1]=1    
        A[(Ny-1)*(Nx-2)+j,(Ny-1)*(Nx-2)+j]=-4

#FOR POINTS NEXT TO THE BOTTOM BOUNDARY
for i in range (1,Nx-2):           
        A[(Ny-1)*i+0,(Ny-1)*(i+1)+0]=1   
        A[(Ny-1)*i+0,(Ny-1)*(i-1)+0]=1   
        A[(Ny-1)*i+0,(Ny-1)*(i)+1]=2   
        A[(Ny-1)*i+0,(Ny-1)*(i)+0]=-4

#FOR POINTS NEXT TO THE TOP BOUNDARY
for i in range (1,Nx-2):           
        A[(Ny-1)*i+Ny-2,(Ny-1)*(i+1)+Ny-2]=1  
        A[(Ny-1)*i+Ny-2,(Ny-1)*(i-1)+Ny-2]=1    
        A[(Ny-1)*i+Ny-2,(Ny-1)*(i)+Ny-3]=2    
        A[(Ny-1)*i+Ny-2,(Ny-1)*(i)+Ny-2]=-4

#FOR CORNER POINTS
A[(Ny-1)*0+0,(Ny-1)*(1)+0]=2
A[(Ny-1)*0+0,(Ny-1)*(0)+1]=2
A[(Ny-1)*0+0,(Ny-1)*(0)+0]=-4

A[(Ny-1)*0+Ny-2,(Ny-1)*(1)+Ny-2]=2
A[(Ny-1)*0+Ny-2,(Ny-1)*(0)+Ny-3]=2
A[(Ny-1)*0+Ny-2,(Ny-1)*(0)+Ny-2]=-4

A[(Ny-1)*(Nx-2)+0,(Ny-1)*(Nx-3)+0]=2
A[(Ny-1)*(Nx-2)+0,(Ny-1)*(Nx-2)+1]=2
A[(Ny-1)*(Nx-2)+0,(Ny-1)*(Nx-2)+0]=-4

#This conditions are for the value of /mu at the top right to be not set
A[(Ny-1)*(Nx-2)+Ny-2,(Ny-1)*(Nx-3)+Ny-2]=2
A[(Ny-1)*(Nx-2)+Ny-2,(Ny-1)*(Nx-2)+Ny-3]=2
A[(Ny-1)*(Nx-2)+Ny-2,(Ny-1)*(Nx-2)+Ny-2]=-4

#This conditions are for the value of /mu at the top right corner set to zero
#A[(Ny-1)*(Nx-2)+Ny-2,(Ny-1)*(Nx-3)+Ny-2]=1
#A[(Ny-1)*(Nx-2)+Ny-2,(Ny-1)*(Nx-2)+Ny-3]=0
#A[(Ny-1)*(Nx-2)+Ny-2,(Ny-1)*(Nx-2)+Ny-2]=0

# %%
lu = splu(A[0:N2-1,0:N2-1])      # A must be a CSC or CSR sparse matrix

# %%
#parameters of the SC
Gs=1;rs=-1;bs=1;
Gp=1; GdEg=1; GgT1g=1;GdT2g=1;

rA1g = rs; bA1g = bs; aA1g = 1; aA1gT1u = 1; aA1gEg = 1; aA1gT1g = 1; aA1gT2g = 1;
rT1u = -30; aT1u1 = 1.1; aT1u2 = 1; aT1u3 = 1.2; aT1u4 = 1.1; bT1u1 = 1; bT1u2 = 1; bT1u3 = 1; aT1uEg = 1; aT1uT1g = 1.2; aT1uT2g = 1;
aT1uT1g = 1.2; aT1uT2g = 1;
rEg = -40; aEg1 = 1.05; aEg2 = 1; bEg1 = 1; bEg2 = 1; aEgT1g1 = 1.1; aEgT1g2 = 1; aEgT2g1 = 1.1; aEgT2g2 = 1.05;
rT1g = -50; aT1g1 = 1; aT1g2 = 1.05; aT1g3 = 1.1; aT1g4 = 0.95; bT1g1 = 1; bT1g2 = 1; bT1g3 = 1;
aT1gT2g1 = 1; aT1gT2g2 = 1.1; aT1gT2g3 = 1.2;
rT2g = -60; aT2g1 = 0.9; aT2g2 = 0.95; aT2g3 = 0.85; aT2g4 = 0.8; bT2g1 = 1; bT2g2 = 1; bT2g3 = 1;
cA1gT1u1 = 1; cA1gEg1 = 1; cA1gT1g1 = 1; cA1gT2g1 = 1;
cA1gT1u2 = 1; cA1gEg2 = 1; cA1gT1g2 = 1; cA1gT2g2 = 1;

# %%
# Initial conditions:
psi0=np.sqrt(-rs/bs)*np.ones([Nx+1, Ny+1])
psi=np.sqrt(-rs/bs)*np.ones([Nx+1, Ny+1])

Px=np.zeros([Nx+1, Ny+1])
Py=np.zeros([Nx+1, Ny+1])
Pz=np.zeros([Nx+1, Ny+1])

D1=np.zeros([Nx+1, Ny+1])
D2=np.zeros([Nx+1, Ny+1])

D3=np.zeros([Nx+1, Ny+1])
D4=np.zeros([Nx+1, Ny+1])
D5=np.zeros([Nx+1, Ny+1])

G1=np.zeros([Nx+1, Ny+1])
G2=np.zeros([Nx+1, Ny+1])
G3=np.zeros([Nx+1, Ny+1])
mu=np.zeros([Nx+1, Ny+1])
Jst=np.zeros([Nx+1, Ny+1])
Jnt=np.zeros([Nx+1, Ny+1])

# %%
#parameters of the SC for pure s-wave test
Gs=1;rs=-1;bs=1;
Gp=1; GdEg=1; GgT1g=1;GdT2g=1;

rA1g = rs; bA1g = bs; aA1g = 1; aA1gT1u = 0; aA1gEg = 0; aA1gT1g = 0; aA1gT2g = 0;
rT1u = 0; aT1u1 = 0; aT1u2 = 0; aT1u3 = 0; aT1u4 = 0; bT1u1 = 0; bT1u2 = 0; bT1u3 = 0; aT1uEg = 0; aT1uT1g = 0; aT1uT2g = 0;
aT1uT1g = 0; aT1uT2g = 0;
rEg = 0; aEg1 = 0; aEg2 = 0; bEg1 = 0; bEg2 = 0; aEgT1g1 = 0; aEgT1g2 = 0; aEgT2g1 = 0; aEgT2g2 = 0;
rT1g = 0; aT1g1 = 0; aT1g2 = 0; aT1g3 = 0; aT1g4 = 0; bT1g1 = 0; bT1g2 = 0; bT1g3 = 0;
aT1gT2g1 = 0; aT1gT2g2 = 0; aT1gT2g3 = 0;
rT2g = 0; aT2g1 = 0; aT2g2 = 0; aT2g3 = 0; aT2g4 = 0; bT2g1 = 0; bT2g2 = 0; bT2g3 = 0;
cA1gT1u1 = 0; cA1gEg1 = 0; cA1gT1g1 = 0; cA1gT2g1 = 0;
cA1gT1u2 = 0; cA1gEg2 = 0; cA1gT1g2 = 0; cA1gT2g2 = 0;

# %%
#parameters of the SC 2
Gs=1;rs=-1;bs=1;
Gp=1; GdEg=1; GgT1g=1;GdT2g=1;

rA1g = rs; bA1g = bs; aA1g = 1; aA1gT1u = 1; aA1gEg = 1; aA1gT1g = 1; aA1gT2g = 1;
rT1u = -30; aT1u1 = 1; aT1u2 = 1; aT1u3 = 1; aT1u4 = 1; bT1u1 = 1; bT1u2 = 1; bT1u3 = 1; aT1uEg = 1; aT1uT1g = 1; aT1uT2g = 1;
aT1uT1g = 1; aT1uT2g = 1;
rEg = -40; aEg1 = 1; aEg2 = 1; bEg1 = 1; bEg2 = 1; aEgT1g1 = 1; aEgT1g2 = 1; aEgT2g1 = 1; aEgT2g2 = 1;
rT1g = -50; aT1g1 = 1; aT1g2 = 1; aT1g3 = 1; aT1g4 = 1; bT1g1 = 1; bT1g2 = 1; bT1g3 = 1;
aT1gT2g1 = 1; aT1gT2g2 = 1; aT1gT2g3 = 1;
rT2g = -60; aT2g1 = 1; aT2g2 = 1; aT2g3 = 1; aT2g4 = 1; bT2g1 = 1; bT2g2 = 1; bT2g3 = 1;
cA1gT1u1 = 1; cA1gEg1 = 1; cA1gT1g1 = 1; cA1gT2g1 = 1;
cA1gT1u2 = 1; cA1gEg2 = 1; cA1gT1g2 = 1; cA1gT2g2 = 1;

# %%
#parameters of the SC 3
Gs=1;rs=-1;bs=1;
Gp=1; GdEg=1; GgT1g=1;GdT2g=1;

rA1g = rs; bA1g = bs; aA1g = 1; aA1gT1u = 0; aA1gEg = 0; aA1gT1g = 0; aA1gT2g = 0;
rT1u = -30; aT1u1 = 1; aT1u2 = 1; aT1u3 = 1; aT1u4 = 1; bT1u1 = 1; bT1u2 = 1; bT1u3 = 1; aT1uEg = 0; aT1uT1g = 0; aT1uT2g = 0;
aT1uT1g = 0; aT1uT2g = 0;
rEg = -40; aEg1 = 1.5; aEg2 = 1; bEg1 = 1; bEg2 = 1; aEgT1g1 = 0; aEgT1g2 = 0; aEgT2g1 = 0; aEgT2g2 = 0;
rT1g = -50; aT1g1 = 1; aT1g2 = 1; aT1g3 = 1; aT1g4 = 1; bT1g1 = 1; bT1g2 = 1; bT1g3 = 1;
aT1gT2g1 = 0; aT1gT2g2 = 0; aT1gT2g3 = 0;
rT2g = -60; aT2g1 = 1; aT2g2 = 1; aT2g3 = 1; aT2g4 = 1; bT2g1 = 1; bT2g2 = 1; bT2g3 = 1;
cA1gT1u1 = 0; cA1gEg1 = 0; cA1gT1g1 = 0; cA1gT2g1 = 0;
cA1gT1u2 = 0; cA1gEg2 = 0; cA1gT1g2 = 0; cA1gT2g2 = 0;

# %%
Kronecker=np.identity(Nx)

# %%
#Simulation. Note that in this implementation we disregard the dynamics of the scalar potential and set it to zero.
psi_hist0=np.asarray(psi)
Px_hist0=np.asarray(Px)
Py_hist0=np.asarray(Py)
Pz_hist0=np.asarray(Pz)
D1_hist0=np.asarray(D1)
D2_hist0=np.asarray(D2)
D3_hist0=np.asarray(D3)
D4_hist0=np.asarray(D4)
D5_hist0=np.asarray(D5)
G1_hist0=np.asarray(G1)
G2_hist0=np.asarray(G2)
G3_hist0=np.asarray(G3)
mu_hist0=np.asarray(mu)
Js_hist0=np.asarray(Jst)
Jn_hist0=np.asarray(Jnt)

psi_hist=[psi_hist0.tolist()]
Px_hist=[Px_hist0.tolist()]
Py_hist=[Py_hist0.tolist()]
Pz_hist=[Pz_hist0.tolist()]

D1_hist=[D1_hist0.tolist()]
D2_hist=[D2_hist0.tolist()]
D3_hist=[D3_hist0.tolist()]
D4_hist=[D4_hist0.tolist()]
D5_hist=[D5_hist0.tolist()]

G1_hist=[G1_hist0.tolist()]
G2_hist=[G2_hist0.tolist()]
G3_hist=[G3_hist0.tolist()]

mu_hist=[mu_hist0.tolist()]
Js_hist=[Js_hist0.tolist()]
Jn_hist=[Jn_hist0.tolist()]

print(np.shape(Px_hist))
#Simulation cycles
for it in range(0,126000):
     t=it*dt
     #print(it)
     #print(psi[10,10])
     #Ax=np.asarray(applied_vector_potential_x_fun(xg,yg,t))
     Ax=np.transpose(np.asarray(applied_vector_potential_x_fun(xg,yg,t)))
     Ay=np.transpose(np.asarray(applied_vector_potential_y_fun(xg,yg,t)))
     psi_sq1=psi*np.conjugate(psi)
     psi_sq2=psi*psi

     P_sq1=Px*np.conjugate(Px)+Py*np.conjugate(Py)+Pz*np.conjugate(Pz)
     P_sq2=Px*Px+Py*Py+Pz*Pz

     DEg_sq1 = D1*np.conjugate(D1)+D2*np.conjugate(D2)
     DEg_sq2 = D1*D1+D2*D2

     GT1g_sq1 = G1*np.conjugate(G1)+G2*np.conjugate(G2) + G3*np.conjugate(G3)
     GT1g_sq2 = G1*G1+G2*G2+G3*G3

     DT2g_sq1 = D3*np.conjugate(D3)+D4*np.conjugate(D4)+D5*np.conjugate(D5)
     DT2g_sq2 = D3*D3+D4*D4+D5*D5

       
     psi=( (1-psi_sq1)*psi + Laplacian(psi) -1j*Gs*mu*psi - 2*aA1gT1u*(Dx(Px)+Dy(Py)) + 1./2*aA1gEg*(Dxx(D1)-Dxx(D2)-Dyy(D1)-Dyy(D2)) 
          + aA1gT1g*(-Dxy(D3)+Dyx(D3)) + aA1gT2g*(Dxy(G1)+Dyx(G1))   -  cA1gT1u1*P_sq1*psi - cA1gT1u2*P_sq2*np.conjugate(psi) - cA1gEg1*DEg_sq1*psi - cA1gEg2*DEg_sq2*np.conjugate(psi) -  cA1gT1g1*GT1g_sq1*psi - cA1gT1g2*GT1g_sq2*np.conjugate(psi) -  cA1gT2g1*DT2g_sq1*psi - cA1gT2g2*DT2g_sq2*np.conjugate(psi))*dt/Gs+psi
     
     Px=( rT1u*Px - 2*bT1u1*P_sq1*Px - 2*bT1u2*P_sq2*np.conjugate(Px) - bT1u3*(P_sq1-Px*np.conjugate(Px))*Px + aT1u1*Dxx(Px) + aT1u2*Dyy(Px) + aT1u3*Dxy(Py) + aT1u4*Dyx(Py) -1j*Gp*mu*Px + 2*aA1gT1u*Dx(psi) + aT1uEg*(Dx(D1)-Dx(D2)) 
          + 2*aT1uT1g*Dy(D3) - 2*aT1uT2g*Dy(G1)    - cA1gT1u1*psi_sq1*Px - cA1gT1u2*psi_sq2*np.conjugate(Px))*dt/Gp+Px
     Py=( rT1u*Py - 2*bT1u1*P_sq1*Py - 2*bT1u2*P_sq2*np.conjugate(Py) - bT1u3*(P_sq1-Py*np.conjugate(Py))*Py + aT1u1*Dyy(Py) + aT1u2*Dxx(Py) + aT1u3*Dyx(Px) + aT1u4*Dxy(Px) -1j*Gp*mu*Py + 2*aA1gT1u*Dy(psi) + aT1uEg*(-Dy(D1)-Dy(D2)) 
          - 2*aT1uT1g*Dx(D3) - 2*aT1uT2g*Dx(G1)   - cA1gT1u1*psi_sq1*Py - cA1gT1u2*psi_sq2*np.conjugate(Py))*dt/Gp+Py
     Pz=( rT1u*Pz - 2*bT1u1*P_sq1*Pz - 2*bT1u2*P_sq2*np.conjugate(Pz) - bT1u3*(P_sq1-Px*np.conjugate(Pz))*Pz + aT1u2*(Dxx(Pz)+Dyy(Pz)) -1j*Gp*mu*Pz
          + 2*aT1uT1g*(-Dx(D4)-Dy(D5)) + 2*aT1uT2g*(-Dx(G2)-Dy(G3))   - cA1gT1u1*psi_sq1*Pz - cA1gT1u2*psi_sq2*np.conjugate(Pz))*dt/Gp+Pz
     

     D1=( rEg*D1 - 2*bEg1*DEg_sq1*D1 - 2*bEg2*(np.conjugate(D1)*D2-D1*np.conjugate(D2))*D2 + aEg1*(Dxx(D1)+Dyy(D1)) + aEg2*(-Dxx(D1)-Dyy(D1)+sqrt(3)*Dxx(D2)-sqrt(3)*Dyy(D2)) -1j*GdEg*mu*D1 + 1./2.*aA1gEg*(Dxx(psi)-Dyy(psi)) - aT1uEg*(Dx(Px)-Dy(Py)) 
          + aEgT1g1*(-1./3.*Dxy(D3)-1./3.*Dyx(D3)) + aEgT1g2*(-1./3.*Dxy(D3)-1./3.*Dyx(D3)) + aEgT2g1*(1./3.*Dxy(G1)-1./3.*Dyx(G1)) + aEgT2g2*(-1./3.*Dxy(G1)+1./3.*Dyx(G1))    - cA1gEg1*psi_sq1*D1 - cA1gEg2*psi_sq2*np.conjugate(D1))*dt/GdEg+D1
     D2=( rEg*D2 - 2*bEg1*DEg_sq1*D2 + 2*bEg2*(np.conjugate(D1)*D2-D1*np.conjugate(D2))*D1 + aEg1*(Dxx(D2)+Dyy(D2)) + aEg2*(Dxx(D2)+Dyy(D2)+sqrt(3)*Dxx(D1)-sqrt(3)*Dyy(D1)) -1j*GdEg*mu*D2 + 1./2.*aA1gEg*(-Dxx(psi)-Dyy(psi)) - aT1uEg*(-Dx(Px)-Dy(Py)) 
          + aEgT1g1*(Dxy(D3)-Dyx(D3)) + aEgT1g2*(-Dxy(D3)+Dyx(D3)) + aEgT2g1*(-Dxy(G1)-Dyx(G1)) + aEgT2g2*(-Dxy(G1)-Dyx(G1))    - cA1gEg1*psi_sq1*D2 - cA1gEg2*psi_sq2*np.conjugate(D2))*dt/GdEg+D2
     

     G1=( rT1g*G1 - 2*bT1g1*GT1g_sq1*G1 - 2*bT1g2*GT1g_sq2*np.conjugate(G1) - bT1g3*(GT1g_sq1-G1*np.conjugate(G1))*G1 + aT1g1*Dxx(G1) + aT1g2*Dyy(G1) + aT1g3*Dxy(G2) + aT1g4*Dyx(G2) -1j*GgT1g*mu*G1 + aA1gT1g*(-Dyx(psi)+Dxy(psi)) + 2*aT1uT1g*(-Dy(Px)+Dx(Py)) 
          + aEgT1g1*(-1./3.*Dyx(D1)-1./3.*Dxy(D1)+Dyx(D2)-Dxy(D2)) + aEgT1g2*(-1./3.*Dyx(D1)-1./3.*Dxy(D1)-Dyx(D2)+Dxy(D2)) + aT1gT2g1*(-Dxx(D3)+Dyy(D3))   - cA1gT1g1*psi_sq1*G1 - cA1gT1g2*psi_sq2*np.conjugate(G1))*dt/GgT1g+G1
     G2=( rT1g*G2 - 2*bT1g1*GT1g_sq1*G2 - 2*bT1g2*GT1g_sq2*np.conjugate(G2) - bT1g3*(GT1g_sq1-G2*np.conjugate(G2))*G2 + aT1g1*Dyy(G2) + aT1g2*Dxx(G2) + aT1g3*Dyx(G1) + aT1g4*Dxy(G1) -1j*GgT1g*mu*G2 + 2*aT1uT1g*Dx(Pz) 
          - aT1gT2g1*Dxx(D4) - aT1gT2g2*Dxy(D5)- aT1gT2g3*Dyx(D5)    - cA1gT1g1*psi_sq1*G2 - cA1gT1g2*psi_sq2*np.conjugate(G2))*dt/GgT1g+G2
     G3=( rT1g*G3 - 2*bT1g1*GT1g_sq1*G3 - 2*bT1g2*GT1g_sq2*np.conjugate(G3) - bT1g3*(GT1g_sq1-G3*np.conjugate(G3))*G3 + aT1g2*(Dxx(G3)+Dyy(G3)) -1j*GgT1g*mu*G3
          - aT1gT2g1*Dyy(D5) - aT1gT2g3*Dxy(D4)    - cA1gT1g1*psi_sq1*G3 - cA1gT1g2*psi_sq2*np.conjugate(G3))*dt/GgT1g+G3
     

     D3=( rT2g*D3 - 2*bT2g1*DT2g_sq1*D3 - 2*bT2g2*DT2g_sq2*np.conjugate(D3) - bT2g3*(DT2g_sq1-D3*np.conjugate(D3))*D3 + aT2g1*Dxx(D3) + aT2g2*Dyy(D3) + aT2g3*Dxy(D4) + aT2g4*Dyx(D4) -1j*GdT2g*mu*D3 + aA1gT2g*(Dyx(psi)+Dxy(psi)) + 2*aT1uT2g*(Dy(Px)+Dx(Py)) 
          + aEgT2g1*(1./3.*Dyx(D1)-1./3.*Dxy(D1)-Dyx(D2)-Dxy(D2)) + aEgT2g2*(-1./3.*Dyx(D1)+1./3.*Dxy(D1)-Dyx(D2)-Dxy(D2)) + aT1gT2g1*(-Dxx(G1)+Dyy(G1))    - cA1gT2g1*psi_sq1*D3 - cA1gT2g2*psi_sq2*np.conjugate(D3))*dt/GdT2g+D3
     D4=( rT2g*D4 - 2*bT2g1*DT2g_sq1*D4 - 2*bT2g2*DT2g_sq2*np.conjugate(D4) - bT2g3*(DT2g_sq1-D4*np.conjugate(D4))*D4 + aT2g1*Dyy(D4) + aT2g2*Dxx(D4) + aT2g3*Dyx(D3) + aT2g4*Dxy(D3) -1j*GdT2g*mu*D4 + 2*aT1uT2g*Dx(Pz) 
          - aT1gT2g1*Dxx(G2) - aT1gT2g3*Dyx(G3)   - cA1gT2g1*psi_sq1*D4 - cA1gT2g2*psi_sq2*np.conjugate(D4))*dt/GdT2g+D4
     D5=( rT2g*D5 - 2*bT2g1*DT2g_sq1*D5 - 2*bT2g2*DT2g_sq2*np.conjugate(D5) - bT2g3*(DT2g_sq1-D5*np.conjugate(D5))*D5 + aT2g2*(Dxx(D5)+Dyy(D5)) -1j*GdT2g*mu*D5
          - aT1gT2g1*Dyy(G3) - aT1gT2g2*Dyx(G2) - aT1gT2g3*Dxy(G2)    - cA1gT2g1*psi_sq1*D5 - cA1gT2g2*psi_sq2*np.conjugate(D5))*dt/GdT2g+D5

     
     #Boundary conditions:

     choice=2
     if choice==1: #grad =0 condition for all
          #psi[0,:]=psi[1,:]/(1+1j*Ax[0,:]*delx)
          #psi[Nx-1,:]=psi[Nx-2,:]*(1+1j*Ax[Nx-2,:]*delx)
          #psi[:,0]=psi[:,1]/(1+1j*Ay[:,0]*dely)
          #psi[:,Ny-1]=psi[:,Ny-2]*(1+1j*Ay[:,Ny-2]*dely)
          psi[0,:]=psi[2,:] - 1j*Ax[1,:]*psi[1,:]*2*delx
          psi[Nx,:]=psi[Nx-2,:]+1j*Ax[Nx-1,:]*psi[Nx-1,:]*2*delx
          psi[:,0]=psi[:,2]-1j*Ay[:,1]*psi[:,1]*2*dely
          psi[:,Ny]=psi[:,Ny-2]+1j*Ay[:,Ny-1]*psi[Ny-1]*2*dely

          Px[0,:]=Px[2,:] - 1j*Ax[1,:]*Px[1,:]*2*delx
          Px[Nx,:]=Px[Nx-2,:]+1j*Ax[Nx-1,:]*Px[Nx-1,:]*2*delx
          Px[:,0]=Px[:,2]-1j*Ay[:,1]*Px[:,1]*2*dely
          Px[:,Ny]=Px[:,Ny-2]+1j*Ay[:,Ny-1]*Px[Ny-1]*2*dely

          Py[0,:]=Py[2,:] - 1j*Ax[1,:]*Py[1,:]*2*delx
          Py[Nx,:]=Py[Nx-2,:]+1j*Ax[Nx-1,:]*Py[Nx-1,:]*2*delx
          Py[:,0]=Py[:,2]-1j*Ay[:,1]*Py[:,1]*2*dely
          Py[:,Ny]=Py[:,Ny-2]+1j*Ay[:,Ny-1]*Py[Ny-1]*2*dely

          Pz[0,:]=Pz[2,:] - 1j*Ax[1,:]*Pz[1,:]*2*delx
          Pz[Nx,:]=Pz[Nx-2,:]+1j*Ax[Nx-1,:]*Pz[Nx-1,:]*2*delx
          Pz[:,0]=Pz[:,2]-1j*Ay[:,1]*Pz[:,1]*2*dely
          Pz[:,Ny]=Pz[:,Ny-2]+1j*Ay[:,Ny-1]*Pz[Ny-1]*2*dely

          D1[0,:]=D1[2,:] - 1j*Ax[1,:]*D1[1,:]*2*delx
          D1[Nx,:]=D1[Nx-2,:]+1j*Ax[Nx-1,:]*D1[Nx-1,:]*2*delx
          D1[:,0]=D1[:,2]-1j*Ay[:,1]*D1[:,1]*2*dely
          D1[:,Ny]=D1[:,Ny-2]+1j*Ay[:,Ny-1]*D1[Ny-1]*2*dely

          D2[0,:]=D2[2,:] - 1j*Ax[1,:]*D2[1,:]*2*delx
          D2[Nx,:]=D2[Nx-2,:]+1j*Ax[Nx-1,:]*D2[Nx-1,:]*2*delx
          D2[:,0]=D2[:,2]-1j*Ay[:,1]*D2[:,1]*2*dely
          D2[:,Ny]=D2[:,Ny-2]+1j*Ay[:,Ny-1]*D2[Ny-1]*2*dely

          G1[0,:]=G1[2,:] - 1j*Ax[1,:]*G1[1,:]*2*delx
          G1[Nx,:]=G1[Nx-2,:]+1j*Ax[Nx-1,:]*G1[Nx-1,:]*2*delx
          G1[:,0]=G1[:,2]-1j*Ay[:,1]*G1[:,1]*2*dely
          G1[:,Ny]=G1[:,Ny-2]+1j*Ay[:,Ny-1]*G1[Ny-1]*2*dely

          G2[0,:]=G2[2,:] - 1j*Ax[1,:]*G2[1,:]*2*delx
          G2[Nx,:]=G2[Nx-2,:]+1j*Ax[Nx-1,:]*G2[Nx-1,:]*2*delx
          G2[:,0]=G2[:,2]-1j*Ay[:,1]*G2[:,1]*2*dely
          G2[:,Ny]=G2[:,Ny-2]+1j*Ay[:,Ny-1]*G2[Ny-1]*2*dely

          G3[0,:]=G3[2,:] - 1j*Ax[1,:]*G3[1,:]*2*delx
          G3[Nx,:]=G3[Nx-2,:]+1j*Ax[Nx-1,:]*G3[Nx-1,:]*2*delx
          G3[:,0]=G3[:,2]-1j*Ay[:,1]*G3[:,1]*2*dely
          G3[:,Ny]=G3[:,Ny-2]+1j*Ay[:,Ny-1]*G3[Ny-1]*2*dely

          D3[0,:]=D3[2,:] - 1j*Ax[1,:]*D3[1,:]*2*delx
          D3[Nx,:]=D3[Nx-2,:]+1j*Ax[Nx-1,:]*D3[Nx-1,:]*2*delx
          D3[:,0]=D3[:,2]-1j*Ay[:,1]*D3[:,1]*2*dely
          D3[:,Ny]=D3[:,Ny-2]+1j*Ay[:,Ny-1]*D3[Ny-1]*2*dely

          D4[0,:]=D4[2,:] - 1j*Ax[1,:]*D4[1,:]*2*delx
          D4[Nx,:]=D4[Nx-2,:]+1j*Ax[Nx-1,:]*D4[Nx-1,:]*2*delx
          D4[:,0]=D4[:,2]-1j*Ay[:,1]*D4[:,1]*2*dely
          D4[:,Ny]=D4[:,Ny-2]+1j*Ay[:,Ny-1]*D4[Ny-1]*2*dely

          D5[0,:]=D5[2,:] - 1j*Ax[1,:]*D5[1,:]*2*delx
          D5[Nx,:]=D5[Nx-2,:]+1j*Ax[Nx-1,:]*D5[Nx-1,:]*2*delx
          D5[:,0]=D5[:,2]-1j*Ay[:,1]*D5[:,1]*2*dely
          D5[:,Ny]=D5[:,Ny-2]+1j*Ay[:,Ny-1]*D5[Ny-1]*2*dely
     elif choice==2: #grad boundary condition for the s-wave order parameter, zero b.c.s for the rest
          psi[0,:]=psi[2,:] - 1j*Ax[1,:]*psi[1,:]*2*delx
          psi[Nx,:]=psi[Nx-2,:]+1j*Ax[Nx-1,:]*psi[Nx-1,:]*2*delx
          psi[:,0]=psi[:,2]-1j*Ay[:,1]*psi[:,1]*2*dely
          psi[:,Ny]=psi[:,Ny-2]+1j*Ay[:,Ny-1]*psi[Ny-1]*2*dely

          Px[0,:]=0
          Px[Nx,:]=0
          Px[:,0]=0
          Px[:,Ny]=0

          Py[0,:]=0
          Py[Nx,:]=0
          Py[:,0]=0
          Py[:,Ny]=0

          Pz[0,:]=0
          Pz[Nx,:]=0
          Pz[:,0]=0
          Pz[:,Ny]=0

          D1[0,:]=0
          D1[Nx,:]=0
          D1[:,0]=0
          D1[:,Ny]=0

          D2[0,:]=0
          D2[Nx,:]=0
          D2[:,0]=0
          D2[:,Ny]=0

          G1[0,:]=0
          G1[Nx,:]=0
          G1[:,0]=0
          G1[:,Ny]=0

          G2[0,:]=0
          G2[Nx,:]=0
          G2[:,0]=0
          G2[:,Ny]=0

          G3[0,:]=0
          G3[Nx,:]=0
          G3[:,0]=0
          G3[:,Ny]=0

          D3[0,:]=0
          D3[Nx,:]=0
          D3[:,0]=0
          D3[:,Ny]=0

          D4[0,:]=0
          D4[Nx,:]=0
          D4[:,0]=0
          D4[:,Ny]=0

          D5[0,:]=0
          D5[Nx,:]=0
          D5[:,0]=0
          D5[:,Ny]=0
     else: #exact boundary conditions derived from the GL model
          BC=np.zeros((48*(Nx-1),48*(Nx-1)))
          RHS=np.zeros(48*(Nx-1))
          for i in range(0,Nx-1):
               #Left Boundary Conditions:
               BC[48*i,0+48*i]=aA1g*(-1)
               BC[48*i,4+48*i]=1/2.*aA1gEg*(-1)
               BC[48*i,5+48*i]=-1/2.*aA1gEg*(-1)
               RHS[48*i]=aA1g*(-psi[2,i+1]+1j*delx*Ax[1,i+1]*psi[1,i+1]) + 1/2.*aA1gEg*(-D1[2,i+1]+1j*delx*Ax[1,i+1]*D1[1,i+1]) - 1/2.*aA1gEg*(-D2[2,i+1]+1j*delx*Ax[1,i+1]*D2[1,i+1]) - aA1gT1g*(-delx/dely*G1[1,2]+1j*delx*Ay[1,1]*G1[1,1])*Kronecker[i+1,1] - aA1gT1g*(-delx/dely*(G1[1,i+2]-G1[1,i])+1j*delx*Ay[1,i+1]*G1[1,i+1])*(1-Kronecker[i+1,1]-Kronecker[i+1,Nx-1]) - aA1gT1g*(delx/dely*G1[1,Nx-2]+1j*delx*Ay[1,Nx-1]*G1[1,Nx-1])*Kronecker[i+1,Nx-1]    + aA1gT2g*(-delx/dely*D3[1,2]+1j*delx*Ay[1,1]*D3[1,1])*Kronecker[i+1,1] + aA1gT2g*(-delx/dely*(D3[1,i+2]-D3[1,i])+1j*delx*Ay[1,i+1]*D3[1,i+1])*(1-Kronecker[i+1,1]-Kronecker[i+1,Nx-1]) + aA1gT2g*(delx/dely*D3[1,Ny-2]+1j*delx*Ay[1,Ny-1]*D3[1,Ny-1])*Kronecker[i+1,Ny-1]      + aA1gT1u*Px[1,i+1] 
               BC[48*i,30]=-aA1gT1g*(-delx/dely*Kronecker[i+1,1]) 
               BC[48*i,42]=-aA1gT1g*(delx/dely*Kronecker[i+1,Ny-1])
               BC[48*i,33]=aA1gT2g*(-delx/dely*Kronecker[i+1,1]) 
               BC[48*i,45]=aA1gT2g*(delx/dely*Kronecker[i+1,Ny-1])



               BC[48*i+1,1+48*i]=aT1u1*(-1)
               RHS[48*i+1]=aT1u1*(-Px[2,i+1]+1j*delx*Ax[1,i+1]*Px[1,i+1]) + aT1u3*(-delx/dely*Py[1,2]+1j*delx*Ay[1,1]*Py[1,1])*Kronecker[i+1,1] + aT1u3*(-delx/dely*(Py[1,i+2]-Py[1,i])+1j*delx*Ay[1,i+1]*Py[1,i+1])*(1-Kronecker[i+1,1]-Kronecker[i+1,Ny-1]) + aT1u3*(delx/dely*Py[1,Ny-2]+1j*delx*Ay[1,Ny-1]*Py[1,Ny-1])*Kronecker[i+1,Ny-1]     - aA1gT1u*psi[1,i+1] - 1/2.*aT1uEg*D1[1,i+1] + 1/2.*aT1uEg*D2[1,i+1]
               BC[48*i+1,26]=aT1u3*(-delx/dely*Kronecker[i+1,1])
               BC[48*i+1,38]=aT1u3*(delx/dely*Kronecker[i+1,Ny-1])



               BC[48*i+2,2+48*i]=aT1u2*(-1)
               RHS[48*i+2]=aT1u2*(-Py[2,i+1]+1j*delx*Ax[1,i+1]*Py[1,i+1]) + aT1u4*(-delx/dely*(Px[1,i+2]*(1-Kronecker[i+1,Ny-1])-Py[1,i]*(1-Kronecker[i+1,1]))+1j*delx*Ay[1,i+1]*Py[1,i+1])       + aT1uT1g*G1[1,i+1] + aT1uT2g*D3[1,i+1]
               BC[48*i+2,25]=aT1u4*(-delx/dely*Kronecker[i+1,1])
               BC[48*i+2,37]=aT1u4*(delx/dely*Kronecker[i+1,Ny-1])



               BC[48*i+3,3+48*i]=aT1u2*(-1)
               RHS[48*i+3]=aT1u2*(-Pz[2,i+1]+1j*delx*Ax[1,i+1]*Pz[1,i+1]) + aT1uT1g*G2[1,i+1] + aT1uT2g*D4[1,i+1]



               BC[48*i+4,0+48*i]=1/2.*aA1gEg*(-1)
               RHS[48*i+4]=1/2.*aA1gEg*(-psi[2,i+1]+1j*delx*Ax[1,i+1]*psi[1,i+1])  + (aEg1-aEg2)*(-D1[2,i+1]+1j*delx*Ax[1,i+1]*D1[1,i+1])    + sqrt(3)*aEg2*(-D2[2,i+1]+1j*delx*Ax[1,i+1]*D2[1,i+1])    -1/3.*(aEgT1g1+aEgT1g2)*(-delx/dely*(G1[1,i+2]*(1-Kronecker[i+1,Ny-1])-G1[1,i]*(1-Kronecker[i+1,1]))+1j*delx*Ay[1,i+1]*G1[1,i+1])      + 1/3.*(aEgT2g1-aEgT2g2)*(-delx/dely*(D3[1,i+2]*(1-Kronecker[i+1,Ny-1])-D3[1,i]*(1-Kronecker[i+1,1]))+1j*delx*Ay[1,i+1]*D3[1,i+1])     + 1/2.*aT1uEg*Px[1,i+1]
               BC[48*i+4,4+48*i]=(aEg1-aEg2)*(-1)
               BC[48*i+4,5+48*i]=sqrt(3)*aEg2*(-1)
               BC[48*i+4,30]=-1/3.*(aEgT1g1+aEgT1g2)*(-delx/dely*Kronecker[i+1,1])
               BC[48*i+4,42]=-1/3.*(aEgT1g1+aEgT1g2)*(delx/dely*Kronecker[i+1,Ny-1])
               BC[48*i+4,33]=1/3.*(aEgT2g1-aEgT2g2)*(-delx/dely*Kronecker[i+1,1])
               BC[48*i+4,45]=1/3.*(aEgT2g1-aEgT2g2)*(delx/dely*Kronecker[i+1,Ny-1])



               BC[48*i+5,0+48*i]=-1/2.*aA1gEg*(-1)
               RHS[48*i+5]=-1/2.*aA1gEg*(-psi[2,i+1]+1j*delx*Ax[1,i+1]*psi[1,i+1])  + (aEg1+aEg2)*(-D2[2,i+1]+1j*delx*Ax[1,i+1]*D2[1,i+1])    + sqrt(3)*aEg2*(-D1[2,i+1]+1j*delx*Ax[1,i+1]*D1[1,i+1])    +(aEgT1g1-aEgT1g2)*(-delx/dely*(G1[1,i+2]*(1-Kronecker[i+1,Ny-1])-G1[1,i]*(1-Kronecker[i+1,1]))+1j*delx*Ay[1,i+1]*G1[1,i+1])      - (aEgT2g1+aEgT2g2)*(-delx/dely*(D3[1,i+2]*(1-Kronecker[i+1,Ny-1])-D3[1,i]*(1-Kronecker[i+1,1]))+1j*delx*Ay[1,i+1]*D3[1,i+1])     - 1/2.*aT1uEg*Px[1,i+1]
               BC[48*i+5,5+48*i]=(aEg1+aEg2)*(-1)
               BC[48*i+5,4+48*i]=sqrt(3)*aEg2*(-1)
               BC[48*i+5,30]=(aEgT1g1-aEgT1g2)*(-delx/dely*Kronecker[i+1,1])
               BC[48*i+5,42]=(aEgT1g1-aEgT1g2)*(delx/dely*Kronecker[i+1,Ny-1])
               BC[48*i+5,33]=-(aEgT2g1+aEgT2g2)*(-delx/dely*Kronecker[i+1,1])
               BC[48*i+5,45]=-(aEgT2g1+aEgT2g2)*(delx/dely*Kronecker[i+1,Ny-1])



               BC[48*i+6,24]=aA1gT1g*(-delx/dely*Kronecker[i+1,1])
               BC[48*i+6,36]=aA1gT1g*(delx/dely*Kronecker[i+1,Ny-1])
               RHS[48*i+6]=aA1gT1g*(-delx/dely*(psi[1,i+2]*(1-Kronecker[i+1,Ny-1])-psi[1,i]*(1-Kronecker[i+1,1]))+1j*delx*Ay[1,i+1]*psi[1,i+1])     -1/3.*(aEgT1g1+aEgT1g2)*(-delx/dely*(D1[1,i+2]*(1-Kronecker[i+1,Ny-1])-D1[1,i]*(1-Kronecker[i+1,1]))+1j*delx*Ay[1,i+1]*D1[1,i+1])      -(aEgT1g1-aEgT1g2)*(-delx/dely*(D2[1,i+2]*(1-Kronecker[i+1,Ny-1])-D2[1,i]*(1-Kronecker[i+1,1]))+1j*delx*Ay[1,i+1]*D2[1,i+1])     +aT1g1*(-G1[2,i+1]+1j*delx*Ax[1,i+1]*G1[1,i+1])    +aT1g3*(-delx/dely*(G2[1,i+2]*(1-Kronecker[i+1,Ny-1])-G2[1,i]*(1-Kronecker[i+1,1]))+1j*delx*Ay[1,i+1]*G2[1,i+1])      -aT1gT2g1*(-D3[2,i+1]+1j*delx*Ax[1,i+1]*D3[1,i+1])    -aT1uT1g*Py[1,i+1]
               BC[48*i+6,28]=-1/3.*(aEgT1g1+aEgT1g2)*(-delx/dely*Kronecker[i+1,1])
               BC[48*i+6,40]=-1/3.*(aEgT1g1+aEgT1g2)*(delx/dely*Kronecker[i+1,Ny-1])
               BC[48*i+6,29]=-(aEgT1g1-aEgT1g2)*(-delx/dely*Kronecker[i+1,1])
               BC[48*i+6,41]=-(aEgT1g1-aEgT1g2)*(delx/dely*Kronecker[i+1,Ny-1])
               BC[48*i+6,6+48*i]=aT1g1*(-1)
               BC[48*i+6,31]=aT1g3*(-delx/dely*Kronecker[i+1,1])
               BC[48*i+6,43]=aT1g3*(delx/dely*Kronecker[i+1,Ny-1])
               BC[48*i+6,9+48*i]=-aT1gT2g1*(-1)



               BC[48*i+7,7+48*i]=aT1g2*(-1)
               RHS[48*i+7]=aT1g2*(-G2[2,i+1]+1j*delx*Ax[1,i+1]*G2[1,i+1])     +aT1g4*(-delx/dely*(G1[1,i+2]*(1-Kronecker[i+1,Ny-1])-G1[1,i]*(1-Kronecker[i+1,1]))+1j*delx*Ay[1,i+1]*G1[1,i+1])      -aT1gT2g1*(-D4[2,i+1]+1j*delx*Ax[1,i+1]*D4[1,i+1])-aT1gT2g2*(-delx/dely*(D5[1,i+2]*(1-Kronecker[i+1,Ny-1])-D5[1,i]*(1-Kronecker[i+1,1]))+1j*delx*Ay[1,i+1]*D5[1,i+1])    -aT1uT1g*Pz[1,i+1]
               BC[48*i+7,30]=aT1g4*(-delx/dely*Kronecker[i+1,1])
               BC[48*i+7,42]=aT1g4*(delx/dely*Kronecker[i+1,Ny-1])
               BC[48*i+7,10+48*i]=-aT1gT2g1*(-1)
               BC[48*i+7,35]=-aT1gT2g2*(-delx/dely*Kronecker[i+1,1])
               BC[48*i+7,47]=-aT1gT2g2*(delx/dely*Kronecker[i+1,Ny-1])



               BC[48*i+8,8+48*i]=aT1g2*(-1)   
               RHS[48*i+8]=aT1g2*(-G3[2,i+1]+1j*delx*Ax[1,i+1]*G3[1,i+1])      -aT1gT2g3*(-delx/dely*(D4[1,i+2]*(1-Kronecker[i+1,Ny-1])-D4[1,i]*(1-Kronecker[i+1,1]))+1j*delx*Ay[1,i+1]*D4[1,i+1])
               BC[48*i+8,34]=-aT1gT2g3*(-delx/dely*Kronecker[i+1,1])
               BC[48*i+8,46]=-aT1gT2g3*(delx/dely*Kronecker[i+1,Ny-1])



               BC[48*i+9,24]=aA1gT2g*(-delx/dely*Kronecker[i+1,1])
               BC[48*i+9,36]=aA1gT2g*(delx/dely*Kronecker[i+1,Ny-1])
               RHS[48*i+9]=aA1gT2g*(-delx/dely*(psi[1,i+2]*(1-Kronecker[i+1,Ny-1])-psi[1,i]*(1-Kronecker[i+1,1]))+1j*delx*Ay[1,i+1]*psi[1,i+1])     -1/3.*(aEgT2g1-aEgT2g2)*(-delx/dely*(D1[1,i+2]*(1-Kronecker[i+1,Ny-1])-D1[1,i]*(1-Kronecker[i+1,1]))+1j*delx*Ay[1,i+1]*D1[1,i+1])      -(aEgT2g1+aEgT2g2)*(-delx/dely*(D2[1,i+2]*(1-Kronecker[i+1,Ny-1])-D2[1,i]*(1-Kronecker[i+1,1]))+1j*delx*Ay[1,i+1]*D2[1,i+1]) -aT1gT2g1*(-G1[2,i+1]+1j*delx*Ax[1,i+1]*G1[1,i+1])      +aT2g1*(-D3[2,i+1]+1j*delx*Ax[1,i+1]*D3[1,i+1])    +aT2g3*(-delx/dely*(D4[1,i+2]*(1-Kronecker[i+1,Ny-1])-D4[1,i]*(1-Kronecker[i+1,1]))+1j*delx*Ay[1,i+1]*D4[1,i+1])          -aT1uT2g*Py[1,i+1]
               BC[48*i+9,28]=-1/3.*(aEgT2g1-aEgT2g2)*(-delx/dely*Kronecker[i+1,1])
               BC[48*i+9,40]=-1/3.*(aEgT2g1-aEgT2g2)*(delx/dely*Kronecker[i+1,Ny-1])
               BC[48*i+9,29]=-(aEgT2g1+aEgT2g2)*(-delx/dely*Kronecker[i+1,1])
               BC[48*i+9,41]=-(aEgT2g1+aEgT2g2)*(delx/dely*Kronecker[i+1,Ny-1])
               BC[48*i+9,6+48*i]=-aT1gT2g1*(-1)
               BC[48*i+9,9+48*i]=aT2g1*(-1)
               BC[48*i+9,34]=aT2g3*(-delx/dely*Kronecker[i+1,1])
               BC[48*i+9,46]=aT2g3*(delx/dely*Kronecker[i+1,Ny-1])



               BC[48*i+10,7+48*i]=-aT1gT2g1*(-1)
               BC[48*i+10,10+48*i]=aT2g2*(-1)
               RHS[48*i+10]=-aT1gT2g1*(-G2[2,i+1]+1j*delx*Ax[1,i+1]*G2[1,i+1])     +aT2g2*(-D4[2,i+1]+1j*delx*Ax[1,i+1]*D4[1,i+1])     +aT2g4*(-delx/dely*(D3[1,i+2]*(1-Kronecker[i+1,Ny-1])-D3[1,i]*(1-Kronecker[i+1,1]))+1j*delx*Ay[1,i+1]*D3[1,i+1])      -aT1uT2g*Pz[1,i+1]
               BC[48*i+10,33]=aT2g4*(-delx/dely*Kronecker[i+1,1])
               BC[48*i+10,45]=aT2g4*(delx/dely*Kronecker[i+1,Ny-1])



               BC[48*i+11,31]=-aT1gT2g3*(-delx/dely*Kronecker[i+1,1])
               BC[48*i+11,43]=-aT1gT2g3*(delx/dely*Kronecker[i+1,Ny-1])
               RHS[48*i+11]=-aT1gT2g3*(-delx/dely*(G2[1,i+2]*(1-Kronecker[i+1,Ny-1])-G2[1,i]*(1-Kronecker[i+1,1]))+1j*delx*Ay[1,i+1]*G2[1,i+1])      +aT2g2*(-D5[2,i+1]+1j*delx*Ax[1,i+1]*D5[1,i+1])      
               BC[48*i+11,11+48*i]=aT2g2*(-1)   


               #Right Boundary Conditions:
               BC[48*i+12,0+48*i+12]=aA1g*(1)
               BC[48*i+12,4+48*i+12]=1/2.*aA1gEg*(1)
               BC[48*i+12,5+48*i+12]=-1/2.*aA1gEg*(1)
               RHS[48*i+12]=aA1g*(psi[Nx-2,i+1]+1j*delx*Ax[Nx-1,i+1]*psi[Nx-1,i+1]) + 1/2.*aA1gEg*(D1[Nx-2,i+1]+1j*delx*Ax[Nx-1,i+1]*D1[Nx-1,i+1]) - 1/2.*aA1gEg*(D2[Nx-2,i+1]+1j*delx*Ax[Nx-1,i+1]*D2[Nx-1,i+1]) - aA1gT1g*(-delx/dely*G1[Nx-1,2]+1j*delx*Ay[Nx-1,1]*G1[Nx-1,1])*Kronecker[i+1,1] - aA1gT1g*(-delx/dely*(G1[Nx-1,i+2]-G1[Nx-1,i])+1j*delx*Ay[Nx-1,i+1]*G1[Nx-1,i+1])*(1-Kronecker[i+1,1]-Kronecker[i+1,Nx-1]) - aA1gT1g*(delx/dely*G1[Nx-1,Nx-2]+1j*delx*Ay[Nx-1,Nx-1]*G1[Nx-1,Nx-1])*Kronecker[i+1,Nx-1]    + aA1gT2g*(-delx/dely*D3[Nx-1,2]+1j*delx*Ay[Nx-1,1]*D3[Nx-1,1])*Kronecker[i+1,1] + aA1gT2g*(-delx/dely*(D3[Nx-1,i+2]-D3[Nx-1,i])+1j*delx*Ay[Nx-1,i+1]*D3[Nx-1,i+1])*(1-Kronecker[i+1,1]-Kronecker[i+1,Ny-1]) + aA1gT2g*(delx/dely*D3[Nx-1,Ny-2]+1j*delx*Ay[Nx-1,Ny-1]*D3[Nx-1,Ny-1])*Kronecker[i+1,Ny-1]      + aA1gT1u*Px[Nx-1,i+1] 
               BC[48*i+12,30+(Nx-2)*48]=-aA1gT1g*(-delx/dely*Kronecker[i+1,1]) 
               BC[48*i+12,42+(Nx-2)*48]=-aA1gT1g*(-delx/dely*Kronecker[i+1,Ny-1])
               BC[48*i+12,33+(Nx-2)*48]=aA1gT2g*(-delx/dely*Kronecker[i+1,1]) 
               BC[48*i+12,45+(Nx-2)*48]=aA1gT2g*(delx/dely*Kronecker[i+1,Ny-1])



               BC[48*i+13,1+48*i+12]=aT1u1*(1)
               RHS[48*i+13]=aT1u1*(Px[Nx-2,i+1]+1j*delx*Ax[Nx-1,i+1]*Px[Nx-1,i+1]) + aT1u3*(-delx/dely*Py[Nx-1,2]+1j*delx*Ay[Nx-1,1]*Py[Nx-1,1])*Kronecker[i+1,1] + aT1u3*(-delx/dely*(Py[Nx-1,i+2]-Py[Nx-1,i])+1j*delx*Ay[Nx-1,i+1]*Py[Nx-1,i+1])*(1-Kronecker[i+1,1]-Kronecker[i+1,Ny-1]) + aT1u3*(delx/dely*Py[Nx-1,Ny-2]+1j*delx*Ay[Nx-1,Ny-1]*Py[Nx-1,Ny-1])*Kronecker[i+1,Ny-1]     - aA1gT1u*psi[Nx-1,i+1] - 1/2.*aT1uEg*D1[Nx-1,i+1] + 1/2.*aT1uEg*D2[Nx-1,i+1]
               BC[48*i+13,26+48*(Nx-2)]=aT1u3*(-delx/dely*Kronecker[i+1,1])
               BC[48*i+13,38+48*(Nx-2)]=aT1u3*(delx/dely*Kronecker[i+1,Ny-1])



               BC[48*i+14,2+48*i+12]=aT1u2*(1)
               RHS[48*i+14]=aT1u2*(Py[Nx-2,i+1]+1j*delx*Ax[Nx-1,i+1]*Py[Nx-1,i+1]) + aT1u4*(-delx/dely*(Px[Nx-1,i+2]*(1-Kronecker[i+1,Ny-1])-Py[Nx-1,i]*(1-Kronecker[i+1,1]))+1j*delx*Ay[Nx-1,i+1]*Py[Nx-1,i+1])       + aT1uT1g*G1[Nx-1,i+1] + aT1uT2g*D3[Nx-1,i+1]
               BC[48*i+14,25+(Nx-2)*48]=aT1u4*(-delx/dely*Kronecker[i+1,1])
               BC[48*i+14,37+(Nx-2)*48]=aT1u4*(delx/dely*Kronecker[i+1,Ny-1])



               BC[48*i+15,3+48*i+12]=aT1u2*(1)
               RHS[48*i+15]=aT1u2*(Pz[Nx-2,i+1]+1j*delx*Ax[Nx-1,i+1]*Pz[Nx-1,i+1]) + aT1uT1g*G2[Nx-1,i+1] + aT1uT2g*D4[Nx-1,i+1]



               BC[48*i+16,0+48*i+12]=1/2.*aA1gEg*(1)
               RHS[48*i+16]=1/2.*aA1gEg*(psi[Nx-2,i+1]+1j*delx*Ax[Nx-1,i+1]*psi[Nx-1,i+1])  + (aEg1-aEg2)*(D1[Nx-2,i+1]+1j*delx*Ax[Nx-1,i+1]*D1[Nx-1,i+1])    + sqrt(3)*aEg2*(D2[Nx-2,i+1]+1j*delx*Ax[Nx-1,i+1]*D2[Nx-1,i+1])    -1/3.*(aEgT1g1+aEgT1g2)*(-delx/dely*(G1[Nx-1,i+2]*(1-Kronecker[i+1,Ny-1])-G1[Nx-1,i]*(1-Kronecker[i+1,1]))+1j*delx*Ay[Nx-1,i+1]*G1[Nx-1,i+1])      + 1/3.*(aEgT2g1-aEgT2g2)*(-delx/dely*(D3[Nx-1,i+2]*(1-Kronecker[i+1,Ny-1])-D3[Nx-1,i]*(1-Kronecker[i+1,1]))+1j*delx*Ay[Nx-1,i+1]*D3[Nx-1,i+1])     + 1/2.*aT1uEg*Px[Nx-1,i+1]
               BC[48*i+16,4+48*i+12]=(aEg1-aEg2)*(1)
               BC[48*i+16,5+48*i+12]=sqrt(3)*aEg2*(1)
               BC[48*i+16,30+48*(Nx-2)]=-1/3.*(aEgT1g1+aEgT1g2)*(-delx/dely*Kronecker[i+1,1])
               BC[48*i+16,42+48*(Nx-2)]=-1/3.*(aEgT1g1+aEgT1g2)*(delx/dely*Kronecker[i+1,Ny-1])
               BC[48*i+16,33+48*(Nx-2)]=1/3.*(aEgT2g1-aEgT2g2)*(-delx/dely*Kronecker[i+1,1])
               BC[48*i+16,45+48*(Nx-2)]=1/3.*(aEgT2g1-aEgT2g2)*(delx/dely*Kronecker[i+1,Ny-1])



               BC[48*i+17,0+48*i+12]=-1/2.*aA1gEg*(1)
               RHS[48*i+17]=-1/2.*aA1gEg*(psi[Nx-2,i+1]+1j*delx*Ax[Nx-1,i+1]*psi[Nx-1,i+1])  + (aEg1+aEg2)*(-D2[Nx-2,i+1]+1j*delx*Ax[Nx-1,i+1]*D2[Nx-1,i+1])    + sqrt(3)*aEg2*(-D1[Nx-2,i+1]+1j*delx*Ax[Nx-1,i+1]*D1[Nx-1,i+1])    +(aEgT1g1-aEgT1g2)*(-delx/dely*(G1[Nx-1,i+2]*(1-Kronecker[i+1,Ny-1])-G1[Nx-1,i]*(1-Kronecker[i+1,1]))+1j*delx*Ay[Nx-1,i+1]*G1[Nx-1,i+1])      - (aEgT2g1+aEgT2g2)*(-delx/dely*(D3[Nx-1,i+2]*(1-Kronecker[i+1,Ny-1])-D3[Nx-1,i]*(1-Kronecker[i+1,1]))+1j*delx*Ay[Nx-1,i+1]*D3[Nx-1,i+1])     - 1/2.*aT1uEg*Px[Nx-1,i+1]
               BC[48*i+17,5+48*i+12]=(aEg1+aEg2)*(1)
               BC[48*i+17,4+48*i+12]=sqrt(3)*aEg2*(1)
               BC[48*i+17,30+48*(Nx-2)]=(aEgT1g1-aEgT1g2)*(-delx/dely*Kronecker[i+1,1])
               BC[48*i+17,42+48*(Nx-2)]=(aEgT1g1-aEgT1g2)*(delx/dely*Kronecker[i+1,Ny-1])
               BC[48*i+17,33+48*(Nx-2)]=-(aEgT2g1+aEgT2g2)*(-delx/dely*Kronecker[i+1,1])
               BC[48*i+17,45+48*(Nx-2)]=-(aEgT2g1+aEgT2g2)*(delx/dely*Kronecker[i+1,1])



               BC[48*i+18,24+48*(Nx-2)]=aA1gT1g*(-delx/dely*Kronecker[i+1,1])
               BC[48*i+18,36+48*(Nx-2)]=aA1gT1g*(delx/dely*Kronecker[i+1,Ny-1])
               RHS[48*i+18]=aA1gT1g*(-delx/dely*(psi[Nx-1,i+2]*(1-Kronecker[i+1,Ny-1])-psi[Nx-1,i]*(1-Kronecker[i+1,1]))+1j*delx*Ay[Nx-1,i+1]*psi[Nx-1,i+1])     -1/3.*(aEgT1g1+aEgT1g2)*(-delx/dely*(D1[Nx-1,i+2]*(1-Kronecker[i+1,Ny-1])-D1[Nx-1,i]*(1-Kronecker[i+1,1]))+1j*delx*Ay[Nx-1,i+1]*D1[Nx-1,i+1])      -(aEgT1g1-aEgT1g2)*(-delx/dely*(D2[Nx-1,i+2]*(1-Kronecker[i+1,Ny-1])-D2[Nx-1,i]*(1-Kronecker[i+1,1]))+1j*delx*Ay[Nx-1,i+1]*D2[Nx-1,i+1])+aT1g1*(G1[Nx-2,i+1]+1j*delx*Ax[Nx-1,i+1]*G1[Nx-1,i+1])    +aT1g3*(-delx/dely*(G2[Nx-1,i+2]*(1-Kronecker[i+1,Ny-1])-G2[Nx-1,i]*(1-Kronecker[i+1,1]))+1j*delx*Ay[Nx-1,i+1]*G2[Nx-1,i+1])      -aT1gT2g1*(D3[Nx-2,i+1]+1j*delx*Ax[Nx-1,i+1]*D3[Nx-1,i+1])    -aT1uT1g*Py[Nx-1,i+1]
               BC[48*i+18,28+48*(Nx-2)]=-1/3.*(aEgT1g1+aEgT1g2)*(-delx/dely*Kronecker[i+1,1])
               BC[48*i+18,40+48*(Nx-2)]=-1/3.*(aEgT1g1+aEgT1g2)*(delx/dely*Kronecker[i+1,Ny-1])
               BC[48*i+18,29+48*(Nx-2)]=-(aEgT1g1-aEgT1g2)*(-delx/dely*Kronecker[i+1,1])
               BC[48*i+18,41+48*(Nx-2)]=-(aEgT1g1-aEgT1g2)*(delx/dely*Kronecker[i+1,Ny-1])
               BC[48*i+18,6+48*i+12]=aT1g1*(1)
               BC[48*i+18,31+48*(Nx-2)]=aT1g3*(-delx/dely*Kronecker[i+1,1])
               BC[48*i+18,43+48*(Nx-2)]=aT1g3*(delx/dely*Kronecker[i+1,Ny-1])
               BC[48*i+18,9+48*i+12]=-aT1gT2g1*(1)



               BC[48*i+19,7+48*i+12]=aT1g2*(1)
               RHS[48*i+19]=aT1g2*(G2[Nx-2,i+1]+1j*delx*Ax[Nx-1,i+1]*G2[Nx-1,i+1])     +aT1g4*(-delx/dely*(G1[Nx-1,i+2]*(1-Kronecker[i+1,Ny-1])-G1[Nx-1,i]*(1-Kronecker[i+1,1]))+1j*delx*Ay[Nx-1,i+1]*G1[Nx-1,i+1])      -aT1gT2g1*(D4[Nx-2,i+1]+1j*delx*Ax[Nx-1,i+1]*D4[Nx-1,i+1])-aT1gT2g2*(-delx/dely*(D5[Nx-1,i+2]*(1-Kronecker[i+1,Ny-1])-D5[Nx-1,i]*(1-Kronecker[i+1,1]))+1j*delx*Ay[Nx-1,i+1]*D5[Nx-1,i+1])    -aT1uT1g*Pz[Nx-1,i+1]
               BC[48*i+19,30+48*(Nx-2)]=aT1g4*(-delx/dely*Kronecker[i+1,1])
               BC[48*i+19,42+48*(Nx-2)]=aT1g4*(delx/dely*Kronecker[i+1,Ny-1])
               BC[48*i+19,10+48*i+12]=-aT1gT2g1*(1)
               BC[48*i+19,35+48*(Nx-2)]=-aT1gT2g2*(-delx/dely*Kronecker[i+1,1])
               BC[48*i+19,47+48*(Nx-2)]=-aT1gT2g2*(delx/dely*Kronecker[i+1,Ny-1])



               BC[48*i+20,8+48*i+12]=aT1g2*(1)   
               RHS[48*i+20]=aT1g2*(G3[Nx-2,i+1]+1j*delx*Ax[Nx-1,i+1]*G3[Nx-1,i+1])      -aT1gT2g3*(-delx/dely*(D4[Nx-1,i+2]*(1-Kronecker[i+1,Ny-1])-D4[Nx-1,i]*(1-Kronecker[i+1,1]))+1j*delx*Ay[Nx-1,i+1]*D4[Nx-1,i+1])
               BC[48*i+20,34+48*(Nx-2)]=-aT1gT2g3*(-delx/dely*Kronecker[i+1,1])
               BC[48*i+20,46+48*(Nx-2)]=-aT1gT2g3*(delx/dely*Kronecker[i+1,Ny-1])



               BC[48*i+21,24+48*(Nx-2)]=aA1gT2g*(-delx/dely*Kronecker[i+1,1])
               BC[48*i+21,36+48*(Nx-2)]=aA1gT2g*(delx/dely*Kronecker[i+1,Ny-1])
               RHS[48*i+21]=aA1gT2g*(-delx/dely*(psi[Nx-1,i+2]*(1-Kronecker[i+1,Ny-1])-psi[Nx-1,i]*(1-Kronecker[i+1,1]))+1j*delx*Ay[Nx-1,i+1]*psi[Nx-1,i+1])     -1/3.*(aEgT2g1-aEgT2g2)*(-delx/dely*(D1[Nx-1,i+2]*(1-Kronecker[i+1,Ny-1])-D1[Nx-1,i]*(1-Kronecker[i+1,1]))+1j*delx*Ay[Nx-1,i+1]*D1[Nx-1,i+1])      -(aEgT2g1+aEgT2g2)*(-delx/dely*(D2[Nx-1,i+2]*(1-Kronecker[i+1,Ny-1])-D2[Nx-1,i]*(1-Kronecker[i+1,1]))+1j*delx*Ay[Nx-1,i+1]*D2[Nx-1,i+1]) -aT1gT2g1*(G1[Nx-2,i+1]+1j*delx*Ax[Nx-1,i+1]*G1[Nx-1,i+1])      +aT2g1*(D3[Nx-2,i+1]+1j*delx*Ax[Nx-1,i+1]*D3[Nx-1,i+1])    +aT2g3*(-delx/dely*(D4[Nx-1,i+2]*(1-Kronecker[i+1,Ny-1])-D4[Nx-1,i]*(1-Kronecker[i+1,1]))+1j*delx*Ay[Nx-1,i+1]*D4[Nx-1,i+1])          -aT1uT2g*Py[Nx-1,i+1]
               BC[48*i+21,28+48*(Nx-2)]=-1/3.*(aEgT2g1-aEgT2g2)*(-delx/dely*Kronecker[i+1,1])
               BC[48*i+21,40+48*(Nx-2)]=-1/3.*(aEgT2g1-aEgT2g2)*(delx/dely*Kronecker[i+1,Ny-1])
               BC[48*i+21,29+48*(Nx-2)]=-(aEgT2g1+aEgT2g2)*(-delx/dely*Kronecker[i+1,1])
               BC[48*i+21,41+48*(Nx-2)]=-(aEgT2g1+aEgT2g2)*(delx/dely*Kronecker[i+1,Ny-1])
               BC[48*i+21,6+48*i+12]=-aT1gT2g1*(1)
               BC[48*i+21,9+48*i+12]=aT2g1*(1)
               BC[48*i+21,34+48*(Nx-2)]=aT2g3*(-delx/dely*Kronecker[i+1,1])
               BC[48*i+21,46+48*(Nx-2)]=aT2g3*(delx/dely*Kronecker[i+1,Ny-1])



               BC[48*i+22,7+48*i+12]=-aT1gT2g1*(1)
               BC[48*i+22,10+48*i+12]=aT2g2*(1)
               RHS[48*i+22]=-aT1gT2g1*(G2[Nx-2,i+1]+1j*delx*Ax[Nx-1,i+1]*G2[Nx-1,i+1])     +aT2g2*(D4[Nx-2,i+1]+1j*delx*Ax[Nx-1,i+1]*D4[Nx-1,i+1])     +aT2g4*(-delx/dely*(D3[Nx-1,i+2]*(1-Kronecker[i+1,Ny-1])-D3[Nx-1,i]*(1-Kronecker[i+1,1]))+1j*delx*Ay[Nx-1,i+1]*D3[Nx-1,i+1])      -aT1uT2g*Pz[Nx-1,i+1]
               BC[48*i+22,33+48*(Nx-2)]=aT2g4*(-delx/dely*Kronecker[i+1,1])
               BC[48*i+22,45+48*(Nx-2)]=aT2g4*(delx/dely*Kronecker[i+1,Ny-1])



               BC[48*i+23,31+48*(Nx-2)]=-aT1gT2g3*(-delx/dely*Kronecker[i+1,1])
               BC[48*i+23,43+48*(Nx-2)]=-aT1gT2g3*(delx/dely*Kronecker[i+1,Ny-1])
               RHS[48*i+23]=-aT1gT2g3*(-delx/dely*(G2[Nx-1,i+2]*(1-Kronecker[i+1,Ny-1])-G2[Nx-1,i]*(1-Kronecker[i+1,1]))+1j*delx*Ay[Nx-1,i+1]*G2[Nx-1,i+1])      +aT2g2*(D5[Nx-2,i+1]+1j*delx*Ax[Nx-1,i+1]*D5[Nx-1,i+1])      
               BC[48*i+23,11+48*i+12]=aT2g2*(1)   



               #Bottom Boundary Conditions:
               BC[48*i+24,24+48*i]=aA1g*(-1)
               BC[48*i+24,28+48*i]=-1/2.*aA1gEg*(-1)
               BC[48*i+24,29+48*i]=-1/2.*aA1gEg*(-1)
               RHS[48*i+24]=aA1g*(-psi[i+1,2]+1j*dely*Ay[i+1,1]*psi[i+1,1]) - 1/2.*aA1gEg*(-D1[i+1,2]+1j*dely*Ay[i+1,1]*D1[i+1,1]) - 1/2.*aA1gEg*(-D2[i+1,2]+1j*dely*Ay[i+1,1]*D2[i+1,1])    + aA1gT1g*(-dely/delx*(G1[i+2,1]*(1-Kronecker[i+1,Nx-1])-G1[i,1]*(1-Kronecker[i+1,1]))+1j*dely*Ax[i+1,1]*G1[i+1,1])      + aA1gT2g*(-dely/delx*(D3[i+2,1]*(1-Kronecker[i+1,Nx-1])-D3[i,1]*(1-Kronecker[i+1,1]))+1j*dely*Ax[i+1,1]*D3[i+1,1])      + aA1gT1u*Py[i+1,1] 
               BC[48*i+24,6]=aA1gT1g*(-dely/delx*Kronecker[i+1,1]) 
               BC[48*i+24,18]=aA1gT1g*(dely/delx*Kronecker[i+1,Nx-1])
               BC[48*i+24,9]=aA1gT2g*(-dely/delx*Kronecker[i+1,1]) 
               BC[48*i+24,21]=aA1gT2g*(dely/delx*Kronecker[i+1,Nx-1])



               BC[48*i+25,25+48*i]=aT1u2*(-1)
               RHS[48*i+25]=aT1u2*(-Px[i+1,2]+1j*dely*Ay[i+1,1]*Px[i+1,1])   - aT1uT1g*G1[i+1,1]   +aT1uT2g*D3[i+1,1]



               BC[48*i+26,26+48*i]=aT1u1*(-1)
               RHS[48*i+26]=aT1u1*(-Py[i+1,2]+1j*dely*Ay[i+1,1]*Py[i+1,1])     +aT1u3*(-dely/delx*(Px[i+2,1]*(1-Kronecker[i+1,Nx-1])-Px[i,1]*(1-Kronecker[i+1,1]))+1j*dely*Ax[i+1,1]*Px[i+1,1])     -aA1gT1u*psi[i+1,1]   + 1/2.*aT1uEg*D1[i+1,1]   +1/2.*aT1uEg*D2[i+1,1]
               BC[48*i+26,1]=aT1u3*(-dely/delx*Kronecker[i+1,1]) 
               BC[48*i+26,13]=aT1u3*(dely/delx*Kronecker[i+1,Nx-1])



               BC[48*i+27,27+48*i]=aT1u2*(-1)
               RHS[48*i+27]=aT1u2*(-Pz[i+1,2]+1j*dely*Ay[i+1,1]*Pz[i+1,1])   + aT1uT1g*G3[i+1,1] +aT1uT2g*D4[i+1,1]



               BC[48*i+28,24+48*i]=-1/2.*aA1gEg*(-1)
               RHS[48*i+28]=-1/2.*aA1gEg*(-psi[i+1,2]+1j*dely*Ay[i+1,1]*psi[i+1,1])  + (aEg1-aEg2)*(-D1[i+1,2]+1j*dely*Ay[i+1,1]*D1[i+1,1])    - sqrt(3)*aEg2*(-D2[i+1,2]+1j*dely*Ay[i+1,1]*D2[i+1,1])    -1/3.*(aEgT1g1+aEgT1g2)*(-dely/delx*(G1[i+2,1]*(1-Kronecker[i+1,Nx-1])-G1[i,1]*(1-Kronecker[i+1,1]))+1j*dely*Ax[i+1,1]*G1[i+1,1])      - 1/3.*(aEgT2g1-aEgT2g2)*(-dely/delx*(D3[i+2,1]*(1-Kronecker[i+1,Nx-1])-D3[i,1]*(1-Kronecker[i+1,1]))+1j*dely*Ax[i+1,1]*D3[i+1,1])     - 1/2.*aT1uEg*Py[i+1,1]
               BC[48*i+28,28+48*i]=(aEg1-aEg2)*(-1)
               BC[48*i+28,29+48*i]=-sqrt(3)*aEg2*(-1)
               BC[48*i+28,6]=-1/3.*(aEgT1g1+aEgT1g2)*(-dely/delx*Kronecker[i+1,1])
               BC[48*i+28,18]=-1/3.*(aEgT1g1+aEgT1g2)*(dely/delx*Kronecker[i+1,Nx-1])
               BC[48*i+28,9]=-1/3.*(aEgT2g1-aEgT2g2)*(-dely/delx*Kronecker[i+1,1])
               BC[48*i+28,21]=-1/3.*(aEgT2g1-aEgT2g2)*(dely/delx*Kronecker[i+1,Nx-1]) 



               BC[48*i+29,24+48*i]=-1/2.*aA1gEg*(-1)
               RHS[48*i+29]=-1/2.*aA1gEg*(-psi[i+1,2]+1j*dely*Ay[i+1,1]*psi[i+1,1])  + (aEg1+aEg2)*(-D2[i+1,2]+1j*dely*Ay[i+1,1]*D2[i+1,1])    - sqrt(3)*aEg2*(-D1[i+1,2]+1j*dely*Ay[i+1,1]*D1[i+1,1])    -(aEgT1g1-aEgT1g2)*(-dely/delx*(G1[i+2,1]*(1-Kronecker[i+1,Nx-1])-G1[i,1]*(1-Kronecker[i+1,1]))+1j*dely*Ax[i+1,1]*G1[i+1,1])      - (aEgT2g1+aEgT2g2)*(-dely/delx*(D3[i+2,1]*(1-Kronecker[i+1,Nx-1])-D3[i,1]*(1-Kronecker[i+1,1]))+1j*dely*Ax[i+1,1]*D3[i+1,1])     - 1/2.*aT1uEg*Py[i+1,1]
               BC[48*i+29,29+48*i]=(aEg1+aEg2)*(-1)
               BC[48*i+29,28+48*i]=-sqrt(3)*aEg2*(-1)
               BC[48*i+29,6]=-(aEgT1g1-aEgT1g2)*(-dely/delx*Kronecker[i+1,1])
               BC[48*i+29,18]=-(aEgT1g1-aEgT1g2)*(dely/delx*Kronecker[i+1,Nx-1])
               BC[48*i+29,9]=-(aEgT2g1+aEgT2g2)*(-dely/delx*Kronecker[i+1,1])
               BC[48*i+29,21]=-(aEgT2g1+aEgT2g2)*(dely/delx*Kronecker[i+1,Nx-1])   



               BC[48*i+30,0]=-aA1gT1g*(-dely/delx*Kronecker[i+1,1])
               BC[48*i+30,12]=-aA1gT1g*(dely/delx*Kronecker[i+1,Nx-1])
               RHS[48*i+30]=-aA1gT1g*(-dely/delx*(psi[i+2,1]*(1-Kronecker[i+1,Nx-1])-psi[i,1]*(1-Kronecker[i+1,1]))+1j*dely*Ax[i+1,1]*psi[i+1,1])     -1/3.*(aEgT1g1+aEgT1g2)*(-dely/delx*(D1[i+2,1]*(1-Kronecker[i+1,Nx-1])-D1[i,1]*(1-Kronecker[i+1,1]))+1j*dely*Ax[i+1,1]*D1[i+1,1])      +(aEgT1g1-aEgT1g2)*(-dely/delx*(D2[i+2,1]*(1-Kronecker[i+1,Nx-1])-D2[i,1]*(1-Kronecker[i+1,1]))+1j*dely*Ax[i+1,1]*D2[i+1,1])+aT1g2*(-G1[i+1,2]+1j*dely*Ay[i+1,1]*G1[i+1,1])    +aT1g4*(-dely/delx*(G2[i+2,1]*(1-Kronecker[i+1,Nx-1])-G2[i,1]*(1-Kronecker[i+1,1]))+1j*dely*Ax[i+1,1]*G2[i+1,1])      +aT1gT2g1*(-D3[i+1,2]+1j*dely*Ay[i+1,1]*D3[i+1,1])    +aT1uT1g*Px[i+1,1]
               BC[48*i+30,4]=-1/3.*(aEgT1g1+aEgT1g2)*(-dely/delx*Kronecker[i+1,1])
               BC[48*i+30,16]=-1/3.*(aEgT1g1+aEgT1g2)*(dely/delx*Kronecker[i+1,Nx-1])
               BC[48*i+30,5]=(aEgT1g1-aEgT1g2)*(-dely/delx*Kronecker[i+1,1])
               BC[48*i+30,17]=(aEgT1g1-aEgT1g2)*(dely/delx*Kronecker[i+1,Nx-1])
               BC[48*i+30,30+48*i]=aT1g2*(-1)
               BC[48*i+30,7]=aT1g4*(-dely/delx*Kronecker[i+1,1])
               BC[48*i+30,19]=aT1g4*(dely/delx*Kronecker[i+1,Nx-1])
               BC[48*i+30,33+48*i]=aT1gT2g1*(-1) 


               BC[48*i+31,6]=aT1g3*(-dely/delx*Kronecker[i+1,1])
               BC[48*i+31,18]=aT1g3*(dely/delx*Kronecker[i+1,Nx-1])
               RHS[48*i+31]=aT1g3*(-dely/delx*(G1[i+2,1]*(1-Kronecker[i+1,Nx-1])-G1[i,1]*(1-Kronecker[i+1,1]))+1j*dely*Ax[i+1,1]*G1[i+1,1])     +aT1g1*(-G2[i+1,2]+1j*dely*Ay[i+1,1]*G2[i+1,1])    -aT1gT2g3*(-dely/delx*(D5[i+2,1]*(1-Kronecker[i+1,Nx-1])-D5[i,1]*(1-Kronecker[i+1,1]))+1j*dely*Ax[i+1,1]*D5[i+1,1])
               BC[48*i+31,31+48*i]=aT1g1*(-1)
               BC[48*i+31,11]=-aT1gT2g3*(-dely/delx*Kronecker[i+1,1])
               BC[48*i+31,23]=-aT1gT2g3*(dely/delx*Kronecker[i+1,Nx-1])



               BC[48*i+32,32+48*i]=aT1g2*(-1)
               RHS[48*i+32]=aT1g2*(-G3[i+1,2]+1j*dely*Ay[i+1,1]*G3[i+1,1])    -aT1gT2g1*(-D5[i+1,2]+1j*dely*Ay[i+1,1]*D5[i+1,1])    -aT1uT1g*Pz[i+1,1]
               BC[48*i+32,35+48*i]=-aT1gT2g1*(-1)



               BC[48*i+33,0]=aA1gT2g*(-dely/delx*Kronecker[i+1,1])
               BC[48*i+33,12]=aA1gT2g*(dely/delx*Kronecker[i+1,Nx-1])
               RHS[48*i+33]=aA1gT2g*(-dely/delx*(psi[i+2,1]*(1-Kronecker[i+1,Nx-1])-psi[i,1]*(1-Kronecker[i+1,1]))+1j*dely*Ax[i+1,1]*psi[i+1,1])     +1/3.*(aEgT2g1-aEgT2g2)*(-dely/delx*(D1[i+2,1]*(1-Kronecker[i+1,Nx-1])-D1[i,1]*(1-Kronecker[i+1,1]))+1j*dely*Ax[i+1,1]*D1[i+1,1])      -(aEgT2g1+aEgT2g2)*(-dely/delx*(D2[i+2,1]*(1-Kronecker[i+1,Nx-1])-D2[i,1]*(1-Kronecker[i+1,1]))+1j*dely*Ax[i+1,1]*D2[i+1,1])+aT1gT2g1*(-G1[i+1,2]+1j*dely*Ay[i+1,1]*G1[i+1,1])   +aT2g2*(-D3[i+1,2]+1j*dely*Ay[i+1,1]*D3[i+1,1])    +aT2g4*(-dely/delx*(D4[i+2,1]*(1-Kronecker[i+1,Nx-1])-D4[i,1]*(1-Kronecker[i+1,1]))+1j*dely*Ax[i+1,1]*D4[i+1,1])          -aT1uT2g*Px[i+1,1]
               BC[48*i+33,4]=1/3.*(aEgT2g1-aEgT2g2)*(-dely/delx*Kronecker[i+1,1])
               BC[48*i+33,16]=1/3.*(aEgT2g1-aEgT2g2)*(dely/delx*Kronecker[i+1,Nx-1])
               BC[48*i+33,5]=-(aEgT2g1+aEgT2g2)*(-dely/delx*Kronecker[i+1,1])
               BC[48*i+33,17]=-(aEgT2g1+aEgT2g2)*(dely/delx*Kronecker[i+1,Nx-1])
               BC[48*i+33,30+48*i]=aT1gT2g1*(-1)
               BC[48*i+33,33+48*i]=aT2g2*(-1)
               BC[48*i+33,10]=aT2g4*(-dely/delx*Kronecker[i+1,1])
               BC[48*i+33,22]=aT2g4*(dely/delx*Kronecker[i+1,Nx-1])



               BC[48*i+34,8]=-aT1gT2g3*(-dely/delx*Kronecker[i+1,1])
               BC[48*i+34,20]=-aT1gT2g3*(dely/delx*Kronecker[i+1,Nx-1])
               RHS[48*i+34]=-aT1gT2g3*(-dely/delx*(G3[i+2,1]*(1-Kronecker[i+1,Nx-1])-G3[i,1]*(1-Kronecker[i+1,1]))+1j*dely*Ax[i+1,1]*G3[i+1,1])    +aT2g1*(-D4[i+1,2]+1j*dely*Ay[i+1,1]*D4[i+1,1])          +aT2g3*(-dely/delx*(D3[i+2,1]*(1-Kronecker[i+1,Nx-1])-D3[i,1]*(1-Kronecker[i+1,1]))+1j*dely*Ax[i+1,1]*D3[i+1,1])       
               BC[48*i+34,34+48*i]=aT2g1*(-1)
               BC[48*i+34,9]=aT2g3*(-dely/delx*Kronecker[i+1,1])
               BC[48*i+34,21]=aT2g3*(dely/delx*Kronecker[i+1,Nx-1])



               BC[48*i+35,32+48*i]=-aT1gT2g1*(-1)
               RHS[48*i+35]=-aT1gT2g1*(-G3[i+1,2]+1j*dely*Ay[i+1,1]*G3[i+1,1])     -aT1gT2g2*(-dely/delx*(G2[i+2,1]*(1-Kronecker[i+1,Nx-1])-G2[i,1]*(1-Kronecker[i+1,1]))+1j*dely*Ax[i+1,1]*G2[i+1,1])            +aT2g2*(-D5[i+1,2]+1j*dely*Ay[i+1,1]*D5[i+1,1])    -aT1uT2g*Pz[i+1,1]
               BC[48*i+35,7]=-aT1gT2g2*(-dely/delx*Kronecker[i+1,1])
               BC[48*i+35,19]=-aT1gT2g2*(dely/delx*Kronecker[i+1,Nx-1])
               BC[48*i+35,35+48*i]=aT2g2*(-1)



               #Top Boundary Conditions:
               BC[48*i+36,36+48*i]=aA1g*(1)
               BC[48*i+36,40+48*i]=-1/2.*aA1gEg*(1)
               BC[48*i+36,41+48*i]=-1/2.*aA1gEg*(1)
               RHS[48*i+36]=aA1g*(psi[i+1,Ny-2]+1j*dely*Ay[i+1,Ny-1]*psi[i+1,Ny-1]) - 1/2.*aA1gEg*(D1[i+1,Ny-2]+1j*dely*Ay[i+1,Ny-1]*D1[i+1,Ny-1]) - 1/2.*aA1gEg*(D2[i+1,Ny-2]+1j*dely*Ay[i+1,Ny-1]*D2[i+1,Ny-1])    + aA1gT1g*(-dely/delx*(G1[i+2,Ny-1]*(1-Kronecker[i+1,Nx-1])-G1[i,Ny-1]*(1-Kronecker[i+1,1]))+1j*dely*Ax[i+1,Ny-1]*G1[i+1,Ny-1])      + aA1gT2g*(-dely/delx*(D3[i+2,Ny-1]*(1-Kronecker[i+1,Nx-1])-D3[i,Ny-1]*(1-Kronecker[i+1,1]))+1j*dely*Ax[i+1,Ny-1]*D3[i+1,Ny-1])      + aA1gT1u*Py[i+1,Ny-1] 
               BC[48*i+36,6+48*(Ny-2)]=aA1gT1g*(-dely/delx*Kronecker[i+1,1]) 
               BC[48*i+36,18+48*(Ny-2)]=aA1gT1g*(dely/delx*Kronecker[i+1,Nx-1])
               BC[48*i+36,9+48*(Ny-2)]=aA1gT2g*(-dely/delx*Kronecker[i+1,1]) 
               BC[48*i+36,21+48*(Ny-2)]=aA1gT2g*(dely/delx*Kronecker[i+1,Nx-1])



               BC[48*i+37,37+48*i]=aT1u2*(1)
               RHS[48*i+37]=aT1u2*(Px[i+1,Ny-2]+1j*dely*Ay[i+1,Ny-1]*Px[i+1,Ny-1])   - aT1uT1g*G1[i+1,Ny-1]   +aT1uT2g*D3[i+1,Ny-1]



               BC[48*i+38,38+48*i]=aT1u1*(1)
               RHS[48*i+38]=aT1u1*(Py[i+1,Ny-2]+1j*dely*Ay[i+1,Ny-1]*Py[i+1,Ny-1])     +aT1u3*(-dely/delx*(Px[i+2,Ny-1]*(1-Kronecker[i+1,Nx-1])-Px[i,Ny-1]*(1-Kronecker[i+1,1]))+1j*dely*Ax[i+1,Ny-1]*Px[i+1,Ny-1])     -aA1gT1u*psi[i+1,Ny-1]   + 1/2.*aT1uEg*D1[i+1,Ny-1]   +1/2.*aT1uEg*D2[i+1,Ny-1]
               BC[48*i+38,1+48*(Ny-2)]=aT1u3*(-dely/delx*Kronecker[i+1,1]) 
               BC[48*i+38,13+48*(Ny-2)]=aT1u3*(dely/delx*Kronecker[i+1,Nx-1])



               BC[48*i+39,39+48*i]=aT1u2*(1)
               RHS[48*i+39]=aT1u2*(Pz[i+1,Ny-2]+1j*dely*Ay[i+1,Ny-1]*Pz[i+1,Ny-1])   + aT1uT1g*G3[i+1,Ny-1] +aT1uT2g*D4[i+1,Ny-1]



               BC[48*i+40,36+48*i]=-1/2.*aA1gEg*(1)
               RHS[48*i+40]=-1/2.*aA1gEg*(psi[i+1,Ny-2]+1j*dely*Ay[i+1,Ny-1]*psi[i+1,Ny-1])  + (aEg1-aEg2)*(D1[i+1,Ny-2]+1j*dely*Ay[i+1,Ny-1]*D1[i+1,Ny-1])    - sqrt(3)*aEg2*(D2[i+1,Ny-2]+1j*dely*Ay[i+1,Ny-1]*D2[i+1,Ny-1])    -1/3.*(aEgT1g1+aEgT1g2)*(-dely/delx*(G1[i+2,Ny-1]*(1-Kronecker[i+1,Nx-1])-G1[i,Ny-1]*(1-Kronecker[i+1,1]))+1j*dely*Ax[i+1,Ny-1]*G1[i+1,Ny-1])      - 1/3.*(aEgT2g1-aEgT2g2)*(-dely/delx*(D3[i+2,Ny-1]*(1-Kronecker[i+1,Nx-1])-D3[i,Ny-1]*(1-Kronecker[i+1,1]))+1j*dely*Ax[i+1,Ny-1]*D3[i+1,Ny-1])     - 1/2.*aT1uEg*Py[i+1,Ny-1]
               BC[48*i+40,40+48*i]=(aEg1-aEg2)*(1)
               BC[48*i+40,41+48*i]=-sqrt(3)*aEg2*(1)
               BC[48*i+40,6+48*(Ny-2)]=-1/3.*(aEgT1g1+aEgT1g2)*(-dely/delx*Kronecker[i+1,1])
               BC[48*i+40,18+48*(Ny-2)]=-1/3.*(aEgT1g1+aEgT1g2)*(dely/delx*Kronecker[i+1,Nx-1])
               BC[48*i+40,9+48*(Ny-2)]=-1/3.*(aEgT2g1-aEgT2g2)*(-dely/delx*Kronecker[i+1,1])
               BC[48*i+40,21+48*(Ny-2)]=-1/3.*(aEgT2g1-aEgT2g2)*(dely/delx*Kronecker[i+1,Nx-1]) 



               BC[48*i+41,36+48*i]=-1/2.*aA1gEg*(1)
               RHS[48*i+41]=-1/2.*aA1gEg*(psi[i+1,Ny-2]+1j*dely*Ay[i+1,Ny-1]*psi[i+1,Ny-1])  + (aEg1+aEg2)*(D2[i+1,Ny-2]+1j*dely*Ay[i+1,Ny-1]*D2[i+1,Ny-1])    - sqrt(3)*aEg2*(D1[i+1,Ny-2]+1j*dely*Ay[i+1,Ny-1]*D1[i+1,Ny-1])    -(aEgT1g1-aEgT1g2)*(-dely/delx*(G1[i+2,Ny-1]*(1-Kronecker[i+1,Nx-1])-G1[i,Ny-1]*(1-Kronecker[i+1,1]))+1j*dely*Ax[i+1,Ny-1]*G1[i+1,Ny-1])      - (aEgT2g1+aEgT2g2)*(-dely/delx*(D3[i+2,Ny-1]*(1-Kronecker[i+1,Nx-1])-D3[i,Ny-1]*(1-Kronecker[i+1,1]))+1j*dely*Ax[i+1,Ny-1]*D3[i+1,Ny-1])     - 1/2.*aT1uEg*Py[i+1,Ny-1]
               BC[48*i+41,41+48*i]=(aEg1+aEg2)*(1)
               BC[48*i+41,40+48*i]=-sqrt(3)*aEg2*(1)
               BC[48*i+41,6+48*(Ny-2)]=-(aEgT1g1-aEgT1g2)*(-dely/delx*Kronecker[i+1,1])
               BC[48*i+41,18+48*(Ny-2)]=-(aEgT1g1-aEgT1g2)*(dely/delx*Kronecker[i+1,Nx-1])
               BC[48*i+41,9+48*(Ny-2)]=-(aEgT2g1+aEgT2g2)*(-dely/delx*Kronecker[i+1,1])
               BC[48*i+41,21+48*(Ny-2)]=-(aEgT2g1+aEgT2g2)*(dely/delx*Kronecker[i+1,Nx-1])   



               BC[48*i+42,0+48*(Ny-2)]=-aA1gT1g*(-dely/delx*Kronecker[i+1,1])
               BC[48*i+42,12+48*(Ny-2)]=-aA1gT1g*(dely/delx*Kronecker[i+1,Nx-1])
               RHS[48*i+42]=-aA1gT1g*(-dely/delx*(psi[i+2,Ny-1]*(1-Kronecker[i+1,Nx-1])-psi[i,Ny-1]*(1-Kronecker[i+1,1]))+1j*dely*Ax[i+1,Ny-1]*psi[i+1,Ny-1])     -1/3.*(aEgT1g1+aEgT1g2)*(-dely/delx*(D1[i+2,Ny-1]*(1-Kronecker[i+1,Nx-1])-D1[i,Ny-1]*(1-Kronecker[i+1,1]))+1j*dely*Ax[i+1,Ny-1]*D1[i+1,Ny-1])      +(aEgT1g1-aEgT1g2)*(-dely/delx*(D2[i+2,Ny-1]*(1-Kronecker[i+1,Nx-1])-D2[i,Ny-1]*(1-Kronecker[i+1,1]))+1j*dely*Ax[i+1,Ny-1]*D2[i+1,Ny-1])+aT1g2*(G1[i+1,Ny-2]+1j*dely*Ay[i+1,Ny-1]*G1[i+1,Ny-1])    +aT1g4*(-dely/delx*(G2[i+2,Ny-1]*(1-Kronecker[i+1,Nx-1])-G2[i,Ny-1]*(1-Kronecker[i+1,1]))+1j*dely*Ax[i+1,Ny-1]*G2[i+1,Ny-1])      +aT1gT2g1*(D3[i+1,Ny-2]+1j*dely*Ay[i+1,Ny-1]*D3[i+1,Ny-1])    +aT1uT1g*Px[i+1,Ny-1]
               BC[48*i+42,4+48*(Ny-2)]=-1/3.*(aEgT1g1+aEgT1g2)*(-dely/delx*Kronecker[i+1,1])
               BC[48*i+42,16+48*(Ny-2)]=-1/3.*(aEgT1g1+aEgT1g2)*(dely/delx*Kronecker[i+1,Nx-1])
               BC[48*i+42,5+48*(Ny-2)]=(aEgT1g1-aEgT1g2)*(-dely/delx*Kronecker[i+1,1])
               BC[48*i+42,17+48*(Ny-2)]=(aEgT1g1-aEgT1g2)*(dely/delx*Kronecker[i+1,Nx-1])
               BC[48*i+42,42+48*i]=aT1g2*(1)
               BC[48*i+42,7+48*(Ny-2)]=aT1g4*(-dely/delx*Kronecker[i+1,1])
               BC[48*i+42,19+48*(Ny-2)]=aT1g4*(dely/delx*Kronecker[i+1,Nx-1])
               BC[48*i+42,45+48*i]=aT1gT2g1*(1) 


               BC[48*i+43,6+48*(Ny-2)]=aT1g3*(-dely/delx*Kronecker[i+1,1])
               BC[48*i+43,18+48*(Ny-2)]=aT1g3*(dely/delx*Kronecker[i+1,Nx-1])
               RHS[48*i+43]=aT1g3*(-dely/delx*(G1[i+2,Ny-1]*(1-Kronecker[i+1,Nx-1])-G1[i,Ny-1]*(1-Kronecker[i+1,1]))+1j*dely*Ax[i+1,Ny-1]*G1[i+1,Ny-1])     +aT1g1*(G2[i+1,Ny-2]+1j*dely*Ay[i+1,Ny-1]*G2[i+1,Ny-1])    -aT1gT2g3*(-dely/delx*(D5[i+2,Ny-1]*(1-Kronecker[i+1,Nx-1])-D5[i,Ny-1]*(1-Kronecker[i+1,1]))+1j*dely*Ax[i+1,Ny-1]*D5[i+1,Ny-1])
               BC[48*i+43,43+48*i]=aT1g1*(1)
               BC[48*i+43,11+48*(Ny-2)]=-aT1gT2g3*(-dely/delx*Kronecker[i+1,1])
               BC[48*i+43,23+48*(Ny-2)]=-aT1gT2g3*(dely/delx*Kronecker[i+1,Nx-1])



               BC[48*i+44,44+48*i]=aT1g2*(1)
               RHS[48*i+44]=aT1g2*(G3[i+1,Ny-2]+1j*dely*Ay[i+1,Ny-1]*G3[i+1,Ny-1])    -aT1gT2g1*(D5[i+1,Ny-2]+1j*dely*Ay[i+1,Ny-1]*D5[i+1,Ny-1])    -aT1uT1g*Pz[i+1,Ny-1]
               BC[48*i+44,47+48*i]=-aT1gT2g1*(1)



               BC[48*i+45,0+48*(Ny-2)]=aA1gT2g*(-dely/delx*Kronecker[i+1,1])
               BC[48*i+45,12+48*(Ny-2)]=aA1gT2g*(dely/delx*Kronecker[i+1,Nx-1])
               RHS[48*i+45]=aA1gT2g*(-dely/delx*(psi[i+2,Ny-1]*(1-Kronecker[i+1,Nx-1])-psi[i,Ny-1]*(1-Kronecker[i+1,1]))+1j*dely*Ax[i+1,Ny-1]*psi[i+1,Ny-1])     +1/3.*(aEgT2g1-aEgT2g2)*(-dely/delx*(D1[i+2,Ny-1]*(1-Kronecker[i+1,Nx-1])-D1[i,Ny-1]*(1-Kronecker[i+1,1]))+1j*dely*Ax[i+1,Ny-1]*D1[i+1,Ny-1])      -(aEgT2g1+aEgT2g2)*(-dely/delx*(D2[i+2,Ny-1]*(1-Kronecker[i+1,Nx-1])-D2[i,Ny-1]*(1-Kronecker[i+1,1]))+1j*dely*Ax[i+1,Ny-1]*D2[i+1,Ny-1])+aT1gT2g1*(G1[i+1,Ny-2]+1j*dely*Ay[i+1,Ny-1]*G1[i+1,Ny-1])   +aT2g2*(D3[i+1,Ny-2]+1j*dely*Ay[i+1,Ny-1]*D3[i+1,Ny-1])    +aT2g4*(-dely/delx*(D4[i+2,Ny-1]*(1-Kronecker[i+1,Nx-1])-D4[i,Ny-1]*(1-Kronecker[i+1,1]))+1j*dely*Ax[i+1,Ny-1]*D4[i+1,Ny-1])          -aT1uT2g*Px[i+1,Ny-1]
               BC[48*i+45,4+48*(Ny-2)]=1/3.*(aEgT2g1-aEgT2g2)*(-dely/delx*Kronecker[i+1,1])
               BC[48*i+45,16+48*(Ny-2)]=1/3.*(aEgT2g1-aEgT2g2)*(dely/delx*Kronecker[i+1,Nx-1])
               BC[48*i+45,5+48*(Ny-2)]=-(aEgT2g1+aEgT2g2)*(-dely/delx*Kronecker[i+1,1])
               BC[48*i+45,17+48*(Ny-2)]=-(aEgT2g1+aEgT2g2)*(dely/delx*Kronecker[i+1,Nx-1])
               BC[48*i+45,42+48*i]=aT1gT2g1*(1)
               BC[48*i+45,45+48*i]=aT2g2*(1)
               BC[48*i+45,10+48*(Ny-2)]=aT2g4*(-dely/delx*Kronecker[i+1,1])
               BC[48*i+45,22+48*(Ny-2)]=aT2g4*(dely/delx*Kronecker[i+1,Nx-1])



               BC[48*i+46,8+48*(Ny-2)]=-aT1gT2g3*(-dely/delx*Kronecker[i+1,1])
               BC[48*i+46,20+48*(Ny-2)]=-aT1gT2g3*(dely/delx*Kronecker[i+1,Nx-1])
               RHS[48*i+46]=-aT1gT2g3*(-dely/delx*(G3[i+2,Ny-1]*(1-Kronecker[i+1,Nx-1])-G3[i,Ny-1]*(1-Kronecker[i+1,1]))+1j*dely*Ax[i+1,Ny-1]*G3[i+1,Ny-1])    +aT2g1*(D4[i+1,Ny-2]+1j*dely*Ay[i+1,Ny-1]*D4[i+1,Ny-1])          +aT2g3*(-dely/delx*(D3[i+2,Ny-1]*(1-Kronecker[i+1,Nx-1])-D3[i,Ny-1]*(1-Kronecker[i+1,1]))+1j*dely*Ax[i+1,Ny-1]*D3[i+1,Ny-1])       
               BC[48*i+46,46+48*i]=aT2g1*(1)
               BC[48*i+46,9+48*(Ny-2)]=aT2g3*(-dely/delx*Kronecker[i+1,1])
               BC[48*i+46,21+48*(Ny-2)]=aT2g3*(dely/delx*Kronecker[i+1,Nx-1])



               BC[48*i+47,44+48*i]=-aT1gT2g1*(1)
               RHS[48*i+47]=-aT1gT2g1*(G3[i+1,Ny-2]+1j*dely*Ay[i+1,Ny-1]*G3[i+1,Ny-1])     -aT1gT2g2*(-dely/delx*(G2[i+2,Ny-1]*(1-Kronecker[i+1,Nx-1])-G2[i,Ny-1]*(1-Kronecker[i+1,1]))+1j*dely*Ax[i+1,Ny-1]*G2[i+1,Ny-1])            +aT2g2*(D5[i+1,Ny-2]+1j*dely*Ay[i+1,Ny-1]*D5[i+1,Ny-1])    -aT1uT2g*Pz[i+1,Ny-1]
               BC[48*i+47,7+48*(Ny-2)]=-aT1gT2g2*(-dely/delx*Kronecker[i+1,1])
               BC[48*i+47,19+48*(Ny-2)]=-aT1gT2g2*(dely/delx*Kronecker[i+1,Nx-1])
               BC[48*i+47,47+48*i]=aT2g2*(1)


          #BCinv = np.linalg.inv(BC)
          #Sol=np.dot(BCinv,RHS)

          lu2=splu(BC)
          Sol=lu2.solve(RHS)

          Sol2=Sol.reshape(48,Nx-1)
     


          psi[0,1:Ny] = Sol2[0]
     
          Px[0,1:Ny] = Sol2[1]
          Py[0,1:Ny] = Sol2[2]
          Pz[0,1:Ny] = Sol2[3]

          D1[0,1:Ny] = Sol2[4]
          D2[0,1:Ny] = Sol2[5]

          G1[0,1:Ny] = Sol2[6]
          G2[0,1:Ny] = Sol2[7]
          G3[0,1:Ny] = Sol2[8]

          D3[0,1:Ny] = Sol2[9]
          D4[0,1:Ny] = Sol2[10]
          D5[0,1:Ny] = Sol2[11]
     

          psi[Nx,1:Ny] = Sol2[12]
     
          Px[Nx,1:Ny] = Sol2[13]
          Py[Nx,1:Ny] = Sol2[14]
          Pz[Nx,1:Ny] = Sol2[15]

          D1[Nx,1:Ny] = Sol2[16]
          D2[Nx,1:Ny] = Sol2[17]

          G1[Nx,1:Ny] = Sol2[18]
          G2[Nx,1:Ny] = Sol2[19]
          G3[Nx,1:Ny] = Sol2[20]

          D3[Nx,1:Ny] = Sol2[21]
          D4[Nx,1:Ny] = Sol2[22]
          D5[Nx,1:Ny] = Sol2[23]
     

          psi[1:Nx,0] = Sol2[24]
     
          Px[1:Nx,0] = Sol2[25]
          Py[1:Nx,0] = Sol2[26]
          Pz[1:Nx,0] = Sol2[27]

          D1[1:Nx,0] = Sol2[28]
          D2[1:Nx,0] = Sol2[29]

          G1[1:Nx,0] = Sol2[30]
          G2[1:Nx,0] = Sol2[31]
          G3[1:Nx,0] = Sol2[32]

          D3[1:Nx,0] = Sol2[33]
          D4[1:Nx,0] = Sol2[34]
          D5[1:Nx,0] = Sol2[35]




          psi[1:Nx,Ny] = Sol2[36]
     
          Px[1:Nx,Ny] = Sol2[37]
          Py[1:Nx,Ny] = Sol2[38]
          Pz[1:Nx,Ny] = Sol2[39]

          D1[1:Nx,Ny] = Sol2[40]
          D2[1:Nx,Ny] = Sol2[41]

          G1[1:Nx,Ny] = Sol2[42]
          G2[1:Nx,Ny] = Sol2[43]
          G3[1:Nx,Ny] = Sol2[44]

          D3[1:Nx,Ny] = Sol2[45]
          D4[1:Nx,Ny] = Sol2[46]
          D5[1:Nx,Ny] = Sol2[47]



     #mu=solve_mu(psi)
     """###rhs=np.zeros(N2)   #for forward difference formula in b.c.s
     divJs=delx*delx*div_Js(psi,Px,Py,Pz,D1,D2,G1,G2,G3,D3,D4,D5)
     divAt=dx(Ax)+dy(Ay)
     divAtdt=dx(np.transpose(applied_vector_potential_x_fun(xg,yg,t+dt)))+dy(np.transpose(applied_vector_potential_y_fun(xg,yg,t+dt)))
     ddivAbydt=(divAtdt-divAt)/dt*delx*delx
     for i in range(0,Nx-1):  #for forward difference formula in b.c.s
        for j in range(0,Ny-1):
            rhs[(Ny-1)*i+j] = divJs[i+1,j+1] - ddivAbydt[i+1,j+1]
     for j in range(0,Ny-1):
           rhs[(Ny-1)*0+j] = rhs[(Ny-1)*0+j] - (np.transpose(applied_vector_potential_x_fun(xgrid[0],ygrid[j+1],t+dt))-Ax[0,j+1])/dt*delx
           rhs[(Ny-1)*(Nx-2)+j] = rhs[(Ny-1)*0+j] + (np.transpose(applied_vector_potential_x_fun(xgrid[Nx-1],ygrid[j+1],t+dt))-Ax[Nx-1,j+1])/dt*delx
     for i in range(0,Nx-1):
           rhs[(Ny-1)*i+0] = rhs[(Ny-1)*i+0] - (np.transpose(applied_vector_potential_y_fun(xgrid[i+1],ygrid[0],t+dt))-Ay[i+1,0])/dt*dely
           rhs[(Ny-1)*i+Ny-2] = rhs[(Ny-1)*i+Ny-2] + (np.transpose(applied_vector_potential_y_fun(xgrid[i+1],ygrid[Ny-1],t+dt))-Ay[i+1,Ny-1])/dt*dely"""
     
     """for i in range(0,Nx-1): #for central difference formula in b.c.s
        for j in range(0,Ny-1):
            rhs[(Ny-1)*i+j] = divJs[i+1,j+1] - ddivAbydt[i+1,j+1]
     for j in range(0,Ny-1):
           rhs[(Ny-1)*0+j] = rhs[(Ny-1)*0+j] - (np.transpose(applied_vector_potential_x_fun(xgrid[1],ygrid[j+1],t+dt))-Ax[1,j+1])/dt*2*delx
           rhs[(Ny-1)*(Nx-2)+j] = rhs[(Ny-1)*0+j] + (np.transpose(applied_vector_potential_x_fun(xgrid[Nx-1],ygrid[j+1],t+dt))-Ax[Nx-1,j+1])/dt*2*delx
     for i in range(0,Nx-1):
           rhs[(Ny-1)*i+0] = rhs[(Ny-1)*i+0] - (np.transpose(applied_vector_potential_y_fun(xgrid[i+1],ygrid[1],t+dt))-Ay[i+1,1])/dt*2*dely
           rhs[(Ny-1)*i+Ny-2] = rhs[(Ny-1)*i+Ny-2] + (np.transpose(applied_vector_potential_y_fun(xgrid[i+1],ygrid[Ny-1],t+dt))-Ay[i+1,Ny-1])/dt*2*dely"""

     """for i in range(0,Nx-1):
          for j in range(0,Ny-1):
               if i+1==1:
                    rhs[(Ny-1)*i+j] = rhs[(Ny-1)*i+j] - (np.transpose(applied_vector_potential_x_fun(xgrid[i+1],ygrid[j+1],t+dt))-Ax[i+1,j+1])/dt*delx
               elif i+1==Nx-1:
                    rhs[(Ny-1)*i+j] = rhs[(Ny-1)*i+j] + (np.transpose(applied_vector_potential_x_fun(xgrid[i+1],ygrid[j+1],t+dt))-Ax[i+1,j+1])/dt*delx
               elif j+1==1:
                    rhs[(Ny-1)*i+j] = rhs[(Ny-1)*i+j] - (np.transpose(applied_vector_potential_y_fun(xgrid[i+1],ygrid[j+1],t+dt))-Ay[i+1,j+1])/dt*dely
               elif j+1==Ny-1:
                    rhs[(Ny-1)*i+j] = rhs[(Ny-1)*i+j] + (np.transpose(applied_vector_potential_y_fun(xgrid[i+1],ygrid[j+1],t+dt))-Ay[i+1,j+1])/dt*dely
               elif i+1==1 and j+1==1:
                    rhs[(Ny-1)*i+j] = rhs[(Ny-1)*i+j] - (np.transpose(applied_vector_potential_x_fun(xgrid[i+1],ygrid[j+1],t+dt))-Ax[i+1,j+1])/dt*delx - (np.transpose(applied_vector_potential_y_fun(xgrid[i+1],ygrid[j+1],t+dt))-Ay[i+1,j+1])/dt*dely
               elif i+1==1 and j+1==Ny-1:
                    rhs[(Ny-1)*i+j] = rhs[(Ny-1)*i+j] - (np.transpose(applied_vector_potential_x_fun(xgrid[i+1],ygrid[j+1],t+dt))-Ax[i+1,j+1])/dt*delx + (np.transpose(applied_vector_potential_y_fun(xgrid[i+1],ygrid[j+1],t+dt))-Ay[i+1,j+1])/dt*dely
               elif i+1==Nx-1 and j+1==1:
                    rhs[(Ny-1)*i+j] = rhs[(Ny-1)*i+j] + (np.transpose(applied_vector_potential_x_fun(xgrid[i+1],ygrid[j+1],t+dt))-Ax[i+1,j+1])/dt*delx - (np.transpose(applied_vector_potential_y_fun(xgrid[i+1],ygrid[j+1],t+dt))-Ay[i+1,j+1])/dt*dely
               elif i+1==Nx-1 and j+1==Ny-1:
                    rhs[(Ny-1)*i+j] = rhs[(Ny-1)*i+j] + (np.transpose(applied_vector_potential_x_fun(xgrid[i+1],ygrid[j+1],t+dt))-Ax[i+1,j+1])/dt*delx + (np.transpose(applied_vector_potential_y_fun(xgrid[i+1],ygrid[j+1],t+dt))-Ay[i+1,j+1])/dt*dely
               #else:
               #    rhs[(Ny-1)*i+j] = rhs[(Ny-1)*i+j]"""
     """for i in range(0,Nx-1):
        for j in range(0,Ny-1):
            rhs[(Ny-1)*i+j] = divJs[i+1,j+1]"""
     #Res[it]=np.sum(rhs)
     #Res2[it]=np.sum(rhs-Res[it]/N2)
     #rhs=rhs[0:N2-1]-np.sum(rhs)/N2
     #rhs=rhs[0:N2-1]


     ###rhs=rhs-np.sum(rhs)/N2

     #C=np.dot(Ainv,rhs)
     ###C=lu.solve(rhs)
     #C=np.append(C,0)
     
     ###mu[1:Nx,1:Ny]=C.reshape((Nx-1,Ny-1))
     
     """###for j in range (0,Nx+1):
               mu[0,j]=mu[1,j]+(np.transpose(applied_vector_potential_x_fun(xgrid[0],ygrid[j],t+dt))-Ax[0,j])/dt*delx #left Boundary
               mu[Nx,j]=mu[Nx-1,j]-(np.transpose(applied_vector_potential_x_fun(xgrid[Nx-1],ygrid[j],t+dt))-Ax[Nx-1,j])/dt*delx #Right Boundary
     
     for i in range (0,Ny+1):
               mu[i,0]=mu[i,1]+(np.transpose(applied_vector_potential_y_fun(xgrid[i],ygrid[0],t+dt))-Ay[i,0])/dt*dely #Lower Boundary
               mu[i,Ny]=mu[i,Ny-1]-(np.transpose(applied_vector_potential_y_fun(xgrid[i],ygrid[Ny-1],t+dt))-Ay[1,Ny-1])/dt*dely #Upper Boundary"""
     ####mu=np.zeros([Nx+1, Ny+1])
     Jst=Js(psi,Px,Py,Pz,D1,D2,G1,G2,G3,D3,D4,D5)
     Jnt=-dx(mu)-(np.transpose(applied_vector_potential_x_fun(xg,yg,t+dt))-Ax)/dt + 1j*(-dy(mu)-(np.transpose(applied_vector_potential_y_fun(xg,yg,t+dt))-Ay)/dt)
          
     #print(it)
     #save configuration of the relevant physical values every hstep=100 steps
     if it%100==0:       
          psi_hist.append(psi.tolist())
          Px_hist.append(Px.tolist())
          Py_hist.append(Py.tolist())
          Pz_hist.append(Pz.tolist())
          D1_hist.append(D1.tolist())
          D2_hist.append(D2.tolist())
          D3_hist.append(D3.tolist())
          D4_hist.append(D4.tolist())
          D5_hist.append(D5.tolist())
          G1_hist.append(G1.tolist())
          G2_hist.append(G2.tolist())
          G3_hist.append(G3.tolist())
          mu_hist.append(mu.tolist())
          Js_hist.append(Jst.tolist())
          Jn_hist.append(Jnt.tolist())
    #print(psi)

# %%
#convert the arrays in which time-evolution of the physical values is saved to np arrays
psi_hist_nd=np.array(psi_hist)

Px_hist_nd=np.array(Px_hist)
Py_hist_nd=np.array(Py_hist)
Pz_hist_nd=np.array(Pz_hist)

D1_hist_nd=np.array(D1_hist)
D2_hist_nd=np.array(D2_hist)
D3_hist_nd=np.array(D3_hist)
D4_hist_nd=np.array(D4_hist)
D5_hist_nd=np.array(D5_hist)

G1_hist_nd=np.array(G1_hist)
G2_hist_nd=np.array(G2_hist)
G3_hist_nd=np.array(G3_hist)

mu_hist_nd=np.array(mu_hist)

Js_hist_nd=np.array(Js_hist)
Jn_hist_nd=np.array(Jn_hist)
hstep=100

# %%
#save 
np.save('psi_Nx=50_Ny=50_dt=0.0001_L=20_hstep=100_A0x=0.5_w=1_w0=4_linear_muzero.npy', psi_hist_nd)
np.save('Px_Nx=50_Ny=50_dt=0.0001_L=20_hstep=100_A0x=0.5_w=1_w0=4_linear_muzero.npy', Px_hist_nd)
np.save('Py_Nx=50_Ny=50_dt=0.0001_L=20_hstep=100_A0x=0.5_w=1_w0=4_linear_muzero.npy', Py_hist_nd)
np.save('Pz_Nx=50_Ny=50_dt=0.0001_L=20_hstep=100_A0x=0.5_w=1_w0=4_linear_muzero.npy', Pz_hist_nd)
np.save('D1_Nx=50_Ny=50_dt=0.0001_L=20_hstep=100_A0x=0.5_w=1_w0=4_linear_muzero.npy', D1_hist_nd)
np.save('D2_Nx=50_Ny=50_dt=0.0001_L=20_hstep=100_A0x=0.5_w=1_w0=4_linear_muzero.npy', D2_hist_nd)
np.save('D3_Nx=50_Ny=50_dt=0.0001_L=20_hstep=100_A0x=0.5_w=1_w0=4_linear_muzero.npy', D3_hist_nd)
np.save('D4_Nx=50_Ny=50_dt=0.0001_L=20_hstep=100_A0x=0.5_w=1_w0=4_linear_muzero.npy', D4_hist_nd)
np.save('D5_Nx=50_Ny=50_dt=0.0001_L=20_hstep=100_A0x=0.5_w=1_w0=4_linear_muzero.npy', D5_hist_nd)
np.save('G1_Nx=50_Ny=50_dt=0.0001_L=20_hstep=100_A0x=0.5_w=1_w0=4_linear_muzero.npy', G1_hist_nd)
np.save('G2_Nx=50_Ny=50_dt=0.0001_L=20_hstep=100_A0x=0.5_w=1_w0=4_linear_muzero.npy', G2_hist_nd)
np.save('G3_Nx=50_Ny=50_dt=0.0001_L=20_hstep=100_A0x=0.5_w=1_w0=4_linear_muzero.npy', G3_hist_nd)
np.save('G3_Nx=50_Ny=50_dt=0.0001_L=20_hstep=100_A0x=0.5_w=1_w0=4_linear_muzero.npy', G3_hist_nd)
np.save('mu_Nx=50_Ny=50_dt=0.0001_L=20_hstep=100_A0x=0.5_w=1_w0=4_linear_muzero.npy', mu_hist_nd)
np.save('Js_Nx=50_Ny=50_dt=0.0001_L=20_hstep=100_A0x=0.5_w=1_w0=4_linear_muzero.npy', Js_hist_nd)
np.save('Jn_Nx=50_Ny=50_dt=0.0001_L=20_hstep=100_A0x=0.5_w=1_w0=4_linear_muzero.npy', Jn_hist_nd)

# %%
(it+1)/100+1

# %%
#Plot s-wave (psi) OP at the center of the film as a function of time (w=1, thus T=2*pi)
fig, ax = plt.subplots()
ax.set_ylabel(r'$psi$')
ax.set_xlabel('t/T')
plt.plot(hstep*dt/(2*np.pi)*np.array(list(range(np.shape(psi_hist_nd)[0]))),np.abs(psi_hist_nd[:,25,25]))

# %%
psi

# %%
#Plot the absolute values of the OPs at the end of simulations
fig, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9), (ax10, ax11, ax12))  = plt.subplots(4,3,figsize=(15, 15))
contour_psi=ax1.contourf(xgrid, ygrid, np.absolute(psi)-0, levels=50, linewidths=100.0, cmap="PRGn");
contour_DEg1=ax2.contourf(xgrid, ygrid, np.absolute(D1)-0, levels=50, linewidths=100.0, cmap="PRGn");
contour_DEg2=ax3.contourf(xgrid, ygrid, np.absolute(D2)-0, levels=50, linewidths=100.0, cmap="PRGn");
contour_Px=ax4.contourf(xgrid, ygrid, np.abs(Px), levels=50, linewidths=100.0, cmap="PRGn");
contour_Py=ax5.contourf(xgrid, ygrid, np.abs(Py), levels=50, linewidths=100.0, cmap="PRGn");
contour_Pz=ax6.contourf(xgrid, ygrid, np.abs(Pz), levels=50, linewidths=100.0, cmap="PRGn");
contour_DT2g3=ax7.contourf(xgrid, ygrid, np.abs(D3), levels=50, linewidths=100.0, cmap="PRGn");
contour_DT2g4=ax8.contourf(xgrid, ygrid, np.abs(D4), levels=50, linewidths=100.0, cmap="PRGn");
contour_DT2g5=ax9.contourf(xgrid, ygrid, np.abs(D5), levels=50, linewidths=100.0, cmap="PRGn");
contour_GT1g1=ax10.contourf(xgrid, ygrid, np.abs(G1), levels=50, linewidths=100.0, cmap="PRGn");
contour_GT1g2=ax11.contourf(xgrid, ygrid, np.abs(G2), levels=50, linewidths=100.0, cmap="PRGn");
contour_GT1g3=ax12.contourf(xgrid, ygrid, np.abs(G3), levels=50, linewidths=100.0, cmap="PRGn");
plt.colorbar(contour_psi)
plt.colorbar(contour_DEg1)
plt.colorbar(contour_DEg2)
plt.colorbar(contour_Px)
plt.colorbar(contour_Py)
plt.colorbar(contour_Pz)
plt.colorbar(contour_DT2g3)
plt.colorbar(contour_DT2g4)
plt.colorbar(contour_DT2g5)
plt.colorbar(contour_GT1g1)
plt.colorbar(contour_GT1g2)
plt.colorbar(contour_GT1g3)
ax1.title.set_text('Psi')
ax2.title.set_text('D1')
ax3.title.set_text('D2')
ax4.title.set_text('Px')
ax5.title.set_text('Py')
ax6.title.set_text('Pz')
ax7.title.set_text('D3')
ax8.title.set_text('D4')
ax9.title.set_text('D5')
ax10.title.set_text('G1')
ax11.title.set_text('G2')
ax12.title.set_text('G3')
#plt.title('Pxx',fontsize=24,y=1.08);
plt.show();

# %%
#Plot the phases of the OPs at the end of simulations
fig, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9), (ax10, ax11, ax12))  = plt.subplots(4,3,figsize=(15, 15))
contour_psi=ax1.contourf(xgrid, ygrid, np.angle(psi)-0, levels=50, linewidths=100.0, cmap="PRGn");
contour_DEg1=ax2.contourf(xgrid, ygrid, np.angle(D1)-0, levels=50, linewidths=100.0, cmap="PRGn");
contour_DEg2=ax3.contourf(xgrid, ygrid, np.angle(D2)-0, levels=50, linewidths=100.0, cmap="PRGn");
contour_Px=ax4.contourf(xgrid, ygrid, np.angle(Px), levels=50, linewidths=100.0, cmap="PRGn");
contour_Py=ax5.contourf(xgrid, ygrid, np.angle(Py), levels=50, linewidths=100.0, cmap="PRGn");
contour_Pz=ax6.contourf(xgrid, ygrid, np.angle(Pz), levels=50, linewidths=100.0, cmap="PRGn");
contour_DT2g3=ax7.contourf(xgrid, ygrid, np.angle(D3), levels=50, linewidths=100.0, cmap="PRGn");
contour_DT2g4=ax8.contourf(xgrid, ygrid, np.angle(D4), levels=50, linewidths=100.0, cmap="PRGn");
contour_DT2g5=ax9.contourf(xgrid, ygrid, np.angle(D5), levels=50, linewidths=100.0, cmap="PRGn");
contour_GT1g1=ax10.contourf(xgrid, ygrid, np.angle(G1), levels=50, linewidths=100.0, cmap="PRGn");
contour_GT1g2=ax11.contourf(xgrid, ygrid, np.angle(G2), levels=50, linewidths=100.0, cmap="PRGn");
contour_GT1g3=ax12.contourf(xgrid, ygrid, np.angle(G3), levels=50, linewidths=100.0, cmap="PRGn");
plt.colorbar(contour_psi)
plt.colorbar(contour_DEg1)
plt.colorbar(contour_DEg2)
plt.colorbar(contour_Px)
plt.colorbar(contour_Py)
plt.colorbar(contour_Pz)
plt.colorbar(contour_DT2g3)
plt.colorbar(contour_DT2g4)
plt.colorbar(contour_DT2g5)
plt.colorbar(contour_GT1g1)
plt.colorbar(contour_GT1g2)
plt.colorbar(contour_GT1g3)
ax1.title.set_text('Psi')
ax2.title.set_text('D1')
ax3.title.set_text('D2')
ax4.title.set_text('Px')
ax5.title.set_text('Py')
ax6.title.set_text('Pz')
ax7.title.set_text('D3')
ax8.title.set_text('D4')
ax9.title.set_text('D5')
ax10.title.set_text('G1')
ax11.title.set_text('G2')
ax12.title.set_text('G3')
#plt.title('Pxx',fontsize=24,y=1.08);
plt.show();

# %%
#Make an animation for the absolute values of the OPs
fig, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9), (ax10, ax11, ax12))  = plt.subplots(4,3,figsize=(15, 15))

psi_hist_nd=np.array(psi_hist)

Px_hist_nd=np.array(Px_hist)
Py_hist_nd=np.array(Py_hist)
Pz_hist_nd=np.array(Pz_hist)

D1_hist_nd=np.array(D1_hist)
D2_hist_nd=np.array(D2_hist)
D3_hist_nd=np.array(D3_hist)
D4_hist_nd=np.array(D4_hist)
D5_hist_nd=np.array(D5_hist)

G1_hist_nd=np.array(G1_hist)
G2_hist_nd=np.array(G2_hist)
G3_hist_nd=np.array(G3_hist)

mu_hist_nd=np.array(mu_hist)

hstep=100

psi_min=np.min(np.abs(psi_hist_nd))
psi_max=np.max(np.abs(psi_hist_nd))

Px_min=np.min(np.abs(Px_hist_nd))
Px_max=np.max(np.abs(Px_hist_nd))
Py_min=np.min(np.abs(Py_hist_nd))
Py_max=np.max(np.abs(Py_hist_nd))
Pz_min=np.min(np.abs(Pz_hist_nd))
Pz_max=np.max(np.abs(Pz_hist_nd))

D1_min=np.min(np.abs(D1_hist_nd))
D1_max=np.max(np.abs(D1_hist_nd))
D2_min=np.min(np.abs(D2_hist_nd))
D2_max=np.max(np.abs(D2_hist_nd))
D3_min=np.min(np.abs(D3_hist_nd))
D3_max=np.max(np.abs(D3_hist_nd))
D4_min=np.min(np.abs(D4_hist_nd))
D4_max=np.max(np.abs(D4_hist_nd))
D5_min=np.min(np.abs(D5_hist_nd))
D5_max=np.max(np.abs(D5_hist_nd))

G1_min=np.min(np.abs(G1_hist_nd))
G1_max=np.max(np.abs(G1_hist_nd))
G2_min=np.min(np.abs(G2_hist_nd))
G2_max=np.max(np.abs(G2_hist_nd))
G3_min=np.min(np.abs(G3_hist_nd))
G3_max=np.max(np.abs(G3_hist_nd))


contour_psi=ax1.pcolormesh(xg, yg, np.abs(psi_hist[0])-0, cmap="PRGn",shading='gouraud',vmin=psi_min,vmax=psi_max);
contour_DEg1=ax2.pcolormesh(xg, yg, np.abs(D1_hist[0])-0, cmap="PRGn",shading='gouraud',vmin=D1_min,vmax=D1_max);
contour_DEg2=ax3.pcolormesh(xg, yg, np.abs(D2_hist[0])-0, cmap="PRGn",shading='gouraud',vmin=D2_min,vmax=D2_max);
contour_Px=ax4.pcolormesh(xg, yg, np.abs(Px_hist[0]), cmap="PRGn",shading='gouraud',vmin=Px_min,vmax=Px_max);
contour_Py=ax5.pcolormesh(xg, yg, np.abs(Py_hist[0]), cmap="PRGn",shading='gouraud',vmin=Py_min,vmax=Py_max);
contour_Pz=ax6.pcolormesh(xg, yg, np.abs(Pz_hist[0]), cmap="PRGn",shading='gouraud',vmin=Pz_min,vmax=Pz_max);
contour_DT2g3=ax7.pcolormesh(xg, yg, np.abs(D3_hist[0]), cmap="PRGn",shading='gouraud',vmin=D3_min,vmax=D3_max);
contour_DT2g4=ax8.pcolormesh(xg, yg, np.abs(D4_hist[0]), cmap="PRGn",shading='gouraud',vmin=D4_min,vmax=D4_max);
contour_DT2g5=ax9.pcolormesh(xg, yg, np.abs(D5_hist[0]), cmap="PRGn",shading='gouraud',vmin=D5_min,vmax=D5_max);
contour_GT1g1=ax10.pcolormesh(xg, yg, np.abs(G1_hist[0]), cmap="PRGn",shading='gouraud',vmin=G1_min,vmax=G1_max);
contour_GT1g2=ax11.pcolormesh(xg, yg, np.abs(G2_hist[0]), cmap="PRGn",shading='gouraud',vmin=G2_min,vmax=G2_max);
contour_GT1g3=ax12.pcolormesh(xg, yg, np.abs(G3_hist[0]), cmap="PRGn",shading='gouraud',vmin=G3_min,vmax=G3_max);


plt.colorbar(contour_psi)
plt.colorbar(contour_DEg1)
plt.colorbar(contour_DEg2)
plt.colorbar(contour_Px)
plt.colorbar(contour_Py)
plt.colorbar(contour_Pz)
plt.colorbar(contour_DT2g3)
plt.colorbar(contour_DT2g4)
plt.colorbar(contour_DT2g5)
plt.colorbar(contour_GT1g1)
plt.colorbar(contour_GT1g2)
plt.colorbar(contour_GT1g3)

#Fr=ax.pcolormesh(xg,yg,np.abs(psi_hist[0]),cmap="PRGn",shading='gouraud',vmin=0.985,vmax=1)
#plt.colorbar(Fr, ax=ax)
#plt.show()

def animate(i):
       #ax.clear()
       #Fr=ax.contourf(xg,yg,psi_hist[i],levels=50, linewidths=100.0, cmap="PRGn",vmin=0,vmax=1)
       #Fr=ax.pcolormesh(xg,yg,np.abs(psi_hist[i]),cmap="PRGn",shading='gouraud',vmin=0,vmax=1)
       #Fr.set_array(np.abs(psi_hist[i]))
       contour_psi.set_array(np.abs(psi_hist[i]))
       contour_DEg1.set_array(np.abs(D1_hist[i]))
       contour_DEg2.set_array(np.abs(D2_hist[i]))
       contour_Px.set_array(np.abs(Px_hist[i]))
       contour_Py.set_array(np.abs(Py_hist[i]))
       contour_Pz.set_array(np.abs(Pz_hist[i]))
       contour_DT2g3.set_array(np.abs(D3_hist[i]))
       contour_DT2g4.set_array(np.abs(D4_hist[i]))
       contour_DT2g5.set_array(np.abs(D5_hist[i]))
       contour_GT1g1.set_array(np.abs(G1_hist[i]))
       contour_GT1g2.set_array(np.abs(G2_hist[i]))
       contour_GT1g3.set_array(np.abs(G3_hist[i]))

       ax1.set_title(r'$|\eta_{A_{1g}}|$, t/T=%03.4f'%(hstep*i*w*dt/(2*np.pi)))
       ax2.set_title(r'$|\eta_{E_{g}}^1|$, t/T=%03.4f'%(hstep*i*w*dt/(2*np.pi)))
       ax3.set_title(r'$|\eta_{E_{g}}^2|$, t/T=%03.4f'%(hstep*i*w*dt/(2*np.pi)))

       ax4.set_title(r'$|\eta_{T_{1u}}^1|$, t/T=%03.4f'%(hstep*i*w*dt/(2*np.pi)))
       ax5.set_title(r'$|\eta_{T_{1u}}^2|$, t/T=%03.4f'%(hstep*i*w*dt/(2*np.pi)))
       ax6.set_title(r'$|\eta_{T_{1u}}^3|$, t/T=%03.4f'%(hstep*i*w*dt/(2*np.pi)))

       ax7.set_title(r'$|\eta_{T_{2g}}^1|$, t/T=%03.4f'%(hstep*i*w*dt/(2*np.pi)))
       ax8.set_title(r'$|\eta_{T_{2g}}^2|$, t/T=%03.4f'%(hstep*i*w*dt/(2*np.pi)))
       ax9.set_title(r'$|\eta_{T_{2g}}^3|$, t/T=%03.4f'%(hstep*i*w*dt/(2*np.pi)))

       ax10.set_title(r'$|\eta_{T_{1g}}^1|$, t/T=%03.4f'%(hstep*i*w*dt/(2*np.pi)))
       ax11.set_title(r'$|\eta_{T_{1g}}^2|$, t/T=%03.4f'%(hstep*i*w*dt/(2*np.pi)))
       ax12.set_title(r'$|\eta_{T_{1g}}^3|$, t/T=%03.4f'%(hstep*i*w*dt/(2*np.pi)))
       
    
ani = animation.FuncAnimation(fig,animate,1261,interval=30,blit=False)

writer = animation.writers['ffmpeg'](fps=30)
ani.save('psi.mp4', writer=writer, dpi=72)
#video = ani.to_html5_video()
#display(video)

plt.show()

# %%
#Make an animation for the phases of the OPs
fig, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9), (ax10, ax11, ax12))  = plt.subplots(4,3,figsize=(15, 15))


hstep=100

psi_min=-pi
psi_max=pi

Px_min=-pi
Px_max=pi
Py_min=-pi
Py_max=pi
Pz_min=-pi
Pz_max=pi

D1_min=-pi
D1_max=pi
D2_min=-pi
D2_max=pi
D3_min=-pi
D3_max=pi
D4_min=-pi
D4_max=pi
D5_min=-pi
D5_max=pi

G1_min=-pi
G1_max=pi
G2_min=-pi
G2_max=pi
G3_min=-pi
G3_max=pi


contour_psi=ax1.pcolormesh(xg, yg, np.angle(psi_hist_nd[0])-0, cmap=hsluv_anglemap,shading='gouraud',vmin=psi_min,vmax=psi_max);
contour_DEg1=ax2.pcolormesh(xg, yg, np.angle(D1_hist_nd[0])-0, cmap=hsluv_anglemap,shading='gouraud',vmin=D1_min,vmax=D1_max);
contour_DEg2=ax3.pcolormesh(xg, yg, np.angle(D2_hist_nd[0])-0, cmap=hsluv_anglemap,shading='gouraud',vmin=D2_min,vmax=D2_max);
contour_Px=ax4.pcolormesh(xg, yg, np.angle(Px_hist_nd[0]), cmap=hsluv_anglemap,shading='gouraud',vmin=Px_min,vmax=Px_max);
contour_Py=ax5.pcolormesh(xg, yg, np.angle(Py_hist_nd[0]), cmap=hsluv_anglemap,shading='gouraud',vmin=Py_min,vmax=Py_max);
contour_Pz=ax6.pcolormesh(xg, yg, np.angle(Pz_hist_nd[0]), cmap=hsluv_anglemap,shading='gouraud',vmin=Pz_min,vmax=Pz_max);
contour_DT2g3=ax7.pcolormesh(xg, yg, np.angle(D3_hist_nd[0]), cmap=hsluv_anglemap,shading='gouraud',vmin=D3_min,vmax=D3_max);
contour_DT2g4=ax8.pcolormesh(xg, yg, np.angle(D4_hist_nd[0]), cmap=hsluv_anglemap,shading='gouraud',vmin=D4_min,vmax=D4_max);
contour_DT2g5=ax9.pcolormesh(xg, yg, np.angle(D5_hist_nd[0]), cmap=hsluv_anglemap,shading='gouraud',vmin=D5_min,vmax=D5_max);
contour_GT1g1=ax10.pcolormesh(xg, yg, np.angle(G1_hist_nd[0]), cmap=hsluv_anglemap,shading='gouraud',vmin=G1_min,vmax=G1_max);
contour_GT1g2=ax11.pcolormesh(xg, yg, np.angle(G2_hist_nd[0]), cmap=hsluv_anglemap,shading='gouraud',vmin=G2_min,vmax=G2_max);
contour_GT1g3=ax12.pcolormesh(xg, yg, np.angle(G3_hist_nd[0]), cmap=hsluv_anglemap,shading='gouraud',vmin=G3_min,vmax=G3_max);


plt.colorbar(contour_psi)
plt.colorbar(contour_DEg1)
plt.colorbar(contour_DEg2)
plt.colorbar(contour_Px)
plt.colorbar(contour_Py)
plt.colorbar(contour_Pz)
plt.colorbar(contour_DT2g3)
plt.colorbar(contour_DT2g4)
plt.colorbar(contour_DT2g5)
plt.colorbar(contour_GT1g1)
plt.colorbar(contour_GT1g2)
plt.colorbar(contour_GT1g3)

#Fr=ax.pcolormesh(xg,yg,np.angle(psi_hist[0]),cmap="lum-HSLUV",shading='gouraud',vmin=0.985,vmax=1)
#plt.colorbar(Fr, ax=ax)
#plt.show()

def animate(i):
       #ax.clear()
       #Fr=ax.contourf(xg,yg,psi_hist[i],levels=50, linewidths=100.0, cmap="lum-HSLUV",vmin=0,vmax=1)
       #Fr=ax.pcolormesh(xg,yg,np.angle(psi_hist[i]),cmap="lum-HSLUV",shading='gouraud',vmin=0,vmax=1)
       #Fr.set_array(np.angle(psi_hist[i]))
       contour_psi.set_array(np.angle(psi_hist_nd[i]))
       contour_DEg1.set_array(np.angle(D1_hist_nd[i]))
       contour_DEg2.set_array(np.angle(D2_hist_nd[i]))
       contour_Px.set_array(np.angle(Px_hist_nd[i]))
       contour_Py.set_array(np.angle(Py_hist_nd[i]))
       contour_Pz.set_array(np.angle(Pz_hist_nd[i]))
       contour_DT2g3.set_array(np.angle(D3_hist_nd[i]))
       contour_DT2g4.set_array(np.angle(D4_hist_nd[i]))
       contour_DT2g5.set_array(np.angle(D5_hist_nd[i]))
       contour_GT1g1.set_array(np.angle(G1_hist_nd[i]))
       contour_GT1g2.set_array(np.angle(G2_hist_nd[i]))
       contour_GT1g3.set_array(np.angle(G3_hist_nd[i]))

       ax1.set_title(r'$|\eta_{A_{1g}}|$, t/T=%03.4f'%(hstep*i*w*dt/(2*np.pi)))
       ax2.set_title(r'$|\eta_{E_{g}}^1|$, t/T=%03.4f'%(hstep*i*w*dt/(2*np.pi)))
       ax3.set_title(r'$|\eta_{E_{g}}^2|$, t/T=%03.4f'%(hstep*i*w*dt/(2*np.pi)))

       ax4.set_title(r'$|\eta_{T_{1u}}^1|$, t/T=%03.4f'%(hstep*i*w*dt/(2*np.pi)))
       ax5.set_title(r'$|\eta_{T_{1u}}^2|$, t/T=%03.4f'%(hstep*i*w*dt/(2*np.pi)))
       ax6.set_title(r'$|\eta_{T_{1u}}^3|$, t/T=%03.4f'%(hstep*i*w*dt/(2*np.pi)))

       ax7.set_title(r'$|\eta_{T_{2g}}^1|$, t/T=%03.4f'%(hstep*i*w*dt/(2*np.pi)))
       ax8.set_title(r'$|\eta_{T_{2g}}^2|$, t/T=%03.4f'%(hstep*i*w*dt/(2*np.pi)))
       ax9.set_title(r'$|\eta_{T_{2g}}^3|$, t/T=%03.4f'%(hstep*i*w*dt/(2*np.pi)))

       ax10.set_title(r'$|\eta_{T_{1g}}^1|$, t/T=%03.4f'%(hstep*i*w*dt/(2*np.pi)))
       ax11.set_title(r'$|\eta_{T_{1g}}^2|$, t/T=%03.4f'%(hstep*i*w*dt/(2*np.pi)))
       ax12.set_title(r'$|\eta_{T_{1g}}^3|$, t/T=%03.4f'%(hstep*i*w*dt/(2*np.pi)))
    
ani = animation.FuncAnimation(fig,animate,1261,interval=30,blit=False)

writer = animation.writers['ffmpeg'](fps=30)
ani.save('Phase.mp4', writer=writer, dpi=72)
#video = ani.to_html5_video()
#display(video)

plt.show()

# %%



