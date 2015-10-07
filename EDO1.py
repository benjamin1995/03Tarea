import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import ode
from mpl_toolkits.mplot3d import Axes3D


#PARTE1
mu=1.375
#mu=1.375

def f(y,m):
    return m,-y-mu*((y**2)-1)*m


def get_k1(y_n, m_n, h, f):
    f_eval = f(y_n, m_n)
    return h * f_eval[0], h * f_eval[1]


def get_k2(y_n, m_n, h, f):
    k1 = get_k1(y_n, m_n, h, f)
    f_eval = f(y_n + k1[0]/2, m_n + k1[1]/2)
    return k1,(h * f_eval[0], h * f_eval[1])

def get_k3(y_n, m_n, h, f):
    k1,k2=get_k2(y_n,m_n,h,f)
    f_eval=f(y_n-k1[0]+2*k2[0],m_n-k1[1]+2*k2[1])
    return k1,k2,(h*f_eval[0],h*f_eval[1])


def rk3_step(y_n, m_n, h, f):
    k1,k2,k3=get_k3(y_n,m_n,h,f)
    y_n1= y_n +(k1[0]+4*k2[0]+k3[0])/6
    m_n1= m_n + (k1[1]+4*k2[1]+k3[1])/6
    return y_n1,m_n1

N_steps = 40000
h = 20*np.pi / N_steps
#h=1000/N_steps.
y = np.zeros(N_steps)
m = np.zeros(N_steps)

y[0] = 0.001
# mientras mas chico la condicion inicial mas pequenos son los intervalos de yprima con y ademas de y vs s
m[0] = 2
for i in range(1, N_steps):
    y[i], m[i] = rk3_step(y[i-1], m[i-1], h, f)



s_rk = [h * i for i in range(N_steps)]

plt.figure(1)
plt.clf()
plt.plot(y,m, 'g',label='metodo rk3 y0=0')
plt.title(" y` vs y ")
plt.xlabel('$ y $')
plt.ylabel('$ dy/ds $', fontsize=13)
plt.legend()


plt.figure(2)
plt.clf()
plt.plot(s_rk,y, 'b',label='metodo rk3 y0=0')
plt.title('$y$ vs $s_rk$',fontsize=14)
plt.xlabel('$s$')
plt.ylabel('$y$', fontsize=13)
plt.legend()

#y[0] = 4
#m[0] = 0
#for i in range(1, N_steps):
#y[i], m[i] = rk3_step(y[i-1], m[i-1], h, f)



#plt.figure(3)
#plt.clf()
#plt.plot(y,m, 'g',label='metodo rk3 y0=4')
#plt.title(" y` vs y ")
#plt.xlabel('$ y $')
#plt.ylabel('$ dy/ds $', fontsize=13)
#plt.legend()


#plt.figure(4)
#plt.clf()
#plt.plot(s_rk,y, 'b',label='metodo rk3 y0=4')
#plt.title('$y$ vs $s_rk$',fontsize=14)
#plt.xlabel('$s$')
#plt.ylabel('$y$', fontsize=13)
#plt.legend()



#plt.show()
#plt.draw()


#PARTE2

sigma = 10
beta = 8/3
rho = 28


def f_to_solve(t,(x,y,z)):   # arr arreglo de (x,y,z)
    return [sigma*(y-x),x*(rho-z)-y,x*y-beta*z]
# Condiciones inicial

x0 = 1
y0 = 1
z0 = 1

#creamos la solucion usando ode
solution = ode(f_to_solve)
solution.set_integrator('dopri5', max_step=0.1, first_step=0.01)
solution.set_initial_value([x0,y0,z0],0)

t1 = 100
dt = 0.01     #mientras mas chico el delta mas preciso es x,y,z para condicion inicial.
pasos = 1000000
i = 0
t_n = np.zeros(pasos)
sol_n=[np.zeros(pasos),np.zeros(pasos),np.zeros(pasos)]

while solution.successful() and solution.t < t1:  #definido asi para integrar
    t_n[i],(sol_n[0][i],sol_n[1][i],sol_n[2][i])=[solution.t,(solution.integrate(solution.t+dt))]
    i += 1


fig=plt.figure(3)
fig.clf()

ax = fig.add_subplot(111,projection='3d')
ax.plot(sol_n[0],sol_n[1],sol_n[2],'r')
#ax2.plot(t_values, np.log10(np.fabs(analitica(t_values) - x_values)), 'r')
ax.set_xlabel('x(t)')
ax.set_ylabel('y(t)')
ax.set_zlabel('z(t)')
ax.set_title('Atractor de Lorentz')

fig.show()
