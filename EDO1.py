import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import ode
from pylab import *
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


#PARTE1

'''
En esta parte del script se resuelve la ecuacion diferencial del oscilador de Van der pol mediante el metodo de runge-kutta de orden 3

'''




'''
 Se define la funcion matricial f(y(s),w=dy/ds) (independiente de la variable s) en forma matricial reduciendo a orden 1 la ecuacion original de orden 2
'''
def f(y,w):
    return w,-y-mu*((y**2)-1)*w


'''
 Funcion matricial que recibe el y-esimo y w-esimo valor, ademas del paso h arbitrario y la funcion f, para retornar vector k1
 Una componente es para y mientras que la otra para w
'''
def get_k1(y_n, w_n, h, f):
    f_eval = f(y_n, w_n)
    return h * f_eval[0], h * f_eval[1]


'''
 Funcion matricial que recibe el y-esimo y w-esimo valor, ademas del paso h arbitrario y la funcion f, para retornar vector k1 y un vector k2.
 Dos componentes son para y ([0]) ,mientras que las otras para w ([1])
'''
def get_k2(y_n, w_n, h, f):
    k1 = get_k1(y_n, w_n, h, f)
    f_eval = f(y_n + k1[0]/2, w_n + k1[1]/2)
    return k1,(h * f_eval[0], h * f_eval[1])


'''
 Funcion matricial que recibe el y-esimo y w-esimo valor, ademas del paso h arbitrario y la funcion f, para retornar vector k1,k2 y k3
 Al igual que antes, 3 componentes seran de y ([0]), mientras que las otras ([1]) de w.
'''
def get_k3(y_n, w_n, h, f):
    k1,k2=get_k2(y_n,w_n,h,f)
    f_eval=f(y_n-k1[0]+2*k2[0],w_n-k1[1]+2*k2[1])
    return k1,k2,(h*f_eval[0],h*f_eval[1])

'''
 Funcion matricial que recibe el y-esimo y w-esimo valor, ademas del paso h arbitrario y la funcion f, para retornar el y-esimo mas un termino,
 Ademas del w-esimo mas un termino
 De esta forma se van obteniendo los vectores (w,y).
'''
def rk3_step(y_n, w_n, h, f):
    k1,k2,k3=get_k3(y_n,w_n,h,f)
    y_n1= y_n +(k1[0]+4*k2[0]+k3[0])/6
    w_n1= w_n + (k1[1]+4*k2[1]+k3[1])/6
    return y_n1,w_n1




mu=1.375      #mi rut es 18956375-2
N_steps = 40000
h = 20*np.pi / N_steps
y = np.zeros(N_steps)
w = np.zeros(N_steps)
# mientras mas chico la condicion inicial mas pequenos son los intervalos de yprima con y ademas de y vs s
y[0] = 0.1
w[0] = 0
for i in range(1, N_steps):
    y[i], w[i] = rk3_step(y[i-1], w[i-1], h, f)



s_rk = [h * i for i in range(N_steps)]

# Primera condicion inicial para graficar (y0=0.1,w0=0)

plt.figure(1)
plt.clf()
subplot(2,1,1,axisbg='darkslategray')
plt.plot(y,w, 'g',label=' Metodo rk3 para y0=0.1')
plt.grid(True)
plt.title(" $y'$ $v/s$ $y$ ")
plt.xlabel(' y(s) ')
plt.ylabel(' dy/ds ')
plt.legend()


subplot(2,1,2,axisbg='darkblue')
plt.subplots_adjust(hspace=0.4)
plt.plot(s_rk,y, 'b',label=' Metodo rk3 para y0=0.1',markersize=10)
plt.grid(True)
plt.title('$y(s)$ $v/s$ $s$')
plt.xlabel('s')
plt.ylabel('y(s)')
plt.legend()
plt.savefig('primeracondicion.png')



# Segunda condicion inicial para graficar (y0=4,w0=0)

y[0] = 4
w[0] = 0
for i in range(1, N_steps):
    y[i], w[i] = rk3_step(y[i-1], w[i-1], h, f)

plt.figure(2)
plt.clf()
subplot(2,1,1,axisbg='darkgreen')
plt.plot(y,w, 'g',label=' Metodo rk3 para y0=4')
plt.grid(True)
plt.title(" $y'$ $v/s$ $y$ ")
plt.xlabel(' y(s) ')
plt.ylabel(' dy/ds ')
plt.legend()


subplot(2,1,2,axisbg='darkorange')
plt.subplots_adjust(hspace=0.4)
plt.plot(s_rk,y, 'b',label=' Metodo rk3 para y0=4')
plt.grid(True)
plt.title('$y(s)$ $v/s$ $s$')
plt.xlabel('s')
plt.ylabel('y(s)')
plt.legend()

plt.savefig('segundacondicion.png')
plt.show()
plt.draw()


#PARTE2

'''
En esta parte del script se graficara (x(t),y(t),z(t)) obtenido a partir del sistema de ecuaciones de Lorentz
ocupando la libreria de scipy.integrate

'''


# parametros del sistema de lorenz
sigma = 10
beta = 8/3
rho = 28


'''
Se crea funcion que recibe el tiempo y las coordenadas como vector para retornar el sistema de ecuaciones de Lorentz
'''
def f_to_solve(t,(x,y,z)):
    return [sigma*(y-x),x*(rho-z)-y,x*y-beta*z]


# Condiciones iniciales

x0 = 1
y0 = 1
z0 = 1

# creamos la solucion usando ode de scipy y guardamos las condiciones iniciales en la solucion
solution = ode(f_to_solve)
solution.set_integrator('dopri5', max_step=0.1, first_step=0.01)
solution.set_initial_value([x0,y0,z0],0)


'''
 arreglo de tiempo para crear vector sol_n con coordenadas (x,y,z)
 mientras mas chico el delta mas preciso es x,y,z para la condicion inicial(se acerca mas al valor de cada coordenada inicial).
'''
t1 = 100
dt = 0.01
pasos = 1000000
i = 0
t_n = np.zeros(pasos)
sol_n=[np.zeros(pasos),np.zeros(pasos),np.zeros(pasos)]

#definido asi el metodo de ode 'dopri5' para integrar
while solution.successful() and solution.t < t1:
    t_n[i],(sol_n[0][i],sol_n[1][i],sol_n[2][i])=[solution.t,(solution.integrate(solution.t+dt))]
    i += 1

# grafico de (x,y,z) (se anexan 4 graficos con distintas condiciones iniciales)
fig=plt.figure(3)
fig.clf()

ax = fig.add_subplot(111,projection='3d')
ax.plot(sol_n[0],sol_n[1],sol_n[2],'green')
ax.set_xlabel('x(t)')
ax.set_ylabel('y(t)')
ax.set_zlabel('z(t)')
ax.set_title('Atractor de Lorentz (x0,y0,z0)=(1,1,1)')
plt.savefig('atractordelorenz.png')

fig.show()
