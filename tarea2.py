
#Solución a tarea 2
#Librerías
import csv
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import statistics
import pandas
from scipy.stats import norm

#importar archivo
filename = 'datos.csv'
raw_data = open(filename)
datos = np.loadtxt(raw_data, 
            delimiter=",",skiprows=0)


#Obtención del histograma de los datos
plt.hist(datos, bins = 30,  alpha=0.6, 
         color='c')
plt.xlabel('Datos')
plt.ylabel('Número de casos')
plt.show()

# Obtención de la función 
#de densidad de la transformación 
a, loc, scale = stats.gamma.fit(datos)

plt.hist(datos, bins=25, density=True,
         alpha=0.6, color='g')
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = stats.gamma.pdf(x, a, loc, scale)
plt.plot(x, p, 'k', linewidth=2)
plt.title("a = %.2f,  loc = %.2f, scale = %.2f,"
          % (a, loc, scale))
plt.show()

#Cálculo de momentos centrales 
#directamente de los datos
print(np.mean(datos))
print(np.var(datos))
print(np.std(datos))
print(stats.skew(datos))
print(stats.kurtosis(datos))

#Probabilidad acumulada 
#directamente de los datos 
cdf = 0
for i in range(N):
    if(datos[i]>37 and datos[i]<76):
        cdf=cdf+1
print(cdf)
cdf = cdf/N
print(cdf)
#Probabilidad acumulada 
#por función de densidad
p1 = stats.gamma.cdf(37,a, loc, scale)
p2 = stats.gamma.cdf(76,a, loc, scale)
p2 = p2 - p1
print(p2)



#obtención de la transformación
transformacion = np.sqrt(datos) 

#Obtención del histograma de 
#la transformación
plt.hist(transformacion, 
    bins = 25,  alpha=0.6, color='c')
plt.plot()
plt.xlabel('Transformacióm')
plt.ylabel('Número de c')
plt.show()

# Ajuste de transformación a 
#una distribución normal
mu, std = norm.fit(transformacion)
#Gráfico de histograma y PDF
plt.hist(transformacion, bins=25, 
  density=True, alpha=0.6, color='g')
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu, std)
plt.plot(x, p, 'k', linewidth=2)
plt.title("mu = %.2f,  std = %.2f"
          % (mu, std))
plt.show()




