#task1
import numpy as np 
from numpy import zeros, dot, savetxt
from numpy.linalg import norm

def cd(a, b):
   return 1.0 - (dot(a, b) / (norm(a) * norm(b)))

if __name__ == "__main__":
   with open("text.txt") as file:      
       lines = sum(1 for _ in file)
       file.seek(0)      
       import re
       slova = {}  
       lc, wc = 0, 0
       for line in file:          
           p = re.compile(r"[^a-z]+")
           tokens = p.split(line.lower())          
           tokens.pop()
           for token in tokens:
               if token not in slova:
                   slova[token] = {
                       "index": wc,
                       "occurrences": [0] * lines
                   }
                   wc += 1
               elif slova[token]["occurrences"][lc] != 0:
                   continue
               slova[token]["occurrences"][lc] = tokens.count(token)   
           lc += 1
       arr = zeros((lines, len(slova)))
       for slovo in slova:
           i, j = 0, slova[slovo]["index"]
           for occ in slova[slovo]["occurrences"]:
               arr[i, j] = occ
               i += 1
       d = []
       a = arr[0, j]
       for j in range(1, lines):
           b = arr[i-1, j]
           d.append({"index": j, "distance": cd(a, b)})   
      
       d.sort(key=lambda x: x["distance"])
       print("%d %d" % (
           d[0]["index"],
           d[1]["index"],
       ))
#task2
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg

def f(x):
    return np.sin(0.2 * x) * np.exp(0.1 * x) + 5 * np.exp(0.5 * (-x))

I = [[1,1**1,1**2,1**3], [1,4**1,4**2,4**3],[1,10**1,10**2,10**3],[1,15**1,15**2,15**3]]
matrix = np.array (I)
i = [f(1.),f(4.),f(10.),f(15.)]
res = np.linalg.solve(I, i)

def f3(x):
    return res[0] + res[1] + res[2] + res[3]*x

print (res)

fig, ax = plt.subplots()
x = np.arange(1, 15, 0.1)
ax.plot(x, f(x), color="black", label= "Функция оси X")

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.legend()

plt.show()
