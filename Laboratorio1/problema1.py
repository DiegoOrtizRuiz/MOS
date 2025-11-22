import pyomo.environ as pyo
from graficos import plot_problema1_parteA, plot_problema1_parteB


# ----------------------
# Datos de ejemplo
# ----------------------
T = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]  # Tareas
p = {1: 5, 2: 8, 3: 13, 4: 1, 5: 21, 6: 2, 7:8, 8:5, 9:8, 10:13, 11:21}   # puntos por tarea
w = {1: 21, 2: 8, 3: 13, 4: 3, 5: 1, 6: 5, 7:13, 8:13, 9:2, 10:21, 11:13}   # prioridad (Fibonacci)
Capacidad_total = 52

D = [1, 2, 3, 4]   # desarrolladores
Cd = {1: 15, 2: 15, 3: 15, 4: 15}

# ----------------------
# Modelo Parte A
# ----------------------
modelA = pyo.ConcreteModel()
modelA.T = pyo.Set(initialize=T)
modelA.x = pyo.Var(modelA.T, domain=pyo.Binary)

def objA(m):
    return sum(w[i] * m.x[i] for i in m.T)
modelA.obj = pyo.Objective(expr=sum(w[i]*modelA.x[i] for i in modelA.T), sense=pyo.maximize)

def capA(m):
    return sum(p[i] * m.x[i] for i in m.T) <= Capacidad_total
modelA.capacidad = pyo.Constraint(expr=sum(p[i]*modelA.x[i] for i in modelA.T) <= Capacidad_total)

# ----------------------
# Resolver Parte A
# ----------------------
solver = pyo.SolverFactory("glpk")
solver.solve(modelA)
modelA.display()
print("Parte A - Tareas seleccionadas:")
for i in modelA.T:
    if pyo.value(modelA.x[i]) == 1:
        print(f"Tarea {i} seleccionada")
        
resultadosA = {i: w[i] for i in modelA.T if pyo.value(modelA.x[i]) == 1}
cap_usada = sum(p[i] for i in resultadosA.keys())
plot_problema1_parteA(resultadosA, cap_usada, Capacidad_total)

# ----------------------
# Modelo Parte B
# ----------------------
modelB = pyo.ConcreteModel()
modelB.T = pyo.Set(initialize=T)
modelB.D = pyo.Set(initialize=D)
modelB.x = pyo.Var(modelB.T, modelB.D, domain=pyo.Binary)
modelB.y = pyo.Var(modelB.T, domain=pyo.Binary)

def objB(m):
    return sum(w[i] * m.x[i,d] for i in m.T for d in m.D)
modelB.obj = pyo.Objective(expr=sum(w[i]*modelB.x[i,d] for i in T for d in D), sense=pyo.maximize)

def capEquipo(m):
    return sum(p[i] * m.y[i] for i in m.T) <= Capacidad_total
modelB.cap_total = pyo.Constraint(expr=sum(p[i]*modelB.y[i] for i in T) <= Capacidad_total)

def capDev(m, d):
    return sum(p[i] * m.x[i,d] for i in m.T) <= Cd[d]
modelB.capacidad = pyo.Constraint(D, rule=lambda m,d: sum(p[i]*m.x[i,d] for i in T) <= Cd[d])

def asignacion(m, i):
    return sum(m.x[i,d] for d in m.D) == m.y[i]
modelB.asignacion = pyo.Constraint(T, rule=lambda m,i: sum(m.x[i,d] for d in D) == m.y[i])

# ----------------------
# Resolver Parte B
# ----------------------
solver.solve(modelB)
modelB.display()
print("\nParte B - Asignación de tareas a desarrolladores:")
for i in modelB.T:
    for d in modelB.D:
        if pyo.value(modelB.x[i,d]) == 1:
            print(f"Tarea {i} asignada al desarrollador {d}")

asignacionesB = {(i,d): w[i] for i in T for d in D if pyo.value(modelB.x[i,d]) == 1}
cap_usada_dev = {d: sum(p[i]*pyo.value(modelB.x[i,d]) for i in T) for d in D}
plot_problema1_parteB(asignacionesB, cap_usada_dev, Cd)