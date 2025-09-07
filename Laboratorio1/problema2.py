import pyomo.environ as pyo
from graficos import plot_problema2


# ----------------------
# Datos
# ----------------------
W = [1, 2, 3]  # trabajadores
J = [1, 2, 3, 4, 5]  # trabajos
H = {1: 10, 2: 12, 3: 15}  # capacidad horaria
g = {1: 50, 2: 40, 3: 70, 4: 30, 5: 20}  # ganancia
h = {1: 5, 2: 6, 3: 7, 4: 8, 5: 4}       # horas requeridas

solver = pyo.SolverFactory("glpk")

# ----------------------
# Modelo base
# ----------------------
def construir_modelo(restricciones=False):
    m = pyo.ConcreteModel()
    m.W, m.J = pyo.Set(initialize=W), pyo.Set(initialize=J)
    m.x = pyo.Var(m.W, m.J, domain=pyo.Binary)
    m.obj = pyo.Objective(expr=sum(g[j]*m.x[w,j] for w in W for j in J), sense=pyo.maximize)
    m.unique = pyo.Constraint(m.J, rule=lambda m,j: sum(m.x[w,j] for w in W) <= 1)
    m.cap = pyo.Constraint(m.W, rule=lambda m,w: sum(h[j]*m.x[w,j] for j in J) <= H[w])
    if restricciones:
        m.x[2,1].fix(0); m.x[3,1].fix(0); m.x[2,3].fix(0)
    return m

# ----------------------
# Parte A
# ----------------------
modelA = construir_modelo()
solver.solve(modelA)
asignacionesA = {(w,j): int(pyo.value(modelA.x[w,j])) for w in W for j in J}
gananciasA = {w: sum(g[j]*pyo.value(modelA.x[w,j]) for j in J) for w in W}
horas_usadasA = {w: sum(h[j]*pyo.value(modelA.x[w,j]) for j in J) for w in W}
plot_problema2(asignacionesA, gananciasA, horas_usadasA, H)
modelA.display()
print("=== Parte A: Asignación óptima de trabajos ===")
for w in modelA.W:
    for j in modelA.J:
        if pyo.value(modelA.x[w,j]) == 1:
            print(f"Trabajador {w} realiza trabajo {j}")
print("\n")

# ----------------------
# Parte B: Modelo con restricciones adicionales
# ----------------------
modelB = construir_modelo(restricciones=True)
solver.solve(modelB)
asignacionesB = {(w,j): int(pyo.value(modelB.x[w,j])) for w in W for j in J}
gananciasB = {w: sum(g[j]*pyo.value(modelB.x[w,j]) for j in J) for w in W}
horas_usadasB = {w: sum(h[j]*pyo.value(modelB.x[w,j]) for j in J) for w in W}
plot_problema2(asignacionesB, gananciasB, horas_usadasB, H)
modelB.display()
print("=== Parte B: Asignación óptima con restricciones adicionales ===")
for w in modelB.W:
    for j in modelB.J:
        if pyo.value(modelB.x[w,j]) == 1:
            print(f"Trabajador {w} realiza trabajo {j}")
