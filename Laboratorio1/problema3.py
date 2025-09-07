import pyomo.environ as pyo
from graficos import plot_problema3


# ----------------------
# Datos
# ----------------------
R = ["agua", "medicina", "equipos"]
A = [1, 2]

v = {"agua": 5, "medicina": 50, "equipos": 40}  # valor por tonelada
s = {"agua": 100, "medicina": 30, "equipos": 20} # stock disponible
u = {"agua": 1, "medicina": 2, "equipos": 3}     # volumen por tonelada

W = {1: 60, 2: 50}  # capacidad en peso
V = {1: 100, 2: 80} # capacidad en volumen

solver = pyo.SolverFactory("glpk")

# ----------------------
# Modelo básico
# ----------------------
def construir_modelo(restricciones=False):
    m = pyo.ConcreteModel()
    m.R, m.A = pyo.Set(initialize=R), pyo.Set(initialize=A)
    m.x = pyo.Var(m.R, m.A, domain=pyo.NonNegativeIntegers)
    m.y = pyo.Var(m.R, m.A, domain=pyo.Binary)
    m.obj = pyo.Objective(expr=sum(v[r]*m.x[r,a] for r in R for a in A), sense=pyo.maximize)
    m.stock = pyo.Constraint(m.R, rule=lambda m,r: sum(m.x[r,a] for a in A) <= s[r])
    m.peso = pyo.Constraint(m.A, rule=lambda m,a: sum(m.x[r,a] for r in R) <= W[a])
    m.vol = pyo.Constraint(m.A, rule=lambda m,a: sum(u[r]*m.x[r,a] for r in R) <= V[a])
    if restricciones:
        m.x["medicina",1].fix(0)
        m.incompat = pyo.Constraint(m.A, rule=lambda m,a: m.y["equipos",a] + m.y["agua",a] <= 1)
        m.relacion = pyo.Constraint(m.R, m.A, rule=lambda m,r,a: m.x[r,a] <= s[r]*m.y[r,a])
    return m

# ----------------------
# Parte A
# ----------------------
modelA = construir_modelo()
solver.solve(modelA)
asignacionesA = {(r,a): int(pyo.value(modelA.x[r,a])) for r in R for a in A}
peso_usadoA = {a: sum(pyo.value(modelA.x[r,a]) for r in R) for a in A}
vol_usadoA = {a: sum(u[r]*pyo.value(modelA.x[r,a]) for r in R) for a in A}
plot_problema3(asignacionesA, peso_usadoA, W, vol_usadoA, V)
modelA.display()
print("=== Parte A: Asignación óptima de recursos ===")
for r in modelA.R:
    for a in modelA.A:
        cantidad = pyo.value(modelA.x[r,a])
        if cantidad > 0:
            print(f"{cantidad} toneladas de {r} en avión {a}")
print("\n")

# ----------------------
# Parte B
# ----------------------
modelB = construir_modelo(restricciones=True)
solver.solve(modelB)
asignacionesB = {(r,a): int(pyo.value(modelB.x[r,a])) for r in R for a in A}
peso_usadoB = {a: sum(pyo.value(modelB.x[r,a]) for r in R) for a in A}
vol_usadoB = {a: sum(u[r]*pyo.value(modelB.x[r,a]) for r in R) for a in A}
plot_problema3(asignacionesB, peso_usadoB, W, vol_usadoB, V)
modelB.display()
print("=== Parte B: Asignación óptima con restricciones adicionales ===")
for r in modelB.R:
    for a in modelB.A:
        cantidad = pyo.value(modelB.x[r,a])
        if cantidad > 0:
            print(f"{cantidad} toneladas de {r} en avión {a}")
