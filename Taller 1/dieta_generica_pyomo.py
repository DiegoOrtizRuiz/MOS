from pyomo.environ import ConcreteModel, Set, Param, Var, NonNegativeReals, Objective, Constraint, SolverFactory, value
import sys

def build_generic_diet_model(food_ids, nutrient_ids, c, A, L, U):
    """
    Construye y devuelve un modelo Pyomo genérico del problema de la dieta.
    - food_ids: lista/iterable de índices de alimentos (por ejemplo [1,2,3,4])
    - nutrient_ids: lista/iterable de índices de nutrientes (por ejemplo [1..5])
    - c: dict {i: costo_por_porcion}
    - A: dict {(j,i): aporte del alimento i en el nutriente j}
    - L: dict {j: cota inferior (puede ser -inf)}
    - U: dict {j: cota superior (puede ser +inf)}
    """
    model = ConcreteModel(name="DietaGenérica")
    model.F = Set(initialize=list(food_ids))
    model.N = Set(initialize=list(nutrient_ids))

    model.c = Param(model.F, initialize=c, within=None)
    model.A = Param(model.N, model.F, initialize=A, within=None)
    model.L = Param(model.N, initialize=L, within=None)
    model.U = Param(model.N, initialize=U, within=None)

    # Variables: cantidad de porciones (no negativas)
    model.x = Var(model.F, domain=NonNegativeReals)

    # Objetivo: minimizar costo
    def obj_rule(m):
        return sum(m.c[i] * m.x[i] for i in m.F)
    model.OBJ = Objective(rule=obj_rule, sense=1)  # sense=1 -> minimize

    # Restricción: para cada nutriente, L_j <= sum_i A_{j,i} x_i <= U_j
    def lower_rule(m, j):
        # Si L[j] es None o -inf, no creamos la restricción de cota inferior
        Lj = value(m.L[j])
        if Lj == float("-inf"):
            return Constraint.Skip
        return sum(m.A[j, i] * m.x[i] for i in m.F) >= Lj
    model.Lower = Constraint(model.N, rule=lower_rule)

    def upper_rule(m, j):
        Uj = value(m.U[j])
        if Uj == float("inf"):
            return Constraint.Skip
        return sum(m.A[j, i] * m.x[i] for i in m.F) <= Uj
    model.Upper = Constraint(model.N, rule=upper_rule)

    return model

def instantiate_example():
    # Índices de alimentos (1..4)
    foods = [1,2,3,4]  # 1=carne(100g), 2=arroz(1 taza), 3=leche(1 taza), 4=pan(100g)
    nutrients = [1,2,3,4,5]  # 1=Calorías,2=Proteínas,3=Azúcares,4=Grasas,5=Carbohidratos

    # Costos por porción (COP)
    c = {1:3000.0, 2:1000.0, 3:600.0, 4:700.0}

    # Matriz A (a_{j,i}) con filas = nutrientes, columnas = alimentos
    A = {
        (1,1):287.0, (1,2):204.0, (1,3):146.0, (1,4):245.0,
        (2,1):26.0,  (2,2):4.2,   (2,3):8.0,   (2,4):6.0,
        (3,1):0.0,   (3,2):0.01,  (3,3):13.0,  (3,4):25.0,
        (4,1):19.3,  (4,2):0.5,   (4,3):8.0,   (4,4):0.8,
        (5,1):0.0,   (5,2):44.1,  (5,3):11.0,  (5,4):55.0
    }

    # Cotas L (inferior) y U (superior). Usamos +-inf donde aplique.
    neg_inf = float("-inf")
    pos_inf = float("inf")
    L = {1:1500.0, 2:63.0, 3:neg_inf, 4:neg_inf, 5:neg_inf}
    U = {1:pos_inf, 2:pos_inf, 3:25.0,    4:50.0,    5:200.0}

    return foods, nutrients, c, A, L, U

def pretty_print_solution(model):
    print("Solución:")
    try:
        for i in model.F:
            xi = value(model.x[i])
            print(f"  x[{i}] = {xi:.6f}")
        print(f"Costo objetivo = {value(model.OBJ):.2f} COP")
    except Exception as e:
        print("No hay solución disponible o no se resolvió el modelo. Error:", e)


def main():
    foods, nutrients, c, A, L, U = instantiate_example()

    model = build_generic_diet_model(foods, nutrients, c, A, L, U)
    print("Modelo genérico (instanciado) creado con éxito. Intentando resolver...")

    # Intentar resolver con GLPK (u otro solver disponible). Si no hay solver, informar.
    solvers_to_try = ["glpk", "cbc", "cplex", "gurobi", "ipopt"]
    solver_found = None
    for sname in solvers_to_try:
        try:
            solver = SolverFactory(sname)
            if solver.available(exception_flag=False):
                solver_found = sname
                break
        except Exception:
            continue

    if solver_found is None:
        return

    solver = SolverFactory(solver_found)
    results = solver.solve(model, tee=True)
    # Cargar solución en el modelo
    model.solutions.load_from(results)
    print("Solve finalizado. Estado:", results.solver.status, results.solver.termination_condition)
    pretty_print_solution(model)

if __name__ == "__main__":
    main()
