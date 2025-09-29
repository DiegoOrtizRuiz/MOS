"""
Módulo para crear y resolver el modelo de optimización.
"""
from typing import Dict, List, Tuple, Set, Optional

import pyomo.environ as pyo
from pyomo.opt import SolverFactory, SolverStatus, TerminationCondition


def create_model(resources: List[str], planes: List[str],
                val: Dict[str, float], stock: Dict[str, float],
                weight: Dict[str, float], volume: Dict[str, float],
                plane_w: Dict[str, float], plane_v: Dict[str, float],
                comp_pairs: Optional[List[Tuple[str, str]]] = None,
                together_pairs: Optional[List[Tuple[str, str]]] = None,
                forbidden_pairs: Optional[List[Tuple[str, str]]] = None) -> pyo.ConcreteModel:
    """
    Crea el modelo de optimización para el problema de asignación.

    Args:
        resources: Lista de nombres de recursos
        planes: Lista de nombres de aviones
        val: Diccionario con valor por recurso
        stock: Diccionario con stock por recurso
        weight: Diccionario con peso por recurso
        volume: Diccionario con volumen por recurso
        plane_w: Diccionario con capacidad de peso por avión
        plane_v: Diccionario con capacidad de volumen por avión
        comp_pairs: Lista de pares de recursos incompatibles (opcional)
        together_pairs: Lista de pares de recursos que deben ir juntos (opcional) 
        forbidden_pairs: Lista de pares (recurso, avión) prohibidos (opcional)

    Returns:
        Modelo Pyomo listo para resolver
    """
    model = pyo.ConcreteModel()

    # Sets
    model.R = pyo.Set(initialize=resources)  # recursos
    model.P = pyo.Set(initialize=planes)  # aviones

    # Diccionarios de parámetros
    model.val = pyo.Param(model.R, initialize=val)  # valor por recurso
    model.stock = pyo.Param(model.R, initialize=stock)  # stock por recurso 
    model.weight = pyo.Param(model.R, initialize=weight)  # peso por recurso
    model.volume = pyo.Param(model.R, initialize=volume)  # volumen por recurso
    model.plane_w = pyo.Param(model.P, initialize=plane_w)  # cap peso por avión
    model.plane_v = pyo.Param(model.P, initialize=plane_v)  # cap volumen por avión

    # Variables
    model.x = pyo.Var(model.R, model.P, domain=pyo.NonNegativeReals)  # cantidad a transportar

    # Función objetivo: maximizar valor total transportado
    def obj_rule(m):
        return sum(m.val[r] * m.x[r,p] for r in m.R for p in m.P)
    model.obj = pyo.Objective(rule=obj_rule, sense=pyo.maximize)

    # Restricción stock disponible
    def stock_rule(m, r):
        return sum(m.x[r,p] for p in m.P) <= m.stock[r]
    model.stock_constr = pyo.Constraint(model.R, rule=stock_rule)

    # Restricción capacidad peso por avión
    def weight_rule(m, p):
        return sum(m.weight[r] * m.x[r,p] for r in m.R) <= m.plane_w[p]
    model.weight_constr = pyo.Constraint(model.P, rule=weight_rule)

    # Restricción capacidad volumen por avión 
    def volume_rule(m, p):
        return sum(m.volume[r] * m.x[r,p] for r in m.R) <= m.plane_v[p]
    model.volume_constr = pyo.Constraint(model.P, rule=volume_rule)

    # Restricciones de incompatibilidad
    if comp_pairs:
        def incomp_rule(m, r1, r2, p):
            if (r1, r2) in comp_pairs or (r2, r1) in comp_pairs:
                return m.x[r1,p] * m.x[r2,p] == 0
            return pyo.Constraint.Skip
        model.incomp_constr = pyo.Constraint(model.R, model.R, model.P, rule=incomp_rule)

    # Restricciones de recursos que deben ir juntos
    if together_pairs:
        def together_rule(m, r1, r2):
            if (r1, r2) in together_pairs or (r2, r1) in together_pairs:
                # Si hay algo de r1 en un avión, debe haber algo de r2 en el mismo avión
                for p in m.P:
                    if m.x[r1,p].value > 0:
                        return m.x[r2,p] > 0
            return pyo.Constraint.Skip
        model.together_constr = pyo.Constraint(model.R, model.R, rule=together_rule)

    # Restricciones de seguridad (recursos prohibidos en ciertos aviones)
    if forbidden_pairs:
        def forbidden_rule(m, r, p):
            if (r, p) in forbidden_pairs:
                return m.x[r,p] == 0
            return pyo.Constraint.Skip
        model.forbidden_constr = pyo.Constraint(model.R, model.P, rule=forbidden_rule)

    return model


def solve_model(model: pyo.ConcreteModel,
                solver_name: str = 'glpk',
                solver_path: Optional[str] = None,
                timeout: Optional[int] = None) -> Tuple[Optional[float], Dict[str, Dict[str, float]]]:
    """
    Resuelve el modelo de optimización.

    Args:
        model: Modelo Pyomo a resolver
        solver_name: Nombre del solver a usar
        solver_path: Ruta al ejecutable del solver (opcional)
        timeout: Tiempo máximo de ejecución en segundos (opcional)

    Returns:
        (valor_objetivo, asignaciones)
        donde asignaciones es un diccionario con la estructura:
        {recurso: {avion: cantidad, ...}, ...}
        En caso de error retorna (None, {})
    """
    try:
        # Configurar solver
        if solver_path:
            solver = SolverFactory(solver_name, executable=solver_path)
        else:
            solver = SolverFactory(solver_name)

        # Configurar tiempo máximo si se especifica
        if timeout:
            solver.options['tmlim'] = timeout

        # Resolver modelo
        result = solver.solve(model, tee=True)

        # Verificar estado de la solución
        if (result.solver.status == SolverStatus.ok and
            result.solver.termination_condition == TerminationCondition.optimal):

            # Extraer valor objetivo
            obj_value = pyo.value(model.obj)

            # Extraer asignaciones
            assignments = {}
            for r in model.R:
                assignments[r] = {}
                for p in model.P:
                    if abs(pyo.value(model.x[r,p])) > 1e-6:  # filtrar valores muy cercanos a 0
                        assignments[r][p] = pyo.value(model.x[r,p])

            return obj_value, assignments

        else:
            print("Error: No se encontró solución óptima")
            print(f"Estado del solver: {result.solver.status}")
            print(f"Condición de terminación: {result.solver.termination_condition}")
            return None, {}

    except Exception as e:
        print(f"Error al resolver el modelo: {e}")
        return None, {}


def get_sensitivities(model: pyo.ConcreteModel,
                     objective_value: float,
                     assignments: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """
    Calcula análisis de sensibilidad para las restricciones activas.

    Args:
        model: Modelo Pyomo resuelto
        objective_value: Valor de la función objetivo en el óptimo
        assignments: Diccionario con las asignaciones óptimas

    Returns:
        Diccionario con información de sensibilidad por restricción
    """
    sensitivity = {}
    
    # Sensibilidad para restricciones de stock
    sensitivity['stock'] = {}
    for r in model.R:
        if model.stock_constr[r].dual is not None:
            sensitivity['stock'][r] = {
                'shadow_price': model.stock_constr[r].dual,
                'current_value': model.stock[r],
                'slack': (model.stock[r] - 
                         sum(assignments.get(r, {}).get(p, 0) for p in model.P))
            }

    # Sensibilidad para restricciones de peso
    sensitivity['weight'] = {}
    for p in model.P:
        if model.weight_constr[p].dual is not None:
            curr_weight = sum(model.weight[r] * assignments.get(r, {}).get(p, 0)
                            for r in model.R)
            sensitivity['weight'][p] = {
                'shadow_price': model.weight_constr[p].dual,
                'current_value': model.plane_w[p],
                'slack': model.plane_w[p] - curr_weight
            }

    # Sensibilidad para restricciones de volumen
    sensitivity['volume'] = {}
    for p in model.P:
        if model.volume_constr[p].dual is not None:
            curr_volume = sum(model.volume[r] * assignments.get(r, {}).get(p, 0)
                            for r in model.R)
            sensitivity['volume'][p] = {
                'shadow_price': model.volume_constr[p].dual,
                'current_value': model.plane_v[p],
                'slack': model.plane_v[p] - curr_volume
            }

    return sensitivity