# modelo_problema1_viz.py
"""
Versión extendida con:
 - Generación de gráficas (matplotlib)
 - Análisis de sensibilidad básico (+20% capacidad del avión más limitante)
 - Entrada opcional por JSON
Salida:
 - resultado_asignacion.csv
 - resultado_resumen.txt
 - utilizacion_por_avion.png
 - distribucion_valor_por_avion.png
 - matriz_asignacion.png
 - sensibilidad_resumen.txt
Requisitos:
    pip install pyomo pandas matplotlib
    Tener un solver instalado (glpk, cbc, gurobi, ...)
Uso:
    python modelo_problema1_viz.py --solver glpk
    python modelo_problema1_viz.py --data datos_ejemplo.json --solver cbc
"""
import argparse
import json
from collections import defaultdict

import pandas as pd
import matplotlib.pyplot as plt
from pyomo.environ import *

# ---------------------------
# Datos por defecto (enunciado)
# ---------------------------
DEFAULT_RESOURCES = [
    ("Alimentos_Basicos", 250, 45.0, 1.0, 0.8),
    ("Medicinas", 800, 12.0, 1.0, 0.4),
    ("Equipos_Medicos", 2400, 24.0, 3.0, 2.5),
    ("Agua_Potable", 180, 60.0, 1.0, 1.2),
    ("Mantas", 320, 62.5, 2.5, 1.8),
    ("Generadores", 3200, 20.0, 5.0, 3.2),
    ("Tiendas_Campana", 450, 27.0, 1.5, 4.0),
    ("Medicamentos_Especiales", 1200, 8.0, 1.0, 0.3),
    ("Equipos_Comunicacion", 1800, 12.0, 2.0, 1.0),
    ("Material_Construccion", 150, 80.0, 1.0, 0.6),
]

DEFAULT_PLANES = [
    ("Hercules-1", 28, 22),
    ("Hercules-2", 32, 26),
    ("Galaxy-1", 45, 38),
    ("Galaxy-2", 48, 42),
    ("Antonov-1", 65, 55),
    ("Antonov-2", 70, 60),
]

DEFAULT_COMPAT = [
    ("Equipos_Medicos", "Agua_Potable", "incompatible"),
    ("Medicinas", "Medicamentos_Especiales", "together"),
    ("Generadores", "Material_Construccion", "incompatible"),
    ("Equipos_Comunicacion", "Equipos_Medicos", "together"),
    ("Alimentos_Basicos", "Medicinas", "incompatible"),
    ("Tiendas_Campana", "Material_Construccion", "together"),
    ("Agua_Potable", "Alimentos_Basicos", "together"),
]

DEFAULT_SECURITY = [
    ("Medicinas", "Hercules-1", False),
    ("Medicinas", "Hercules-2", False),
    ("Medicamentos_Especiales", "Hercules-1", False),
    ("Medicamentos_Especiales", "Hercules-2", False),
    ("Generadores", "Hercules-1", False),
    ("Generadores", "Hercules-2", False),
    ("Generadores", "Galaxy-1", False),
    ("Generadores", "Galaxy-2", False),
    ("Generadores", "Antonov-1", True),
    ("Generadores", "Antonov-2", True),
    ("Equipos_Comunicacion", "Hercules-1", False),
]

# ---------------------------
# Utilidades
# ---------------------------
def load_data_from_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print("No se pudo leer JSON:", e)
        return None

def build_indexed_data(resources_list, planes_list):
    resources = [r[0] for r in resources_list]
    planes = [p[0] for p in planes_list]
    val = {r[0]: r[1] for r in resources_list}
    stock = {r[0]: r[2] for r in resources_list}
    weight = {r[0]: r[3] for r in resources_list}
    volume = {r[0]: r[4] for r in resources_list}
    plane_w = {p[0]: p[1] for p in planes_list}
    plane_v = {p[0]: p[2] for p in planes_list}
    return resources, planes, val, stock, weight, volume, plane_w, plane_v

def auto_determine_divisibility(resources_list):
    divis = {}
    for name, _, stock, w, vol in resources_list:
        is_indiv = (w >= 2.0) or (vol >= 3.0) or (stock < 30 and w >= 1.5)
        divis[name] = not is_indiv
    divis["Agua_Potable"] = True
    divis["Alimentos_Basicos"] = True
    divis["Material_Construccion"] = True
    return divis

def preprocess_compatibility(comp_list):
    incompat = set()
    together = set()
    for a, b, rel in comp_list:
        if rel.lower().startswith("incompat"):
            pair = tuple(sorted((a, b)))
            incompat.add(pair)
        elif rel.lower().startswith("together"):
            pair = tuple(sorted((a, b)))
            together.add(pair)
    conflic = incompat.intersection(together)
    if conflic:
        raise ValueError(f"Conflicting rules for pairs: {conflic}")
    return incompat, together

def preprocess_security(sec_list, planes, resources):
    allowed = {}
    for r in resources:
        for p in planes:
            allowed[(r, p)] = True
    for r, p, flag in sec_list:
        if (r, p) in allowed:
            allowed[(r, p)] = flag
    return allowed

# ---------------------------
# Model builder
# ---------------------------
def build_pyomo_model(resources, planes, val, stock, weight, volume, plane_w, plane_v, divisibles, incompat_pairs, together_pairs, allowed):
    model = ConcreteModel()
    model.R = Set(initialize=resources)
    model.A = Set(initialize=planes)
    model.value = Param(model.R, initialize=lambda m, r: val[r])
    model.stock = Param(model.R, initialize=lambda m, r: stock[r])
    model.w = Param(model.R, initialize=lambda m, r: weight[r])
    model.vol = Param(model.R, initialize=lambda m, r: volume[r])
    model.cap_w = Param(model.A, initialize=lambda m, a: plane_w[a])
    model.cap_v = Param(model.A, initialize=lambda m, a: plane_v[a])
    model.x = Var(model.R, model.A, domain=NonNegativeReals)
    model.y = Var(model.R, model.A, domain=Binary)

    def bigM_link(m, r, a):
        return m.x[r, a] <= m.stock[r] * m.y[r, a]
    model.link_bigM = Constraint(model.R, model.A, rule=bigM_link)

    def cap_weight_rule(m, a):
        return sum(m.w[r] * m.x[r, a] for r in m.R) <= m.cap_w[a]
    model.cap_weight = Constraint(model.A, rule=cap_weight_rule)

    def cap_volume_rule(m, a):
        return sum(m.vol[r] * m.x[r, a] for r in m.R) <= m.cap_v[a]
    model.cap_volume = Constraint(model.A, rule=cap_volume_rule)

    def stock_rule(m, r):
        return sum(m.x[r, a] for a in m.A) <= m.stock[r]
    model.stock_constr = Constraint(model.R, rule=stock_rule)

    model.incompat_constraints = ConstraintList()
    for (i, j) in incompat_pairs:
        for a in planes:
            model.incompat_constraints.add(model.y[i, a] + model.y[j, a] <= 1)

    model.together_constraints = ConstraintList()
    for (i, j) in together_pairs:
        for a in planes:
            model.together_constraints.add(model.y[i, a] - model.y[j, a] == 0)

    model.security_constraints = ConstraintList()
    for r in resources:
        for a in planes:
            if not allowed.get((r, a), True):
                model.security_constraints.add(model.y[r, a] == 0)

    indivisible = [r for r in resources if not divisibles.get(r, True)]
    if indivisible:
        model.z = Var(indivisible, model.A, domain=NonNegativeIntegers)
        def link_z_x(m, r, a):
            return m.z[r, a] == m.x[r, a]
        model.link_z = Constraint(indivisible, model.A, rule=link_z_x)

    def obj_rule(m):
        return sum(m.value[r] * m.x[r, a] for r in m.R for a in m.A)
    model.obj = Objective(rule=obj_rule, sense=maximize)
    return model

# ---------------------------
# Plots (matplotlib)
# ---------------------------
def plot_utilization(model, out_png="utilizacion_por_avion.png"):
    R = list(model.R); A = list(model.A)
    caps_w = [value(model.cap_w[a]) for a in A]
    weights = [sum(value(model.w[r])*value(model.x[r,a]) for r in R) for a in A]
    util_percent = [ (weights[i]/caps_w[i])*100.0 if caps_w[i]>0 else 0.0 for i in range(len(A)) ]
    fig, ax = plt.subplots(figsize=(8,4))
    ax.bar(A, util_percent)
    ax.set_xlabel("Avión"); ax.set_ylabel("Utilización de peso (%)")
    ax.set_title("Utilización de peso por avión (%)")
    fig.tight_layout(); fig.savefig(out_png); plt.close(fig)

def plot_value_distribution(model, out_png="distribucion_valor_por_avion.png"):
    R = list(model.R); A = list(model.A)
    vals = [ sum(value(model.value[r])*value(model.x[r,a]) for r in R) for a in A ]
    fig, ax = plt.subplots(figsize=(6,6))
    if sum(vals) <= 0:
        ax.text(0.5,0.5,"Sin valor transportado", horizontalalignment='center', verticalalignment='center')
    else:
        ax.pie(vals, labels=A, autopct='%1.1f%%')
    ax.set_title("Distribución del valor transportado por avión")
    fig.tight_layout(); fig.savefig(out_png); plt.close(fig)

def plot_assignment_matrix(model, out_png="matriz_asignacion.png"):
    import numpy as np
    R = list(model.R); A = list(model.A)
    M = [[ value(model.x[r,a]) for a in A ] for r in R ]
    M = np.array(M)
    fig, ax = plt.subplots(figsize=(9,6))
    im = ax.imshow(M, aspect='auto')
    ax.set_yticks(range(len(R))); ax.set_yticklabels(R)
    ax.set_xticks(range(len(A))); ax.set_xticklabels(A, rotation=45, ha='right')
    ax.set_title("Matriz de asignación (unidades por avión)")
    fig.colorbar(im, ax=ax); fig.tight_layout(); fig.savefig(out_png); plt.close(fig)

# ---------------------------
# Sensitivity analysis
# ---------------------------
def sensitivity_increase_plane_and_solve(resources_list, planes_list, model_builder, divisibles, incompat_pairs, together_pairs, allowed, solver_name="glpk"):
    resources, planes, val, stock, weight, volume, plane_w, plane_v = build_indexed_data(resources_list, planes_list)
    avg_w = sum(weight[r] for r in resources)/len(resources)
    avg_v = sum(volume[r] for r in resources)/len(resources)
    scores = {}
    for p in planes:
        scores[p] = min(plane_w[p]/(avg_w+1e-6), plane_v[p]/(avg_v+1e-6))
    most_limiting = min(scores, key=lambda k: scores[k])
    modified_planes = []
    for name, cw, cv in planes_list:
        if name == most_limiting:
            modified_planes.append((name, cw*1.2, cv*1.2))
        else:
            modified_planes.append((name, cw, cv))
    # Build and solve original and modified (if solver available)
    resources_o, planes_o, val_o, stock_o, weight_o, volume_o, plane_w_o, plane_v_o = build_indexed_data(resources_list, planes_list)
    model_o = model_builder(resources_o, planes_o, val_o, stock_o, weight_o, volume_o, plane_w_o, plane_v_o, divisibles, incompat_pairs, together_pairs, allowed)
    resources_m, planes_m, val_m, stock_m, weight_m, volume_m, plane_w_m, plane_v_m = build_indexed_data(resources_list, modified_planes)
    model_m = model_builder(resources_m, planes_m, val_m, stock_m, weight_m, volume_m, plane_w_m, plane_v_m, divisibles, incompat_pairs, together_pairs, allowed)
    solver = SolverFactory(solver_name)
    val_o = None; val_m = None
    try:
        solver.solve(model_o, tee=False)
        val_o = value(model_o.obj)
    except Exception as e:
        print("No se pudo resolver modelo original en sensibilidad:", e)
    try:
        solver.solve(model_m, tee=False)
        val_m = value(model_m.obj)
    except Exception as e:
        print("No se pudo resolver modelo modificado en sensibilidad:", e)
    return {"most_limiting_plane": most_limiting, "orig_value": val_o, "mod_value": val_m}

# ---------------------------
# Main
# ---------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=None, help='Archivo JSON opcional')
    parser.add_argument('--solver', type=str, default='glpk', help='Solver a usar')
    args = parser.parse_args()

    data = None
    if args.data:
        data = load_data_from_json(args.data)

    resources_list = DEFAULT_RESOURCES
    planes_list = DEFAULT_PLANES
    comp_list = DEFAULT_COMPAT
    sec_list = DEFAULT_SECURITY

    resources, planes, val, stock, weight, volume, plane_w, plane_v = build_indexed_data(resources_list, planes_list)
    divisibles = auto_determine_divisibility(resources_list)
    incompat_pairs, together_pairs = preprocess_compatibility(comp_list)
    allowed = preprocess_security(sec_list, planes, resources)

    model = build_pyomo_model(resources, planes, val, stock, weight, volume, plane_w, plane_v, divisibles, incompat_pairs, together_pairs, allowed)

    # Solve (if solver present). If not, warn.
    solver = SolverFactory(args.solver)
    solved = False
    try:
        solver.solve(model, tee=False)
        solved = True
    except Exception as e:
        print("Advertencia: no se pudo ejecutar el solver aquí. Error:", e)
        solved = False

    if solved:
        R = list(model.R); A = list(model.A)
        rows = []
        for r in R:
            for a in A:
                q = value(model.x[r,a]) or 0.0
                if q > 1e-6:
                    rows.append({"recurso": r, "avion": a, "unidades": float(q), "valor_usd": float(q*value(model.value[r]))})
        df = pd.DataFrame(rows)
        df.to_csv("resultado_asignacion.csv", index=False)
        total_val = df["valor_usd"].sum() if not df.empty else 0.0
        with open("resultado_resumen.txt", "w", encoding="utf-8") as f:
            f.write(f"Valor total transportado (USD): {total_val}\n")
            for a in A:
                w_used = sum(value(model.w[r]) * value(model.x[r,a]) for r in R)
                v_used = sum(value(model.vol[r]) * value(model.x[r,a]) for r in R)
                f.write(f"{a}: peso={w_used}, volumen={v_used}\n")
        plot_utilization(model, out_png="utilizacion_por_avion.png")
        plot_value_distribution(model, out_png="distribucion_valor_por_avion.png")
        plot_assignment_matrix(model, out_png="matriz_asignacion.png")
        print("Resultados y gráficas generadas.")
    else:
        with open("resultado_resumen.txt", "w", encoding="utf-8") as f:
            f.write("No se resolvió el modelo en este entorno. Ejecuta localmente con un solver.\n")
        print("No se resolvió. Ejecuta localmente con solver para obtener salida completa.")

    sens = sensitivity_increase_plane_and_solve(resources_list, planes_list, build_pyomo_model, divisibles, incompat_pairs, together_pairs, allowed, solver_name=args.solver)
    with open("sensibilidad_resumen.txt", "w", encoding="utf-8") as f:
        f.write(f"Most limiting plane: {sens.get('most_limiting_plane')}\n")
        f.write(f"Original objective: {sens.get('orig_value')}\n")
        f.write(f"Modified (+20% capacity) objective: {sens.get('mod_value')}\n")
    print("Análisis de sensibilidad guardado en sensibilidad_resumen.txt")

if __name__ == '__main__':
    main()
