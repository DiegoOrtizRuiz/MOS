"""
Módulo principal que integra todos los componentes del programa.
"""
import argparse
from pathlib import Path
from typing import Optional

from preprocessing import load_data_from_json, build_indexed_data
from model import create_model, solve_model, get_sensitivities
from visualization import (plot_value_distribution, plot_utilization,
                         plot_assignment_matrix)
from reporting import (save_assignments_csv, generate_summary,
                      write_sensitivity_report)

# Datos por defecto si no se especifica archivo JSON
DEFAULT_RESOURCES = [
    ("R1", 1000, 100, 1.5, 2.0),  # nombre, valor, stock, peso, volumen
    ("R2", 800, 80, 1.2, 1.8),
    ("R3", 1200, 120, 1.8, 2.2)
]

DEFAULT_PLANES = [
    ("P1", 50, 70),  # nombre, cap_peso, cap_volumen
    ("P2", 60, 85),
    ("P3", 45, 65)
]

DEFAULT_COMPAT = [
    ("R1", "R2", "incompatible"),  # recurso1, recurso2, tipo_regla
    ("R2", "R3", "together")
]

DEFAULT_SECURITY = [
    ("R3", "P1", False)  # recurso, avión, permitido
]


def main(json_path: Optional[str] = None,
         solver: str = 'glpk',
         solver_path: Optional[str] = None,
         timeout: Optional[int] = None):
    """
    Función principal que ejecuta el flujo completo del programa.

    Args:
        json_path: Ruta al archivo JSON con datos de entrada (opcional)
        solver: Nombre del solver a utilizar
        solver_path: Ruta al ejecutable del solver (opcional)
        timeout: Tiempo máximo de ejecución en segundos (opcional)
    """
    print("\n=== INICIANDO PROGRAMA DE ASIGNACIÓN DE RECURSOS ===\n")

    # Cargar datos
    if json_path:
        print(f"Cargando datos desde {json_path}...")
        resources, planes, comp, sec = load_data_from_json(json_path)
        if not all([resources, planes]):
            print("Error al cargar datos del JSON. Usando valores por defecto.")
            resources, planes = DEFAULT_RESOURCES, DEFAULT_PLANES
            comp, sec = DEFAULT_COMPAT, DEFAULT_SECURITY
    else:
        print("Usando datos por defecto...")
        resources, planes = DEFAULT_RESOURCES, DEFAULT_PLANES
        comp, sec = DEFAULT_COMPAT, DEFAULT_SECURITY

    # Construir estructuras de datos indexadas
    print("\nPreparando datos para el modelo...")
    resources_idx, planes_idx, val, stock, weight, volume, plane_w, plane_v = \
        build_indexed_data(resources, planes)

    # Procesar reglas de compatibilidad
    comp_pairs = []  # pares incompatibles
    together_pairs = []  # pares que deben ir juntos
    if comp:
        for r1, r2, rule in comp:
            if rule == "incompatible":
                comp_pairs.append((r1, r2))
            elif rule == "together":
                together_pairs.append((r1, r2))

    # Procesar reglas de seguridad
    forbidden_pairs = []  # pares (recurso, avión) prohibidos
    if sec:
        forbidden_pairs = [(r, p) for r, p, allowed in sec if not allowed]

    # Crear y resolver modelo
    print("\nCreando modelo de optimización...")
    model = create_model(resources_idx, planes_idx, val, stock, weight, volume,
                        plane_w, plane_v, comp_pairs, together_pairs, forbidden_pairs)

    print(f"\nResolviendo modelo usando solver {solver}...")
    obj_value, assignments = solve_model(model, solver, solver_path, timeout)

    if obj_value is None:
        print("\nError: No se pudo resolver el modelo.")
        return

    print(f"\nSolución encontrada! Valor total: ${obj_value:,.2f}")

    # Análisis de sensibilidad
    print("\nRealizando análisis de sensibilidad...")
    sensitivity = get_sensitivities(model, obj_value, assignments)

    # Generar visualizaciones
    print("\nGenerando gráficos...")
    plot_value_distribution(assignments, val)
    plot_utilization(assignments, plane_w, plane_v, weight, volume)
    plot_assignment_matrix(assignments)

    # Generar reportes
    print("\nGenerando reportes...")
    save_assignments_csv(assignments)
    generate_summary(assignments, val, weight, volume, plane_w, plane_v, obj_value)
    write_sensitivity_report(sensitivity)

    print("\n=== PROGRAMA COMPLETADO CON ÉXITO ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Programa de asignación óptima de recursos a aviones."
    )
    parser.add_argument(
        "--json",
        type=str,
        help="Ruta al archivo JSON con datos de entrada"
    )
    parser.add_argument(
        "--solver",
        type=str,
        default="glpk",
        help="Solver a utilizar (default: glpk)"
    )
    parser.add_argument(
        "--solver-path",
        type=str,
        help="Ruta al ejecutable del solver"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        help="Tiempo máximo de ejecución en segundos"
    )

    args = parser.parse_args()
    main(args.json, args.solver, args.solver_path, args.timeout)