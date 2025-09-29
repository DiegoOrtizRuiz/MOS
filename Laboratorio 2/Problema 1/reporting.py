"""
Funciones utilitarias para análisis de resultados y generación de reportes.
"""
from typing import Dict, List, Tuple
import csv


def save_assignments_csv(assignments: Dict[str, Dict[str, float]],
                        filepath: str = "resultado_asignacion.csv"):
    """
    Guarda la matriz de asignación en un archivo CSV.

    Args:
        assignments: Diccionario con asignaciones {recurso: {avion: cantidad}}
        filepath: Ruta del archivo CSV a generar
    """
    # Obtener lista única de aviones
    planes = list({p for d in assignments.values() for p in d.keys()})
    
    # Escribir CSV
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # Escribir encabezados
        writer.writerow(['Recurso'] + planes)
        
        # Escribir datos
        for resource in assignments:
            row = [resource]
            for plane in planes:
                row.append(assignments[resource].get(plane, 0))
            writer.writerow(row)


def generate_summary(assignments: Dict[str, Dict[str, float]],
                    values: Dict[str, float],
                    weights: Dict[str, float],
                    volumes: Dict[str, float],
                    plane_weights: Dict[str, float],
                    plane_volumes: Dict[str, float],
                    obj_value: float,
                    filepath: str = "resultado_resumen.txt"):
    """
    Genera un resumen detallado de la solución.

    Args:
        assignments: Diccionario con asignaciones {recurso: {avion: cantidad}}
        values: Diccionario con valor por unidad de cada recurso
        weights: Diccionario con peso por unidad de cada recurso
        volumes: Diccionario con volumen por unidad de cada recurso
        plane_weights: Diccionario con capacidad de peso por avión
        plane_volumes: Diccionario con capacidad de volumen por avión
        obj_value: Valor de la función objetivo
        filepath: Ruta del archivo de resumen a generar
    """
    with open(filepath, 'w', encoding='utf-8') as f:
        # Valor total
        f.write(f"Valor Total de la Solución: ${obj_value:,.2f}\n\n")
        
        # Resumen por avión
        f.write("Resumen por Avión:\n")
        f.write("-" * 50 + "\n")
        
        for plane in plane_weights.keys():
            f.write(f"\nAvión: {plane}\n")
            
            # Calcular peso y volumen usado
            weight_used = sum(weights[r] * amt 
                            for r, p_data in assignments.items() 
                            for p, amt in p_data.items() 
                            if p == plane)
            volume_used = sum(volumes[r] * amt 
                            for r, p_data in assignments.items() 
                            for p, amt in p_data.items() 
                            if p == plane)
            
            # Calcular valor transportado
            value = sum(values[r] * amt 
                       for r, p_data in assignments.items() 
                       for p, amt in p_data.items() 
                       if p == plane)
            
            # Escribir estadísticas
            f.write(f"  Valor transportado: ${value:,.2f}\n")
            f.write(f"  Peso usado: {weight_used:.2f} / {plane_weights[plane]:.2f} ton ")
            f.write(f"({(weight_used/plane_weights[plane])*100:.1f}%)\n")
            f.write(f"  Volumen usado: {volume_used:.2f} / {plane_volumes[plane]:.2f} m³ ")
            f.write(f"({(volume_used/plane_volumes[plane])*100:.1f}%)\n")
            
            # Detalle de recursos
            f.write("  Recursos transportados:\n")
            for resource in assignments:
                if plane in assignments[resource]:
                    amt = assignments[resource][plane]
                    if amt > 0:
                        f.write(f"    - {resource}: {amt:.2f} ton\n")
        
        f.write("\nNota: Cantidades pueden tener pequeñas variaciones por redondeo.\n")


def write_sensitivity_report(sensitivity: Dict[str, Dict[str, Dict[str, float]]],
                           filepath: str = "sensibilidad_resumen.txt"):
    """
    Genera un reporte detallado del análisis de sensibilidad.

    Args:
        sensitivity: Diccionario con información de sensibilidad por restricción
        filepath: Ruta del archivo a generar
    """
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("ANÁLISIS DE SENSIBILIDAD\n")
        f.write("=" * 50 + "\n\n")
        
        # Restricciones de stock
        f.write("RESTRICCIONES DE STOCK\n")
        f.write("-" * 50 + "\n")
        for resource, data in sensitivity['stock'].items():
            f.write(f"\nRecurso: {resource}\n")
            f.write(f"  Valor actual: {data['current_value']:.2f} ton\n")
            f.write(f"  Holgura: {data['slack']:.2f} ton\n")
            f.write(f"  Precio sombra: ${data['shadow_price']:.2f}/ton\n")
            
            if abs(data['shadow_price']) < 1e-6:
                f.write("  → Esta restricción no es limitante\n")
            else:
                f.write(f"  → Aumentar el stock en 1 ton incrementaría ")
                f.write(f"el valor objetivo en ${data['shadow_price']:.2f}\n")
        
        # Restricciones de peso
        f.write("\nRESTRICCIONES DE PESO\n")
        f.write("-" * 50 + "\n")
        for plane, data in sensitivity['weight'].items():
            f.write(f"\nAvión: {plane}\n")
            f.write(f"  Capacidad actual: {data['current_value']:.2f} ton\n")
            f.write(f"  Holgura: {data['slack']:.2f} ton\n")
            f.write(f"  Precio sombra: ${data['shadow_price']:.2f}/ton\n")
            
            if abs(data['shadow_price']) < 1e-6:
                f.write("  → Esta restricción no es limitante\n")
            else:
                f.write(f"  → Aumentar la capacidad en 1 ton incrementaría ")
                f.write(f"el valor objetivo en ${data['shadow_price']:.2f}\n")
        
        # Restricciones de volumen
        f.write("\nRESTRICCIONES DE VOLUMEN\n")
        f.write("-" * 50 + "\n")
        for plane, data in sensitivity['volume'].items():
            f.write(f"\nAvión: {plane}\n")
            f.write(f"  Capacidad actual: {data['current_value']:.2f} m³\n")
            f.write(f"  Holgura: {data['slack']:.2f} m³\n")
            f.write(f"  Precio sombra: ${data['shadow_price']:.2f}/m³\n")
            
            if abs(data['shadow_price']) < 1e-6:
                f.write("  → Esta restricción no es limitante\n")
            else:
                f.write(f"  → Aumentar la capacidad en 1 m³ incrementaría ")
                f.write(f"el valor objetivo en ${data['shadow_price']:.2f}\n")