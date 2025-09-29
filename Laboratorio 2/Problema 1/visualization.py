"""
Módulo para visualización de resultados mediante gráficos.
"""
from typing import Dict, List
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_value_distribution(assignments: Dict[str, Dict[str, float]],
                          values: Dict[str, float],
                          output_path: str = "distribucion_valor_por_avion.png"):
    """
    Genera gráfico de barras mostrando la distribución del valor por avión.

    Args:
        assignments: Diccionario con asignaciones {recurso: {avion: cantidad}}
        values: Diccionario con valor por unidad de cada recurso
        output_path: Ruta donde guardar el gráfico
    """
    # Calcular valor total por avión
    plane_values = {}
    for resource, plane_data in assignments.items():
        for plane, amount in plane_data.items():
            if plane not in plane_values:
                plane_values[plane] = 0
            plane_values[plane] += amount * values[resource]

    # Crear gráfico
    plt.figure(figsize=(10, 6))
    plt.bar(plane_values.keys(), plane_values.values())
    plt.title("Distribución del Valor por Avión")
    plt.xlabel("Avión")
    plt.ylabel("Valor Total (USD)")
    plt.xticks(rotation=45)
    
    # Añadir valores sobre las barras
    for i, v in enumerate(plane_values.values()):
        plt.text(i, v, f'${v:,.0f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_utilization(assignments: Dict[str, Dict[str, float]],
                    plane_weights: Dict[str, float],
                    plane_volumes: Dict[str, float],
                    weights: Dict[str, float],
                    volumes: Dict[str, float],
                    output_path: str = "utilizacion_por_avion.png"):
    """
    Genera gráfico de utilización de peso y volumen por avión.

    Args:
        assignments: Diccionario con asignaciones {recurso: {avion: cantidad}}
        plane_weights: Diccionario con capacidad de peso por avión
        plane_volumes: Diccionario con capacidad de volumen por avión
        weights: Diccionario con peso por unidad de cada recurso
        volumes: Diccionario con volumen por unidad de cada recurso
        output_path: Ruta donde guardar el gráfico
    """
    # Calcular utilización por avión
    utilization = {}
    for plane in plane_weights.keys():
        # Calcular peso y volumen usado
        weight_used = sum(weights[r] * amt 
                         for r, p_data in assignments.items() 
                         for p, amt in p_data.items() 
                         if p == plane)
        volume_used = sum(volumes[r] * amt 
                         for r, p_data in assignments.items() 
                         for p, amt in p_data.items() 
                         if p == plane)
        
        # Calcular porcentajes de utilización
        utilization[plane] = {
            'Peso': (weight_used / plane_weights[plane]) * 100,
            'Volumen': (volume_used / plane_volumes[plane]) * 100
        }

    # Convertir a DataFrame para facilitar el plotting
    df = pd.DataFrame(utilization).T

    # Crear gráfico
    plt.figure(figsize=(10, 6))
    
    x = np.arange(len(utilization))
    width = 0.35
    
    plt.bar(x - width/2, df['Peso'], width, label='Peso')
    plt.bar(x + width/2, df['Volumen'], width, label='Volumen')
    
    plt.title("Utilización de Capacidad por Avión")
    plt.xlabel("Avión")
    plt.ylabel("Porcentaje de Utilización")
    plt.xticks(x, df.index, rotation=45)
    plt.legend()
    
    # Añadir porcentajes sobre las barras
    for i in range(len(df)):
        plt.text(i - width/2, df['Peso'].iloc[i], 
                f'{df["Peso"].iloc[i]:.1f}%', 
                ha='center', va='bottom')
        plt.text(i + width/2, df['Volumen'].iloc[i],
                f'{df["Volumen"].iloc[i]:.1f}%',
                ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_assignment_matrix(assignments: Dict[str, Dict[str, float]],
                         output_path: str = "matriz_asignacion.png"):
    """
    Genera mapa de calor mostrando la matriz de asignación.

    Args:
        assignments: Diccionario con asignaciones {recurso: {avion: cantidad}}
        output_path: Ruta donde guardar el gráfico
    """
    # Convertir asignaciones a DataFrame
    resources = list(assignments.keys())
    planes = list(set(p for d in assignments.values() for p in d.keys()))
    
    data = []
    for r in resources:
        row = []
        for p in planes:
            row.append(assignments[r].get(p, 0))
        data.append(row)
    
    df = pd.DataFrame(data, index=resources, columns=planes)
    
    # Crear mapa de calor
    plt.figure(figsize=(12, 8))
    sns.heatmap(df, annot=True, fmt='.2f', cmap='YlOrRd')
    plt.title("Matriz de Asignación")
    plt.xlabel("Avión")
    plt.ylabel("Recurso")
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()