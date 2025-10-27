import matplotlib.pyplot as plt
import numpy as np

# ---------------------------
# Problema 1 – Sprint Planning
# ---------------------------
def plot_problema1_parteA(resultados, capacidades_usada, capacidad_total):
    """
    resultados: dict {tarea: prioridad}
    capacidades_usada: puntos totales usados
    capacidad_total: capacidad máxima
    """
    tareas = list(resultados.keys())
    prioridades = list(resultados.values())

    plt.figure(figsize=(8,5))
    plt.barh(tareas, prioridades, color="skyblue")
    plt.xlabel("Prioridad (peso Fibonacci)")
    plt.ylabel("Tareas")
    plt.title("Parte A - Tareas seleccionadas con su prioridad")
    plt.show()

    # Capacidad usada vs disponible
    plt.figure(figsize=(5,5))
    plt.pie([capacidades_usada, capacidad_total - capacidades_usada],
            labels=["Usada", "Libre"],
            colors=["tomato", "lightgrey"],
            autopct="%1.1f%%")
    plt.title("Capacidad del equipo")
    plt.show()


def plot_problema1_parteB(asignaciones, capacidades_dev, capacidades_max):
    """
    asignaciones: dict {(tarea, dev): prioridad}
    capacidades_dev: dict {dev: capacidad usada}
    capacidades_max: dict {dev: capacidad máxima}
    """
    # Asignaciones por desarrollador
    plt.figure(figsize=(8,5))
    for (t, d), prio in asignaciones.items():
        plt.barh(f"Dev {d}", prio, left=0, label=f"Tarea {t}")
    plt.xlabel("Prioridad acumulada")
    plt.title("Parte B - Asignación de tareas por desarrollador")
    plt.show()

    # Capacidad por desarrollador
    devs = list(capacidades_dev.keys())
    usadas = list(capacidades_dev.values())
    maxs = [capacidades_max[d] for d in devs]

    plt.figure(figsize=(7,5))
    plt.bar(devs, usadas, color="steelblue")
    plt.plot(devs, maxs, "r--", label="Capacidad máxima")
    plt.xlabel("Desarrolladores")
    plt.ylabel("Capacidad usada")
    plt.legend()
    plt.title("Capacidad por desarrollador")
    plt.show()

# ---------------------------
# Problema 2 – Asignación de trabajos
# ---------------------------
def plot_problema2(asignaciones, ganancias, horas_usadas, horas_max):
    """
    asignaciones: dict {(trabajador, trabajo): 1/0}
    ganancias: dict {trabajador: total ganado}
    horas_usadas: dict {trabajador: horas usadas}
    horas_max: dict {trabajador: horas máximas}
    """
    # Ganancias
    plt.figure(figsize=(7,5))
    plt.bar(ganancias.keys(), ganancias.values(), color="seagreen")
    plt.xlabel("Trabajador")
    plt.ylabel("Ganancia")
    plt.title("Ganancia por trabajador")
    plt.show()

    # Horas usadas vs capacidad
    w = list(horas_usadas.keys())
    usadas = list(horas_usadas.values())
    maxs = [horas_max[i] for i in w]

    plt.figure(figsize=(7,5))
    plt.bar(w, usadas, color="cornflowerblue")
    plt.plot(w, maxs, "r--", label="Capacidad máxima")
    plt.xlabel("Trabajador")
    plt.ylabel("Horas")
    plt.legend()
    plt.title("Horas usadas vs capacidad")
    plt.show()

    # Heatmap asignaciones
    matriz = np.zeros((len(horas_usadas), len({j for _, j in asignaciones.keys()})))
    for (w, j), val in asignaciones.items():
        if val == 1:
            matriz[w-1, j-1] = 1

    plt.figure(figsize=(6,5))
    plt.imshow(matriz, cmap="Greens", aspect="auto")
    plt.colorbar(label="Asignado (1) / No (0)")
    plt.xlabel("Trabajos")
    plt.ylabel("Trabajadores")
    plt.title("Matriz de asignaciones")
    plt.show()

# ---------------------------
# Problema 3 – Transporte en aviones
# ---------------------------
def plot_problema3(asignaciones, peso_usado, peso_max, vol_usado, vol_max):
    """
    asignaciones: dict {(recurso, avion): cantidad}
    peso_usado: dict {avion: peso usado}
    peso_max: dict {avion: peso máximo}
    vol_usado: dict {avion: volumen usado}
    vol_max: dict {avion: volumen máximo}
    """
    # Stacked bar chart recursos por avión
    recursos = sorted({r for r, _ in asignaciones.keys()})
    aviones = sorted({a for _, a in asignaciones.keys()})

    bottom = np.zeros(len(aviones))
    plt.figure(figsize=(8,6))
    for r in recursos:
        valores = [asignaciones.get((r,a), 0) for a in aviones]
        plt.bar(aviones, valores, bottom=bottom, label=r)
        bottom += np.array(valores)
    plt.xlabel("Aviones")
    plt.ylabel("Toneladas")
    plt.title("Carga por avión (stacked)")
    plt.legend()
    plt.show()

    # Uso de capacidad en peso y volumen
    fig, ax = plt.subplots(1, 2, figsize=(12,5))
    
    ax[0].bar(peso_usado.keys(), peso_usado.values(), color="dodgerblue")
    ax[0].plot(peso_max.keys(), peso_max.values(), "r--")
    ax[0].set_title("Peso usado vs capacidad")

    ax[1].bar(vol_usado.keys(), vol_usado.values(), color="orange")
    ax[1].plot(vol_max.keys(), vol_max.values(), "r--")
    ax[1].set_title("Volumen usado vs capacidad")

    plt.show()
