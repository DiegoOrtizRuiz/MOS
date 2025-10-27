#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Script académico: Comparación entre Newton-Raphson y Gradiente Descendente.

Este script replica el flujo principal del cuaderno Jupyter, implementando:
    - Definición simbólica de la función objetivo f(x,y)
    - Cálculo de gradiente y Hessiana mediante SymPy
    - Funciones numéricas f, grad_f, hess_f (lambdify)
    - Métodos de optimización: Newton-Raphson y Gradiente Descendente
    - Ejecución experimental (multi-alpha para GD)
    - Estimación de un mínimo de referencia x_ref a partir de múltiples inicios Newton
    - Construcción y presentación de una tabla comparativa (pandas)

Uso:
    python comparacion_newton_gd.py

Requisitos:
    numpy, sympy, pandas, time
    (matplotlib opcional para gráficos; no requerido para este script de consola)

Notas:
    Se evita el uso de librerías de optimización black-box. El enfoque es pedagógico.
"""
import numpy as np
import sympy as sp
import time
import pandas as pd

# ---------------------------------------------------------------------------
# 1. Definición simbólica y derivados
# ---------------------------------------------------------------------------
x, y = sp.symbols('x y', real=True)
f_sym = (x - 2)**2 * (y + 2)**2 + (x + 1)**2 + (y - 1)**2
fx_sym = sp.diff(f_sym, x)
fy_sym = sp.diff(f_sym, y)
hess_sym = sp.hessian(f_sym, (x, y))

# Lambdify para evaluación numérica
f_num = sp.lambdify((x, y), f_sym, 'numpy')
grad_num = sp.lambdify((x, y), [fx_sym, fy_sym], 'numpy')
hess_num = sp.lambdify((x, y), hess_sym, 'numpy')

# ---------------------------------------------------------------------------
# 2. Funciones numéricas reutilizables
# ---------------------------------------------------------------------------
def f(v: np.ndarray) -> float:
    """Evalúa la función objetivo en un punto v=[x,y].

    Parámetros
    ----------
    v : array_like
        Punto en R^2 donde se evalúa f.

    Retorna
    -------
    float
        Valor escalar f(x,y).
    """
    v = np.asarray(v, dtype=float)
    return float(f_num(v[0], v[1]))


def grad_f(v: np.ndarray) -> np.ndarray:
    """Calcula el gradiente de f en v.

    Parámetros
    ----------
    v : array_like
        Punto en R^2.

    Retorna
    -------
    np.ndarray (shape=(2,))
        Vector gradiente [df/dx, df/dy].
    """
    v = np.asarray(v, dtype=float)
    g = grad_num(v[0], v[1])
    return np.array(g, dtype=float)


def hess_f(v: np.ndarray) -> np.ndarray:
    """Calcula la matriz Hessiana de f en v.

    Parámetros
    ----------
    v : array_like
        Punto en R^2.

    Retorna
    -------
    np.ndarray (shape=(2,2))
        Matriz Hessiana.
    """
    v = np.asarray(v, dtype=float)
    H = hess_num(v[0], v[1])
    return np.array(H, dtype=float)


# ---------------------------------------------------------------------------
# 3. Método de Newton-Raphson
# ---------------------------------------------------------------------------
def newton_raphson(x0: np.ndarray, tol_grad: float = 1e-6, max_iter: int = 200,
                   cond_threshold: float = 1e12, reg_lambda: float = 1e-8) -> dict:
    """Minimiza f mediante Newton-Raphson en R^2.

    Implementa regularización y uso de pseudo-inversa si la Hessiana está mal
    condicionada (cond(H) > cond_threshold) o no es definida positiva.

    Parámetros
    ----------
    x0 : array_like
        Punto inicial.
    tol_grad : float, opcional
        Tolerancia para la norma del gradiente (criterio de parada).
    max_iter : int, opcional
        Máximo de iteraciones permitidas.
    cond_threshold : float, opcional
        Umbral de mal condicionamiento.
    reg_lambda : float, opcional
        Término de regularización diagonal si se detecta curvatura no positiva.

    Retorna
    -------
    dict
        Diccionario con trayectoria, valores de f, normas de gradiente,
        condiciones de Hessiana, tiempos por iteración, tiempo total y estado final.
    """
    x = np.asarray(x0, dtype=float)
    traj = [x.copy()]
    f_values = [f(x)]
    grad_norms = []
    hess_conds = []
    iter_times = []
    start = time.perf_counter()
    status = 'running'
    for _ in range(max_iter):
        t0 = time.perf_counter()
        g = grad_f(x)
        gn = np.linalg.norm(g)
        grad_norms.append(gn)
        H = hess_f(x)
        eigs = np.linalg.eigvalsh(H)
        if np.min(eigs) <= 0:
            H = H + reg_lambda * np.eye(2)
        try:
            condH = np.linalg.cond(H)
        except Exception:
            condH = np.inf
        hess_conds.append(condH)
        if gn < tol_grad:
            status = 'converged'
            break
        if (not np.isfinite(condH)) or condH > cond_threshold:
            H_inv = np.linalg.pinv(H + reg_lambda * np.eye(2))
        else:
            H_inv = np.linalg.inv(H)
        x = x - H_inv @ g
        traj.append(x.copy())
        f_values.append(f(x))
        iter_times.append(time.perf_counter() - t0)
    total_time = time.perf_counter() - start
    if status == 'running':
        status = 'converged' if np.linalg.norm(grad_f(x)) < tol_grad else 'max_iter'
    return {
        'traj': np.array(traj),
        'f_values': np.array(f_values),
        'grad_norms': np.array(grad_norms),
        'hess_conds': np.array(hess_conds),
        'iter_times': np.array(iter_times),
        'total_time': total_time,
        'status': status
    }


# ---------------------------------------------------------------------------
# 4. Método de Gradiente Descendente
# ---------------------------------------------------------------------------
def gradient_descent(x0: np.ndarray, alpha: float, tol_grad: float = 1e-6,
                     max_iter: int = 10000) -> dict:
    """Minimiza f mediante Gradiente Descendente con paso fijo.

    Parámetros
    ----------
    x0 : array_like
        Punto inicial.
    alpha : float
        Tamaño de paso constante.
    tol_grad : float, opcional
        Tolerancia para la norma del gradiente.
    max_iter : int, opcional
        Máximo de iteraciones.

    Retorna
    -------
    dict
        Traectoria, valores de f, normas del gradiente, tiempo total y estado.
    """
    x = np.asarray(x0, dtype=float)
    traj = [x.copy()]
    f_values = [f(x)]
    grad_norms = []
    start = time.perf_counter()
    status = 'running'
    for k in range(max_iter):
        g = grad_f(x)
        gn = np.linalg.norm(g)
        grad_norms.append(gn)
        if gn < tol_grad:
            status = 'converged'
            break
        x = x - alpha * g
        traj.append(x.copy())
        f_values.append(f(x))
        # Heurística de divergencia para pasos grandes
        if k > 5 and f_values[-1] > f_values[-2] and alpha > 0.2:
            status = 'diverge'
            break
    total_time = time.perf_counter() - start
    if status == 'running':
        status = 'max_iter'
    return {
        'traj': np.array(traj),
        'f_values': np.array(f_values),
        'grad_norms': np.array(grad_norms),
        'total_time': total_time,
        'status': status,
        'alpha': alpha
    }


# ---------------------------------------------------------------------------
# 5. Estimación de mínimo de referencia x_ref (multi-inicio Newton)
# ---------------------------------------------------------------------------
def estimar_referencia(puntos_iniciales) -> np.ndarray:
    """Obtiene un punto de referencia ejecutando Newton desde varios inicios.

    Parámetros
    ----------
    puntos_iniciales : list[array_like]
        Lista de puntos de partida para Newton.

    Retorna
    -------
    np.ndarray
        Punto final con menor valor de f.
    """
    candidatos = []
    for p in puntos_iniciales:
        r = newton_raphson(np.array(p))
        candidatos.append({'x_final': r['traj'][-1], 'f_final': r['f_values'][-1], 'status': r['status']})
    mejor = min(candidatos, key=lambda d: d['f_final'])
    return np.array(mejor['x_final'])


# ---------------------------------------------------------------------------
# 6. Ejecución principal
# ---------------------------------------------------------------------------
def main():
    x0 = np.array([-2., -3.])
    print('--- Ejecución Newton-Raphson ---')
    newton_res = newton_raphson(x0)
    NR_iters = len(newton_res['traj']) - 1
    NR_time = newton_res['total_time']
    NR_f_final = newton_res['f_values'][-1]
    NR_grad_final = newton_res['grad_norms'][-1]
    finite_conds = newton_res['hess_conds'][np.isfinite(newton_res['hess_conds'])]
    NR_cond_mean = float(np.mean(finite_conds)) if finite_conds.size > 0 else float('nan')
    print(f"Newton: iteraciones={NR_iters}, tiempo_total={NR_time:.6f}s, f_final={NR_f_final:.6e}, ||grad||={NR_grad_final:.3e}, cond_mean={NR_cond_mean:.3e}, status={newton_res['status']}")

    # Estimar referencia si se desea mayor rigor (multi-inicio)
    x_ref = estimar_referencia([[-2., -3.], [0., 0.], [-5., 2.], [3., -4.]])
    print(f"x_ref estimado = {x_ref}, f(x_ref) = {f(x_ref):.6e}, ||grad||={np.linalg.norm(grad_f(x_ref)):.3e}")

    print('\n--- Ejecución Gradiente Descendente (multi-α) ---')
    alphas = [0.001, 0.01, 0.05, 0.1, 0.2, 0.5]
    gd_rows = []
    for a in alphas:
        gd = gradient_descent(x0, alpha=a)
        iters = len(gd['traj']) - 1
        f_final = gd['f_values'][-1]
        grad_final = gd['grad_norms'][-1] if gd['grad_norms'].size > 0 else float('nan')
        dist_ref = np.linalg.norm(gd['traj'][-1] - x_ref)
        print(f"GD α={a}: iters={iters}, f_final={f_final:.6e}, ||grad||={grad_final:.3e}, dist_ref={dist_ref:.3e}, status={gd['status']}")
        gd_rows.append({
            'Método': f'GD (α={a})',
            'Iteraciones': iters,
            'Tiempo_total_s': gd['total_time'],
            'Tiempo_promedio_s': gd['total_time']/max(iters,1),
            'f_final': f_final,
            'norm_grad_final': grad_final,
            'dist_x_ref': dist_ref,
            'Robustez': 'Variable',
            'Ventajas': 'Sencillo',
            'Desventajas': 'Sensibilidad α'
        })

    # Fila Newton
    dist_ref_nr = np.linalg.norm(newton_res['traj'][-1] - x_ref)
    NR_row = {
        'Método': 'Newton-Raphson',
        'Iteraciones': NR_iters,
        'Tiempo_total_s': NR_time,
        'Tiempo_promedio_s': NR_time / max(NR_iters, 1),
        'f_final': NR_f_final,
        'norm_grad_final': NR_grad_final,
        'dist_x_ref': dist_ref_nr,
        'Robustez': 'Alta',
        'Ventajas': 'Convergencia rápida',
        'Desventajas': 'Costo Hessiana'
    }

    rows = [NR_row] + gd_rows
    df = pd.DataFrame(rows)
    print('\n--- Tabla Comparativa ---')
    print(df.to_string(index=False))

    # Resumen adicional: mejor GD por f_final
    mejor_gd = min(gd_rows, key=lambda r: r['f_final']) if gd_rows else None
    if mejor_gd:
        ratio_iters = mejor_gd['Iteraciones'] / max(NR_iters, 1)
        print('\n--- Resumen Métricas ---')
        print(f"Mejor GD: {mejor_gd['Método']} con f_final={mejor_gd['f_final']:.6e}")
        print(f"Iteraciones Newton / GD: {NR_iters} / {mejor_gd['Iteraciones']} (ratio={ratio_iters:.2f})")
        print(f"Distancia a x_ref (Newton)={dist_ref_nr:.3e}, (Mejor GD)={mejor_gd['dist_x_ref']:.3e}")


if __name__ == '__main__':
    main()
