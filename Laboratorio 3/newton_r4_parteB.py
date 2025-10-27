"""Script académico: Método de Newton-Raphson en R^4 (Parte B)
Reproduce la lógica del cuaderno Problema3B.ipynb.
Requisitos: numpy, sympy, matplotlib, pandas (para tabla opcional).
"""
from __future__ import annotations
import numpy as np
import sympy as sp
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Any, List

# =============================
# 1. Definición simbólica
# =============================
w, x, y, z = sp.symbols('w x y z', real=True)
f_sym = (w-1)**2 + (x-2)**2 + (y-3)**2 + (z-4)**2
grad_sym = sp.Matrix([sp.diff(f_sym, v) for v in (w,x,y,z)])
H_sym = sp.hessian(f_sym, (w,x,y,z))

# Lambdify
f_num = sp.lambdify((w,x,y,z), f_sym, 'numpy')
grad_num = sp.lambdify((w,x,y,z), grad_sym, 'numpy')
hess_num = sp.lambdify((w,x,y,z), H_sym, 'numpy')

# =============================
# 2. Funciones numéricas
# =============================
def f_vec(v: np.ndarray) -> float:
    w_,x_,y_,z_ = v
    return float(f_num(w_,x_,y_,z_))

def grad_vec(v: np.ndarray) -> np.ndarray:
    w_,x_,y_,z_ = v
    return np.asarray(grad_num(w_,x_,y_,z_), dtype=float).reshape(-1)

def hess_mat(v: np.ndarray) -> np.ndarray:
    w_,x_,y_,z_ = v
    return np.asarray(hess_num(w_,x_,y_,z_), dtype=float)

def cond_number(H: np.ndarray) -> float:
    try:
        vals = np.linalg.eigvalsh(H)
        min_abs = np.min(np.abs(vals))
        max_abs = np.max(np.abs(vals))
        if min_abs == 0:
            return float('inf')
        return float(max_abs / min_abs)
    except np.linalg.LinAlgError:
        return float('inf')

# =============================
# 3. Método de Newton-Raphson
# =============================
def newton_raphson(x0: np.ndarray, tol: float = 1e-6, max_iter: int = 200,
                   reg_eps: float = 1e-10, cond_warn: float = 1e8,
                   use_pinv: bool = True, store_all: bool = True) -> Dict[str, Any]:
    """Ejecuta el método de Newton-Raphson para la función f.

    Parámetros
    ----------
    x0 : np.ndarray (4,)
        Punto inicial.
    tol : float
        Tolerancia de parada en la norma del gradiente.
    max_iter : int
        Máximo de iteraciones.
    reg_eps : float
        Término de regularización diagonal.
    cond_warn : float
        Umbral de advertencia para número de condición.
    use_pinv : bool
        Si True usa pseudo-inversa cuando la Hessiana es singular o mal condicionada.
    store_all : bool
        Si True guarda todas las iteraciones.

    Retorna
    -------
    dict con trayectoria y métricas.
    """
    xk = np.array(x0, dtype=float)
    gk = grad_vec(xk)
    trajectory: List[np.ndarray] = [xk.copy()]
    f_values: List[float] = [f_vec(xk)]
    grad_norms: List[float] = [np.linalg.norm(gk)]
    cond_numbers: List[float] = []
    stopped_by = 'max_iter'

    for k in range(max_iter):
        if grad_norms[-1] < tol:
            stopped_by = 'tolerance'
            break
        Hk = hess_mat(xk)
        cn = cond_number(Hk)
        cond_numbers.append(cn)
        if not np.isfinite(cn) or cn > cond_warn:
            if use_pinv:
                H_inv = np.linalg.pinv(Hk)
            else:
                Hk = Hk + reg_eps * np.eye(4)
                H_inv = np.linalg.inv(Hk)
        else:
            H_inv = np.linalg.inv(Hk)
        pk = H_inv @ gk
        xk = xk - pk
        gk = grad_vec(xk)
        if store_all:
            trajectory.append(xk.copy())
            f_values.append(f_vec(xk))
            grad_norms.append(np.linalg.norm(gk))
        else:
            trajectory = [xk.copy()]
            f_values = [f_vec(xk)]
            grad_norms = [np.linalg.norm(gk)]

    return {
        'trajectory': trajectory,
        'f_values': f_values,
        'grad_norms': grad_norms,
        'cond_numbers': cond_numbers,
        'iterations': len(trajectory)-1,
        'stopped_by': stopped_by
    }

# =============================
# 4. Ejecución principal
# =============================
def main():
    x0_a = np.array([0.0, 0.0, 0.0, 0.0])
    x0_b = np.array([0.0, 10.0, -5.0, 2.0])

    res_a = newton_raphson(x0_a)
    res_b = newton_raphson(x0_b)

    print("== Resultados Run A ==")
    print("Iteraciones:", res_a['iterations'], "Parada:", res_a['stopped_by'])
    print("Último punto:", res_a['trajectory'][-1])
    print("f final:", res_a['f_values'][-1], "||grad|| final:", res_a['grad_norms'][-1])
    print("\n== Resultados Run B ==")
    print("Iteraciones:", res_b['iterations'], "Parada:", res_b['stopped_by'])
    print("Último punto:", res_b['trajectory'][-1])
    print("f final:", res_b['f_values'][-1], "||grad|| final:", res_b['grad_norms'][-1])

    # Tabla resumen
    def build_records(res, label):
        rows = []
        for k,(xk,fk,gk) in enumerate(zip(res['trajectory'], res['f_values'], res['grad_norms'])):
            rows.append({'run': label, 'k': k, 'w': xk[0], 'x': xk[1], 'y': xk[2], 'z': xk[3], 'f': fk, '||grad||': gk})
        return rows
    df = pd.DataFrame(build_records(res_a,'A') + build_records(res_b,'B'))
    print("\nResumen primeras filas:")
    print(df.head().to_string(index=False))

    # Gráficas de convergencia
    fig, axes = plt.subplots(1,2, figsize=(10,4))
    for res,label,color in [(res_a,'A','blue'), (res_b,'B','red')]:
        axes[0].plot(range(len(res['f_values'])), res['f_values'], '-o', color=color, label=f'Run {label}')
        axes[1].plot(range(len(res['grad_norms'])), res['grad_norms'], '-o', color=color, label=f'Run {label}')
    axes[0].set_title('Evolución f(x_k)')
    axes[0].set_xlabel('Iteración'); axes[0].set_ylabel('f')
    axes[0].grid(alpha=0.3); axes[0].legend()
    axes[1].set_title('Evolución ||∇f||')
    axes[1].set_xlabel('Iteración'); axes[1].set_ylabel('||∇f||')
    axes[1].set_yscale('log'); axes[1].grid(alpha=0.3); axes[1].legend()
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
