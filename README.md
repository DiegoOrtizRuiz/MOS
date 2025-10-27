# Modelado, Simulación y Optimización (MOS) 2025-2

Autor: Diego Fernando Ortiz Ruiz

Este repositorio contiene los laboratorios y talleres del curso de **Modelado, Simulación y Optimización**. Cada laboratorio incluye cuadernos Jupyter (`.ipynb`) y scripts Python (`.py`) que desarrollan los problemas asignados con enfoque académico, utilizando principalmente `numpy`, `sympy`, `matplotlib`, `pandas` y `pyomo` (donde corresponde).

## Índice de Contenido
- [Laboratorio 1](#laboratorio-1) – Introducción y primeros modelos analíticos.
- [Laboratorio 2](#laboratorio-2) – (Descripción pendiente si se agregan notebooks.)
- [Laboratorio 3](#laboratorio-3) – Métodos de optimización (Newton-Raphson vs Gradiente Descendente) y problemas adicionales.
- [Taller 1](#taller-1) – Modelo de dieta genérica con Pyomo.

---
## Requisitos Generales de Entorno
Instalar dependencias básicas:

```
pip install numpy sympy pandas matplotlib pyomo
```

Para ejecutar cualquier notebook:
1. Abrir el archivo `.ipynb` en Jupyter / VS Code.
2. Ejecutar todas las celdas en orden (Kernel limpio).
3. Cerrar/aceptar cada gráfica para ver la siguiente si se genera en ventana emergente.

Para scripts Python:
```
python nombre_script.py
```

---
## Laboratorio 1
Ubicación: `Laboratorio1/`

Archivos principales:
- `problema1.py`, `problema2.py`, `problema3.py`
- Notebook(s) (si se incorporan posteriormente)

Objetivos generales:
- Familiarización con la estructura del curso.
- Resolución de problemas base de optimización / modelado.

Ejecución:
```
python Laboratorio1/problema1.py
python Laboratorio1/problema2.py
python Laboratorio1/problema3.py
```
Cada script imprime resultados numéricos y genera gráficos según el problema.

---
## Laboratorio 2
Ubicación: `Laboratorio 2/`

Contenido actual: Archivos JSON de datos (`datos_no_transportados.json`, `ejemplo_datos.json`). No hay notebooks todavía en el repositorio listados. Este laboratorio parece orientado a preparación de datos. Cuando se añadan notebooks, actualizar esta sección con:
- Objetivo del problema
- Método aplicado
- Métricas y visualizaciones

---
## Laboratorio 3
Ubicación: `Laboratorio 3/`

Notebooks identificados:
- `Problema1.ipynb`
- `Problema2.ipynb`
- `Problema3A.ipynb`
- `Problema3B.ipynb`
- `Problema4A.ipynb`
- `Problema4B.ipynb` (comparación Newton-Raphson vs Gradiente Descendente)

Scripts complementarios:
- `comparacion_newton_gd.py` (versión académica del Problema4B)
- `newton_r4_parteB.py`
- Otros scripts de apoyo (p.ej. `comparacion_newton_gd.py`)

### Resumen por Notebook (Laboratorio 3)
| Notebook | Objetivo | Técnicas | Salidas Principales |
|----------|----------|----------|---------------------|
| Problema1.ipynb | (Agregar breve descripción) | (Métodos usados) | Gráficas y métricas específicas |
| Problema2.ipynb | (Agregar breve descripción) | (Métodos usados) | Resultados numéricos / gráficos |
| Problema3A.ipynb | (Agregar breve descripción) | (Métodos usados) | Trayectorias / tablas |
| Problema3B.ipynb | (Agregar breve descripción) | (Métodos usados) | Comparaciones / análisis |
| Problema4A.ipynb | (Agregar breve descripción) | (Métodos usados) | Visualizaciones iniciales |
| Problema4B.ipynb | Comparar Newton-Raphson y GD en función con curvatura variable | Simbólico con SymPy, Newton, GD, análisis de condicionamiento | Tabla comparativa, figuras de trayectorias, error log, evolución f y ||grad|| |

### Detalle Problema4B
Función:
$$f(x,y) = (x-2)^2 (y+2)^2 + (x+1)^2 + (y-1)^2$$

Secciones incluidas: Portada, Introducción, Desarrollo Teórico, Implementación, Resultados, Análisis Comparativo, Conclusiones.

Métricas principales:
- Newton: pocas iteraciones, costo por iteración más alto (inversión Hessiana).
- GD: muchas iteraciones, sensibilidad al parámetro de paso α.
- Referente numérico obtenido vía multi-inicio Newton.

Ejecutar notebook para reproducir figuras:
```
# Abrir en VS Code o Jupyter y ejecutar todas las celdas
```
Ejecutar script equivalente:
```
python "Laboratorio 3/comparacion_newton_gd.py"
```

---
## Taller 1
Ubicación: `Taller 1/`

Archivo principal: `dieta_generica_pyomo.py`
Objetivo: Modelo de optimización de dieta usando Pyomo.

Ejecución:
```
python "Taller 1/dieta_generica_pyomo.py"
```
Editar parámetros dentro del script para probar diferentes configuraciones nutricionales.

---
## Buenas Prácticas
- Mantener kernels limpios antes de ejecutar notebooks.
- Verificar instalación de paquetes antes de correr (especialmente `sympy`, `pyomo`).
- Usar control de versiones para justificar cambios en modelos.

## Próximas Mejoras
- Completar descripciones faltantes en Problemas 1–4A.
- Añadir columna distancia a x_ref en tabla comparativa dentro del notebook (ya disponible en variables, falta volcado si se desea).
- Incorporar guardado automático de figuras.

## Contacto
Para dudas sobre la implementación o ampliaciones, contactar al autor del repositorio.
