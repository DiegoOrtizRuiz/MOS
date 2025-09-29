# modelo_problema1_viz.py
"""
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
import numpy as np
from pyomo.environ import (
    ConcreteModel, Set, Param, Var, NonNegativeReals, NonNegativeIntegers, Binary,
    Constraint, ConstraintList, Objective, SolverFactory,
    maximize, value
)

# ---------------------------
# ---------------------------
# Datos por defecto (Default Data)
# ---------------------------

# Recursos disponibles (Available Resources):
# Formato: (nombre, valor_usd, stock_ton, peso_ton/un, volumen_m3/un)
# - nombre: Nombre del recurso
# - valor_usd: Valor en dólares por unidad
# - stock_ton: Stock disponible en toneladas
# - peso_ton/un: Peso en toneladas por unidad
# - volumen_m3/un: Volumen en metros cúbicos por unidad

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

# Capacidades de la flota de transporte disponible: (avion, capacidad_peso_ton, capacidad_volumen_m3)
DEFAULT_PLANES = [
    ("Hercules-1", 28, 22),
    ("Hercules-2", 32, 26),
    ("Galaxy-1", 45, 38),
    ("Galaxy-2", 48, 42),
    ("Antonov-1", 65, 55),
    ("Antonov-2", 70, 60),
]

# Reglas de compatibilidad: (recurso1, recurso2, "incompatible"|"together")
DEFAULT_COMPAT = [
    ("Equipos_Medicos", "Agua_Potable", "incompatible"),
    ("Medicinas", "Medicamentos_Especiales", "together"),
    ("Generadores", "Material_Construccion", "incompatible"),
    ("Equipos_Comunicacion", "Equipos_Medicos", "together"),
    ("Alimentos_Basicos", "Medicinas", "incompatible"),
    ("Tiendas_Campana", "Material_Construccion", "together"),
    ("Agua_Potable", "Alimentos_Basicos", "together"),
]

# Reglas de seguridad: (recurso, avion, permitido_bool)
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

def validate_json_structure(data):
    """Valida que el JSON tenga la estructura correcta para el modelo.

    Args:
        data (dict): Datos cargados del JSON

    Returns:
        bool: True si la estructura es válida, False si no

    Raises:
        ValueError: Si falta alguna sección obligatoria o hay datos inválidos
    """
    required_sections = ['resources', 'planes']
    for section in required_sections:
        if section not in data:
            raise ValueError(f"Falta la sección obligatoria: {section}")
    
    # Validar recursos
    for r in data['resources']:
        required_fields = ['name', 'value_usd', 'stock_ton', 'weight_ton', 'volume_m3']
        for field in required_fields:
            if field not in r:
                raise ValueError(f"Falta campo {field} en recurso {r.get('name', '???')}")
    
    # Validar aviones
    for p in data['planes']:
        required_fields = ['name', 'weight_capacity', 'volume_capacity']
        for field in required_fields:
            if field not in p:
                raise ValueError(f"Falta campo {field} en avión {p.get('name', '???')}")
    
    return True

def convert_json_to_model_format(data):
    """Convierte los datos del JSON al formato usado por el modelo.

    Args:
        data (dict): Datos cargados y validados del JSON

    Returns:
        tuple: (resources_list, planes_list, comp_list, sec_list)
    """
    # Convertir recursos
    resources_list = [
        (r['name'], r['value_usd'], r['stock_ton'], r['weight_ton'], r['volume_m3'])
        for r in data['resources']
    ]

    # Convertir aviones
    planes_list = [
        (p['name'], p['weight_capacity'], p['volume_capacity'])
        for p in data['planes']
    ]

    # Convertir reglas de compatibilidad
    comp_list = []
    if 'compatibility_rules' in data:
        comp_list = [
            (r['resource1'], r['resource2'], r['rule'])
            for r in data['compatibility_rules']
        ]

    # Convertir reglas de seguridad
    sec_list = []
    if 'security_rules' in data:
        sec_list = [
            (r['resource'], r['plane'], r['allowed'])
            for r in data['security_rules']
        ]

    return resources_list, planes_list, comp_list, sec_list

def load_data_from_json(path):
    """Carga y valida datos desde un archivo JSON para la entrada del modelo.

    El archivo JSON debe tener la siguiente estructura:
    {
        "resources": [
            {
                "name": str,
                "value_usd": number,
                "stock_ton": number,
                "weight_ton": number,
                "volume_m3": number
            },
            ...
        ],
        "planes": [
            {
                "name": str,
                "weight_capacity": number,
                "volume_capacity": number
            },
            ...
        ],
        "compatibility_rules": [  // opcional
            {
                "resource1": str,
                "resource2": str,
                "rule": "incompatible"|"together"
            },
            ...
        ],
        "security_rules": [  // opcional
            {
                "resource": str,
                "plane": str,
                "allowed": boolean
            },
            ...
        ],
        "divisibility_rules": {  // opcional
            "resource_name": boolean,
            ...
        }
    }

    Args:
        path (str): Ruta al archivo JSON que contiene los datos de entrada

    Returns:
        tuple: (resources_list, planes_list, comp_list, sec_list) en el formato
               esperado por el modelo, o (None, None, None, None) si hay error

    Raises:
        Exception: Si hay un error al leer o procesar el archivo JSON
        ValueError: Si el JSON no tiene la estructura correcta
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Validar estructura del JSON
        if validate_json_structure(data):
            # Convertir a formato del modelo
            return convert_json_to_model_format(data)

    except Exception as e:
        print(f"Error al cargar datos desde {path}: {e}")
        return None, None, None, None

def build_indexed_data(resources_list, planes_list):
    """Construye estructuras de datos indexadas para el modelo Pyomo.

    Args:
        resources_list (list): Lista de tuplas con información de recursos
        planes_list (list): Lista de tuplas con información de aviones

    Returns:
        tuple: Contiene las siguientes estructuras:
            - resources (list): Lista de nombres de recursos
            - planes (list): Lista de nombres de aviones
            - val (dict): Diccionario de valores por recurso
            - stock (dict): Diccionario de stock por recurso
            - weight (dict): Diccionario de pesos por recurso
            - volume (dict): Diccionario de volúmenes por recurso
            - plane_w (dict): Diccionario de capacidades de peso por avión
            - plane_v (dict): Diccionario de capacidades de volumen por avión
    """
    resources = [r[0] for r in resources_list] # nombres de recursos
    planes = [p[0] for p in planes_list] # nombres de aviones
    val = {r[0]: r[1] for r in resources_list} # valor por recurso
    stock = {r[0]: r[2] for r in resources_list} # stock por recurso
    weight = {r[0]: r[3] for r in resources_list} # peso por recurso
    volume = {r[0]: r[4] for r in resources_list} # volumen por recurso
    plane_w = {p[0]: p[1] for p in planes_list} # capacidad peso por avión
    plane_v = {p[0]: p[2] for p in planes_list} # capacidad volumen por avión
    return resources, planes, val, stock, weight, volume, plane_w, plane_v

def determine_divisibility(json_data=None):
    """Determina la divisibilidad de los recursos.

    Si se proporciona json_data, usa las reglas definidas en el JSON.
    Si no, usa las reglas predefinidas por defecto.

    Args:
        json_data (dict, optional): Datos cargados del JSON que pueden incluir
                                   la sección 'divisibility_rules'

    Returns:
        dict: Diccionario que mapea nombres de recursos a booleanos:
            - True: El recurso es divisible
            - False: El recurso debe asignarse en unidades enteras
    """
    # Reglas predeterminadas
    default_rules = {
        'Alimentos_Basicos': True,
        'Medicinas': True,
        'Equipos_Medicos': False,
        'Agua_Potable': True,
        'Mantas': False,
        'Generadores': False,
        'Tiendas_Campana': False,
        'Medicamentos_Especiales': True,
        'Equipos_Comunicacion': False,
        'Material_Construccion': True
    }

    # Si hay datos JSON y contienen reglas de divisibilidad, usarlas
    if json_data and 'divisibility_rules' in json_data:
        return json_data['divisibility_rules']
    
    return default_rules

def preprocess_compatibility(comp_list, available_resources):
    """Preprocesa las reglas de compatibilidad entre recursos.

    Args:
        comp_list (list): Lista de tuplas (recurso1, recurso2, tipo_relación)
            donde tipo_relación puede ser 'incompatible' o 'together'
        available_resources (list): Lista de nombres de recursos disponibles

    Returns:
        tuple: Contiene dos conjuntos:
            - incompat (set): Pares de recursos incompatibles
            - together (set): Pares de recursos que deben ir juntos

    Raises:
        ValueError: Si hay reglas conflictivas para el mismo par de recursos
        ValueError: Si se referencia un recurso que no existe
    """
    available_resources = set(available_resources)  # convertir a set para búsqueda más rápida
    incompat = set() # pares incompatibles
    together = set() # pares que deben ir juntos
    
    for a, b, rel in comp_list:
        # Validar que ambos recursos existan
        if a not in available_resources:
            print(f"Advertencia: Recurso '{a}' en regla de compatibilidad no existe. Se ignorará la regla.")
            continue
        if b not in available_resources:
            print(f"Advertencia: Recurso '{b}' en regla de compatibilidad no existe. Se ignorará la regla.")
            continue
            
        if rel.lower().startswith("incompat"):
            pair = tuple(sorted((a, b)))
            incompat.add(pair) # evitar duplicados (a,b) y (b,a)
        elif rel.lower().startswith("together"):
            pair = tuple(sorted((a, b)))
            together.add(pair)
    
    conflic = incompat.intersection(together)
    if conflic: 
        raise ValueError(f"Conflicting rules for pairs: {conflic}")  # sanity check
    
    return incompat, together

def preprocess_security(sec_list, planes, resources):
    """Preprocesa las reglas de seguridad para recursos y aviones.

    Args:
        sec_list (list): Lista de tuplas (recurso, avión, permitido)
        planes (list): Lista de nombres de aviones disponibles
        resources (list): Lista de nombres de recursos disponibles

    Returns:
        dict: Diccionario que mapea tuplas (recurso, avión) a booleanos:
            - True: El recurso puede transportarse en el avión
            - False: El recurso no puede transportarse en el avión
    """
    planes_set = set(planes)  # convertir a set para búsqueda más rápida
    resources_set = set(resources)  # convertir a set para búsqueda más rápida
    allowed = {}
    
    # Inicializar todas las combinaciones como permitidas
    for r in resources:
        for p in planes:
            allowed[(r, p)] = True
    
    # Aplicar reglas de seguridad
    for r, p, flag in sec_list:
        # Validar que el recurso y el avión existan
        if r not in resources_set:
            print(f"Advertencia: Recurso '{r}' en regla de seguridad no existe. Se ignorará la regla.")
            continue
        if p not in planes_set:
            print(f"Advertencia: Avión '{p}' en regla de seguridad no existe. Se ignorará la regla.")
            continue
            
        allowed[(r, p)] = flag
    
    return allowed

# ---------------------------
# Model builder
# ---------------------------

def build_pyomo_model(resources, planes, val, stock, weight, volume, plane_w, plane_v, divisibles, incompat_pairs, together_pairs, allowed):
    """Construye el modelo de optimización usando Pyomo.

    El modelo matemático implementado es:

    Maximizar:
        ∑ᵣ∑ₐ valor[r] * x[r,a]

    Sujeto a:
        1. Restricciones de capacidad de peso:
           ∑ᵣ peso[r] * x[r,a] ≤ cap_peso[a]  ∀a∈A

        2. Restricciones de capacidad de volumen:
           ∑ᵣ volumen[r] * x[r,a] ≤ cap_volumen[a]  ∀a∈A

        3. Restricciones de stock:
           ∑ₐ x[r,a] ≤ stock[r]  ∀r∈R

        4. Restricciones Big-M para variables binarias:
           x[r,a] ≤ stock[r] * y[r,a]  ∀r∈R, ∀a∈A

        5. Restricciones de incompatibilidad:
           y[i,a] + y[j,a] ≤ 1  ∀(i,j)∈incompatibles, ∀a∈A

        6. Restricciones de recursos juntos:
           y[i,a] = y[j,a]  ∀(i,j)∈juntos, ∀a∈A

        7. Restricciones de seguridad:
           y[r,a] = 0  ∀(r,a)∈no_permitidos

        8. Restricciones de integralidad (para recursos indivisibles):
           x[r,a] = z[r,a], z[r,a] ∈ ℤ⁺  ∀r∈indivisibles, ∀a∈A

    Donde:
        x[r,a] ≥ 0  ∀r∈R, ∀a∈A  (Variables continuas no negativas)
        y[r,a] ∈ {0,1}  ∀r∈R, ∀a∈A  (Variables binarias)

    Args:
        resources (list): Lista de nombres de recursos (conjunto R)
        planes (list): Lista de nombres de aviones (conjunto A)
        val (dict): Valores por recurso {r: valor_r}
        stock (dict): Stock disponible por recurso {r: stock_r}
        weight (dict): Peso por unidad de recurso {r: peso_r}
        volume (dict): Volumen por unidad de recurso {r: volumen_r}
        plane_w (dict): Capacidad de peso por avión {a: cap_peso_a}
        plane_v (dict): Capacidad de volumen por avión {a: cap_volumen_a}
        divisibles (dict): Indica si cada recurso es divisible {r: bool}
        incompat_pairs (set): Pares de recursos incompatibles {(i,j)}
        together_pairs (set): Pares de recursos que deben ir juntos {(i,j)}
        allowed (dict): Matriz de compatibilidad recurso-avión {(r,a): bool}

    Returns:
        ConcreteModel: Modelo de Pyomo configurado con todas las restricciones
                     y la función objetivo a maximizar

    Note:
        El modelo usa dos tipos principales de variables:
        - x[r,a]: Cantidad del recurso r asignada al avión a (≥ 0)
        - y[r,a]: 1 si el recurso r se asigna al avión a, 0 si no
        Para recursos indivisibles se añade:
        - z[r,a]: Variable entera no negativa igual a x[r,a]
    """
    model = ConcreteModel() # Modelo concreto
    model.R = Set(initialize=resources) # Recursos
    model.A = Set(initialize=planes) # Aviones
    model.value = Param(model.R, initialize=lambda m, r: val[r]) # Valor por recurso
    model.stock = Param(model.R, initialize=lambda m, r: stock[r]) # Stock por recurso
    model.w = Param(model.R, initialize=lambda m, r: weight[r]) # Peso por recurso
    model.vol = Param(model.R, initialize=lambda m, r: volume[r]) # Volumen por recurso
    model.cap_w = Param(model.A, initialize=lambda m, a: plane_w[a]) # Capacidad peso por avión
    model.cap_v = Param(model.A, initialize=lambda m, a: plane_v[a]) # Capacidad de volumen por avión
    model.x = Var(model.R, model.A, domain=NonNegativeReals) # Cantidad asignada
    model.y = Var(model.R, model.A, domain=Binary) # Indicador de uso

    
    def big_m_link(m, r, a):
        """Implementa la restricción Big-M que vincula variables continuas y binarias.

        Esta restricción asegura que:
        1. Si y[r,a] = 0, entonces x[r,a] = 0 (no se asigna el recurso)
        2. Si y[r,a] = 1, entonces x[r,a] ≤ stock[r] (asignación limitada por stock)

        La constante Big-M usada es el stock disponible del recurso, que es el
        máximo valor posible para x[r,a].

        Restricción matemática:
            x[r,a] ≤ stock[r] * y[r,a]  ∀r∈R, ∀a∈A

        Args:
            m (ConcreteModel): Modelo Pyomo que contiene las variables
            r (str): Índice del recurso en el conjunto R
            a (str): Índice del avión en el conjunto A

        Returns:
            Expression: Restricción Pyomo x[r,a] ≤ stock[r] * y[r,a]
        """
        return m.x[r, a] <= m.stock[r] * m.y[r, a]
    model.link_bigM = Constraint(model.R, model.A, rule=big_m_link)

    def cap_weight_rule(m, a):
        """Implementa la restricción de capacidad de peso para cada avión.

        Esta restricción asegura que el peso total de los recursos asignados
        a cada avión no exceda su capacidad máxima de carga.

        Restricción matemática:
            ∑ᵣ peso[r] * x[r,a] ≤ cap_peso[a]  ∀a∈A

        Donde:
        - peso[r]: Peso por unidad del recurso r
        - x[r,a]: Cantidad del recurso r asignada al avión a
        - cap_peso[a]: Capacidad máxima de peso del avión a

        Args:
            m (ConcreteModel): Modelo Pyomo que contiene las variables
            a (str): Índice del avión en el conjunto A

        Returns:
            Expression: Restricción Pyomo ∑(peso[r] * x[r,a]) ≤ cap_peso[a]
        """
        return sum(m.w[r] * m.x[r, a] for r in m.R) <= m.cap_w[a]
    model.cap_weight = Constraint(model.A, rule=cap_weight_rule)

    def cap_volume_rule(m, a):
        """Implementa la restricción de capacidad de volumen para cada avión.

        Esta restricción asegura que el volumen total de los recursos asignados
        a cada avión no exceda su capacidad máxima de volumen.

        Restricción matemática:
            ∑ᵣ volumen[r] * x[r,a] ≤ cap_volumen[a]  ∀a∈A

        Donde:
        - volumen[r]: Volumen por unidad del recurso r
        - x[r,a]: Cantidad del recurso r asignada al avión a
        - cap_volumen[a]: Capacidad máxima de volumen del avión a

        Args:
            m (ConcreteModel): Modelo Pyomo que contiene las variables
            a (str): Índice del avión en el conjunto A

        Returns:
            Expression: Restricción Pyomo ∑(volumen[r] * x[r,a]) ≤ cap_volumen[a]
        """
        return sum(m.vol[r] * m.x[r, a] for r in m.R) <= m.cap_v[a]
    model.cap_volume = Constraint(model.A, rule=cap_volume_rule)

    def stock_rule(m, r):
        """Implementa la restricción de stock disponible para cada recurso.

        Esta restricción asegura que la cantidad total asignada de cada recurso
        a todos los aviones no exceda el stock disponible de ese recurso.

        Restricción matemática:
            ∑ₐ x[r,a] ≤ stock[r]  ∀r∈R

        Donde:
        - x[r,a]: Cantidad del recurso r asignada al avión a
        - stock[r]: Cantidad disponible del recurso r

        Args:
            m (ConcreteModel): Modelo Pyomo que contiene las variables
            r (str): Índice del recurso en el conjunto R

        Returns:
            Expression: Restricción Pyomo ∑(x[r,a]) ≤ stock[r]
        """
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
            """Implementa la restricción de integralidad para recursos indivisibles.

            Esta restricción asegura que las cantidades asignadas de recursos
            indivisibles sean valores enteros, vinculando la variable continua x[r,a]
            con una variable entera z[r,a].

            Restricción matemática:
                x[r,a] = z[r,a]  ∀r∈indivisibles, ∀a∈A
                z[r,a] ∈ ℤ⁺ (enteros no negativos)

            Args:
                m (ConcreteModel): Modelo Pyomo que contiene las variables
                r (str): Índice del recurso indivisible
                a (str): Índice del avión

            Returns:
                Expression: Restricción Pyomo z[r,a] = x[r,a]

            Note:
                Esta restricción se aplica solo a los recursos marcados como
                indivisibles en el diccionario divisibles
            """
            return m.z[r, a] == m.x[r, a]
        model.link_z = Constraint(indivisible, model.A, rule=link_z_x)

    def obj_rule(m):
        """Define la función objetivo del modelo de optimización.

        Maximiza el valor total transportado, calculado como la suma del valor
        de cada unidad de recurso multiplicado por la cantidad asignada
        a cada avión.

        Función objetivo matemática:
            max ∑ᵣ∑ₐ valor[r] * x[r,a]

        Donde:
        - valor[r]: Valor por unidad del recurso r
        - x[r,a]: Cantidad del recurso r asignada al avión a

        Args:
            m (ConcreteModel): Modelo Pyomo que contiene las variables

        Returns:
            Expression: Función objetivo Pyomo ∑∑(valor[r] * x[r,a])
        """
        return sum(m.value[r] * m.x[r, a] for r in m.R for a in m.A)
    model.obj = Objective(rule=obj_rule, sense=maximize)
    return model

# ---------------------------
# Plots (matplotlib)
# ---------------------------
def plot_utilization(model, out_png="utilizacion_por_avion.png"):
    """Genera un gráfico de barras mostrando la utilización de peso por avión.

    Args:
        model (ConcreteModel): Modelo Pyomo resuelto
        out_png (str, optional): Ruta donde guardar el gráfico. 
            Default: "utilizacion_por_avion.png"
    """
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
    """Genera un gráfico circular mostrando la distribución del valor transportado por avión.

    Args:
        model (ConcreteModel): Modelo Pyomo resuelto
        out_png (str, optional): Ruta donde guardar el gráfico. 
            Default: "distribucion_valor_por_avion.png"
    """
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
    """Genera una matriz de calor mostrando la asignación de recursos a aviones.

    Args:
        model (ConcreteModel): Modelo Pyomo resuelto
        out_png (str, optional): Ruta donde guardar el gráfico. 
            Default: "matriz_asignacion.png"
    """
    # Numpy ya importado al inicio del archivo
    R = list(model.R); A = list(model.A)
    M = [[ value(model.x[r,a]) for a in A ] for r in R ]
    M = np.array(M)
    fig, ax = plt.subplots(figsize=(9,6))
    im = ax.imshow(M, aspect='auto')
    ax.set_yticks(range(len(R))); ax.set_yticklabels(R)
    ax.set_xticks(range(len(A))); ax.set_xticklabels(A, rotation=45, ha='right')
    ax.set_title("Matriz de asignación (unidades por avión)")
    fig.colorbar(im, ax=ax); fig.tight_layout(); fig.savefig(out_png); plt.close(fig)

def plot_untransported_resources(model, out_png="recursos_no_transportados.png"):
    """
    Visualiza los recursos que no pudieron ser transportados por restricciones.
    Muestra un diagrama de barras con el total disponible vs el total transportado.
    Args:
        model (ConcreteModel): Modelo Pyomo resuelto
        out_png (str): Ruta donde guardar el gráfico
    """
    recursos = list(model.R)
    stock = [value(model.stock[r]) for r in recursos]
    transportado = [sum(value(model.x[r, a]) for a in model.A) for r in recursos]
    no_transportado = [s - t for s, t in zip(stock, transportado)]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(recursos, stock, label='Total disponible', color='lightgray')
    ax.bar(recursos, transportado, label='Total transportado', color='steelblue')
    for i, (s, t) in enumerate(zip(stock, transportado)):
        if s - t > 0.01:
            ax.text(i, t + (s * 0.02), f"No transp.: {s-t:.1f}", ha='center', va='bottom', color='red')
    ax.set_ylabel('Cantidad (toneladas)')
    ax.set_title('Recursos: disponible vs transportado')
    ax.legend()
    plt.xticks(rotation=45, ha='right')
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)

# ---------------------------
# Sensitivity analysis
# ---------------------------
def sensitivity_increase_plane_and_solve(resources_list, planes_list, model_builder, divisibles, incompat_pairs, together_pairs, allowed, solver_name="glpk"):
    """Realiza análisis de sensibilidad aumentando la capacidad del avión más limitante.

    Args:
        resources_list (list): Lista de recursos disponibles
        planes_list (list): Lista de aviones disponibles
        model_builder (function): Función que construye el modelo Pyomo
        divisibles (dict): Indica si cada recurso es divisible
        incompat_pairs (set): Pares de recursos incompatibles
        together_pairs (set): Pares de recursos que deben ir juntos
        allowed (dict): Indica si un recurso puede transportarse en un avión
        solver_name (str, optional): Nombre del solver a utilizar. Default: "glpk"

    Returns:
        dict: Resultados del análisis de sensibilidad:
            - most_limiting_plane: Nombre del avión más limitante
            - orig_value: Valor objetivo original
            - mod_value: Valor objetivo con capacidad aumentada
    """
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
def init_data_from_json(json_path):
    """Inicializa los datos del modelo desde un archivo JSON.

    Args:
        json_path (str): Ruta al archivo JSON

    Returns:
        tuple: (resources_list, planes_list, comp_list, sec_list, json_data)
               o (None, None, None, None, None) si hay error
    """
    try:
        # Cargar y validar datos JSON
        resources_list, planes_list, comp_list, sec_list = load_data_from_json(json_path)
        with open(json_path, "r", encoding="utf-8") as f:
            json_data = json.load(f)
        
        if not all([resources_list, planes_list]):
            raise ValueError("Datos JSON incompletos o inválidos")
        
        return resources_list, planes_list, comp_list, sec_list, json_data
    except Exception as e:
        print(f"Error al cargar datos JSON: {e}")
        print("Se usarán valores por defecto.")
        return None, None, None, None, None

def main():
    """Función principal que ejecuta el proceso completo de optimización.
    
    Pasos:
    1. Procesa argumentos de línea de comandos
    2. Carga datos (por defecto o desde JSON)
    3. Construye y resuelve el modelo de optimización
    4. Genera reportes y gráficos con los resultados
    5. Realiza análisis de sensibilidad
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=None, help='Archivo JSON opcional')
    parser.add_argument('--solver', type=str, default='glpk', help='Solver a usar')
    args = parser.parse_args()

    # Inicializar datos (desde JSON o por defecto)
    json_data = None
    resources_list = DEFAULT_RESOURCES
    planes_list = DEFAULT_PLANES
    comp_list = DEFAULT_COMPAT
    sec_list = DEFAULT_SECURITY

    if args.data:
        loaded_data = init_data_from_json(args.data)
        if loaded_data[0]:  # Si se cargaron datos válidos
            resources_list, planes_list, comp_list, sec_list, json_data = loaded_data

    # Construir estructuras de datos para el modelo
    resources, planes, val, stock, weight, volume, plane_w, plane_v = build_indexed_data(resources_list, planes_list)
    divisibles = determine_divisibility(json_data)
    # Preprocesar reglas con validación
    incompat_pairs, together_pairs = preprocess_compatibility(comp_list, resources)
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
        plot_untransported_resources(model, out_png="recursos_no_transportados.png")
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
