"""
Configuración y datos por defecto para el modelo de optimización.
"""

# Recursos disponibles: (nombre, valor_usd, stock_ton, peso_ton/un, volumen_m3/un)
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

def determine_divisibility():
    """Determina la divisibilidad de los recursos basado en reglas predefinidas.

    Returns:
        dict: Diccionario que mapea nombres de recursos a booleanos:
            - True: El recurso es divisible
            - False: El recurso debe asignarse en unidades enteras
    """
    divis = {}
    divis['Alimentos_Basicos'] = True
    divis['Medicinas'] = True
    divis['Equipos_Medicos'] = False
    divis['Agua_Potable'] = True
    divis['Mantas'] = False
    divis['Generadores'] = False
    divis['Tiendas_Campana'] = False
    divis['Medicamentos_Especiales'] = True
    divis['Equipos_Comunicacion'] = False
    divis['Material_Construccion'] = True
    return divis