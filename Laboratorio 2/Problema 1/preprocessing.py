"""
Módulo para el manejo de datos JSON y preprocesamiento.
"""
import json
from typing import Dict, List, Set, Tuple, Optional, Any


def validate_json_structure(data: Dict[str, Any]) -> bool:
    """Valida que el JSON tenga la estructura correcta para el modelo.

    Args:
        data: Datos cargados del JSON

    Returns:
        True si la estructura es válida

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

def convert_json_to_model_format(data: Dict[str, Any]) -> Tuple[List, List, List, List]:
    """Convierte los datos del JSON al formato usado por el modelo.

    Args:
        data: Datos cargados y validados del JSON

    Returns:
        (resources_list, planes_list, comp_list, sec_list)
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

def load_data_from_json(path: str) -> Tuple[Optional[List], Optional[List], Optional[List], Optional[List]]:
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
        ]
    }

    Args:
        path: Ruta al archivo JSON que contiene los datos de entrada

    Returns:
        (resources_list, planes_list, comp_list, sec_list) en el formato
        esperado por el modelo, o (None, None, None, None) si hay error
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


def build_indexed_data(resources_list: List[Tuple], planes_list: List[Tuple]):
    """Construye estructuras indexadas para Pyomo.

    Args:
        resources_list: Lista de tuplas con información de recursos
        planes_list: Lista de tuplas con información de aviones

    Returns:
        Estructuras de datos indexadas para el modelo Pyomo
    """
    resources = [r[0] for r in resources_list]  # nombres de recursos
    planes = [p[0] for p in planes_list]  # nombres de aviones
    val = {r[0]: r[1] for r in resources_list}  # valor por recurso
    stock = {r[0]: r[2] for r in resources_list}  # stock por recurso
    weight = {r[0]: r[3] for r in resources_list}  # peso por recurso
    volume = {r[0]: r[4] for r in resources_list}  # volumen por recurso
    plane_w = {p[0]: p[1] for p in planes_list}  # capacidad peso por avión
    plane_v = {p[0]: p[2] for p in planes_list}  # capacidad volumen por avión
    return resources, planes, val, stock, weight, volume, plane_w, plane_v