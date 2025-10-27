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