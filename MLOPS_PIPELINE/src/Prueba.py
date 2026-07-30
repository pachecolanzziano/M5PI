import requests
import json

url = "http://localhost:8000/predict"
data = {
    "capital_prestado": 3000000.0,
    "plazo_meses": 12,
    "edad_cliente": 35,
    "salario_cliente": 2000000.0,
    "total_otros_prestamos": 500000.0,
    "cuota_pactada": 250000.0,
    "creditos_sectorFinanciero": 2,
    "creditos_sectorCooperativo": 1,
    "creditos_sectorReal": 0,
    "promedio_ingresos_datacredito": 1800000.0
}

response = requests.post(url, json=data)
print(response.json())