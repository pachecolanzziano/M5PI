# librerías
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import FunctionTransformer

from cargar_datos import cargarDatos

# Cargamos los datos
df = cargarDatos()

# vista previa de los datos
print(df.head())
print(df.info())
print(df.describe())

# split de Features/Target
X = df.drop('Pago_atiempo', axis=1)   # Features
y = df['Pago_atiempo']                # Target

# 3. Definimos los tipos de variables
num_features = X.select_dtypes('number').columns
cat_features = X.select_dtypes('object').columns

print(f'Features numéricas: {num_features}')
print(f'Features categóricas: {cat_features}')

# 4. Creamos pipelines para cada ruta (numérica y categórica)
## Ruta 1: numéricas
num_transformer = Pipeline(
    steps=[
        ('inputer', SimpleImputer(strategy='mean'))
    ]
)

## Ruta 2: categóricas
cat_transformer = Pipeline(
    steps=[
        ('to_str', FunctionTransformer(lambda x: x.astype(str))),
        ('inputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ]
)

# 5. Combinamos las rutas en ColumnTransformer

preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_transformer, num_features),
        ('cat', cat_transformer, cat_features)
    ]
)

# 6. Dividimos los datos en train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 7. Aplicamos el preprocesamiento a los datos
X_train_preprocessed = preprocessor.fit_transform(X_train)
X_test_preprocessed = preprocessor.transform(X_test)

# 8. Imprimimos las resultados de los datos preprocesados
print(f'X_train_preprocessed shape: {X_train_preprocessed}')
print(f'X_test_preprocessed shape: {X_test_preprocessed}')

# 9. Construimos una función para encapsular todo el proceso de preprocesamiento
def preprocesar_datos():
    # 1. Split Features/Target
    X = df.drop('Pago_atiempo', axis=1)
    y = df['Pago_atiempo']

    # 2. Definimos los tipos de variables
    num_features = X.select_dtypes('number').columns
    cat_features = X.select_dtypes('object').columns

    # 3. Creamos pipelines para cada ruta (numérica y categórica)
    num_transformer = Pipeline(
        steps=[
            ('inputer', SimpleImputer(strategy='mean'))
        ]
    )

    cat_transformer = Pipeline(
        steps=[
            ('to_str', FunctionTransformer(lambda x: x.astype(str))),
            ('inputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ]
    )

    # 4. Combinamos las rutas en ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, num_features),
            ('cat', cat_transformer, cat_features)
        ]
    )

    # 5. Dividimos los datos en train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 6. Aplicamos el preprocesamiento a los datos
    X_train_preprocessed = preprocessor.fit_transform(X_train)
    X_test_preprocessed = preprocessor.transform(X_test)

    return preprocessor
