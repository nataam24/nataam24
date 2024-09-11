# Instalar el módulo faker, que se utiliza para generar datos falsos como nombres, direcciones, etc. (solo es necesario correr esto una vez en tu entorno)
pip install faker  

# Importar las librerías necesarias
import sqlite3  # Para manejar bases de datos SQLite
import pandas as pd  # Para manipular datos en forma de DataFrames
import requests  # Para hacer solicitudes HTTP, en este caso a la API de randomuser
import random  # Para generar datos aleatorios
import uuid  # Para generar identificadores únicos
from faker import Faker  # Para generar datos falsos como nombres y direcciones
import numpy as np  # Para operaciones numéricas, incluyendo generación de valores aleatorios
import shutil  # Para copiar archivos en el sistema de archivos
from google.colab import drive  # Para interactuar con Google Drive en Google Colab

# Montar Google Drive en el entorno de Colab
drive.mount('/content/drive')

# Inicializar el generador de datos falsos
fake = Faker()

# Crear la conexión a la base de datos SQLite
conn = sqlite3.connect('financial_data.db')

# Definir una función para ejecutar consultas SQL
def execute_query(query, conn):
    with conn:
        conn.execute(query)

# Crea una tabla llamada customers en la base de datos con columnas para ID de cliente, nombre, dirección, número de teléfono y correo electrónico. Si la tabla ya existe, no se recrea
execute_query('''CREATE TABLE IF NOT EXISTS customers (
                 customer_id TEXT PRIMARY KEY,
                 name TEXT,
                 address TEXT,
                 phone_number TEXT,
                 email TEXT);''', conn)
# Crea una tabla llamada branches para almacenar la información de las sucursales, incluyendo su ID, ubicación, nombre del gerente y número de contacto
execute_query('''CREATE TABLE IF NOT EXISTS branches (
                 branch_id TEXT PRIMARY KEY,
                 branch_location TEXT,
                 manager_name TEXT,
                 contact_number TEXT);''', conn)
# Crea una tabla llamada transaction_types que almacena los diferentes tipos de transacciones y sus descripciones.
execute_query('''CREATE TABLE IF NOT EXISTS transaction_types (
                 transaction_type TEXT PRIMARY KEY,
                 description TEXT);''', conn)
# Crea una tabla llamada transactions para almacenar transacciones financieras, incluyendo su ID, ID del cliente, fecha, monto, ubicación, tipo de transacción, si es fraudulenta, y la sucursal asociada. Además, define relaciones (FOREIGN KEY) entre esta tabla y las tablas customers, transaction_types y branches
execute_query('''CREATE TABLE IF NOT EXISTS transactions (
                 transaction_id TEXT PRIMARY KEY,
                 customer_id TEXT,
                 transaction_date TEXT,
                 transaction_amount REAL,
                 transaction_location TEXT,
                 transaction_type TEXT,
                 fraudulent INTEGER,
                 branch_id TEXT,
                 FOREIGN KEY(customer_id) REFERENCES customers(customer_id),
                 FOREIGN KEY(transaction_type) REFERENCES transaction_types(transaction_type),
                 FOREIGN KEY(branch_id) REFERENCES branches(branch_id));''', conn)

# Define una función para obtener datos de usuarios aleatorios de la API randomuser. Si la solicitud es exitosa (código 200), devuelve los resultados; de lo contrario, muestra un mensaje de error
def get_random_users(num_users=300):
    url = f"https://randomuser.me/api/?results={num_users}&nat=us"
    response = requests.get(url)
    if response.status_code == 200:
        users = response.json()['results']
        return users
    else:
        print("Error fetching data from randomuser.me")
        return []

# Define una función para crear una tabla de clientes en la base de datos. Genera datos falsos para los clientes utilizando randomuser y uuid. Guarda los datos en la tabla customers en SQLite y los devuelve como un DataFrame de Pandas
def create_customers_table(num_customers=300):
    users = get_random_users(num_customers)
    customers_data = {
        "customer_id": [str(uuid.uuid4()) for _ in range(num_customers)],
        "name": [f"{user['name']['first']} {user['name']['last']}" for user in users],
        "address": [f"{user['location']['street']['number']} {user['location']['street']['name']}, {user['location']['city']}, {user['location']['state']}, {user['location']['postcode']}" for user in users],
        "phone_number": [user['phone'] for user in users],
        "email": [user['email'] for user in users]
    }
    customers_df = pd.DataFrame(customers_data)
    customers_df.to_sql('customers', conn, if_exists='replace', index=False)
    return customers_df
 # Llama a la función create_customers_table para crear 100 registros de clientes y almacenarlos en la base de datos
  customers_df = create_customers_table(num_customers=100)

# Define una función para crear una tabla de sucursales en la base de datos. Genera datos falsos para las sucursales y los guarda en la tabla branches en SQLite, devolviendo los datos como un DataFrame.
  def create_branches_table(num_branches=10):
    branch_data = {
        "branch_id": [str(uuid.uuid4()) for _ in range(num_branches)],
        "branch_location": [fake.city() for _ in range(num_branches)],
        "manager_name": [fake.name() for _ in range(num_branches)],
        "contact_number": [fake.phone_number() for _ in range(num_branches)]
    }
    branches_df = pd.DataFrame(branch_data)
    branches_df.to_sql('branches', conn, if_exists='replace', index=False)
    return branches_df

# Llama a la función create_branches_table para crear 20 registros de sucursales y almacenarlos en la base de datos.
branches_df = create_branches_table(num_branches=20)
# Define una función para crear una tabla de tipos de transacciones en la base de datos. Guarda los datos en la tabla transaction_types en SQLite y los devuelve como un DataFrame
def create_transaction_types_table():
    transaction_types_data = {
        "transaction_type": ["online", "in-store"],
        "description": ["Transaction made online via the internet",
                        "Transaction made at a physical store location"]
    }
    transaction_types_df = pd.DataFrame(transaction_types_data)
    transaction_types_df.to_sql('transaction_types', conn, if_exists='replace', index=False)
    return transaction_types_df


# Llama a la función create_transaction_types_table para crear la tabla de tipos de transacciones en la base de datos
transaction_types_df = create_transaction_types_table()

# Crear la tabla de transacciones y guardarla en SQLite
def create_transactions_table(customers_df, branches_df, num_transactions=1000):
    transaction_data = {
        "transaction_id": [str(uuid.uuid4()) for _ in range(num_transactions)],
        "customer_id": [random.choice(customers_df['customer_id']) for _ in range(num_transactions)],
        "transaction_date": [fake.date_time_this_year().isoformat() for _ in range(num_transactions)],
        "transaction_amount": [round(random.uniform(10.0, 1000.0), 2) for _ in range(num_transactions)],
        "transaction_location": [fake.city() for _ in range(num_transactions)],
        "transaction_type": [random.choice(["online", "in-store"]) for _ in range(num_transactions)],
        "fraudulent": [0] * num_transactions
    }
    transactions_df = pd.DataFrame(transaction_data)

  # Introducir transacciones fraudulentas
    n_fraud = 10  # Número de transacciones fraudulentas
    fraud_indices = np.random.choice(transactions_df.index, n_fraud, replace=False)
    transactions_df.loc[fraud_indices, 'fraudulent'] = 1
    transactions_df.loc[fraud_indices, 'transaction_amount'] = [round(random.uniform(1000.0, 5000.0), 2) for _ in range(n_fraud)]
    transactions_df.loc[fraud_indices, 'transaction_type'] = "online"

  # Asignar branch_id solo para transacciones "in-store"
    in_store_indices = transactions_df[transactions_df['transaction_type'] == 'in-store'].index
    transactions_df.loc[in_store_indices, 'branch_id'] = np.random.choice(branches_df['branch_id'], size=len(in_store_indices))

  # Para transacciones online, branch_id se puede establecer en None
    transactions_df['branch_id'].fillna('None', inplace=True)

    transactions_df.to_sql('transactions', conn, if_exists='replace', index=False)
    return transactions_df

# Crear la tabla de transacciones
transactions_df = create_transactions_table(customers_df, branches_df, num_transactions=500)

# Verificar la creación de tablas en SQLite
print("Tablas en la base de datos SQLite:")
query = "SELECT name FROM sqlite_master WHERE type='table';"
tables = pd.read_sql(query, conn)
print(tables)

# Cerrar la conexión
conn.close()

# Ruta en Google Drive donde deseas guardar la base de datos
drive_db_path = '/content/drive/My Drive/financial_data.db'

# Copiar el archivo de la base de datos a Google Drive
shutil.copy('financial_data.db', drive_db_path)

print(f'Base de datos guardada en: {drive_db_path}')


# Este código se utiliza para simular y crear una base de datos financiera en SQLite, donde se almacenan datos de clientes, sucursales, tipos de transacciones y transacciones realizadas. La base de datos se genera automáticamente usando datos ficticios, como nombres de clientes y ubicaciones, obtenidos a través del módulo Faker y la API randomuser. Las tablas generadas se guardan en Google Drive, lo que facilita su acceso y manejo en futuros análisis o procesos.

# Este código es útil en la creación de entornos de prueba o en la enseñanza de conceptos de bases de datos, simulando un entorno financiero realista sin necesidad de datos reales.


# se agregó lo siguiente para importar librerías y poder crear las gráficas
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
import seaborn as sns

# 1. Conectar a la base de datos SQLite
try:
    conn = sqlite3.connect("/content/drive/MyDrive/financial_data.db")
    print("Conexión a la base de datos exitosa.")
except Exception as e:
    print(f"Error al conectar a la base de datos: {e}")
    raise

# 2. Leer la tabla 'db_temp_Azure_vw_CLIENTES' en un DataFrame
try:
    df = pd.read_sql_query("SELECT * FROM transactions", conn)
    print("Datos leídos exitosamente.")
except Exception as e:
    print(f"Error al leer la tabla: {e}")
    conn.close()
    raise

# 3. Verificación de completitud y visualización gráfica
completitud = df.notnull().mean() * 100
completitud.plot(kind='bar', figsize=(12, 6), color='skyblue')
plt.title('Completitud de las variables (% de valores no nulos)')
plt.ylabel('Porcentaje de completitud')
plt.xticks(rotation=45)
plt.show()

# 4. Estadísticas descriptivas para variables numéricas y visualización de la matriz
estadisticas_descriptivas = df.describe().transpose()
plt.figure(figsize=(12, 8))
sns.heatmap(estadisticas_descriptivas, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Matriz de estadísticas descriptivas')
plt.show()

# 5. Distribución de las variables numéricas
variables_numericas = df.select_dtypes(include=['float64', 'int64']).columns

for var in variables_numericas:
    plt.figure(figsize=(10, 6))
    sns.histplot(df[var].dropna(), kde=True, bins=30, color='cornflowerblue')
    plt.title(f'Distribución de la variable: {var}')
    plt.xlabel(var)
    plt.ylabel('Frecuencia')
    plt.show()

# 6. Matriz de correlación para variables numéricas
plt.figure(figsize=(10, 8))
correlacion = df[variables_numericas].corr()
sns.heatmap(correlacion, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Matriz de correlación')
plt.show()

# 7. Cerrar la conexión a la base de datos
conn.close()
print("Conexión a la base de datos cerrada.")

# resultados
# valores nulos
<img width="537" alt="image" src="https://github.com/user-attachments/assets/d421f00c-d5fd-4581-8e9c-2dfaf2809d5c">
La mayoría de las variables, si no todas, tienen un porcentaje de completitud muy cercano al 100%. Esto indica que hay muy pocos valores faltantes o nulos en el conjunto de datos.

# Estadísticas descriptivas
<img width="453" alt="image" src="https://github.com/user-attachments/assets/14a9eb6c-7aec-4203-adfe-89815a71c5f1">
La gráfica es una matriz de calor (heatmap) de estadísticas descriptivas. Este tipo de visualizaciones es útil para comparar de manera rápida y visual diferentes estadísticas de una o más variables. En este caso, el gráfico muestra dos variables principales: transaction_amount y fraudulent.

# Distribución de la variable transaction_amount
<img width="419" alt="image" src="https://github.com/user-attachments/assets/02d5d3e4-5efd-43f4-b2c3-d9812093450f">
La distribución de los montos de las transacciones parece ser asimétrica a la derecha o positivamente sesgada. Esto significa que la mayoría de las transacciones tienen un monto relativamente bajo, pero hay algunas transacciones con montos mucho más altos que "jalan" la media hacia la derecha.

# Distribución de la variable fraudolent
<img width="428" alt="image" src="https://github.com/user-attachments/assets/b7a23f48-f0eb-4efc-9aff-3fce3c8cadc0">
Desbalance de clases: La gráfica muestra un claro desbalance entre las clases. La gran mayoría de las transacciones son clasificadas como no fraudulentas (valor 0 en el eje X), mientras que solo una pequeña proporción se clasifica como fraudulenta (valor 1).
Pocos casos de fraude: La altura de la barra correspondiente al valor 1 es mucho más pequeña que la de la barra correspondiente al valor 0, lo que indica que los casos de fraude son relativamente pocos en comparación con las transacciones legítimas.

# Matriz de correlación
<img width="383" alt="image" src="https://github.com/user-attachments/assets/dc9b8c46-779e-4bc8-9245-b050f2c60099">
Existe una correlación positiva fuerte entre el monto de la transacción (transaction_amount) y la probabilidad de que sea fraudulenta (fraudulent). Esto significa que, en general, las transacciones de mayor monto tienen más probabilidad de ser fraudulentas.






