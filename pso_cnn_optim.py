import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, Input

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

from tensorflow.keras import backend as K

from pyswarms.single.global_best import GlobalBestPSO

import csv


def prep_data(df):
    # Dividir las columnas numéricas y categóricas
    numerical_features = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    categorical_features = df.select_dtypes(include=['object']).columns.tolist()
    # Definir el preprocesador para las columnas numéricas
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),  # Imputación con la media
        ('scaler', MinMaxScaler())                    # Escalado Min-Max
    ])

    # Definir el preprocesador para las columnas categóricas
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),  # Imputación con la moda
        ('onehot', OneHotEncoder(handle_unknown='ignore'))     # Codificación One-Hot
    ])

    # Combinar ambos preprocesadores en un ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    return preprocessor.fit_transform(df), preprocessor

def create_1d_cnn(num_layers, filter_size, num_filters, input_shape, num_classes):
    model = Sequential()
    
    # Use Input layer as the first layer
    model.add(Input(shape=input_shape))

    # Add the first Conv1D layer
    model.add(Conv1D(filters=num_filters, kernel_size=filter_size, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    
    # Capas convolucionales adicionales, según se indique en num_layers
    for _ in range(num_layers - 1):
        model.add(Conv1D(filters=num_filters, kernel_size=filter_size, activation='relu'))
        model.add(MaxPooling1D(pool_size=2))
    
    # Capa de flatten y clasificación
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))  # Capa densa intermedia
    model.add(Dropout(0.5))  # Regularización para evitar sobreajuste
    model.add(Dense(num_classes, activation='softmax'))  # Capa de salida
    
    # Compilación del modelo
    model.compile(optimizer='adam', 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])
    
    return model

def simplified_model_evaluation(params, df, target_column = 'class', n_splits=5, n_repeats=3, epochs=5, batch_size=32):
    # Extraer las características (X) y etiquetas (y)
    X = df.drop(columns=[target_column])#.values
    y = df[target_column].values
    # Crear el objeto RepeatedKFold
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.2)

    X_train, preprocessor = prep_data(X_train)
    X_val = preprocessor.transform(X_val)

    # Asegurar que los datos tengan la forma correcta: (n_samples, timesteps, channels)
    X_train = tf.convert_to_tensor(X_train[..., np.newaxis])
    X_val = tf.convert_to_tensor(X_val[..., np.newaxis])

    num_layers,filter_size, num_filters, batch_size = map(int, params.flatten())
    input_shape = (X_train.shape[1], 1) # Longitud de secuencia de entrada y 1 canal
    num_classes = 2       # Número de clases a predecir

    model = create_1d_cnn(num_layers, filter_size, num_filters, input_shape, num_classes)
    
    if model.optimizer is None:
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    # Entrenar el modelo
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0);
    
    y_pred = model.predict(X_val);
    y_pred_classes = np.argmax(y_pred, axis=1)

    f1 = f1_score(y_val, y_pred_classes)

    K.clear_session()

    return -f1


def fitness_function(params):
    # 'params' is a 2D array: (n_particles, n_dimensions)
    results = []

    # Loop through each particle's parameters
    for particle in params:
        # Unpack parameters for each particle
        #num_layers, filter_size, num_filters, batch_size = map(int, particle)
        # Collect the negative accuracy (for minimization)
        results.append(simplified_model_evaluation(particle, df= df))

    # Return the results as a numpy array
    return np.array(results)



def run_pso_optimization(df, options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}, n_particles = 2, iters = 2, verbose = True):
 
    # Definir función lambda para pasar parámetros adicionales a la función objetivo
    #fitness_function = lambda params: repeated_kfold_cross_validation(params, df= df)
    lb = [1, 1, 16, 16]  # Límites inferiores (capas, filtros, tamaño del filtro, batch size)
    ub = [2, 4, 64, 128]  # Límites superiores
    # Ejecutar PSO
    optimizer =  GlobalBestPSO(n_particles = n_particles, 
                               dimensions = 4,
                               options=options, 
                               bounds= (lb, ub))
    
    best_score, best_params = optimizer.optimize(fitness_function, iters = iters, verbose = verbose)

    if verbose:
        print(f'Mejores Hiperparámetros: {best_params}')
        print(f'Mejor f1: {-best_score}')
    
    return best_params, -best_score

def experimento_clase(df, swarms=2, iters=2, n_particles=2, output_file='metrics.csv', pso_params_dict = {'Ws':[0.9],
                                                                                                          'c1s':[2],
                                                                                                          'c2s':[2]}):
    
    Ws = pso_params_dict['Ws']
    c1s = pso_params_dict['c1s']
    c2s = pso_params_dict['c2s']

    # Escribir el encabezado al inicio
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['W', 'C1', 'C2', 'f1_score', 'best_params', 'swarm'])

    # Iterar por cada combinación de hiperparámetros
    for w in Ws:
        for c1 in c1s:
            for c2 in c2s:
                for i in range(swarms):
                    print(f"    Realizando el enjambre {i+1}/{swarms} \n")

                    # Definir las opciones para el enjambre
                    options = {'c1': c1, 'c2': c2, 'w': w}

                    # Ejecutar la optimización PSO
                    best_params, f1 = run_pso_optimization(
                        df, options=options, iters=iters, n_particles=n_particles, verbose=False
                    )

                    # Formatear los mejores parámetros como cadena
                    best_params_str = (
                        f"num_layers:{int(best_params[0])}, "
                        f"filter_size:{int(best_params[1])}, "
                        f"num_filters:{int(best_params[2])}, "
                        f"batch_size:{int(best_params[3])}"
                    )

                    # Abrir el archivo para escribir una línea por vez
                    with open(output_file, mode='a', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow([w, c1, c2, f1, best_params_str, i])

                print("     ---------- Terminados los enjambres \n")

    print(f"Experimento completado. Resultados guardados en {output_file}")


if __name__ == '__main__':

    # Carga de datos
    df = pd.read_csv('df_credit_scoring.csv')

    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Disable TensorFlow INFO and WARNING logs

    # Disable progress bars in TensorFlow
    from tensorflow.keras.utils import disable_interactive_logging
    disable_interactive_logging()

    pso_params_dict = {'Ws':[0.9],
                       'c1s':[2, 1.5],
                       'c2s':[2, 1.5]}

    experimento_clase(df, swarms=10, n_particles=6, iters=10)