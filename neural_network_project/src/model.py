"""
Módulo de Modelo Neural
Este módulo contém a arquitetura da rede neural e funções relacionadas.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from typing import Tuple, Optional


def criar_modelo_simples(input_shape: Tuple[int, ...], 
                        num_classes: int,
                        activation: str = 'relu') -> keras.Model:
    """
    Cria um modelo de rede neural simples (MLP - Multi-Layer Perceptron).
    
    Args:
        input_shape: Formato dos dados de entrada (ex: (28, 28) para MNIST)
        num_classes: Número de classes para classificação
        activation: Função de ativação para camadas ocultas
    
    Returns:
        model: Modelo Keras compilado
    """
    
    model = models.Sequential([
        # Camada de entrada - achata a imagem 2D em vetor 1D
        layers.Flatten(input_shape=input_shape, name='camada_entrada'),
        
        # Primeira camada oculta com 128 neurônios
        layers.Dense(128, activation=activation, name='camada_oculta_1'),
        
        # Dropout para evitar overfitting (desativa 20% dos neurônios aleatoriamente)
        layers.Dropout(0.2),
        
        # Segunda camada oculta com 64 neurônios
        layers.Dense(64, activation=activation, name='camada_oculta_2'),
        
        # Mais dropout
        layers.Dropout(0.2),
        
        # Camada de saída com softmax para classificação multiclasse
        layers.Dense(num_classes, activation='softmax', name='camada_saida')
    ])
    
    return model


def criar_modelo_cnn(input_shape: Tuple[int, int, int], 
                     num_classes: int) -> keras.Model:
    """
    Cria um modelo CNN (Convolutional Neural Network) para classificação de imagens.
    
    Args:
        input_shape: Formato dos dados de entrada (altura, largura, canais)
        num_classes: Número de classes para classificação
    
    Returns:
        model: Modelo CNN Keras
    """
    
    model = models.Sequential([
        # Primeira camada convolucional
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape, 
                     padding='same', name='conv2d_1'),
        layers.MaxPooling2D((2, 2), name='maxpool_1'),
        
        # Segunda camada convolucional
        layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='conv2d_2'),
        layers.MaxPooling2D((2, 2), name='maxpool_2'),
        
        # Terceira camada convolucional
        layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='conv2d_3'),
        
        # Achata para conectar com camadas densas
        layers.Flatten(name='flatten'),
        
        # Camadas densas (fully connected)
        layers.Dense(64, activation='relu', name='dense_1'),
        layers.Dropout(0.5),
        
        # Camada de saída
        layers.Dense(num_classes, activation='softmax', name='output')
    ])
    
    return model


def compilar_modelo(model: keras.Model, 
                   learning_rate: float = 0.001,
                   loss: str = 'sparse_categorical_crossentropy',
                   metrics: list = ['accuracy']) -> keras.Model:
    """
    Compila o modelo com otimizador, função de perda e métricas.
    
    Args:
        model: Modelo Keras não compilado
        learning_rate: Taxa de aprendizado
        loss: Função de perda
        metrics: Lista de métricas para acompanhar
    
    Returns:
        model: Modelo compilado
    """
    
    # Usa o otimizador Adam com a taxa de aprendizado especificada
    optimizer = Adam(learning_rate=learning_rate)
    
    # Compila o modelo
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics
    )
    
    return model


def resumo_modelo(model: keras.Model) -> None:
    """
    Exibe um resumo detalhado do modelo.
    
    Args:
        model: Modelo Keras
    """
    print("\n" + "="*50)
    print("RESUMO DO MODELO")
    print("="*50)
    model.summary()
    print("="*50 + "\n")


def salvar_modelo(model: keras.Model, 
                 caminho: str,
                 incluir_optimizer: bool = True) -> None:
    """
    Salva o modelo em formato H5 ou SavedModel.
    
    Args:
        model: Modelo treinado
        caminho: Caminho onde salvar o modelo
        incluir_optimizer: Se deve incluir o estado do otimizador
    """
    if caminho.endswith('.h5'):
        # Salva em formato H5
        model.save(caminho, include_optimizer=incluir_optimizer)
    else:
        # Salva em formato SavedModel (padrão do TensorFlow)
        model.save(caminho, include_optimizer=incluir_optimizer)
    
    print(f"Modelo salvo em: {caminho}")


def carregar_modelo(caminho: str) -> keras.Model:
    """
    Carrega um modelo salvo.
    
    Args:
        caminho: Caminho do modelo salvo
    
    Returns:
        model: Modelo carregado
    """
    model = keras.models.load_model(caminho)
    print(f"Modelo carregado de: {caminho}")
    return model