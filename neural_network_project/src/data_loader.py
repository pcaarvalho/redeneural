"""
Módulo de Carregamento de Dados
Este módulo é responsável por carregar e preprocessar datasets.
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
from typing import Tuple, Optional
import urllib.request
import gzip
import shutil


def carregar_mnist() -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """
    Carrega o dataset MNIST de dígitos manuscritos.
    
    O MNIST contém 70,000 imagens de dígitos (0-9) em escala de cinza de 28x28 pixels.
    - 60,000 para treino
    - 10,000 para teste
    
    Returns:
        (x_train, y_train), (x_test, y_test): Dados de treino e teste
    """
    print("Carregando dataset MNIST...")
    
    # Carrega o dataset usando Keras
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    
    print(f"Dados de treino: {x_train.shape} imagens")
    print(f"Dados de teste: {x_test.shape} imagens")
    print(f"Formato das imagens: {x_train[0].shape}")
    print(f"Classes: {np.unique(y_train)}")
    
    return (x_train, y_train), (x_test, y_test)


def carregar_cifar10() -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """
    Carrega o dataset CIFAR-10 de imagens coloridas.
    
    O CIFAR-10 contém 60,000 imagens coloridas de 32x32 pixels em 10 classes:
    - avião, automóvel, pássaro, gato, cervo, cachorro, sapo, cavalo, navio, caminhão
    
    Returns:
        (x_train, y_train), (x_test, y_test): Dados de treino e teste
    """
    print("Carregando dataset CIFAR-10...")
    
    # Carrega o dataset usando Keras
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    
    # CIFAR-10 retorna labels como array 2D, vamos achatar
    y_train = y_train.flatten()
    y_test = y_test.flatten()
    
    classes = ['avião', 'automóvel', 'pássaro', 'gato', 'cervo', 
               'cachorro', 'sapo', 'cavalo', 'navio', 'caminhão']
    
    print(f"Dados de treino: {x_train.shape} imagens")
    print(f"Dados de teste: {x_test.shape} imagens")
    print(f"Formato das imagens: {x_train[0].shape}")
    print(f"Classes: {classes}")
    
    return (x_train, y_train), (x_test, y_test)


def carregar_fashion_mnist() -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """
    Carrega o dataset Fashion-MNIST de imagens de roupas.
    
    Similar ao MNIST mas com imagens de roupas em vez de dígitos.
    
    Returns:
        (x_train, y_train), (x_test, y_test): Dados de treino e teste
    """
    print("Carregando dataset Fashion-MNIST...")
    
    # Carrega o dataset usando Keras
    (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
    
    classes = ['Camiseta', 'Calça', 'Suéter', 'Vestido', 'Casaco',
               'Sandália', 'Camisa', 'Tênis', 'Bolsa', 'Bota']
    
    print(f"Dados de treino: {x_train.shape} imagens")
    print(f"Dados de teste: {x_test.shape} imagens")
    print(f"Formato das imagens: {x_train[0].shape}")
    print(f"Classes: {classes}")
    
    return (x_train, y_train), (x_test, y_test)


def preprocessar_dados(x_train: np.ndarray, 
                      x_test: np.ndarray,
                      normalizar: bool = True,
                      expandir_dims: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Preprocessa os dados para treino.
    
    Args:
        x_train: Dados de treino
        x_test: Dados de teste
        normalizar: Se deve normalizar os pixels para [0, 1]
        expandir_dims: Se deve adicionar dimensão de canal (para CNNs)
    
    Returns:
        x_train, x_test: Dados preprocessados
    """
    print("\nPreprocessando dados...")
    
    # Converte para float32 para eficiência
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    
    # Normaliza pixels de [0, 255] para [0, 1]
    if normalizar:
        x_train = x_train / 255.0
        x_test = x_test / 255.0
        print("✓ Dados normalizados para intervalo [0, 1]")
    
    # Adiciona dimensão de canal se necessário (para CNNs)
    if expandir_dims and len(x_train.shape) == 3:
        x_train = np.expand_dims(x_train, -1)
        x_test = np.expand_dims(x_test, -1)
        print(f"✓ Dimensão de canal adicionada. Novo formato: {x_train.shape}")
    
    return x_train, x_test


def criar_data_augmentation() -> keras.Sequential:
    """
    Cria um pipeline de aumento de dados para melhorar a generalização.
    
    Returns:
        data_augmentation: Pipeline de transformações
    """
    data_augmentation = keras.Sequential([
        # Rotação aleatória de até 10 graus
        keras.layers.RandomRotation(0.1),
        
        # Zoom aleatório
        keras.layers.RandomZoom(0.1),
        
        # Deslocamento aleatório
        keras.layers.RandomTranslation(0.1, 0.1),
        
        # Flip horizontal aleatório (útil para alguns datasets)
        # keras.layers.RandomFlip("horizontal"),
    ])
    
    return data_augmentation


def dividir_validacao(x_train: np.ndarray, 
                     y_train: np.ndarray,
                     val_split: float = 0.1) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Divide os dados de treino em treino e validação.
    
    Args:
        x_train: Dados de treino
        y_train: Labels de treino
        val_split: Proporção para validação (padrão 10%)
    
    Returns:
        x_train, y_train, x_val, y_val: Dados divididos
    """
    # Calcula o ponto de divisão
    val_size = int(len(x_train) * val_split)
    
    # Embaralha os índices
    indices = np.random.permutation(len(x_train))
    
    # Divide os dados
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]
    
    x_val = x_train[val_indices]
    y_val = y_train[val_indices]
    x_train = x_train[train_indices]
    y_train = y_train[train_indices]
    
    print(f"\nDados divididos:")
    print(f"Treino: {len(x_train)} amostras")
    print(f"Validação: {len(x_val)} amostras")
    
    return x_train, y_train, x_val, y_val


def criar_tf_dataset(x: np.ndarray, 
                    y: np.ndarray,
                    batch_size: int = 32,
                    shuffle: bool = True,
                    augmentation: Optional[keras.Sequential] = None) -> tf.data.Dataset:
    """
    Cria um tf.data.Dataset otimizado para performance.
    
    Args:
        x: Dados de entrada
        y: Labels
        batch_size: Tamanho do batch
        shuffle: Se deve embaralhar os dados
        augmentation: Pipeline de aumento de dados (opcional)
    
    Returns:
        dataset: tf.data.Dataset pronto para treino
    """
    # Cria dataset básico
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    
    # Embaralha se necessário
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(x))
    
    # Aplica aumento de dados se fornecido
    if augmentation is not None:
        dataset = dataset.map(lambda x, y: (augmentation(x), y),
                            num_parallel_calls=tf.data.AUTOTUNE)
    
    # Cria batches
    dataset = dataset.batch(batch_size)
    
    # Otimizações de performance
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset


def baixar_dataset_personalizado(url: str, 
                               destino: str,
                               extrair: bool = True) -> str:
    """
    Baixa um dataset de uma URL.
    
    Args:
        url: URL do dataset
        destino: Pasta de destino
        extrair: Se deve extrair arquivo comprimido
    
    Returns:
        caminho: Caminho do dataset baixado/extraído
    """
    os.makedirs(destino, exist_ok=True)
    
    filename = os.path.basename(url)
    filepath = os.path.join(destino, filename)
    
    if not os.path.exists(filepath):
        print(f"Baixando dataset de {url}...")
        urllib.request.urlretrieve(url, filepath)
        print(f"Download concluído: {filepath}")
    else:
        print(f"Dataset já existe: {filepath}")
    
    # Extrai se necessário
    if extrair and filepath.endswith('.gz'):
        extracted_path = filepath[:-3]  # Remove .gz
        if not os.path.exists(extracted_path):
            print("Extraindo arquivo...")
            with gzip.open(filepath, 'rb') as f_in:
                with open(extracted_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            print(f"Extraído para: {extracted_path}")
        return extracted_path
    
    return filepath