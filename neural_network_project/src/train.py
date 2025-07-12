"""
Módulo de Treinamento
Este módulo contém funções para treinar e monitorar modelos de rede neural.
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import json


def criar_callbacks(model_dir: str, 
                   patience: int = 5,
                   monitor: str = 'val_loss') -> List[keras.callbacks.Callback]:
    """
    Cria callbacks úteis para o treinamento.
    
    Args:
        model_dir: Diretório para salvar checkpoints
        patience: Paciência para early stopping
        monitor: Métrica para monitorar
    
    Returns:
        callbacks: Lista de callbacks configurados
    """
    os.makedirs(model_dir, exist_ok=True)
    
    callbacks = []
    
    # 1. ModelCheckpoint - Salva o melhor modelo durante o treino
    checkpoint_path = os.path.join(model_dir, 'best_model.h5')
    checkpoint = keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        monitor=monitor,
        save_best_only=True,
        verbose=1,
        mode='min' if 'loss' in monitor else 'max'
    )
    callbacks.append(checkpoint)
    
    # 2. EarlyStopping - Para o treino se não houver melhora
    early_stop = keras.callbacks.EarlyStopping(
        monitor=monitor,
        patience=patience,
        verbose=1,
        restore_best_weights=True,  # Restaura os melhores pesos ao final
        mode='min' if 'loss' in monitor else 'max'
    )
    callbacks.append(early_stop)
    
    # 3. ReduceLROnPlateau - Reduz taxa de aprendizado quando estagna
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor=monitor,
        factor=0.5,  # Reduz pela metade
        patience=3,
        min_lr=1e-7,
        verbose=1,
        mode='min' if 'loss' in monitor else 'max'
    )
    callbacks.append(reduce_lr)
    
    # 4. TensorBoard - Para visualização do treino
    log_dir = os.path.join('logs', datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard = keras.callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=1,
        write_graph=True,
        update_freq='epoch'
    )
    callbacks.append(tensorboard)
    
    print(f"\nCallbacks configurados:")
    print(f"✓ ModelCheckpoint salvando em: {checkpoint_path}")
    print(f"✓ EarlyStopping com paciência: {patience}")
    print(f"✓ ReduceLROnPlateau")
    print(f"✓ TensorBoard logs em: {log_dir}")
    
    return callbacks


def treinar_modelo(model: keras.Model,
                  x_train: np.ndarray,
                  y_train: np.ndarray,
                  x_val: np.ndarray,
                  y_val: np.ndarray,
                  epochs: int = 30,
                  batch_size: int = 32,
                  callbacks: Optional[List] = None) -> keras.callbacks.History:
    """
    Treina o modelo com os dados fornecidos.
    
    Args:
        model: Modelo compilado
        x_train, y_train: Dados de treino
        x_val, y_val: Dados de validação
        epochs: Número de épocas
        batch_size: Tamanho do batch
        callbacks: Lista de callbacks (opcional)
    
    Returns:
        history: Histórico do treinamento
    """
    print("\n" + "="*50)
    print("INICIANDO TREINAMENTO")
    print("="*50)
    print(f"Épocas: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Amostras de treino: {len(x_train)}")
    print(f"Amostras de validação: {len(x_val)}")
    print("="*50 + "\n")
    
    # Treina o modelo
    history = model.fit(
        x_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(x_val, y_val),
        callbacks=callbacks,
        verbose=1
    )
    
    print("\n✓ Treinamento concluído!")
    
    return history


def treinar_com_tf_data(model: keras.Model,
                       train_dataset: tf.data.Dataset,
                       val_dataset: tf.data.Dataset,
                       epochs: int = 30,
                       steps_per_epoch: Optional[int] = None,
                       validation_steps: Optional[int] = None,
                       callbacks: Optional[List] = None) -> keras.callbacks.History:
    """
    Treina o modelo usando tf.data.Dataset (mais eficiente).
    
    Args:
        model: Modelo compilado
        train_dataset: Dataset de treino
        val_dataset: Dataset de validação
        epochs: Número de épocas
        steps_per_epoch: Passos por época (opcional)
        validation_steps: Passos de validação (opcional)
        callbacks: Lista de callbacks
    
    Returns:
        history: Histórico do treinamento
    """
    print("\n" + "="*50)
    print("INICIANDO TREINAMENTO (tf.data)")
    print("="*50)
    print(f"Épocas: {epochs}")
    print("="*50 + "\n")
    
    # Treina o modelo
    history = model.fit(
        train_dataset,
        epochs=epochs,
        validation_data=val_dataset,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        callbacks=callbacks,
        verbose=1
    )
    
    print("\n✓ Treinamento concluído!")
    
    return history


def plotar_historico(history: keras.callbacks.History,
                    save_path: Optional[str] = None) -> None:
    """
    Plota gráficos do histórico de treinamento.
    
    Args:
        history: Histórico retornado pelo fit()
        save_path: Caminho para salvar o gráfico (opcional)
    """
    # Cria figura com subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot 1: Loss (Perda)
    ax1.plot(history.history['loss'], label='Treino', linewidth=2)
    ax1.plot(history.history['val_loss'], label='Validação', linewidth=2)
    ax1.set_title('Perda ao Longo das Épocas', fontsize=14)
    ax1.set_xlabel('Época')
    ax1.set_ylabel('Perda')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Accuracy (Acurácia)
    if 'accuracy' in history.history:
        ax2.plot(history.history['accuracy'], label='Treino', linewidth=2)
        ax2.plot(history.history['val_accuracy'], label='Validação', linewidth=2)
        ax2.set_title('Acurácia ao Longo das Épocas', fontsize=14)
        ax2.set_xlabel('Época')
        ax2.set_ylabel('Acurácia')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Salva se caminho fornecido
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Gráfico salvo em: {save_path}")
    
    plt.show()


def salvar_historico(history: keras.callbacks.History,
                    filepath: str) -> None:
    """
    Salva o histórico de treinamento em JSON.
    
    Args:
        history: Histórico do treinamento
        filepath: Caminho para salvar
    """
    # Converte para dicionário serializável
    history_dict = history.history
    
    # Converte arrays numpy para listas
    for key in history_dict:
        history_dict[key] = [float(val) for val in history_dict[key]]
    
    # Salva em JSON
    with open(filepath, 'w') as f:
        json.dump(history_dict, f, indent=4)
    
    print(f"Histórico salvo em: {filepath}")


def carregar_historico(filepath: str) -> Dict:
    """
    Carrega histórico de treinamento de arquivo JSON.
    
    Args:
        filepath: Caminho do arquivo
    
    Returns:
        history_dict: Dicionário com histórico
    """
    with open(filepath, 'r') as f:
        history_dict = json.load(f)
    
    return history_dict


def encontrar_melhor_epoca(history: keras.callbacks.History,
                          metrica: str = 'val_loss') -> Tuple[int, float]:
    """
    Encontra a melhor época baseada em uma métrica.
    
    Args:
        history: Histórico do treinamento
        metrica: Métrica para avaliar
    
    Returns:
        best_epoch, best_value: Melhor época e valor
    """
    valores = history.history[metrica]
    
    if 'loss' in metrica:
        # Para loss, queremos o mínimo
        best_epoch = np.argmin(valores)
    else:
        # Para accuracy, queremos o máximo
        best_epoch = np.argmax(valores)
    
    best_value = valores[best_epoch]
    
    print(f"\nMelhor época para {metrica}: {best_epoch + 1}")
    print(f"Valor: {best_value:.4f}")
    
    return best_epoch + 1, best_value


def ajuste_fino(model: keras.Model,
               x_train: np.ndarray,
               y_train: np.ndarray,
               x_val: np.ndarray,
               y_val: np.ndarray,
               learning_rate: float = 0.0001,
               epochs: int = 10) -> keras.callbacks.History:
    """
    Realiza ajuste fino do modelo com taxa de aprendizado menor.
    
    Args:
        model: Modelo pré-treinado
        x_train, y_train: Dados de treino
        x_val, y_val: Dados de validação
        learning_rate: Taxa de aprendizado reduzida
        epochs: Número de épocas para ajuste
    
    Returns:
        history: Histórico do ajuste fino
    """
    print("\n" + "="*50)
    print("AJUSTE FINO (Fine-tuning)")
    print("="*50)
    
    # Recompila com taxa de aprendizado menor
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=model.loss,
        metrics=model.metrics
    )
    
    print(f"Nova taxa de aprendizado: {learning_rate}")
    
    # Treina por mais algumas épocas
    history = model.fit(
        x_train, y_train,
        batch_size=32,
        epochs=epochs,
        validation_data=(x_val, y_val),
        verbose=1
    )
    
    return history