"""
Módulo de Avaliação
Este módulo contém funções para avaliar o desempenho dos modelos.
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from typing import List, Tuple, Optional, Dict
import cv2


def avaliar_modelo(model: keras.Model,
                  x_test: np.ndarray,
                  y_test: np.ndarray,
                  batch_size: int = 32) -> Tuple[float, float]:
    """
    Avalia o modelo nos dados de teste.
    
    Args:
        model: Modelo treinado
        x_test: Dados de teste
        y_test: Labels de teste
        batch_size: Tamanho do batch para avaliação
    
    Returns:
        loss, accuracy: Perda e acurácia no teste
    """
    print("\n" + "="*50)
    print("AVALIAÇÃO DO MODELO")
    print("="*50)
    
    # Avalia o modelo
    resultados = model.evaluate(x_test, y_test, batch_size=batch_size, verbose=1)
    
    # Extrai métricas
    if isinstance(resultados, list):
        loss = resultados[0]
        accuracy = resultados[1] if len(resultados) > 1 else None
    else:
        loss = resultados
        accuracy = None
    
    print(f"\nResultados no conjunto de teste:")
    print(f"Perda: {loss:.4f}")
    if accuracy is not None:
        print(f"Acurácia: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    return loss, accuracy


def fazer_predicoes(model: keras.Model,
                   x_test: np.ndarray,
                   batch_size: int = 32) -> Tuple[np.ndarray, np.ndarray]:
    """
    Faz predições no conjunto de teste.
    
    Args:
        model: Modelo treinado
        x_test: Dados de teste
        batch_size: Tamanho do batch
    
    Returns:
        y_pred_probs: Probabilidades para cada classe
        y_pred: Classes preditas
    """
    print("\nFazendo predições...")
    
    # Predições com probabilidades
    y_pred_probs = model.predict(x_test, batch_size=batch_size, verbose=1)
    
    # Converte probabilidades em classes
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    print(f"✓ Predições concluídas para {len(x_test)} amostras")
    
    return y_pred_probs, y_pred


def matriz_confusao(y_true: np.ndarray,
                   y_pred: np.ndarray,
                   class_names: Optional[List[str]] = None,
                   save_path: Optional[str] = None) -> np.ndarray:
    """
    Cria e plota a matriz de confusão.
    
    Args:
        y_true: Labels verdadeiros
        y_pred: Labels preditos
        class_names: Nomes das classes (opcional)
        save_path: Caminho para salvar a figura (opcional)
    
    Returns:
        cm: Matriz de confusão
    """
    # Calcula matriz de confusão
    cm = confusion_matrix(y_true, y_pred)
    
    # Configura o plot
    plt.figure(figsize=(10, 8))
    
    # Usa class_names se fornecido, senão usa números
    if class_names is None:
        class_names = [str(i) for i in range(len(cm))]
    
    # Cria heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names,
                cbar_kws={'label': 'Quantidade'})
    
    plt.title('Matriz de Confusão', fontsize=16)
    plt.xlabel('Predito', fontsize=12)
    plt.ylabel('Verdadeiro', fontsize=12)
    
    # Salva se caminho fornecido
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Matriz de confusão salva em: {save_path}")
    
    plt.show()
    
    return cm


def relatorio_classificacao(y_true: np.ndarray,
                          y_pred: np.ndarray,
                          class_names: Optional[List[str]] = None) -> str:
    """
    Gera relatório detalhado de classificação.
    
    Args:
        y_true: Labels verdadeiros
        y_pred: Labels preditos
        class_names: Nomes das classes (opcional)
    
    Returns:
        report: Relatório em string
    """
    print("\n" + "="*50)
    print("RELATÓRIO DE CLASSIFICAÇÃO")
    print("="*50)
    
    # Gera relatório
    report = classification_report(y_true, y_pred, 
                                 target_names=class_names,
                                 digits=4)
    
    print(report)
    
    return report


def visualizar_predicoes(model: keras.Model,
                        x_test: np.ndarray,
                        y_test: np.ndarray,
                        y_pred: np.ndarray,
                        num_amostras: int = 16,
                        class_names: Optional[List[str]] = None,
                        save_path: Optional[str] = None) -> None:
    """
    Visualiza amostras com suas predições.
    
    Args:
        model: Modelo usado
        x_test: Imagens de teste
        y_test: Labels verdadeiros
        y_pred: Labels preditos
        num_amostras: Número de amostras para mostrar
        class_names: Nomes das classes
        save_path: Caminho para salvar
    """
    # Seleciona amostras aleatórias
    indices = np.random.choice(len(x_test), num_amostras, replace=False)
    
    # Configura grid de subplots
    cols = int(np.sqrt(num_amostras))
    rows = int(np.ceil(num_amostras / cols))
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 15))
    axes = axes.ravel()
    
    for i, idx in enumerate(indices):
        # Pega a imagem
        img = x_test[idx]
        
        # Remove dimensão extra se existir
        if img.shape[-1] == 1:
            img = img.squeeze()
        
        # Mostra a imagem
        axes[i].imshow(img, cmap='gray' if len(img.shape) == 2 else None)
        
        # Título com predição
        true_label = y_test[idx]
        pred_label = y_pred[idx]
        
        if class_names:
            true_name = class_names[true_label]
            pred_name = class_names[pred_label]
            title = f'Real: {true_name}\nPred: {pred_name}'
        else:
            title = f'Real: {true_label}\nPred: {pred_label}'
        
        # Cor do título indica se acertou (verde) ou errou (vermelho)
        color = 'green' if true_label == pred_label else 'red'
        axes[i].set_title(title, color=color, fontsize=10)
        
        axes[i].axis('off')
    
    # Remove subplots extras
    for i in range(num_amostras, len(axes)):
        fig.delaxes(axes[i])
    
    plt.suptitle('Visualização de Predições', fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualização salva em: {save_path}")
    
    plt.show()


def analisar_erros(x_test: np.ndarray,
                  y_test: np.ndarray,
                  y_pred: np.ndarray,
                  class_names: Optional[List[str]] = None) -> Dict:
    """
    Analisa os erros de classificação.
    
    Args:
        x_test: Dados de teste
        y_test: Labels verdadeiros
        y_pred: Labels preditos
        class_names: Nomes das classes
    
    Returns:
        analise: Dicionário com análise dos erros
    """
    # Identifica erros
    erros_mask = y_test != y_pred
    indices_erros = np.where(erros_mask)[0]
    
    print(f"\nAnálise de Erros:")
    print(f"Total de amostras: {len(y_test)}")
    print(f"Total de erros: {len(indices_erros)}")
    print(f"Taxa de erro: {len(indices_erros)/len(y_test)*100:.2f}%")
    
    # Conta erros por classe
    erros_por_classe = {}
    for i in range(len(np.unique(y_test))):
        mask_classe = y_test == i
        erros_classe = np.sum((y_test == i) & (y_test != y_pred))
        total_classe = np.sum(mask_classe)
        
        nome_classe = class_names[i] if class_names else str(i)
        erros_por_classe[nome_classe] = {
            'total': int(total_classe),
            'erros': int(erros_classe),
            'taxa_erro': float(erros_classe / total_classe * 100) if total_classe > 0 else 0
        }
    
    # Matriz de confusão dos erros
    print("\nErros por classe:")
    for classe, info in erros_por_classe.items():
        print(f"  {classe}: {info['erros']}/{info['total']} ({info['taxa_erro']:.2f}%)")
    
    return {
        'total_erros': len(indices_erros),
        'taxa_erro_global': len(indices_erros)/len(y_test)*100,
        'erros_por_classe': erros_por_classe,
        'indices_erros': indices_erros.tolist()
    }


def calcular_metricas_por_classe(y_true: np.ndarray,
                               y_pred: np.ndarray,
                               class_names: Optional[List[str]] = None) -> Dict:
    """
    Calcula métricas detalhadas por classe.
    
    Args:
        y_true: Labels verdadeiros
        y_pred: Labels preditos
        class_names: Nomes das classes
    
    Returns:
        metricas: Dicionário com métricas por classe
    """
    from sklearn.metrics import precision_recall_fscore_support
    
    # Calcula métricas
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None
    )
    
    # Organiza em dicionário
    metricas = {}
    for i in range(len(precision)):
        nome = class_names[i] if class_names else f"Classe {i}"
        metricas[nome] = {
            'precision': float(precision[i]),
            'recall': float(recall[i]),
            'f1-score': float(f1[i]),
            'support': int(support[i])
        }
    
    # Adiciona médias
    metricas['média'] = {
        'precision': float(np.mean(precision)),
        'recall': float(np.mean(recall)),
        'f1-score': float(np.mean(f1)),
        'support': int(np.sum(support))
    }
    
    return metricas


def visualizar_filtros_cnn(model: keras.Model,
                         layer_name: str = None,
                         save_path: Optional[str] = None) -> None:
    """
    Visualiza os filtros de uma camada convolucional.
    
    Args:
        model: Modelo CNN
        layer_name: Nome da camada (usa primeira conv se None)
        save_path: Caminho para salvar
    """
    # Encontra camadas convolucionais
    conv_layers = [layer for layer in model.layers 
                   if isinstance(layer, keras.layers.Conv2D)]
    
    if not conv_layers:
        print("Nenhuma camada convolucional encontrada!")
        return
    
    # Seleciona camada
    if layer_name:
        layer = model.get_layer(layer_name)
    else:
        layer = conv_layers[0]
    
    # Pega os pesos (filtros)
    filters, biases = layer.get_weights()
    
    # Normaliza filtros para visualização
    f_min, f_max = filters.min(), filters.max()
    filters = (filters - f_min) / (f_max - f_min)
    
    # Plota filtros
    n_filters = min(filters.shape[3], 64)  # Máximo 64 filtros
    cols = 8
    rows = n_filters // cols + (1 if n_filters % cols else 0)
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols*2, rows*2))
    axes = axes.ravel() if n_filters > 1 else [axes]
    
    for i in range(n_filters):
        # Pega o filtro
        f = filters[:, :, :, i]
        
        # Se tem múltiplos canais, pega apenas o primeiro
        if f.shape[2] > 1:
            f = f[:, :, 0]
        else:
            f = f[:, :, 0]
        
        axes[i].imshow(f, cmap='gray')
        axes[i].set_title(f'Filtro {i}', fontsize=8)
        axes[i].axis('off')
    
    # Remove subplots extras
    for i in range(n_filters, len(axes)):
        fig.delaxes(axes[i])
    
    plt.suptitle(f'Filtros da camada: {layer.name}', fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Filtros salvos em: {save_path}")
    
    plt.show()