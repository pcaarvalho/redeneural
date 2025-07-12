"""
Script Principal - Rede Neural para Classificação de Imagens
Este script demonstra o fluxo completo de treino e avaliação de uma rede neural.
"""

import os
import sys
import argparse
import tensorflow as tf
from datetime import datetime

# Adiciona o diretório src ao path para importar nossos módulos
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Importa nossos módulos
from data_loader import (
    carregar_mnist, carregar_cifar10, carregar_fashion_mnist,
    preprocessar_dados, dividir_validacao
)
from model import (
    criar_modelo_simples, criar_modelo_cnn,
    compilar_modelo, resumo_modelo, salvar_modelo
)
from train import (
    criar_callbacks, treinar_modelo,
    plotar_historico, salvar_historico
)
from evaluate import (
    avaliar_modelo, fazer_predicoes,
    matriz_confusao, relatorio_classificacao,
    visualizar_predicoes, analisar_erros
)


def main(args):
    """
    Função principal que executa todo o pipeline.
    """
    print("\n" + "="*60)
    print("REDE NEURAL PARA CLASSIFICAÇÃO DE IMAGENS")
    print("="*60)
    print(f"Dataset: {args.dataset}")
    print(f"Modelo: {args.model_type}")
    print(f"Épocas: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print("="*60 + "\n")
    
    # 1. CARREGAMENTO DE DADOS
    print("\n[ETAPA 1] Carregando dados...")
    
    if args.dataset == 'mnist':
        (x_train, y_train), (x_test, y_test) = carregar_mnist()
        num_classes = 10
        input_shape = (28, 28)
        class_names = [str(i) for i in range(10)]
    
    elif args.dataset == 'fashion_mnist':
        (x_train, y_train), (x_test, y_test) = carregar_fashion_mnist()
        num_classes = 10
        input_shape = (28, 28)
        class_names = ['Camiseta', 'Calça', 'Suéter', 'Vestido', 'Casaco',
                      'Sandália', 'Camisa', 'Tênis', 'Bolsa', 'Bota']
    
    elif args.dataset == 'cifar10':
        (x_train, y_train), (x_test, y_test) = carregar_cifar10()
        num_classes = 10
        input_shape = (32, 32, 3)
        class_names = ['avião', 'automóvel', 'pássaro', 'gato', 'cervo', 
                      'cachorro', 'sapo', 'cavalo', 'navio', 'caminhão']
    
    else:
        raise ValueError(f"Dataset não suportado: {args.dataset}")
    
    
    # 2. PREPROCESSAMENTO
    print("\n[ETAPA 2] Preprocessando dados...")
    
    # Normaliza os dados
    expandir_dims = args.model_type == 'cnn' and len(input_shape) == 2
    x_train, x_test = preprocessar_dados(x_train, x_test, 
                                        normalizar=True,
                                        expandir_dims=expandir_dims)
    
    # Divide treino em treino/validação
    x_train, y_train, x_val, y_val = dividir_validacao(x_train, y_train, 
                                                       val_split=0.1)
    
    # Atualiza input_shape se expandimos dimensões
    if expandir_dims:
        input_shape = input_shape + (1,)
    
    
    # 3. CRIAÇÃO DO MODELO
    print("\n[ETAPA 3] Criando modelo...")
    
    if args.model_type == 'simples':
        model = criar_modelo_simples(input_shape, num_classes)
    elif args.model_type == 'cnn':
        # Para CNN, precisa ter 3 dimensões
        if len(input_shape) == 2:
            input_shape = input_shape + (1,)
        model = criar_modelo_cnn(input_shape, num_classes)
    else:
        raise ValueError(f"Tipo de modelo não suportado: {args.model_type}")
    
    # Compila o modelo
    model = compilar_modelo(model, learning_rate=args.learning_rate)
    
    # Mostra resumo do modelo
    resumo_modelo(model)
    
    
    # 4. TREINAMENTO
    print("\n[ETAPA 4] Treinando modelo...")
    
    # Cria callbacks
    model_dir = os.path.join('models', f'{args.dataset}_{args.model_type}')
    callbacks = criar_callbacks(model_dir, patience=5)
    
    # Treina o modelo
    history = treinar_modelo(
        model,
        x_train, y_train,
        x_val, y_val,
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks
    )
    
    # Salva histórico e gráficos
    results_dir = os.path.join('results', f'{args.dataset}_{args.model_type}')
    os.makedirs(results_dir, exist_ok=True)
    
    # Salva histórico em JSON
    hist_path = os.path.join(results_dir, 'history.json')
    salvar_historico(history, hist_path)
    
    # Plota e salva gráficos
    plot_path = os.path.join(results_dir, 'training_curves.png')
    plotar_historico(history, save_path=plot_path)
    
    
    # 5. AVALIAÇÃO
    print("\n[ETAPA 5] Avaliando modelo...")
    
    # Avalia no conjunto de teste
    test_loss, test_accuracy = avaliar_modelo(model, x_test, y_test)
    
    # Faz predições
    y_pred_probs, y_pred = fazer_predicoes(model, x_test)
    
    # Gera relatório de classificação
    relatorio = relatorio_classificacao(y_test, y_pred, class_names)
    
    # Salva relatório
    with open(os.path.join(results_dir, 'classification_report.txt'), 'w') as f:
        f.write(relatorio)
    
    # Matriz de confusão
    cm_path = os.path.join(results_dir, 'confusion_matrix.png')
    matriz_confusao(y_test, y_pred, class_names, save_path=cm_path)
    
    # Visualiza algumas predições
    pred_path = os.path.join(results_dir, 'sample_predictions.png')
    visualizar_predicoes(model, x_test, y_test, y_pred, 
                        num_amostras=16, class_names=class_names,
                        save_path=pred_path)
    
    # Análise de erros
    erros = analisar_erros(x_test, y_test, y_pred, class_names)
    
    
    # 6. SALVAMENTO DO MODELO
    print("\n[ETAPA 6] Salvando modelo final...")
    
    # Salva modelo completo
    model_path = os.path.join(model_dir, 'modelo_final.h5')
    salvar_modelo(model, model_path)
    
    # Salva também em formato SavedModel
    savedmodel_path = os.path.join(model_dir, 'saved_model')
    salvar_modelo(model, savedmodel_path)
    
    
    # RESUMO FINAL
    print("\n" + "="*60)
    print("TREINAMENTO CONCLUÍDO COM SUCESSO!")
    print("="*60)
    print(f"✓ Acurácia no teste: {test_accuracy*100:.2f}%")
    print(f"✓ Modelo salvo em: {model_path}")
    print(f"✓ Resultados salvos em: {results_dir}")
    print(f"✓ Logs do TensorBoard em: logs/")
    print("\nPara visualizar no TensorBoard:")
    print("  tensorboard --logdir logs")
    print("="*60 + "\n")


def criar_argumentos():
    """
    Cria parser de argumentos para o script.
    """
    parser = argparse.ArgumentParser(
        description='Script principal para treinar redes neurais'
    )
    
    # Dataset
    parser.add_argument(
        '--dataset',
        type=str,
        default='mnist',
        choices=['mnist', 'fashion_mnist', 'cifar10'],
        help='Dataset para usar (padrão: mnist)'
    )
    
    # Tipo de modelo
    parser.add_argument(
        '--model_type',
        type=str,
        default='simples',
        choices=['simples', 'cnn'],
        help='Tipo de modelo: simples (MLP) ou cnn (padrão: simples)'
    )
    
    # Hiperparâmetros
    parser.add_argument(
        '--epochs',
        type=int,
        default=20,
        help='Número de épocas (padrão: 20)'
    )
    
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Tamanho do batch (padrão: 32)'
    )
    
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.001,
        help='Taxa de aprendizado (padrão: 0.001)'
    )
    
    return parser


if __name__ == '__main__':
    # Configura TensorFlow para não mostrar tantos warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    
    # Verifica GPU disponível
    print(f"\nGPUs disponíveis: {len(tf.config.list_physical_devices('GPU'))}")
    if len(tf.config.list_physical_devices('GPU')) > 0:
        print("✓ GPU detectada! O treinamento será acelerado.")
    else:
        print("✗ Nenhuma GPU detectada. Usando CPU.")
    
    # Parse argumentos
    parser = criar_argumentos()
    args = parser.parse_args()
    
    # Executa programa principal
    try:
        main(args)
    except KeyboardInterrupt:
        print("\n\nTreinamento interrompido pelo usuário.")
    except Exception as e:
        print(f"\nErro durante execução: {e}")
        raise