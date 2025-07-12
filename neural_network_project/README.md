# Projeto de Rede Neural para Classificação de Imagens

Um projeto completo e bem estruturado para criar, treinar e avaliar redes neurais usando TensorFlow/Keras. Ideal para iniciantes e intermediários em Python que querem aprender Deep Learning na prática.

## 📋 Índice

- [Características](#características)
- [Estrutura do Projeto](#estrutura-do-projeto)
- [Instalação](#instalação)
- [Uso Rápido](#uso-rápido)
- [Datasets Disponíveis](#datasets-disponíveis)
- [Arquiteturas de Modelo](#arquiteturas-de-modelo)
- [Exemplos de Uso](#exemplos-de-uso)
- [Personalização](#personalização)
- [Boas Práticas](#boas-práticas)
- [Troubleshooting](#troubleshooting)

## 🚀 Características

- **Modular e Extensível**: Código organizado em módulos reutilizáveis
- **Múltiplos Datasets**: Suporte para MNIST, Fashion-MNIST, CIFAR-10 e datasets personalizados
- **Duas Arquiteturas**: Rede Neural Simples (MLP) e Convolucional (CNN)
- **Visualizações Completas**: Gráficos de treino, matriz de confusão, predições
- **Callbacks Inteligentes**: Early stopping, checkpoints, TensorBoard
- **Notebook Interativo**: Jupyter notebook com exemplos passo a passo
- **Código Comentado**: Explicações detalhadas em português

## 📁 Estrutura do Projeto

```
neural_network_project/
│
├── src/                    # Código fonte modular
│   ├── __init__.py
│   ├── data_loader.py     # Carregamento e preprocessamento de dados
│   ├── model.py           # Arquiteturas de rede neural
│   ├── train.py           # Funções de treinamento
│   └── evaluate.py        # Avaliação e métricas
│
├── notebooks/             # Jupyter notebooks
│   └── exemplo_completo.ipynb
│
├── models/                # Modelos salvos
├── data/                  # Datasets (criado automaticamente)
├── results/               # Resultados e gráficos
├── logs/                  # Logs do TensorBoard
│
├── main.py               # Script principal
├── requirements.txt      # Dependências
└── README.md            # Este arquivo
```

## 🔧 Instalação

### 1. Clone o repositório ou copie os arquivos

```bash
cd neural_network_project
```

### 2. Crie um ambiente virtual (recomendado)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### 3. Instale as dependências

```bash
pip install -r requirements.txt
```

### 4. Verifique a instalação

```bash
python -c "import tensorflow as tf; print(f'TensorFlow {tf.__version__} instalado com sucesso!')"
```

## 🎯 Uso Rápido

### Comando Básico

```bash
python main.py
```

### Com Parâmetros Personalizados

```bash
# Treinar CNN no Fashion-MNIST por 30 épocas
python main.py --dataset fashion_mnist --model_type cnn --epochs 30

# Treinar modelo simples no CIFAR-10 com batch size maior
python main.py --dataset cifar10 --model_type simples --batch_size 64

# Com taxa de aprendizado customizada
python main.py --learning_rate 0.0001 --epochs 50
```

### Parâmetros Disponíveis

- `--dataset`: Escolha entre 'mnist', 'fashion_mnist', 'cifar10' (padrão: mnist)
- `--model_type`: 'simples' (MLP) ou 'cnn' (padrão: simples)
- `--epochs`: Número de épocas de treino (padrão: 20)
- `--batch_size`: Tamanho do batch (padrão: 32)
- `--learning_rate`: Taxa de aprendizado (padrão: 0.001)

## 📊 Datasets Disponíveis

### 1. MNIST
- **Descrição**: Dígitos manuscritos (0-9)
- **Tamanho**: 70,000 imagens (60k treino, 10k teste)
- **Formato**: 28x28 pixels, escala de cinza
- **Classes**: 10 (dígitos 0-9)

### 2. Fashion-MNIST
- **Descrição**: Imagens de roupas e acessórios
- **Tamanho**: 70,000 imagens
- **Formato**: 28x28 pixels, escala de cinza
- **Classes**: Camiseta, Calça, Suéter, Vestido, Casaco, Sandália, Camisa, Tênis, Bolsa, Bota

### 3. CIFAR-10
- **Descrição**: Imagens coloridas de objetos
- **Tamanho**: 60,000 imagens (50k treino, 10k teste)
- **Formato**: 32x32 pixels, RGB
- **Classes**: Avião, Automóvel, Pássaro, Gato, Cervo, Cachorro, Sapo, Cavalo, Navio, Caminhão

### Como Baixar Outros Datasets

```python
# Exemplo com TensorFlow Datasets
import tensorflow_datasets as tfds

# Lista datasets disponíveis
tfds.list_builders()

# Carrega um dataset específico
dataset, info = tfds.load('oxford_flowers102', with_info=True)
```

## 🏗️ Arquiteturas de Modelo

### Modelo Simples (MLP)
```
Input (28x28) → Flatten → Dense(128) → Dropout → Dense(64) → Dropout → Dense(10)
```

### Modelo CNN
```
Input → Conv2D(32) → MaxPool → Conv2D(64) → MaxPool → Conv2D(64) → Flatten → Dense(64) → Dense(10)
```

## 💡 Exemplos de Uso

### 1. Treinamento Básico

```python
from src.data_loader import carregar_mnist, preprocessar_dados
from src.model import criar_modelo_simples, compilar_modelo
from src.train import treinar_modelo

# Carrega dados
(x_train, y_train), (x_test, y_test) = carregar_mnist()

# Preprocessa
x_train, x_test = preprocessar_dados(x_train, x_test)

# Cria e compila modelo
modelo = criar_modelo_simples((28, 28), 10)
modelo = compilar_modelo(modelo)

# Treina
history = treinar_modelo(modelo, x_train, y_train, x_test, y_test)
```

### 2. Usando o Notebook Jupyter

```bash
jupyter notebook notebooks/exemplo_completo.ipynb
```

O notebook contém:
- Exploração visual dos dados
- Experimentação com diferentes arquiteturas
- Análise detalhada dos resultados
- Exemplos de personalização

### 3. Visualizando com TensorBoard

```bash
tensorboard --logdir logs
```

Acesse http://localhost:6006 no navegador.

## 🎨 Personalização

### Adicionar Novo Dataset

```python
def carregar_meu_dataset():
    """Carrega dataset personalizado"""
    # Seu código aqui
    x_train = ...  # numpy array (n_samples, height, width)
    y_train = ...  # numpy array (n_samples,)
    return (x_train, y_train), (x_test, y_test)
```

### Criar Nova Arquitetura

```python
def criar_minha_rede(input_shape, num_classes):
    """Cria arquitetura personalizada"""
    model = keras.Sequential([
        # Suas camadas aqui
    ])
    return model
```

### Adaptar para Outros Problemas

#### Regressão
```python
# Mude a última camada e loss
model.add(Dense(1))  # Sem ativação
model.compile(loss='mse', metrics=['mae'])
```

#### Multi-label
```python
# Múltiplas saídas
model.add(Dense(num_classes, activation='sigmoid'))
model.compile(loss='binary_crossentropy')
```

## 📚 Boas Práticas

### 1. **Sempre normalize os dados**
```python
x_train = x_train / 255.0  # Pixels de [0,255] para [0,1]
```

### 2. **Use validação**
```python
# Separe 10-20% dos dados de treino para validação
x_train, x_val = split_data(x_train, val_split=0.2)
```

### 3. **Monitore overfitting**
- Compare loss de treino vs validação
- Use dropout e regularização
- Implemente early stopping

### 4. **Experimente com hiperparâmetros**
- Learning rate: teste 0.1, 0.01, 0.001, 0.0001
- Batch size: 16, 32, 64, 128
- Arquitetura: número de camadas e neurônios

### 5. **Use callbacks**
```python
callbacks = [
    EarlyStopping(patience=5),
    ModelCheckpoint('best_model.h5'),
    ReduceLROnPlateau(factor=0.5)
]
```

## 🔍 Troubleshooting

### Erro: "No module named 'tensorflow'"
```bash
pip install --upgrade tensorflow
```

### GPU não detectada
```bash
# Verifica GPUs disponíveis
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# Instala versão GPU (NVIDIA)
pip install tensorflow-gpu
```

### Memória insuficiente
- Reduza o batch_size
- Use geradores de dados
- Implemente treinamento em lotes

### Loss não diminui
- Verifique se os dados estão normalizados
- Experimente learning rates menores
- Adicione mais camadas ou neurônios
- Verifique se há desbalanceamento de classes

## 🚀 Próximos Passos

1. **Transfer Learning**: Use modelos pré-treinados (VGG, ResNet, etc.)
2. **Data Augmentation**: Aumente seus dados com transformações
3. **Ensemble**: Combine múltiplos modelos
4. **Hyperparameter Tuning**: Use Keras Tuner ou Optuna
5. **Deploy**: Exporte para TensorFlow Lite ou TensorFlow.js

## 📝 Melhorias Sugeridas

- [ ] Adicionar suporte para mais datasets
- [ ] Implementar mais arquiteturas (ResNet, VGG, etc.)
- [ ] Criar API REST para servir o modelo
- [ ] Adicionar interface gráfica
- [ ] Implementar técnicas avançadas (attention, batch norm, etc.)
- [ ] Suporte para treinamento distribuído
- [ ] Exportação para mobile (TFLite)

## 🤝 Contribuições

Sinta-se à vontade para:
- Reportar bugs
- Sugerir melhorias
- Adicionar novos recursos
- Melhorar a documentação

---

**Dica Final**: Comece com o notebook Jupyter para entender o fluxo completo, depois use o script principal para experimentos mais sérios. Não hesite em modificar o código - a melhor forma de aprender é experimentando!

Bom aprendizado! 🎓