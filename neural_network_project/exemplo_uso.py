#!/usr/bin/env python3
"""
Exemplo de como carregar e usar o modelo treinado
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

# 1. CARREGAR O MODELO TREINADO
print("Carregando modelo...")
model = tf.keras.models.load_model('models/mnist_simples/best_model.h5')

print("Modelo carregado com sucesso!")
print("\nResumo do modelo:")
model.summary()

# 2. CARREGAR DADOS DE TESTE
print("\nCarregando dados de teste...")
(_, _), (x_test, y_test) = mnist.load_data()

# Normalizar os dados (mesmo preprocessamento do treinamento)
x_test = x_test.astype('float32') / 255.0

print(f"Dados de teste: {x_test.shape}")

# 3. FAZER PREDIÇÕES
print("\nFazendo predições...")

# Predição em uma amostra
sample_idx = 0
sample_image = x_test[sample_idx]
sample_label = y_test[sample_idx]

# Expandir dimensões para o modelo (batch dimension)
sample_input = np.expand_dims(sample_image, axis=0)

# Fazer predição
prediction = model.predict(sample_input, verbose=0)
predicted_class = np.argmax(prediction[0])
confidence = np.max(prediction[0])

print(f"Imagem real: {sample_label}")
print(f"Predição: {predicted_class}")
print(f"Confiança: {confidence:.2%}")

# 4. AVALIAR MODELO NO CONJUNTO DE TESTE
print("\nAvaliando modelo...")
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"Acurácia no teste: {test_accuracy:.2%}")
print(f"Loss no teste: {test_loss:.4f}")

# 5. FAZER PREDIÇÕES EM LOTE
print("\nPredições em lote (primeiras 10 imagens):")
batch_predictions = model.predict(x_test[:10], verbose=0)
predicted_classes = np.argmax(batch_predictions, axis=1)

for i in range(10):
    real = y_test[i]
    pred = predicted_classes[i]
    conf = np.max(batch_predictions[i])
    status = "✓" if real == pred else "✗"
    print(f"Imagem {i}: Real={real}, Pred={pred}, Conf={conf:.2%} {status}")

print("\n" + "="*50)
print("EXEMPLO DE USO CONCLUÍDO!")
print("="*50)