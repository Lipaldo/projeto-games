# projeto-games
# 📘 Projeto de Regressão com Rede Neural Artificial (TensorFlow.js)

Este projeto demonstra a construção de um sistema completo de **Regressão Linear Múltipla** utilizando **TensorFlow.js**, rodando **100% no navegador**, sem necessidade de backend ou servidores Python. O objetivo principal foi aprender e aplicar conceitos fundamentais de Machine Learning, criando um pipeline completo de carregamento de dados, pré-processamento, modelagem, treinamento e visualização.

---

🚀 Tecnologias Utilizadas

✅ **TensorFlow.js**

* Treinamento da rede neural diretamente no navegador.
* Construção de um modelo **Sequential** com camadas densas.
* Normalização dos dados e cálculo de métricas (MAE, MSE, R²).

### ✅ **Chart.js**

* Exibição de **gráficos de dispersão** para os dados reais.
* Construção da **linha de regressão** baseada nas previsões do modelo.

### ✅ **JavaScript (ES6+)**

* Manipulação dos dados carregados do CSV.
* Implementação manual de normalização Z-score.
* Construção de matrizes X e y para regressão múltipla.
* Cálculo estatístico das métricas.

### ✅ **HTML5 + CSS3**

* Estrutura da página.
* Layout responsivo.
* Cartões e seções organizadas para visualização dos gráficos e métricas.

---

## 🎯 Objetivo do Projeto

Criar um ambiente intuitivo e totalmente visual para testar a regressão linear com redes neurais, permitindo:

* Carregamento automático de um arquivo CSV com dados reais.
* Seleção de **múltiplas features** (regressão múltipla).
* Escolha da variável alvo (target).
* Visualização em tempo real dos resultados.

---

## 🧠 O que foi aprendido

Durante o desenvolvimento deste projeto, foram explorados e compreendidos diversos conceitos importantes:

### ✅ **1. Como treinar redes neurais 100% no navegador**

Sem instalar nada, sem Python, sem back-end.
TensorFlow.js permitiu criar um ambiente totalmente interativo.

### ✅ **2. Como manipular datasets CSV com JavaScript**

* Leitura e parse de arquivos.
* Conversão manual dos dados.
* Tratamento e preparação das features.

### ✅ **3. Normalização Z-score**

Implementação matemática de:

```
z = (x - média) / desvio padrão
```

Fundamental para modelos de regressão neural funcionarem bem.

### ✅ **4. Construção de matrizes para Regressão Múltipla**

Transformando o dataset em:

* **X (n × m)** → múltiplas features
* **y (n × 1)** → variável alvo

### ✅ **5. Visualização com Chart.js**

Criação de gráficos profissionais:

* Dispersão dos dados reais
* Linha de regressão das previsões
* Identificação clara das variáveis usadas

### ✅ **6. Cálculo de métricas estatísticas**

* **MAE (Mean Absolute Error)**
* **MSE (Mean Squared Error)**
* **R² (Coeficiente de Determinação)**

Com isso, foi possível medir o desempenho do modelo neural de forma objetiva.

---

## 📊 Resultado Final

O projeto final permite:

✅ Carregar automaticamente o arquivo **games.csv**
✅ Selecionar qualquer número de features para regressão múltipla
✅ Treinar uma rede neural **totalmente no navegador**
✅ Gerar previsões e exibir a **linha de regressão**
✅ Visualizar métricas reais de desempenho
✅ Interface limpa, organizada e intuitiva

Além disso, o sistema pode ser expandido facilmente para:

* Regressão não linear
* Exportação do modelo treinado
* Dashboard completo de análise de dados
* Novos gráficos (correlação, histograma etc.)

---

## 🏁 Conclusão

Este projeto demonstra como é possível aplicar machine learning avançado usando apenas **JavaScript**, sem bibliotecas pesadas ou ambientes complexos. O resultado é uma ferramenta educacional poderosa, visual e acessível, ideal tanto para estudos quanto para demonstrações.