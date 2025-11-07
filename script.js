// script.js
// Regressão linear 100% no navegador usando TensorFlow.js + Chart.js
// Pressupostos/heurísticas:
// - Arquivo esperado: games.csv (na mesma pasta do projeto)
// - O CSV deve conter cabeçalho. Procuramos colunas numéricas.
// - Se existirem exatamente 2 colunas numéricas usamos a 1ª como X e a 2ª como Y.
// - Se existirem >2, usamos a primeira numérica como X e a última numérica como Y.

const loadBtn = document.getElementById('loadBtn');
const statusEl = document.getElementById('status');
const metricsEl = document.getElementById('metrics');
const ctx = document.getElementById('scatterPlot').getContext('2d');
let chart; // instancia do Chart.js

loadBtn.addEventListener('click', () => {
  resetUI();
  loadCSV('games.csv');
});

function resetUI() {
  statusEl.textContent = '';
  metricsEl.innerHTML = '';
  if (chart) chart.destroy();
}

async function loadCSV(path) {
  status('Carregando ' + path + ' ...');
  try {
    const res = await fetch(path);
    if (!res.ok) throw new Error('Não foi possível buscar ' + path + ' — verifique se o arquivo existe e se você está servindo o projeto por um servidor (ex.: Live Server no VSCode).');
    const text = await res.text();
    const { headers, rows } = parseCSV(text);
    status('CSV carregado — detectando colunas numéricas...');

    const numericCols = detectNumericColumns(headers, rows);
    if (numericCols.length < 2) {
      status('Erro: foram detectadas menos de 2 colunas numéricas. Verifique o CSV.');
      return;
    }

    const xCol = numericCols[0];
    const yCol = numericCols[numericCols.length - 1];
    status(`Usando X = "${xCol}" e Y = "${yCol}"`);

    const xVals = rows.map(r => parseFloat(r[xCol]));
    const yVals = rows.map(r => parseFloat(r[yCol]));

    // Remover NaNs
    const data = xVals.map((x, i) => ({ x, y: yVals[i] })).filter(p => isFinite(p.x) && isFinite(p.y));
    if (data.length === 0) {
      status('Erro: sem pares numéricos após limpeza.');
      return;
    }

    await runRegression(data, xCol, yCol);
  } catch (err) {
    status(err.message || err);
    console.error(err);
  }
}

function status(txt) {
  statusEl.textContent = txt;
}

function parseCSV(text) {
  const lines = text.split(/\r?\n/).filter(l => l.trim() !== '');
  const headers = lines[0].split(',').map(h => h.trim());
  const rows = lines.slice(1).map(line => {
    const parts = line.split(',');
    const obj = {};
    headers.forEach((h, i) => (obj[h] = parts[i] ? parts[i].trim() : ''));
    return obj;
  });
  return { headers, rows };
}

function detectNumericColumns(headers, rows) {
  const numeric = [];
  headers.forEach(h => {
    let countNum = 0;
    for (let i = 0; i < rows.length && i < 30; i++) {
      const v = rows[i][h];
      if (v === undefined || v === '') continue;
      if (!isNaN(parseFloat(v))) countNum++;
    }
    // se a maioria (metade) das primeiras linhas for numérica consideramos coluna numérica
    if (countNum >= Math.min(10, Math.floor(rows.length / 2))) numeric.push(h);
  });
  return numeric;
}

async function runRegression(data, xLabel, yLabel) {
  status('Preparando tensores e normalizando...');

  const xs = tf.tensor1d(data.map(p => p.x));
  const ys = tf.tensor1d(data.map(p => p.y));

  // Normalização simples
  const xMin = xs.min();
  const xMax = xs.max();
  const yMin = ys.min();
  const yMax = ys.max();

  const xsNorm = xs.sub(xMin).div(xMax.sub(xMin));
  const ysNorm = ys.sub(yMin).div(yMax.sub(yMin));

  // Modelo simples: camada densa com 1 saída (regressão linear)
  const model = tf.sequential();
  model.add(tf.layers.dense({ inputShape: [1], units: 1 }));
  model.compile({ optimizer: tf.train.adam(0.1), loss: 'meanSquaredError', metrics: ['mse'] });

  status('Treinando modelo no navegador...');
  const epochs = 250;
  const history = [];
  await model.fit(xsNorm.reshape([xsNorm.shape[0], 1]), ysNorm.reshape([ysNorm.shape[0], 1]), {
    epochs,
    shuffle: true,
    callbacks: {
      onEpochEnd: (epoch, logs) => {
        if (epoch % 25 === 0) status(`Treinando... epoch ${epoch}/${epochs} — loss: ${logs.loss.toFixed(6)}`);
        history.push({ epoch, loss: logs.loss });
      }
    }
  });

  status('Treinamento concluído — gerando previsões...');

  // Predições em escopo original
  const predsNorm = model.predict(xsNorm.reshape([xsNorm.shape[0], 1]));
  const preds = predsNorm.mul(yMax.sub(yMin)).add(yMin).dataSync();
  const xArr = xs.dataSync();
  const yArr = ys.dataSync();

  // Métricas: MSE, MAE, R^2
  const metrics = computeMetrics(yArr, preds);

  showMetrics(metrics, xLabel, yLabel);
  drawChart(xArr, yArr, preds, xLabel, yLabel);

  // limpeza
  xs.dispose(); ys.dispose(); xsNorm.dispose(); ysNorm.dispose(); predsNorm.dispose();
  status('Tudo pronto — veja o gráfico e métricas.');
}

function computeMetrics(yTrue, yPred) {
  const n = yTrue.length;
  let se = 0; // sum squared errors
  let ae = 0; // sum absolute errors
  let mean = 0;
  for (let i = 0; i < n; i++) mean += yTrue[i];
  mean /= n;
  let ssTot = 0;
  for (let i = 0; i < n; i++) {
    const diff = yTrue[i] - yPred[i];
    se += diff * diff;
    ae += Math.abs(diff);
    const diffMean = yTrue[i] - mean;
    ssTot += diffMean * diffMean;
  }
  const mse = se / n;
  const mae = ae / n;
  const r2 = 1 - (se / ssTot);
  return { mse, mae, r2 };
}

function showMetrics(metrics, xLabel, yLabel) {
  metricsEl.innerHTML = `
    <strong>Variáveis:</strong> X = ${escapeHtml(xLabel)}, Y = ${escapeHtml(yLabel)}<br>
    <strong>MSE:</strong> ${metrics.mse.toFixed(4)}<br>
    <strong>MAE:</strong> ${metrics.mae.toFixed(4)}<br>
    <strong>R²:</strong> ${metrics.r2.toFixed(4)}
  `;
}

function drawChart(xArr, yArr, preds, xLabel, yLabel) {
  // Ordenar por X para desenhar linha
  const pts = xArr.map((x, i) => ({ x, y: yArr[i], p: preds[i] }));
  pts.sort((a, b) => a.x - b.x);

  const scatterData = pts.map(p => ({ x: p.x, y: p.y }));
  const lineData = pts.map(p => ({ x: p.x, y: p.p }));

  const data = {
    datasets: [
      {
        label: 'Dados (observados)',
        data: scatterData,
        showLine: false,
        pointRadius: 4,
        tension: 0.1
      },
      {
        label: 'Regressão (predição)',
        data: lineData,
        type: 'line',
        fill: false,
        borderWidth: 2,
        borderDash: [6, 4],
        pointRadius: 0
      }
    ]
  };

  const options = {
    responsive: true,
    plugins: {
      legend: { position: 'top' }
    },
    scales: {
      x: {
        type: 'linear',
        title: { display: true, text: xLabel }
      },
      y: {
        title: { display: true, text: yLabel }
      }
    }
  };

  chart = new Chart(ctx, { type: 'scatter', data, options });
}

function escapeHtml(s) {
  return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

