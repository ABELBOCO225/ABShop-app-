const tf = require('@tensorflow/tfjs');

const CONFIG = {
  embedDim: 16,
  ffDim: 64,
  maxLen: 20
};

function buildModel(vocabSize) {
  const { embedDim, ffDim, maxLen } = CONFIG;

  const model = tf.sequential({ name: 'abshop' });

  model.add(tf.layers.embedding({
    inputDim: vocabSize, outputDim: embedDim,
    inputLength: maxLen, name: 'embed'
  }));

  model.add(tf.layers.flatten({ name: 'flat' }));

  model.add(tf.layers.dense({
    units: ffDim, activation: 'relu', name: 'h1'
  }));

  model.add(tf.layers.dense({
    units: maxLen * vocabSize, activation: 'linear', name: 'out'
  }));

  model.add(tf.layers.reshape({
    targetShape: [maxLen, vocabSize], name: 'reshape'
  }));

  // softmaxCrossEntropy s'attend à des logits bruts
  model.compile({
    optimizer: tf.train.adam(0.01),
    loss: (yTrue, yPred) => {
      return tf.losses.softmaxCrossEntropy(yTrue, yPred);
    },
    metrics: ['accuracy']
  });

  return model;
}

function saveWeights(model, fs, dir) {
  const data = model.getWeights().map(w => ({
    shape: w.shape,
    values: Array.from(w.dataSync())
  }));
  fs.writeFileSync(`${dir}/weights.json`, JSON.stringify(data));
  console.log(`✓ Poids sauvegardés (${data.length} tenseurs)`);
}

function loadWeights(model, fs, dir) {
  const data    = JSON.parse(fs.readFileSync(`${dir}/weights.json`, 'utf8'));
  const weights = data.map(d => tf.tensor(d.values, d.shape));
  model.setWeights(weights);
  console.log(`✓ Poids chargés (${data.length} tenseurs)`);
}

module.exports = { buildModel, saveWeights, loadWeights, CONFIG };
