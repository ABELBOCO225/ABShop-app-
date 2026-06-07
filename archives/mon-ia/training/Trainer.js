const tf = require('@tensorflow/tfjs');

class Trainer {
  constructor(model, learningRate = 0.005) {
    this.model     = model;
    this.optimizer = tf.train.adam(learningRate);
    this.losses    = [];
  }

  trainStep(X, Y) {
    let lossVal;

    this.optimizer.minimize(() => {
      const logits = this.model.forward(X);
      const loss   = tf.losses.softmaxCrossEntropy(Y, logits);
      lossVal      = loss.dataSync()[0];
      return loss;
    });

    return lossVal;
  }

  async train(dataset, epochs = 200, batchSize = 8, onProgress = null) {
    console.log(`Démarrage entraînement — ${epochs} epochs\n`);

    for (let epoch = 0; epoch < epochs; epoch++) {
      const batches   = dataset.getAllBatches(batchSize);
      let   totalLoss = 0;

      for (const { X, Y } of batches) {
        const loss = this.trainStep(X, Y);
        totalLoss += loss;
        X.dispose();
        Y.dispose();
      }

      const avgLoss = (totalLoss / batches.length).toFixed(4);
      this.losses.push(avgLoss);

      if ((epoch + 1) % 20 === 0)
        console.log(`Epoch ${epoch + 1}/${epochs} — Loss : ${avgLoss}`);

      if (onProgress) onProgress({ epoch: epoch + 1, epochs, loss: avgLoss });
      if ((epoch + 1) % 10 === 0) await tf.nextFrame();
    }

    console.log(`\nEntraînement terminé !`);
    console.log(`Loss finale : ${this.losses[this.losses.length - 1]}`);
    return this.losses;
  }
}

module.exports = Trainer;
