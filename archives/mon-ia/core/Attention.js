// Attention.js — mécanisme Self-Attention (coeur du Transformer)

const tf = require('@tensorflow/tfjs');

class SelfAttention {
  constructor(embedDim, numHeads = 2) {
    this.embedDim  = embedDim;
    this.numHeads  = numHeads;
    this.headDim   = Math.floor(embedDim / numHeads);

    // Projections Q, K, V
    this.Wq = tf.layers.dense({ units: embedDim, useBias: false, name: 'Wq' });
    this.Wk = tf.layers.dense({ units: embedDim, useBias: false, name: 'Wk' });
    this.Wv = tf.layers.dense({ units: embedDim, useBias: false, name: 'Wv' });

    // Projection de sortie
    this.Wo = tf.layers.dense({ units: embedDim, useBias: false, name: 'Wo' });
  }

  // x : [batch, seqLen, embedDim]
  call(x) {
    return tf.tidy(() => {
      const [batch, seqLen] = [x.shape[0], x.shape[1]];

      // Calcul Q, K, V
      const Q = this.Wq.apply(x);
      const K = this.Wk.apply(x);
      const V = this.Wv.apply(x);

      // Scores d'attention : Q * K^T / sqrt(headDim)
      const scale  = Math.sqrt(this.headDim);
      const scores = tf.div(
        tf.matMul(Q, K, false, true),
        tf.scalar(scale)
      );

      // Masque causal — le token i ne voit pas j > i
      const mask = tf.linalg.bandPart(
        tf.ones([seqLen, seqLen]), -1, 0
      );
      const maskedScores = tf.add(
        scores,
        tf.mul(tf.sub(tf.scalar(1), mask), tf.scalar(-1e9))
      );

      // Softmax → poids d'attention
      const weights = tf.softmax(maskedScores, -1);

      // Contexte : weights * V
      const context = tf.matMul(weights, V);

      // Projection finale
      return this.Wo.apply(context);
    });
  }
}

module.exports = SelfAttention;
