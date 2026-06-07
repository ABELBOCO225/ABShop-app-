// TransformerBlock.js — un bloc complet : Attention + FeedForward + LayerNorm

const tf = require('@tensorflow/tfjs');
const SelfAttention = require('./Attention');

class TransformerBlock {
  constructor(embedDim, numHeads = 2, ffDim = 128) {
    this.attention = new SelfAttention(embedDim, numHeads);

    // FeedForward : deux couches denses
    this.ff1 = tf.layers.dense({ units: ffDim,     activation: 'relu', name: 'ff1' });
    this.ff2 = tf.layers.dense({ units: embedDim,  activation: 'linear', name: 'ff2' });

    // Normalisation
    this.norm1 = tf.layers.layerNormalization({ name: 'norm1' });
    this.norm2 = tf.layers.layerNormalization({ name: 'norm2' });

    // Dropout léger
    this.drop1 = tf.layers.dropout({ rate: 0.1, name: 'drop1' });
    this.drop2 = tf.layers.dropout({ rate: 0.1, name: 'drop2' });
  }

  // x : [batch, seqLen, embedDim]
  call(x, training = false) {
    return tf.tidy(() => {
      // 1. Self-Attention + résidu
      const attnOut  = this.attention.call(x);
      const dropped1 = this.drop1.apply(attnOut, { training });
      const normed1  = this.norm1.apply(tf.add(x, dropped1));

      // 2. FeedForward + résidu
      const ffOut    = this.ff2.apply(this.ff1.apply(normed1));
      const dropped2 = this.drop2.apply(ffOut, { training });
      const normed2  = this.norm2.apply(tf.add(normed1, dropped2));

      return normed2;
    });
  }
}

module.exports = TransformerBlock;
