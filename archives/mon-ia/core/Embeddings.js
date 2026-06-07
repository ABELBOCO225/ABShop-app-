// Embeddings.js — convertit indices → vecteurs denses + encodage positionnel

const tf = require('@tensorflow/tfjs');

class Embeddings {
  constructor(vocabSize, embedDim, maxLen = 20) {
    this.embedDim = embedDim;
    this.maxLen   = maxLen;

    // Embedding des tokens
    this.tokenEmbed = tf.layers.embedding({
      inputDim:    vocabSize,
      outputDim:   embedDim,
      maskZero:    true,
      name:        'token_embedding'
    });

    // Embedding positionnel
    this.posEmbed = tf.layers.embedding({
      inputDim:    maxLen,
      outputDim:   embedDim,
      name:        'position_embedding'
    });
  }

  // indices [batch, seqLen] → vecteurs [batch, seqLen, embedDim]
  call(indices) {
    return tf.tidy(() => {
      const seqLen = indices.shape[1];

      // Positions 0,1,2,...,seqLen-1
      const positions = tf.tile(
        tf.expandDims(tf.range(0, seqLen, 1, 'int32'), 0),
        [indices.shape[0], 1]
      );

      const tokVecs = this.tokenEmbed.apply(indices);
      const posVecs = this.posEmbed.apply(positions);

      // Token + position
      return tf.add(tokVecs, posVecs);
    });
  }
}

module.exports = Embeddings;
