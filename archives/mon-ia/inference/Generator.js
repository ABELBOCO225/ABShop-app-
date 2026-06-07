// Generator.js — génère une réponse token par token

const tf = require('@tensorflow/tfjs');

class Generator {
  constructor(model, vocab, tokenizer, maxLen = 20) {
    this.model     = model;
    this.vocab     = vocab;
    this.tokenizer = tokenizer;
    this.maxLen    = maxLen;
  }

  // Génère une réponse à partir d'un texte d'entrée
  generate(inputText, temperature = 0.7) {
    return tf.tidy(() => {
      // Encode l'entrée
      const inputIds = this.tokenizer.encode(inputText, this.maxLen);
      const input    = tf.tensor2d([inputIds], [1, this.maxLen], 'int32');

      // Forward pass
      const logits = this.model.forward(input, false);

      // Récupère les tokens générés
      const outputIds = [];
      for (let i = 0; i < this.maxLen; i++) {
        // Logits du token à la position i
        const stepLogits = tf.slice(logits, [0, i, 0], [1, 1, this.vocab.size])
          .reshape([this.vocab.size]);

        // Applique la température
        const scaled = tf.div(stepLogits, tf.scalar(temperature));
        const probs  = tf.softmax(scaled);

        // Échantillonnage
        const tokenId = this._sample(probs.dataSync());
        outputIds.push(tokenId);

        // Arrête à <EOS>
        if (tokenId === this.vocab.encode('<EOS>')) break;
      }

      return this.tokenizer.decode(outputIds);
    });
  }

  // Échantillonne un token selon les probabilités
  _sample(probs) {
    const rand = Math.random();
    let cumSum = 0;
    for (let i = 0; i < probs.length; i++) {
      cumSum += probs[i];
      if (rand < cumSum) return i;
    }
    return probs.length - 1;
  }

  // Trouve le token le plus probable (greedy)
  _argmax(probs) {
    return probs.indexOf(Math.max(...probs));
  }

  // Génère avec stratégie greedy (plus déterministe)
  generateGreedy(inputText) {
    return tf.tidy(() => {
      const inputIds = this.tokenizer.encode(inputText, this.maxLen);
      const input    = tf.tensor2d([inputIds], [1, this.maxLen], 'int32');
      const logits   = this.model.forward(input, false);

      const outputIds = [];
      for (let i = 0; i < this.maxLen; i++) {
        const stepLogits = tf.slice(logits, [0, i, 0], [1, 1, this.vocab.size])
          .reshape([this.vocab.size]);
        const tokenId = this._argmax(Array.from(stepLogits.dataSync()));
        outputIds.push(tokenId);
        if (tokenId === this.vocab.encode('<EOS>')) break;
      }

      return this.tokenizer.decode(outputIds);
    });
  }
}

module.exports = Generator;
