// Dataset.js — charge et prépare les batchs d'entraînement

const tf = require('@tensorflow/tfjs');

class Dataset {
  constructor(vocab, tokenizer, maxLen = 20) {
    this.vocab     = vocab;
    this.tokenizer = tokenizer;
    this.maxLen    = maxLen;
    this.pairs     = [];
  }

  // Charge conversations.json et encode chaque paire
  load(fs, path) {
    const raw = JSON.parse(fs.readFileSync(path, 'utf8'));
    this.pairs = raw.conversations.map(conv => 
      this.tokenizer.encodePair(conv.input, conv.output, this.maxLen)
    );
    console.log(`Dataset chargé : ${this.pairs.length} paires`);
  }

  // Mélange les paires aléatoirement
  shuffle() {
    for (let i = this.pairs.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [this.pairs[i], this.pairs[j]] = [this.pairs[j], this.pairs[i]];
    }
  }

  // Retourne un batch de tenseurs
  // X = input,  Y = output décalé d'un token (next token prediction)
  getBatch(batchSize = 4) {
    this.shuffle();
    const batch = this.pairs.slice(0, batchSize);

    const X = tf.tensor2d(
      batch.map(p => p.input),
      [batch.length, this.maxLen],
      'int32'
    );

    // Y = output décalé : on prédit le token suivant
    const Y = tf.tensor3d(
      batch.map(p => {
        const shifted = [...p.output.slice(1), this.vocab.encode('<PAD>')];
        return shifted.map(idx => {
          const oneHot = new Array(this.vocab.size).fill(0);
          oneHot[idx] = 1;
          return oneHot;
        });
      }),
      [batch.length, this.maxLen, this.vocab.size]
    );

    return { X, Y };
  }

  // Retourne tous les batchs pour une epoch
  getAllBatches(batchSize = 4) {
    const batches = [];
    for (let i = 0; i < this.pairs.length; i += batchSize) {
      const batch = this.pairs.slice(i, i + batchSize);
      if (batch.length === 0) continue;

      const X = tf.tensor2d(
        batch.map(p => p.input),
        [batch.length, this.maxLen],
        'int32'
      );

      const Y = tf.tensor3d(
        batch.map(p => {
          const shifted = [...p.output.slice(1), this.vocab.encode('<PAD>')];
          return shifted.map(idx => {
            const oneHot = new Array(this.vocab.size).fill(0);
            oneHot[idx] = 1;
            return oneHot;
          });
        }),
        [batch.length, this.maxLen, this.vocab.size]
      );

      batches.push({ X, Y });
    }
    return batches;
  }
}

module.exports = Dataset;
