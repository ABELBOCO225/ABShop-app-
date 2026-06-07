// Tokenizer.js — convertit texte <-> séquences d'indices

class Tokenizer {
  constructor(vocab) {
    this.vocab = vocab;
  }

  // Texte → indices  ex: "bonjour" → [2, 5, 3]
  encode(text, maxLen = 20) {
    const tokens = this.vocab.tokenize(text);
    const indices = [
      this.vocab.encode('<SOS>'),
      ...tokens.map(w => this.vocab.encode(w)),
      this.vocab.encode('<EOS>')
    ];
    return this._pad(indices, maxLen);
  }

  // Indices → texte  ex: [2, 5, 3] → "bonjour"
  decode(indices) {
    return indices
      .map(i => this.vocab.decode(i))
      .filter(w => !['<PAD>', '<SOS>', '<EOS>'].includes(w))
      .join(' ');
  }

  // Padding / troncature à longueur fixe
  _pad(indices, maxLen) {
    const padIdx = this.vocab.encode('<PAD>');
    if (indices.length >= maxLen) return indices.slice(0, maxLen);
    return [...indices, ...Array(maxLen - indices.length).fill(padIdx)];
  }

  // Prépare une paire input/output pour l'entraînement
  encodePair(input, output, maxLen = 20) {
    return {
      input:  this.encode(input, maxLen),
      output: this.encode(output, maxLen)
    };
  }
}

module.exports = Tokenizer;
