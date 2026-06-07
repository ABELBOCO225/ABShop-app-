// Vocabulary.js — construit et gère le vocabulaire token → index

class Vocabulary {
  constructor() {
    this.word2idx = {};
    this.idx2word = {};
    this.size = 0;

    // Tokens spéciaux
    this._addToken('<PAD>'); // 0 — padding
    this._addToken('<UNK>'); // 1 — mot inconnu
    this._addToken('<SOS>'); // 2 — début de séquence
    this._addToken('<EOS>'); // 3 — fin de séquence
  }

  _addToken(token) {
    if (!(token in this.word2idx)) {
      this.word2idx[token] = this.size;
      this.idx2word[this.size] = token;
      this.size++;
    }
  }

  // Transforme un texte en liste de mots
  tokenize(text) {
    return text.toLowerCase()
      .normalize('NFD').replace(/[\u0300-\u036f]/g, '')
      .replace(/[^a-z0-9\s]/g, ' ')
      .trim().split(/\s+/)
      .filter(w => w.length > 0);
  }

  // Construit le vocab depuis les conversations
  build(conversations) {
    conversations.forEach(conv => {
      [...this.tokenize(conv.input),
       ...this.tokenize(conv.output)].forEach(w => this._addToken(w));
    });
    console.log(`Vocabulaire construit : ${this.size} tokens`);
  }

  // Mot → index
  encode(word) {
    return this.word2idx[word] ?? this.word2idx['<UNK>'];
  }

  // Index → mot
  decode(idx) {
    return this.idx2word[idx] ?? '<UNK>';
  }

  // Sauvegarde vocab.json
  save(fs, path) {
    const data = JSON.stringify({ word2idx: this.word2idx, idx2word: this.idx2word, size: this.size }, null, 2);
    fs.writeFileSync(path, data);
    console.log(`Vocabulaire sauvegardé : ${path}`);
  }

  // Charge vocab.json
  load(fs, path) {
    const data = JSON.parse(fs.readFileSync(path, 'utf8'));
    this.word2idx = data.word2idx;
    this.idx2word = data.idx2word;
    this.size = data.size;
    console.log(`Vocabulaire chargé : ${this.size} tokens`);
  }
}

module.exports = Vocabulary;
