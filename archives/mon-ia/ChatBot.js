const fs       = require('fs');
const tf       = require('@tensorflow/tfjs');
const readline = require('readline');
const Vocabulary = require('./core/Vocabulary');
const Tokenizer  = require('./core/Tokenizer');
const { buildModel, loadWeights } = require('./core/MiniLLM');
const Retriever  = require('./inference/Retriever');

async function main() {
  console.log('=== AbShop ChatBot ===\n');

  const vocab = new Vocabulary();
  vocab.load(fs, './data/vocab.json');
  const tokenizer = new Tokenizer(vocab);

  const model = buildModel(vocab.size);
  loadWeights(model, fs, './model-saved');

  // Charge les conversations pour l'index
  const raw = JSON.parse(fs.readFileSync('./data/conversations.json', 'utf8'));

  // Construit l'index de similarité
  const retriever = new Retriever(model, vocab, tokenizer, 20);
  retriever.buildIndex(raw.conversations);

  console.log('\nPrêt !\n' + '─'.repeat(40));

  const rl = readline.createInterface({ input: process.stdin, output: process.stdout });

  const ask = () => rl.question('\nVous : ', txt => {
    txt = txt.trim();
    if (!txt) return ask();
    if (txt === 'quit') { rl.close(); return; }

    const response = retriever.retrieve(txt);
    console.log(`\nAbShop : ${response}\n` + '─'.repeat(40));
    ask();
  });

  ask();
}

main().catch(console.error);
