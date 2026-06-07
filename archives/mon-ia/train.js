const fs = require('fs');
const tf = require('@tensorflow/tfjs');
const Vocabulary = require('./core/Vocabulary');
const Tokenizer  = require('./core/Tokenizer');
const { buildModel, saveWeights } = require('./core/MiniLLM');
const Dataset = require('./training/Dataset');

async function main() {
  console.log('=== AbShop MiniLLM ===\n');

  const vocab = new Vocabulary();
  const raw   = JSON.parse(fs.readFileSync('./data/conversations.json', 'utf8'));
  vocab.build(raw.conversations);
  vocab.save(fs, './data/vocab.json');

  const tokenizer = new Tokenizer(vocab);
  const dataset   = new Dataset(vocab, tokenizer, 20);
  dataset.load(fs, './data/conversations.json');

  const model = buildModel(vocab.size);
  console.log(`Vocab: ${vocab.size} tokens\n`);

  const { X, Y } = dataset.getAllBatches(15)[0];

  await model.fit(X, Y, {
    epochs: 100, batchSize: 4, shuffle: true,
    callbacks: {
      onEpochEnd: (epoch, logs) => {
        if ((epoch + 1) % 20 === 0)
          console.log(`Epoch ${epoch+1}/100 — Loss: ${logs.loss.toFixed(4)} — Acc: ${(logs.acc*100).toFixed(1)}%`);
      }
    }
  });

  saveWeights(model, fs, './model-saved');
  X.dispose(); Y.dispose();
  console.log('\nDone ! Lance : node ChatBot.js');
}

main().catch(console.error);
