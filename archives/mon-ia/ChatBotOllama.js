const fs       = require('fs');
const readline = require('readline');

const SYSTEM_PROMPT = `Tu es l'assistant virtuel d'AbShop, une boutique en ligne algérienne.
Tu réponds uniquement en français, de façon naturelle et chaleureuse.
Tu es serviable, concis (2-3 phrases max) et toujours poli.

Informations sur AbShop :
- Produits : électronique, vêtements, accessoires
- Livraison : partout en Algérie, 3 à 5 jours ouvrables
- Livraison gratuite à partir de 5000 DA
- Paiement : carte bancaire, virement, paiement à la livraison
- Retours : 14 jours après réception
- Remboursement : sous 5 jours après retour reçu
- Support : contact@abshop.com, réponse sous 24h
- Promos : -10% sur la première commande via newsletter
- Suivi commande : numéro de suivi envoyé par email après expédition

Règles :
- Si la question ne concerne pas AbShop, réponds poliment que tu es uniquement l'assistant AbShop.
- Ne jamais inventer de prix spécifiques.
- Toujours proposer contact@abshop.com pour les cas complexes.`;

const HISTORY_FILE = './data/ollama-history.json';
const MAX_HISTORY  = 10;
const MODEL        = 'gemma3:1b';
const OLLAMA_URL   = 'http://localhost:11434/api/chat';

function loadHistory() {
  try { return JSON.parse(fs.readFileSync(HISTORY_FILE, 'utf8')); }
  catch { return []; }
}

function saveHistory(history) {
  fs.writeFileSync(HISTORY_FILE, JSON.stringify(history, null, 2));
}

async function askOllama(history, userMessage) {
  const messages = [
    { role: 'system', content: SYSTEM_PROMPT },
    ...history,
    { role: 'user',   content: userMessage }
  ];

  const response = await fetch(OLLAMA_URL, {
    method:  'POST',
    headers: { 'Content-Type': 'application/json' },
    body:    JSON.stringify({
      model:    MODEL,
      messages: messages,
      stream:   false,
      options:  { temperature: 0.7, num_predict: 200 }
    })
  });

  if (!response.ok) {
    const err = await response.text();
    throw new Error(err || `Erreur Ollama (${response.status})`);
  }

  const data = await response.json();
  return data.message.content;
}

async function main() {
  console.log('=== AbShop ChatBot (Ollama local) ===\n');
  console.log(`Modèle : ${MODEL} — 100% local, pas d'internet requis\n`);

  // Vérifie qu'Ollama est bien lancé
  try {
    await fetch('http://localhost:11434');
  } catch {
    console.error('❌ Ollama n\'est pas lancé !');
    console.error('Lance dans un autre terminal : ollama serve');
    process.exit(1);
  }

  let history = loadHistory();
  console.log(`Historique : ${history.length / 2} échanges précédents\n`);
  console.log('Commandes : "reset" pour effacer, "quit" pour quitter\n');
  console.log('─'.repeat(40));

  const rl = readline.createInterface({ input: process.stdin, output: process.stdout });

  const ask = () => rl.question('\nVous : ', async txt => {
    txt = txt.trim();
    if (!txt) return ask();

    if (txt === 'quit') {
      saveHistory(history);
      console.log('\nHistorique sauvegardé. Au revoir !');
      rl.close();
      return;
    }

    if (txt === 'reset') {
      history = [];
      saveHistory(history);
      console.log('Historique effacé !\n' + '─'.repeat(40));
      return ask();
    }

    try {
      process.stdout.write('\nAbShop : ⏳ ');
      const reply = await askOllama(history, txt);

      process.stdout.clearLine(0);
      process.stdout.cursorTo(0);
      console.log(`\nAbShop : ${reply.trim()}`);
      console.log('─'.repeat(40));

      history.push({ role: 'user',      content: txt   });
      history.push({ role: 'assistant', content: reply });

      if (history.length > MAX_HISTORY * 2)
        history = history.slice(-MAX_HISTORY * 2);

    } catch (err) {
      console.log(`\n❌ Erreur : ${err.message}`);
    }

    ask();
  });

  ask();
}

main().catch(console.error);
