const fs       = require('fs');
const readline = require('readline');

const SYSTEM_PROMPT = `Tu es l'assistant virtuel d'AbShop, une boutique en ligne algérienne.
Tu réponds uniquement en français, de façon naturelle et chaleureuse.
Tu es serviable, concis et toujours poli.

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

const HISTORY_FILE = './data/gemini-history.json';
const MAX_HISTORY  = 10;
const MODELS = ['gemini-flash-lite-latest', 'gemini-flash-latest'];

function loadHistory() {
  try { return JSON.parse(fs.readFileSync(HISTORY_FILE, 'utf8')); }
  catch { return []; }
}

function saveHistory(history) {
  fs.writeFileSync(HISTORY_FILE, JSON.stringify(history, null, 2));
}

async function askGemini(history, userMessage, modelIndex = 0) {
  if (modelIndex >= MODELS.length) throw new Error('Tous les modèles sont épuisés.');

  const model  = MODELS[modelIndex];
  const apiKey = process.env.GEMINI_API_KEY;
  const url    = `https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent?key=${apiKey}`;

  const contents = [
    { role: 'user',  parts: [{ text: SYSTEM_PROMPT }] },
    { role: 'model', parts: [{ text: 'Compris ! Je suis l\'assistant AbShop.' }] },
    ...history.map(m => ({
      role:  m.role === 'assistant' ? 'model' : 'user',
      parts: [{ text: m.content }]
    })),
    { role: 'user', parts: [{ text: userMessage }] }
  ];

  const response = await fetch(url, {
    method:  'POST',
    headers: { 'Content-Type': 'application/json' },
    body:    JSON.stringify({
      contents,
      generationConfig: { maxOutputTokens: 300, temperature: 0.7 }
    })
  });

  if (!response.ok) {
    const err = await response.json();
    const msg = err.error?.message || '';
    if (msg.includes('quota') || msg.includes('RESOURCE_EXHAUSTED') || msg.includes('not found')) {
      console.log(`\n  ⚠ ${model} indisponible, essai avec ${MODELS[modelIndex + 1] || 'aucun'}...`);
      return askGemini(history, userMessage, modelIndex + 1);
    }
    throw new Error(msg || `Erreur API (${response.status})`);
  }

  const data = await response.json();
  const text = data.candidates?.[0]?.content?.parts?.[0]?.text;
  if (!text) throw new Error('Réponse vide.');
  console.log(`  [${model}]`);
  return text;
}

async function main() {
  console.log('=== AbShop ChatBot (Gemini AI) ===\n');

  if (!process.env.GEMINI_API_KEY) {
    console.error('❌ GEMINI_API_KEY manquante !');
    console.error('Lance : export GEMINI_API_KEY=ta_clé_ici');
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
      process.stdout.write('\nAbShop : ...');
      const reply = await askGemini(history, txt);
      process.stdout.clearLine(0);
      process.stdout.cursorTo(0);
      console.log(`\nAbShop : ${reply}`);
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
