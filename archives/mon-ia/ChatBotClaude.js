const fs       = require('fs');
const readline = require('readline');

// System prompt AbShop — personnalise selon ton business
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

const HISTORY_FILE = './data/claude-history.json';
const MAX_HISTORY  = 10; // Garde les 10 derniers échanges

// Charge l'historique de conversation
function loadHistory() {
  try {
    return JSON.parse(fs.readFileSync(HISTORY_FILE, 'utf8'));
  } catch {
    return [];
  }
}

// Sauvegarde l'historique
function saveHistory(history) {
  fs.writeFileSync(HISTORY_FILE, JSON.stringify(history, null, 2));
}

// Appel à l'API Claude
async function askClaude(history, userMessage) {
  const messages = [
    ...history,
    { role: 'user', content: userMessage }
  ];

  const response = await fetch('https://api.anthropic.com/v1/messages', {
    method: 'POST',
    headers: {
      'Content-Type':         'application/json',
      'x-api-key':            process.env.ANTHROPIC_API_KEY,
      'anthropic-version':    '2023-06-01'
    },
    body: JSON.stringify({
      model:      'claude-haiku-4-5-20251001', // Modèle le moins cher
      max_tokens: 300,
      system:     SYSTEM_PROMPT,
      messages:   messages
    })
  });

  if (!response.ok) {
    const err = await response.json();
    throw new Error(err.error?.message || `Erreur API (${response.status})`);
  }

  const data = await response.json();
  return data.content[0].text;
}

async function main() {
  console.log('=== AbShop ChatBot (Claude AI) ===\n');

  if (!process.env.ANTHROPIC_API_KEY) {
    console.error('❌ ANTHROPIC_API_KEY manquante !');
    console.error('Lance : export ANTHROPIC_API_KEY=ta_clé_ici');
    process.exit(1);
  }

  let history = loadHistory();
  console.log(`Historique chargé : ${history.length / 2} échanges précédents\n`);
  console.log('Tapez votre message ("quit" pour quitter, "reset" pour effacer l\'historique)\n');
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
      const reply = await askClaude(history, txt);

      // Efface "..." et affiche la réponse
      process.stdout.clearLine(0);
      process.stdout.cursorTo(0);
      console.log(`\nAbShop : ${reply}`);
      console.log('─'.repeat(40));

      // Met à jour l'historique
      history.push({ role: 'user',      content: txt   });
      history.push({ role: 'assistant', content: reply });

      // Garde seulement les MAX_HISTORY derniers échanges
      if (history.length > MAX_HISTORY * 2) {
        history = history.slice(-MAX_HISTORY * 2);
      }

    } catch (err) {
      console.log(`\n❌ Erreur : ${err.message}`);
    }

    ask();
  });

  ask();
}

main().catch(console.error);
