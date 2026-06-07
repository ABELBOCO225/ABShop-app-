// api/chat.js — Vercel Serverless Function

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

export default async function handler(req, res) {
  // CORS — autorise ton domaine Vercel
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

  if (req.method === 'OPTIONS') return res.status(200).end();
  if (req.method !== 'POST') return res.status(405).json({ error: 'Méthode non autorisée' });

  const { message, history = [] } = req.body;
  if (!message) return res.status(400).json({ error: 'Message manquant' });

  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey) return res.status(500).json({ error: 'Clé API manquante' });

  const url = `https://generativelanguage.googleapis.com/v1beta/models/gemini-flash-lite-latest:generateContent?key=${apiKey}`;

  const contents = [
    { role: 'user',  parts: [{ text: SYSTEM_PROMPT }] },
    { role: 'model', parts: [{ text: 'Compris ! Je suis l\'assistant AbShop.' }] },
    ...history.map(m => ({
      role:  m.role === 'assistant' ? 'model' : 'user',
      parts: [{ text: m.content }]
    })),
    { role: 'user', parts: [{ text: message }] }
  ];

  try {
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
      return res.status(500).json({ error: err.error?.message || 'Erreur Gemini' });
    }

    const data  = await response.json();
    const reply = data.candidates?.[0]?.content?.parts?.[0]?.text || 'Pas de réponse.';
    return res.status(200).json({ reply });

  } catch (err) {
    return res.status(500).json({ error: err.message });
  }
}
