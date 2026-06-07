const tf = require('@tensorflow/tfjs');

class Retriever {
  constructor(model, vocab, tokenizer, maxLen = 20) {
    this.model     = model;
    this.vocab     = vocab;
    this.tokenizer = tokenizer;
    this.maxLen    = maxLen;
    this.index     = [];

    // Mots-clés → réponse directe
    this.keywords = [
      {
        words: ['prix', 'tarif', 'coute', 'combien', 'cher', 'coût', 'montant'],
        response: "Les prix sont affichés sur chaque fiche produit de notre site AbShop."
      },
      {
        words: ['livraison', 'livrer', 'expédition', 'délai', 'jours', 'recevoir', 'colis'],
        response: "Nous livrons partout en Algérie sous 3 à 5 jours ouvrables."
      },
      {
        words: ['payer', 'paiement', 'carte', 'virement', 'règlement'],
        response: "Vous pouvez payer par carte bancaire, virement ou à la livraison."
      },
      {
        words: ['retour', 'retourner', 'rembourser', 'remboursement', 'renvoyer'],
        response: "Vous avez 14 jours pour retourner un article. Contactez-nous à contact@abshop.com."
      },
      {
        words: ['suivi', 'suivre', 'tracker', 'tracking', 'où', 'colis'],
        response: "Votre numéro de suivi est envoyé par email après expédition."
      },
      {
        words: ['contact', 'email', 'support', 'aide', 'joindre', 'parler'],
        response: "Contactez-nous à contact@abshop.com, réponse sous 24h."
      },
      {
        words: ['promo', 'réduction', 'solde', 'offre', 'discount', 'code'],
        response: "Inscrivez-vous à la newsletter pour recevoir -10% sur votre première commande !"
      },
      {
        words: ['vendre', 'vendez', 'produit', 'article', 'catalogue', 'acheter', 'achat'],
        response: "AbShop vend des produits électroniques, vêtements et accessoires."
      },
      {
        words: ['compte', 'inscription', 'inscrire', 'créer', 'register'],
        response: "Cliquez sur S'inscrire en haut du site et remplissez le formulaire."
      },
      {
        words: ['mot de passe', 'password', 'oublié', 'connexion', 'connecter'],
        response: "Cliquez sur Mot de passe oublié sur la page de connexion."
      },
      {
        words: ['bonjour', 'salut', 'bonsoir', 'hello', 'hey', 'coucou'],
        response: "Bonjour ! Bienvenue sur AbShop, comment puis-je vous aider ?"
      },
      {
        words: ['merci', 'super', 'parfait', 'nickel', 'génial', 'ok', 'accord'],
        response: "Avec plaisir ! N'hésitez pas si vous avez d'autres questions."
      },
      {
        words: ['au revoir', 'bye', 'bientôt', 'ciao', 'quit'],
        response: "Au revoir ! Bonne visite sur AbShop."
      }
    ];
  }

  encode(text) {
    const ids        = this.tokenizer.encode(text, this.maxLen);
    const input      = tf.tensor2d([ids], [1, this.maxLen], 'int32');
    const embedLayer = this.model.getLayer('embed');
    const embedOut   = embedLayer.apply(input);
    const vector     = tf.mean(embedOut, 1).reshape([-1]);
    const result     = Array.from(vector.dataSync());
    input.dispose(); embedOut.dispose(); vector.dispose();
    return result;
  }

  cosineSim(a, b) {
    let dot = 0, normA = 0, normB = 0;
    for (let i = 0; i < a.length; i++) {
      dot   += a[i] * b[i];
      normA += a[i] * a[i];
      normB += b[i] * b[i];
    }
    return dot / (Math.sqrt(normA) * Math.sqrt(normB) + 1e-8);
  }

  buildIndex(conversations) {
    console.log('Construction de l\'index...');
    this.index = conversations.map(conv => ({
      input:    conv.input,
      vector:   this.encode(conv.input),
      response: conv.output
    }));
    console.log(`Index construit : ${this.index.length} entrées`);
  }

  // Recherche par mots-clés
  keywordMatch(text) {
    const normalized = text.toLowerCase()
      .normalize('NFD').replace(/[\u0300-\u036f]/g, '')
      .replace(/[^a-z0-9\s]/g, ' ');

    let bestMatch = null;
    let bestCount = 0;

    for (const kw of this.keywords) {
      const count = kw.words.filter(w => normalized.includes(w)).length;
      if (count > bestCount) {
        bestCount = count;
        bestMatch = kw;
      }
    }

    return bestCount > 0 ? bestMatch.response : null;
  }

  retrieve(inputText, threshold = 0.75) {
    // 1. Similarité cosinus
    const queryVec = this.encode(inputText);
    let bestScore  = -1;
    let bestEntry  = null;

    for (const entry of this.index) {
      const score = this.cosineSim(queryVec, entry.vector);
      if (score > bestScore) {
        bestScore = score;
        bestEntry = entry;
      }
    }

    console.log(`  → Cosinus: "${bestEntry.input}" (${bestScore.toFixed(3)})`);

    // Score élevé → réponse directe
    if (bestScore >= threshold) {
      return bestEntry.response;
    }

    // Score moyen → mots-clés
    const kwResponse = this.keywordMatch(inputText);
    if (kwResponse) {
      console.log(`  → Mots-clés détectés`);
      return kwResponse;
    }

    // Rien trouvé
    console.log(`  → Aucune correspondance`);
    return "Je ne suis pas sûr de comprendre votre question. Pouvez-vous reformuler, ou contactez-nous à contact@abshop.com.";
  }
}

module.exports = Retriever;
