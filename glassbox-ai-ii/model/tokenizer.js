export const SPECIAL_TOKENS = ["<BOS>", "<EOS>", "<UNK>"];

export function splitText(text) {
  return (String(text).toLowerCase().match(/[a-z]+(?:'[a-z]+)?|[.,!?;:]/g) ?? []);
}

export class SimpleTokenizer {
  constructor(corpus, vocabulary = null) {
    const words = vocabulary ?? [...new Set(corpus.flatMap(splitText))].sort((a, b) => a.localeCompare(b));
    this.vocabulary = vocabulary ? [...vocabulary] : [...SPECIAL_TOKENS, ...words.filter((word) => !SPECIAL_TOKENS.includes(word))];
    if (this.vocabulary.length < SPECIAL_TOKENS.length || SPECIAL_TOKENS.some((token, i) => this.vocabulary[i] !== token)) {
      throw new Error("Vocabulary must start with <BOS>, <EOS>, <UNK>");
    }
    if (new Set(this.vocabulary).size !== this.vocabulary.length) throw new Error("Vocabulary contains duplicates");
    this.tokenToId = new Map(this.vocabulary.map((token, id) => [token, id]));
  }

  tokenize(text) {
    return splitText(text);
  }

  encode(text, { addBos = false, addEos = false } = {}) {
    const tokens = this.tokenize(text);
    if (addBos) tokens.unshift("<BOS>");
    if (addEos) tokens.push("<EOS>");
    return tokens.map((token) => this.tokenToId.get(token) ?? this.tokenToId.get("<UNK>"));
  }

  decode(ids) {
    return ids.map((id) => this.vocabulary[id] ?? "<UNK>");
  }

  detokenize(idsOrTokens) {
    const tokens = typeof idsOrTokens[0] === "number" ? this.decode(idsOrTokens) : idsOrTokens;
    return tokens.filter((token) => token !== "<BOS>").reduce((text, token) => {
      if (token === "<EOS>") return text;
      if (/^[.,!?;:]$/.test(token)) return `${text}${token}`;
      return text ? `${text} ${token}` : token;
    }, "");
  }
}
