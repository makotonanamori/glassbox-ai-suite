import { SeededRandom } from "../utils/rng.js";
import { SimpleTokenizer } from "../model/tokenizer.js";

export const DEFAULT_CORPUS = [
  "the cat eats fish .",
  "the cat likes milk .",
  "the dog eats meat .",
  "the dog likes bones .",
  "the bird eats seeds .",
  "the bird likes trees .",
  "a cat chases mice .",
  "a dog chases cats .",
  "a bird sees trees .",
  "the fish swims .",
  "cats like milk .",
  "dogs like bones .",
];

export class LanguageDataset {
  constructor(corpus = DEFAULT_CORPUS, contextLength = 8, tokenizer = null) {
    this.corpus = [...corpus];
    this.contextLength = contextLength;
    this.tokenizer = tokenizer ?? new SimpleTokenizer(corpus);
    this.samples = corpus.map((sentence) => {
      const tokens = this.tokenizer.encode(sentence);
      const bos = this.tokenizer.tokenToId.get("<BOS>");
      const eos = this.tokenizer.tokenToId.get("<EOS>");
      const inputIds = [bos, ...tokens].slice(0, contextLength);
      const targetIds = [...tokens, eos].slice(0, contextLength);
      return { sentence, inputIds, targetIds };
    });
  }

  sampleAt(step, seed = 42) {
    const rng = new SeededRandom(`${seed}:training:${step}`);
    return this.samples[rng.integer(this.samples.length)];
  }
}
