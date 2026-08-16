export const FORWARD_STAGES = [
  ["tokenizer", "Tokenizer", "文字列を単語・句読点へ分割し、Token IDを確定します。"],
  ["tokenEmbeddings", "Token Embedding", "各Token IDに対応する学習可能な8次元ベクトルを取得します。"],
  ["positionEmbeddings", "Position Embedding", "位置0〜7に対応する学習可能なベクトルを取得します。"],
  ["initialRepresentation", "Embedding Addition", "Tokenベクトルと位置ベクトルを要素ごとに加算します。"],
  ["layerNorm1", "LayerNorm 1", "各Tokenの8次元を平均0付近・分散1付近へ正規化します。"],
  ["qkv", "Q / K / V", "正規化表現を各HeadのWq・Wk・Wvへ掛けます。"],
  ["rawScores", "Raw Attention Scores", "QとKの内積をsqrt(d_head)で割ります。"],
  ["maskedScores", "Causal Mask", "未来Tokenのscoreを参照禁止としてマスクします。"],
  ["attentionWeights", "Attention Softmax", "参照可能なscoreを行ごとの確率へ変換します。"],
  ["attentionOutput", "Attention Output", "Attention weightでVを加重平均し、2 Headを結合・射影します。"],
  ["residual1", "Residual 1", "Attention前の表現へAttention出力を加算します。"],
  ["mlp", "LayerNorm 2 + MLP", "8→16→8のMLPをGELU活性化付きで計算します。"],
  ["residual2", "Residual 2", "MLP前の表現へMLP出力を加算します。"],
  ["finalNorm", "Final LayerNorm", "Vocabulary射影前の表現を正規化します。"],
  ["logits", "Vocabulary Logits", "各位置の8次元表現をVocabulary全体のscoreへ射影します。"],
  ["probabilities", "Probabilities", "安定化Softmaxで次Token確率へ変換します。"],
].map(([key, label, description], index) => ({ index, key, label, description }));

export class ForwardStepEngine {
  constructor(trace = null) {
    this.load(trace);
  }

  load(trace) {
    this.trace = trace;
    this.index = trace ? 0 : -1;
    return this.current();
  }

  current() {
    if (!this.trace || this.index < 0) return null;
    return { ...FORWARD_STAGES[this.index], trace: this.trace };
  }

  next() {
    if (this.trace) this.index = Math.min(FORWARD_STAGES.length - 1, this.index + 1);
    return this.current();
  }

  previous() {
    if (this.trace) this.index = Math.max(0, this.index - 1);
    return this.current();
  }

  run() {
    if (this.trace) this.index = FORWARD_STAGES.length - 1;
    return this.current();
  }

  reset() {
    if (this.trace) this.index = 0;
    return this.current();
  }
}
