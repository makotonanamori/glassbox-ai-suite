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
].map(([key, label, description], index) => ({ index, key, label, description, observation: [
  '文章を、ひとつずつ扱えるまとまりに分けました。各まとまりの番号は、意味の優劣を表す得点ではありません。',
  '語の番号を使って、その語に対応する8個の数字を取り出しました。学習で変わるのはこの数字です。',
  '何番目にある語かを、8個の数字で表しました。同じ語でも位置が違えば、この数字は変わります。',
  '語の数字と位置の数字を足しました。これで「何の語か」と「どこにあるか」を一緒に計算できます。',
  '8個の数字の平均とばらつきを使って、大きさを整えました。下の表で変換前後を比べられます。',
  '同じ入力から、照合する数字Q・照合される数字K・渡す内容の数字Vを作りました。2つの組で別々に計算します。',
  'QとKの数字を掛けて足し、各語の組み合わせの得点を作りました。得点はまだ割合ではありません。',
  '今の位置より後ろの語を参照できなくしました。学習中も、予測する答えを先に見ないためです。',
  '参照できる語の得点を、行ごとに合計100%の割合へ変えました。未来の語の割合は0です。',
  '各語が持つVの数字を、今作った割合で混ぜました。2つの組の結果を合わせ、8個の数字へ戻します。',
  '語同士を混ぜる前の数字へ、混ぜた結果を足しました。元の情報も次の計算へ渡ります。',
  '数字の大きさを整え、8個から16個へ広げ、曲線による変換をして8個へ戻しました。語同士を混ぜる処理とは別です。',
  '直前の変換で得た数字を、変換前の数字へ足しました。元の情報を残したまま次へ進みます。',
  '候補ごとの得点を作る前に、8個の数字の大きさをもう一度整えました。',
  '8個の数字から、知っているすべての候補の得点を計算しました。負の得点もあり、まだ確率ではありません。',
  '得点を合計100%の割合へ変えました。最後の位置の割合が、文章の次の候補です。生成では開始記号を候補から外します。',
][index] }));

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

  goToStage(key) {
    const index = FORWARD_STAGES.findIndex((stage) => stage.key === key);
    if (!this.trace || index < 0) return null;
    this.index = index;
    return this.current();
  }
}
