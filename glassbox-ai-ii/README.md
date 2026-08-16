# Glassbox AI II — Transparent Language Model

小型のDecoder-only Transformerを、外部API・学習済みモデル・MLライブラリなしで実装したローカル実験装置です。

これは実際の大規模言語モデルそのものではなく、Transformer型autoregressive language modelの主要機構を、人間が観察可能な規模へ縮小したモデルです。性能よりも「この数値がどの入力・Parameter・演算から来たか」を追跡できることを優先しています。

Glassbox AI IIのcanonicalな責務は次Token予測とTransformer内部Traceです。教師あり5→4→3ネットワークは[Glassbox AI I](../glassbox-ai/)、環境報酬から学ぶ強化学習は[Glassbox AI III](../glassbox-ai-iii/)が担当します。

## 起動方法

Windows PowerShellでこのフォルダへ移動し、専用launcherを実行します。

```powershell
.\start-glassbox-ai-ii.cmd
```

既定ではブラウザで `http://127.0.0.1:8102/` を開きます。ビルド、インストール、外部通信は不要です。ES Modulesを使うため、`index.html`の直接ダブルクリックではなくローカルHTTPサーバーから開いてください。IIIとは別directory・別portなので単独でも同時でも起動できます。

手動起動する場合は`python -m http.server 8102 --bind 127.0.0.1`も利用できます。

## モデル構成

| 項目 | 値 |
|---|---:|
| Transformer block | 1 |
| Context length | 8 tokens |
| `d_model` | 8 |
| Attention heads | 2 |
| `d_head` | 4 |
| MLP hidden | 16 |
| Activation | GELU tanh approximation |
| Position | Learned embedding |
| Normalization | Pre-LayerNorm + Final LayerNorm |
| Optimizer | SGD |
| Default learning rate | 0.03 |
| Default global gradient clip | 1.0 |
| Default seed | 42 |
| Vocabulary | 固定Corpusから決定的に生成（初期Corpusでは24） |
| Parameter count | 初期Corpusでは1,088 |

Token embeddingとVocabulary projectionは重み共有をせず、別々のParameterとして観察できます。

## Tensor shape

`T`は入力長（1〜8）、`V`はVocabulary sizeです。

| Trace | Shape |
|---|---|
| Token / Position / Initial representation | `[T, 8]` |
| Q / K / V（各Head） | `[T, 4]` |
| Raw / Masked score | `[T, T]` |
| Attention weight | `[T, T]` |
| 各Head output | `[T, 4]` |
| Concatenated / Projected attention | `[T, 8]` |
| MLP pre-activation / GELU | `[T, 16]` |
| MLP output / Residual / Final norm | `[T, 8]` |
| Logits / Probability | `[T, V]` |

## Forward Pass

Token IDを`t_i`、位置を`i`とすると、初期表現は次です。

```text
x_i = E_token[t_i] + E_pos[i]
u_i = LN1(x_i)
```

各Attention Head `h`で次を計算します。

```text
Q_h = u Wq_h + bq_h
K_h = u Wk_h + bk_h
V_h = u Wv_h + bv_h

score_h(i,j) = (Q_h[i] · K_h[j]) / sqrt(d_head)
```

Causal Maskは未来位置を参照不能にします。

```text
masked_score_h(i,j) = -Infinity   if j > i
                      score_h(i,j) otherwise
```

Softmaxは数値安定化のため行の最大値`m_i`を引きます。

```text
m_i = max_j masked_score_h(i,j)       （参照可能位置のみ）
A_h(i,j) = exp(masked_score_h(i,j) - m_i)
           / sum_k exp(masked_score_h(i,k) - m_i)
```

Maskされた位置のweightは厳密に0、各行の合計は1です。

```text
head_h[i] = sum_j A_h(i,j) V_h[j]
attention_output = concat(head_1, head_2) Wo + bo
h1 = x + attention_output
```

Pre-LayerNorm後にMLPと2つ目のResidualを適用します。

```text
u2 = LN2(h1)
pre = u2 W1 + b1
hidden = GELU(pre)
mlp_output = hidden W2 + b2
h2 = h1 + mlp_output
y = LN_final(h2)
```

GELUは次のtanh近似です。

```text
GELU(x) = 0.5 x [1 + tanh(sqrt(2/pi) (x + 0.044715 x^3))]
```

Vocabulary射影と確率は次です。

```text
logits = y W_vocab + b_vocab
P(token_j | previous_tokens) = exp(logit_j - max(logits))
                                / sum_k exp(logit_k - max(logits))
```

## Training

文`[the, cat, eats, fish, .]`は次の1例になります。

```text
Input : [<BOS>, the, cat, eats, fish, .]
Target: [the, cat, eats, fish, ., <EOS>]
```

各位置のCross Entropyを平均します。

```text
L_i = -log P(target_i | input_0 ... input_i)
L = (1 / T) sum_i L_i
```

自作reverse-mode autogradが同じForward graphを逆順に辿り、全ParameterのGradientを計算します。global normは次です。

```text
g_norm = sqrt(sum_parameter sum_element gradient^2)
clip_scale = min(1, clip_norm / g_norm)
g_clipped = gradient * clip_scale
```

SGD更新は次です。

```text
parameter_new = parameter_old - learning_rate * g_clipped
```

Adamやミニバッチは実装していません。Loss → Gradient → Updateの関係を直接観察するためです。

## UI操作

### Playground

1. Token列を入力して「入力を確定してForward準備」を押します。
2. `Next` / `Previous`で16段階を移動します。
3. Tokenセルを選ぶと、同じTokenのEmbeddingからFinal representationまでを縦に追えます。
4. `Run Forward`でProbabilityまで進みます。
5. GenerationはGreedyまたはTemperatureを選び、1 TokenまたはN Token生成します。Context 8を超えると左端から落とします。`<BOS>`は生成候補から除外します。

### Attention

- Head 1 / Head 2、Raw Score / Masked Score / Softmax Weightを切り替えます。
- Matrixセルを選ぶと、Q・K、要素積、内積、`sqrt(d_head)`、mask、softmax分子・分母、最終weightを表示します。
- Headの意味は断定せず、観測値だけを並べます。

### Training

- `Train One Step`は同一例の更新前後Loss、位置別Loss、Gradient norm、clip scaleを表示します。
- 10 / 100 StepとAuto Trainも同じ学習処理を繰り返します。Auto TrainはEvent Loopへ定期的に制御を返し、Pauseできます。
- Loss HistoryはSVGで描画します。

### Parameters

- 28個のParameter Tensor、合計1,088要素を一覧表示します。
- Shape、Count、Mean、Std、Min、Max、Gradient normを確認できます。
- 選択Parameterでは各要素のold / gradient / delta / newを確認できます。

### Architecture / Debug

- Architectureは現在のForward stageを強調します。
- Debugで選択Parameter要素の中央差分Gradient Checkを実行できます。
- STEP 0、10、100、500は自動Snapshotされ、任意Snapshotも追加できます。同一PromptのAttention・Prediction・Probability・固定サンプルLossを現在状態と比較します。
- JSON Export / Importは構成、Vocabulary、全Parameter、Seed、学習設定、Step、Loss履歴を保存・検証します。不正形式、Shape不一致、NaN/Infinityを拒否します。

## Trace

`model.forward()`はUI用のplain object Traceを返します。主なフィールドは次です。

```text
tokens, tokenEmbeddings, positionEmbeddings, initialRepresentation,
layerNorm1, heads[].q/k/v/rawScores/maskedScores/attentionWeights/output,
attentionOutput, residual1, layerNorm2, mlpPreActivation, mlpActivation,
mlpOutput, residual2, finalNorm, logits, probabilities, loss
```

Trace配列はForward時にcloneされます。学習でParameterが更新されても、過去Traceは変化しません。UIはTraceと公開Model APIを使い、表示専用の疑似数値を作りません。

## 再現性

Seed付きPRNGは自作Mulberry32系生成器とBox–Muller Normalを使い、`Math.random()`へ依存しません。同じSeed、Vocabulary、Corpus、Training step数なら、初期Parameter、サンプル順、更新後Parameter、Logitsが一致します。

## テスト方法

Node.js 20以降を推奨します。外部packageのインストールは不要です。

```powershell
npm test
```

テスト対象：

- matmul、LayerNorm、GELUとBackward
- 極端な値のSoftmax安定性、確率和1
- Causal Mask未来weight 0、Attention各行の和1
- 全主要Tensor shape
- 同一SeedのParameter / Logits再現
- Export → Import後のLogits一致
- Trace snapshot不変性、Step Engine前後移動
- Vocabulary projection、Q、V、MLP、Embeddingの有限差分Gradient Check
- 同一Seed・同一step数の学習再現
- 500 stepでNaN/Infinityなし、評価Loss低下

Gradient Checkは中央差分を使います。

```text
g_numerical = [L(theta + epsilon) - L(theta - epsilon)] / (2 epsilon)
relative_error = |g_analytical - g_numerical|
                 / max(1e-12, |g_analytical| + |g_numerical|)
epsilon = 1e-5
```

中央差分の打切り誤差は`O(epsilon^2)`ですが、`epsilon`を小さくしすぎるとJavaScript Numberの丸め誤差が増えます。本モデルでは`1e-5`を実測上安定する妥協点とし、通常演算の合格基準を`relative error < 1e-3`としています。

## ファイル構成

```text
glassbox-ai-ii/
├─ index.html
├─ css/style.css
├─ js/app.js
├─ model/
│  ├─ tensor.js
│  ├─ transformer.js
│  ├─ optimizer.js
│  ├─ tokenizer.js
│  └─ step-engine.js
├─ training/
│  ├─ dataset.js
│  └─ trainer.js
├─ visualization/renderers.js
├─ utils/
│  ├─ math.js
│  ├─ rng.js
│  └─ serialization.js
├─ tests/
│  ├─ tensor.test.js
│  ├─ transformer.test.js
│  ├─ gradient.test.js
│  └─ training.test.js
├─ package.json
├─ IMPLEMENTATION_PLAN.md
└─ README.md
```

## 意図的に単純化した部分・既知の制限

- word/punctuation tokenizerであり、BPE・subword tokenizerではありません。
- Context 8、1層、2 Head、Vocabulary 24の固定小型構成です。
- Corpusはアプリ同梱の12文だけです。任意Corpus編集UIはありません。
- 1例ずつのSGDで、ミニバッチ、Adam、GPU、WebGPU、KV cacheはありません。
- GenerationはGreedy / Temperatureのみで、Top-p、Beam Searchはありません。
- Snapshot比較はブラウザメモリ内です。JSON Exportは現在のモデル状態を保存しますが、複数Snapshot履歴そのものは含めません。
- 小さなCorpusと確率的な1例更新のため、個々のTrain One Stepで同じ例のLossが下がっても、Corpus全体のLossは単調には下がりません。長期傾向をLoss Curveで確認してください。
- 教材用のため、高速Tensor kernelではなく追跡可能なJavaScript配列演算です。

## 非目標と将来案

実用LLM、ChatGPT互換、大規模Corpus、BPE、Multi-layer、GPU、RLHF、RAG、Agent等は非目標です。将来拡張する場合も、まずCorpus編集とTokenizer比較、複数blockの差分表示、Optimizer比較を独立実験として追加し、初版の固定小型モデルとTrace再現性を残す方針が適切です。

## 最初の2分

ページ上部の初回ガイドには二つの入口があります。

1. **Forwardを観察**: Playgroundへ移動し、`Next`でTokenからProbabilityまでの16段階を進めます。`Run Forward`なら同じTraceを最後まで表示します。
2. **1回学習を観察**: Trainingへ移動し、`Train One Step`でForward → Loss → Backward → Gradient → SGD Updateを実行します。LossとPredictionの学習前後比較まで同じ画面で確認できます。

ガイドは対象Tabを選び、次に押す実操作へfocusします。表示用の別計算は追加しておらず、Forward表示は引き続きclone保存されたTraceだけを読みます。初回ガイドと外部依存境界の静的検査を加え、自動テストは現在16件です。
