# Glassbox AI II — Project Instructions

このprojectで作業する前に、次をすべて読むこと。

1. `../AGENTS.md` — workspace全体の安全・記録ルール
2. `../AGENTS-Glassbox.md` — Glassboxシリーズ共通方針
3. このファイル — `glassbox-ai-ii`固有の実装境界

## 現在のscope

このfolderは、外部API、学習済みmodel、ML libraryなしで動く独立した小型Decoder-only Transformer言語モデルである。

- Context length 8
- `d_model=8`、2 heads、`d_head=4`
- MLP hidden 16、GELU tanh approximation
- Learned Token / Position embedding
- Pre-LayerNorm、causal mask、Attention、Residual、Vocabulary projection
- Reverse-mode autograd、cross entropy、SGD、global gradient clipping
- 16 stage Forward Trace、Attention inspector、Training、Generation、Snapshot、JSON

このfolderがGlassbox AI IIのcanonical実装である。Reinforcement Learning codeは混在させず、番号IIを別の責務へ転用しない。

## 変更境界

- UIは`model.forward()`がclone保存したTrace snapshotを唯一の表示sourceとする。
- Token、Embedding、Q/K/V、Attention、Residual、MLP、Logits、Probabilityの因果鎖を途切れさせない。
- Causal maskの未来weight 0、Attention各rowの和1、Probability総和1を維持する。
- 過去TraceとSnapshotをParameter updateで書き換えない。
- `<BOS>`を生成候補へ戻さない。
- RL、Q-learning、Grid Worldをこのprojectへ追加しない。
- 外部API、学習済みmodel、TensorFlow.js、Transformers.js等を追加しない。

## 記録

変更理由、教材上の影響、検証結果、残る制限をIssue、PR、READMEのいずれかへ追跡可能に残す。

## 検証

```powershell
npm test
```

数学変更では、主要Parameterのfinite-difference gradient check、Tensor shape、serialization round-trip、seed determinism、500 training stepのNaN / Infinity不在とevaluation loss低下を確認する。

UI変更時は実browserで次を確認する。

- 16 stage ForwardのNext / Previous / Run
- Head 1 / Head 2、Raw / Masked / Weight
- Attention cell計算展開
- Training、Gradient、Snapshot、Generation
- narrow width
- Console warning / error

## 現在の優先事項

- 次Token予測を生成AIの基礎として正確に説明する
- TokenからProbabilityまでの数値追跡を維持する
- word / punctuation Tokenizer、Context 8、1 blockという単純化を明示する
- Language Modelの学習とReinforcement Learningを混同しない

複数block化、BPE、WebGPU、Adam、large corpusは現行初版の範囲外であり、独立提案とする。
