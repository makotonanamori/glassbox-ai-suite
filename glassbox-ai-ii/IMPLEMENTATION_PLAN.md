# Glassbox AI II 実装計画

## 目的

外部API・学習済みモデル・MLライブラリを使わず、1層Decoder-only Transformerの実計算を、TokenからSGD更新まで追跡できるローカル実験装置を作る。

## 変更境界

- 新規フォルダ`glassbox-ai-ii`内だけにアプリ本体を作る。
- 既存`glassbox-ai`、Lucy関連、`gpt-5.2現行\現行バックアップ`は変更しない。
- HTML/CSS/ES Modulesのみで起動し、Node.jsは自動テストにだけ使用する。

## 実装順序

1. Seeded RNG、Tokenizer、固定Corpusを実装する。
2. Tensor/autogradと数値安全な演算を実装する。
3. 1層TransformerのForwardとimmutable Traceを実装する。
4. Cross Entropy、Backward、gradient clipping、SGD、Trainerを実装する。
5. 数学テスト、Gradient Check、500 step安定性テストを通す。
6. 6タブUI、16段階Forward、AttentionセルInspectorを実装する。
7. Training、Parameter Browser、Snapshot比較、生成、Export/Import、Debugを接続する。
8. README、実ブラウザQA、決定記録と残課題を整える。

## 完了判定

- Attentionの未来weightが0、各rowとVocabulary probabilityの総和が1。
- selected parameterの解析勾配と中央差分の相対誤差が原則`1e-3`未満。
- 同じseed・Corpus・training step数で同じlogitsを再現。
- 500 training stepsでNaN/Infinityがなく、評価Lossが初期値より低下。
- ForwardのPrevious/Next/Run、Train One Step、生成、Snapshot比較、JSON Export/Importをブラウザで実測。

