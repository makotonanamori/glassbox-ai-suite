# Glassbox AI Suite Roadmap

このRoadmapは、Glassbox AI Suiteを「高性能AI」ではなく、AI内部の因果関係を実数値で追える教材として改善するための優先順位を示します。

予定は品質検証と利用者フィードバックに応じて変更します。日付や機能を保証するrelease scheduleではありません。

## 判断原則

1. 数学的な正しさを見栄えより優先する。
2. UIは実計算またはclone保存したTrace / snapshotだけを表示する。
3. I、II、IIIの学習信号とcanonical責務を混同しない。
4. 同一Seedから同じ観測結果を再現できる状態を守る。
5. 外部API、学習済みモデル、MLライブラリを前提にしない。

## Current baseline — 公開済み

- [x] Glassbox AI I：5→4→3教師あり学習と139 step数学timeline
- [x] Glassbox AI II：1 block Decoder-only Transformerと16 stage Forward Trace
- [x] Glassbox AI III：7×7環境、探索/活用、報酬、REINFORCE専用timeline
- [x] 有限差分による勾配check、Seed再現、JSON round-trip、500 step数値安定性
- [x] I / II / IIIの独立起動とGitHub Pages公開
- [x] Windows / Ubuntu CI、MIT License、貢献・Security文書
- [x] 実画面Screenshot / overview GIFとrepository metadata

## Now — v1.x OSS readiness

- [x] I / II / IIIへ「自動で見る」→「1ステップずつ詳しく見る」の共通入口を実装する
- [x] Glassbox AI IIIで連続Episodeの試行錯誤を時間方向に表示する
- [x] Glassbox AI Iで139 step教師あり学習を時間方向に表示する
- [x] Glassbox AI IIで次Token候補・選択・文末追加・反復を時間方向に表示し、内部Traceを詳細へ分ける
- [x] Glassbox AI IIで同じPromptの学習前／500回後を実Trainingし、候補確率と生成文の変化を並べる
- [x] CHANGELOGを作成し、検証済み公開baselineを履歴化する
- [ ] 初回release / tagを作成し、versioned releaseを固定する
- [ ] 第三者が説明なしでI → II → IIIを完走する初心者user testを行う
- [ ] Keyboardだけで主要操作とMatrix / Parameter選択を完了できるか監査する
- [ ] 360px、標準PC幅、拡大表示でのaccessibility確認手順を文書化する
- [ ] Issueから再現可能なSeed、入力、step、browser情報を収集する運用を定着させる

## Next — 教材としての追跡性

- [ ] 選択値について「入力 → Parameter → 演算 → 出力」の参照先を共有可能なURLまたはTrace exportで表す
- [ ] 学習前後Snapshotの差分をI / II / IIIそれぞれで同じ語彙と表示規則へ揃える
- [ ] 数値異常、Import拒否、Causal Mask等の失敗例を安全な教材scenarioとして追加する
- [ ] Property-basedな数値検証を、外部ML libraryなしで導入できる範囲を評価する
- [ ] 日本語UIを正本に保ちながら、英語教材導線の必要性を第三者利用状況から判断する

## Later — 独立性を保った拡張候補

- [ ] I / II / IIIのTraceを統合せず、学習信号だけを横断比較する読み取り専用view
- [ ] word tokenizerと極小subword tokenizerの分割差を比較する独立実験
- [ ] REINFORCEのbaseline有無を同一Seedで比較する分散観察モード
- [ ] Screenshot / GIFの再生成とvisual regressionを再現可能な保守手順へする

## Non-goals

- 実用LLM、ChatGPT互換、学習済みモデル配布
- 外部AI API、クラウド学習、ユーザーアカウント、telemetry
- GPU、WebGPU、分散学習、大規模Dataset
- Transformer多層化、RLHF、Agent、RAG、Function Calling
- 観測値から「Headが文法を理解した」等の意味を自動断定する表示

## 提案の受け付け方

提案は[Feature request](https://github.com/makotonanamori/glassbox-ai-suite/issues/new?template=feature_request.yml)から受け付けます。教材上の目的、観察したい因果関係、実数値で検証する方法、既存I / II / IIIのどこに属するかを記載してください。

実装前に、可視性、数値正確性、再現性、独立起動、初学者への説明責任を満たすかを確認します。
