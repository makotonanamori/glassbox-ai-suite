# Glassbox AI — Series Instructions

## 1. この文書の役割

このファイルは、Glassbox AIシリーズに共通する設計・教材・OSS運用方針の正本である。

各プロジェクトで作業する際は、次の順序で指示を確認する。

1. workspace rootの`AGENTS.md`
2. この`AGENTS-Glassbox.md`
3. 対象project直下の`AGENTS.md`

project直下の指示は、そのprojectの現在の構造、テスト、変更境界を補足する。上位文書の安全ルールや記録ルールを無効化しない。

## 2. Project Overview

Glassbox AIは、AIの内部動作を小規模なモデルとインタラクティブな可視化によって観察可能にする教育用OSSプロジェクトである。

目的は高性能なAIを作ることではない。

**「AIが何をしているのかを、ブラックボックスのまま使わず観察できるようにすること」**を最優先する。

各デモは可能な限り小規模、単純、決定論的または再現可能な構成とし、利用者が入力、内部状態、計算、学習、出力の因果関係を追跡できることを重視する。

基本思想:

> AIを作るためのAIではなく、AIを理解するためのAI。

## 3. Core Design Principles

### Observability

- 内部状態を可能な限り可視化する。
- 結果だけを表示しない。
- 入力から出力、学習信号、更新後状態まで追跡可能にする。
- 重要な表示値について「どの入力、Parameter、数式から来たか」を確認可能にする。

### Educational clarity

- 数学的正確性を維持しつつ、理解可能性を優先する。
- 不必要に高度なalgorithmへ置換しない。
- 「単純だが原理が見える」実装を優先する。
- 初心者向けであっても、不正確な擬人化や過度な単純化を行わない。

### Small-scale implementation

- 巨大モデル、大量dataset、GPUを前提にしない。
- 一般的なPCとローカルbrowserで全計算を観察できる規模を維持する。
- 規模を広げる前に、現行の中間値を完全に説明できることを優先する。

### Explicit state

- Weight、Bias、Activation、Loss、Probability、Reward、Gradient、Parameter update等を隠さない。
- 数値とHeatmap、線、色、状態表示の対応を示す。
- 色だけに意味を依存させず、数値、label、線種も併用する。

### No fake intelligence

- 実在しない推論過程や内部状態を表示しない。
- 説明用のdummy値を実計算値のように見せない。
- UIは実際の計算結果またはclone保存したTrace / snapshotを読む。
- 「AIが考えた」「理解した」といった観測不能な解釈を事実として表示しない。

### Reproducibility

- 乱数を使用する場合はseedを指定可能にする。
- 同じseed、入力、設定、学習stepから同じ結果を再現する。
- 内部計算では丸めず、丸めは表示時だけ行う。

### Minimal dependency

- 外部AI API、学習済みmodel、ML libraryを使用しない。
- 必要以上に巨大なframeworkやdependencyを追加しない。
- dependency追加が必要な場合は、教育上の利益、license、offline動作、保守負担を先に説明し、確認を得る。

## 4. Canonical Project Mapping

番号、責務、公開directoryは次の一対一対応を正本とする。

### Glassbox AI I — `glassbox-ai` — Supervised Learning

- 5入力→4中間→3出力のニューラルネット
- tanh、softmax、cross entropy、backpropagation、SGD
- 教師あり139 step数学timelineと全39 Parameter update

### Glassbox AI II — `glassbox-ai-ii` — Transparent Language Model

独立した1層Decoder-only Transformer言語モデルである。

- Context length 8
- `d_model=8`
- 2 Attention heads
- MLP hidden 16
- Pre-LayerNorm
- Learned position embedding
- Autoregressive next-token prediction
- Reverse-mode autograd、SGD、gradient clipping

Language Model固有のToken、Embedding、Attention、MLP、Logits、Probability、Loss、Gradientを観察する。Reinforcement Learning実装は混在していない。

### Glassbox AI III — `glassbox-ai-iii` — Reinforcement Learning

- 7×7 Grid Worldと5 sensor
- 温度付き確率方策、探索/活用、環境報酬、割引Return
- REINFORCEと全39 Parameter update
- 教師あり139 stepとは独立したRL因果timeline

IとIIIには分離前のcodeがlegacy compatibility implementationとして一部残る。通常UIでは非canonical panelを表示せず、保存形式・数値回帰のため凍結保持する。新機能をlegacy側へ追加せず、IのRL機能はIII、IIIの教師あり機能はIを正本として変更する。

### 将来project

- Glassbox AI IV: Image Recognition — Planned
- Glassbox AI V: Generative AI / Diffusion — Exploring
- Glassbox AI VI: RAG — Exploring
- Glassbox AI VII: AI Agent — Exploring

将来案を、現在実装済みであるかのようにREADMEやUIへ表示しない。

## 5. Series Learning Path

シリーズ全体では、次の共通骨格と学習信号の違いを観察可能にする。

```text
入力
↓
Parameterを使った計算
↓
確率または行動
↓
学習信号
↓
Gradient
↓
Parameter update
↓
次回の出力変化
```

- Glassbox AI I / 教師あり学習: 人間が指定した正解labelとの誤差
- Glassbox AI II / Language Model: 実際に次に現れたTokenとの予測誤差
- Glassbox AI III / 強化学習: 環境遷移後に得たRewardと割引Return

共通骨格を示しても、強化学習を生成AIの通常の事前学習そのものとは説明しない。Transformer言語モデルとRLは、観察対象と学習信号を分ける。

## 6. Visualization Rules

表示する値は、実際のmodel stateまたは計算Traceに対応させる。

```text
Input
↓
Internal State
↓
Computation
↓
Prediction / Action
↓
Error / Reward
↓
Update
↓
Changed State
```

避けるもの:

- 意味のないparticle effect
- 実計算と無関係なbrain animation
- 「AIが考えています」というblack-box表示
- 実在しないChain of Thought
- 観測値から根拠なく役割や意図を断定する説明
- 見栄えのためだけのfake animation

Animationは計算対象、順序、状態変化の補助に限定する。停止、再開、前後移動がある場合は内部stateと表示stateを一致させる。

## 7. Accuracy Policy

Glassbox AIは教育用途であるため、見た目より技術的正確性を優先する。

単純化を行う場合は、`simplified model`、`toy implementation`、既知の非目標を明示する。

例えばREINFORCEやQ-learningの教材を、現代的LLM post-training全体と同等であるかのように説明しない。小型Transformerを、実際の大規模言語モデルと同じ性能・scaleであるかのように説明しない。

観測と解釈を分離する。

- 許可: 「Head 2でToken AからToken BへのAttention weightは0.41」
- 禁止: 「Head 2は文法を理解している」

## 8. Development Behavior

作業開始時は推測でarchitectureを変更せず、対象projectの次を確認する。

- current directory structure
- execution path
- dependencies
- existing tests
- mathematical data flow
- visualization flow
- model stateとUI stateの境界
- README、IMPLEMENTATION_PLAN、既知の制限

変更前:

1. 関連codeとtestsを読む。
2. 現在の挙動を再現する。
3. 変更対象と非対象を限定する。
4. regression riskを確認する。
5. Issue、PR、READMEのいずれかへ変更理由と検証根拠を追跡可能に残す。

変更中:

- 既存coding styleを維持する。
- Model logicとDOM / visualization logicを分離する。
- 不要なlibrary、magic number、premature abstractionを増やさない。
- UIからmodel内部変数を直接書き換えない。
- 過去TraceやsnapshotがParameter updateで書き換わらないようcloneする。
- 大規模rewrite、directory再編、共通core抽出は、明確な必要性と承認なしに行わない。

変更後:

- syntax check
- unit test
- gradient checkまたは数値検証
- runtime verification
- UI変更時の実browser確認
- narrow-width確認
- Console warning / error確認

を変更リスクに応じて実行する。失敗、未確認、既知の制限を隠さず、Issue、PR、READMEへ記録する。

## 9. Refactoring Policy

次の場合に限ってrefactorを検討する。

- 責務混在により計算の由来を追えない
- Model stateとUI stateの結合がbug原因になっている
- 同じ数学処理が複数箇所で不一致になっている
- 安全な修正に必要な最小範囲である
- 複数Glassbox間に、実証済みの明確な共通処理がある

「綺麗になる」「将来使うかもしれない」だけを理由に大規模rewriteしない。既存の`glassbox-ai`からRLを分離したり、monorepoへ再編したりする作業は、独立した設計判断として事前承認を得る。

## 10. README Requirements

READMEには最低限、次を含める。

- Projectの目的
- 何を実装しているか
- 何を実装していないか
- Quick Start
- Userが操作できる入力
- 観察できる内部値
- 数式と実装の対応
- Test方法
- Seedによる再現性
- File structure
- Known limitations
- Roadmap上の現在地

Browser demoを公開する場合は、初回表示、sample input、reset、step execution、値変更、比較の導線を確認する。ScreenshotやGIFは実際のUIと実計算を使い、fake dataを使わない。

## 11. Repository Structure

このsuite repositoryを公開用正本とし、三つのcanonical appは独立性を保ったsubdirectoryとして収録する。

```text
glassbox-ai-suite/
├─ index.html
├─ AGENTS.md
├─ AGENTS-Glassbox.md
├─ glassbox-ai/                  I / Supervised Learning
├─ glassbox-ai-ii/               II / Language Model
└─ glassbox-ai-iii/              III / Reinforcement Learning
```

Series landingはnavigationだけを担当し、三つのappの数学coreやtimelineを統合しない。shared/core抽出やlegacy compatibility codeの物理削除は、互換性を確認する独立変更とする。

## 12. License

OSS公開用正本としてMIT Licenseを採用済みである。新しい依存や素材を追加する前に次を確認する。

- dependencies
- bundled assets
- fonts
- images
- datasets / corpus
- copiedまたはadapted code

不明なlicenseの素材をMIT扱いしない。必要に応じてNOTICEまたはREADMEへ出典とlicenseを記録する。

## 13. GitHub Project Operation

公開後は通常のOSSとして運用する。

- Issue、Release、Roadmapを応募用に水増ししない。
- 実際の作業から自然な履歴を残す。
- Releaseは意味のある変更単位にする。
- Semantic Versioningを基本とする。
- Pull Requestでは数学、Trace、UI、testの対応を説明する。
- Public demo、CI、Security窓口は実在する状態だけを記載する。

## 14. Codex for Open Source

将来的にOpenAI Codex for Open Sourceへの応募を検討する。ただし応募条件に見せかけるための開発はしない。

優先順位:

1. 教育的価値
2. OSSとしての品質
3. 実際に利用可能であること
4. 継続的な改善
5. Project documentation
6. Community usability
7. Codex for OSS application

選考されること自体をprojectの目的にしない。Glassbox AIを良質なOSSとして育てた結果として、応募可能な状態になることを目標とする。

## 15. Current Priority

### Priority 1 — Glassbox AI Iの教材品質

- 初見利用者が作者の説明なしで起動できる
- 5→4→3教師あり139 step数学timelineを正本として維持する
- legacy Grid World / RL panelへ新機能を追加しない
- Error handling、accessibility、documentation、test、reproducibilityを維持する
- MIT Licenseと第三者素材境界を維持する

### Priority 2 — `glassbox-ai-ii`の教材品質

- TokenからProbabilityまでの16 stage Traceを維持する
- Attention cellの数値由来を追える状態を守る
- Language ModelとRLの説明を混同しない
- Corpus、Tokenizer、Context 8という単純化を明示する

### Priority 3 — Glassbox AI IIIの教材品質

- 環境履歴、報酬、探索/活用、REINFORCE因果timelineを正本として維持する
- legacy教師ありpanelへ新機能を追加しない
- I、II、IIIの学習信号を混同しない

### Priority 4 — Series navigation

三つのアプリを統合せず、series landingからI → II → IIIの番号順と学習信号の違いを案内する。

## 16. Immediate Task Policy

固定された移行作業をImmediate Taskとして持たない。各依頼ごとに現行code、tests、README、実browser状態を確認し、最小変更案を決める。

特に、次を現状確認なしに開始しない。

- I / IIIのlegacy compatibility codeを物理削除する作業
- public directory名またはI / II / III番号を変更する作業
- shared/coreを新設する作業
- folder名またはpublic URLを変更する作業
- LICENSEを追加する作業

これらはすべて構造または運用方針を変えるため、理由、利点、互換性、移行手順を提示してユーザー確認を得る。
