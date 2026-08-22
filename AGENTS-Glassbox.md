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

## 17. Ultra-Beginner First UX — シリーズ共通の最優先UX方針

### 17.1 方針と対象利用者

Glassbox AI Suite全体では、ターゲットユーザーを従来よりさらに初心者側へ明確に寄せる。

Glassboxの目的は、AI・機械学習について知識のないユーザーが、説明を読んで理解することから始めるのではなく、まず触って現象を観察し、その後で概念名を知る順序でAIの基本原理を理解できるようにすることである。

主対象は次のようなユーザーとする。

- AIについてほぼ知識がない
- ChatGPTなどを使った経験がある程度
- ニューラルネットワーク、Token、Weight、Probability Distribution、Rewardなどの用語を知らない
- Python、GitHub、programming経験を前提にできない
- 丁寧な文章で説明されても、未知の専門用語が連続すると理解できない
- 数式や抽象概念から入ると離脱する
- 画面上で変化を見せれば直感的に理解できる可能性がある

最終目標は、AIについて何も知らない人でも、画面を触っているうちに「ああ、こういうことか」と理解できる状態である。

これは内容を子供向けに不正確にすることを意味しない。技術的正確性を維持しつつ、理解に必要な抽象度、認知負荷、前提知識を下げる。

一般的なAI講座や講義形式を模倣せず、Glassbox固有の強みである「AI内部の抽象概念を、操作可能な現象として見せること」へ集中する。

### 17.2 最重要設計原則

UX、初回導線、説明順序、表示階層を判断するときは、原則として次の順序を最優先する。

```text
体験
↓
観察
↓
因果関係の理解
↓
名前を知る
↓
詳細説明
```

専門用語、説明、理解、実験という従来型教材の順序には極力しない。

たとえばWeightの定義から始めず、まず線の強さを変え、出力が変化する様子を見せる。ユーザーが「この線の強さが結果に影響している」と理解した後で、その強さを「Weight（重み）」と呼ぶことを提示する。

この節はシリーズ共通の最優先UX方針である。既存の技術的正確性、安全、実装境界、再現性、実値表示に関する規則は引き続き維持する。両者が緊張する場合は、正確性を損なわずに体験順序と情報階層を改善する案を優先する。

### 17.3 UX基本ルール

#### 最初の操作まで説明を要求しない

起動後、長文説明を読まなければ何をすればよいか分からないUIを避ける。

理想は次の状態である。

- 見れば押す場所が分かる
- 押せば何か起こる
- 起きた変化が目で分かる

#### 1画面1概念

初心者向け導線では、同時に複数の概念を理解させようとしない。入力、重み、学習、予測、確率、報酬などを一度に説明しない。

既存の観察機能を削除するという意味ではない。初回導線とLevel 1表示で焦点を絞り、追加情報を段階的に開示する。

#### 1操作1因果

ユーザーが何かを操作した場合、自分が何を変えたから何が変わったのかを明確にする。複数の内部状態が同時に変わり、因果関係が見えなくなるUIを避ける。

#### 専門用語は現象の後に出す

初心者導線では、可能な限り最初に専門用語を出さない。

たとえば「Learning Rateを変更してください」ではなく、「1回でどれくらい大きく直すかを変えてみよう」と先に案内する。その現象を体験した後で、「この値をLearning Rate（学習率）と呼びます」と正式名称へ接続する。俗語だけで終わらせない。

#### 数式は詳細情報に置く

数式を初心者が最初に理解するための必須条件にしない。必要な数式と正確な定義は、詳細表示、Advanced、詳しく知る等のLevel 3へ退避可能ならそうする。

数式、実値、Trace自体はGlassboxの重要な観察対象であり、削除しない。

#### 説明文を短くする

特に操作前の説明は短くする。一度に読む文章量そのものが認知負荷になるため、長く丁寧な説明だけで初心者対応を解決しようとしない。

#### 視覚変化を因果説明に使う

色、太さ、大きさ、移動、数値変化、グラフ、強調、Animationなどを使い、内部状態の変化を文字より先に認識できる設計を優先する。

装飾目的や実計算と無関係な演出には使わない。色だけに意味を依存せず、数値、label、線種、文言も併用する。

### 17.4 初心者導線の目標

各Glassboxは、初心者が無理なく最初の操作を始め、その結果として意味のある変化を観察できる状態を目標とする。一律の制限時間は設けない。

説明文をほぼ読まなくても何となく何が起きているか分かり、その変化から「次は何が起きるのか」「次も見たい」「自分でも試したい」と興味が続くことを重要な評価軸とする。

素早く完了させることよりも、一つの現象を理解したユーザーが自然に次の操作や観察へ進みたくなることを優先する。次の操作は押しつけず、現在の変化とのつながりが分かる形で示す。

具体的には、現在の現象を観察したあと、「次はどうなる？」という興味が生まれ、その疑問を確かめられる次の操作が、現在の現象との関係を理解できる形で提示されているかを確認する。単なる「次へ」ではなく、いま何が起きたか、次に何を変えられるか、変えると何が起きそうかがつながる案内を優先する。

「意味のある変化」とは、単に画面が動くことではない。ユーザーの操作またはAuto Runと、予測、誤り、修正、候補、行動、結果、報酬などの因果関係を少なくとも一つ観察できることを指す。

### 17.5 各モジュールの初心者向け責務

既存の数学的責務とモジュール境界は維持する。

#### Glassbox AI I — Neural Network

テーマは「学習とは何か」とする。

初心者が最終的に、AIは最初から答えを知っているのではなく、間違いを見ながら内部の値を少しずつ変えていくという直感を持てれば成功である。

初回導線では、Neuron、Gradient Descent、Backpropagation、Activation Functionなどの詳細理解を要求しない。

まず次の変化そのものを見せる。

```text
予測する
↓
間違う
↓
少し直す
↓
だんだん良くなる
```

#### Glassbox AI II — Language Model

テーマは「予測とは何か」とする。このモジュールは初心者が理解しづらいため、現在のUI、Auto Run、説明導線を特に厳しく評価する。

初心者が最終的に、文章生成AIは完成した文章を頭の中から取り出しているのではなく、「次に何が来そうか」を繰り返し選んで文章を作っているという直感を得られれば成功である。

可能なら、「今日はとても」のような入力に対し、次の候補と割合を視覚的に示す。

```text
暑い    45%
楽しい  20%
寒い    15%
…
```

そのうえで、次の連続過程を直接観察できる設計を優先する。

```text
候補から一つ選ぶ
↓
文章に追加される
↓
次の候補が変わる
```

Token、Probability Distribution、Sampling、Temperatureなどは、その現象を体験した後で正式名称として導入する。

必要なら構成変更を提案できるが、既存機能を無意味に全面破壊しない。構成変更は理由、変更案、互換性、影響範囲を示し、承認を得てから実施する。

#### Glassbox AI III — Reinforcement Learning

テーマは「行動を学ぶとは何か」とする。

初心者が、AIは正解を直接教えられなくても、良かった、悪かったという結果を使って行動を変えていけると理解できれば成功である。

現在の視覚的変化が理解されやすい長所を維持し、次の因果が一目で分かる設計を優先する。

```text
行動
↓
結果
↓
ごほうび / ペナルティ
↓
次の行動が変わる
```

Reward、Policy、Q-value等の名称は現象の後から導入する。現行実装はREINFORCEであり、実装していないQ-valueを実在する内部状態として表示しない。

### 17.6 Auto Runの位置づけ

Auto Runは単なる自動実行機能ではない。初心者が何を操作すればよいか分からない場合でも、現象を観察できる導線として扱う。

理想は次の構造である。

1. Auto Runを押す
2. 何かが動く
3. 視覚的に変化する
4. 「何が起きた？」という短い説明が出る
5. 次にユーザー自身で同じ箇所を操作できる

自動で進みすぎて、何が起きたのか分からないまま終了する状態を避ける。必要に応じて速度調整、一時停止、段階停止、現在の変化の強調を使う。

### 17.7 情報階層

説明は可能な限り三段階に分離する。

#### Level 1 — 見れば分かる

- Button
- 数値変化
- 動き
- 短い一文
- 視覚feedback

#### Level 2 — 名前を知る

現象を体験した後で、「この線の強さを重み（Weight）と呼びます」のように正式名称へ接続する。

#### Level 3 — 詳しく知る

- 正式な技術説明
- 数式
- より正確な定義
- 関連概念

初心者がLevel 3を読まなくても基本体験を完了できるようにする。

### 17.8 避ける方向

次の方向へ安易に変更しない。

- 長文tutorial化
- 教科書化
- Quiz中心化
- 用語集中心化
- 動画教材化
- 講義形式化
- 最初に説明をすべて読むことを要求する構成
- 一般的AI講座の模倣
- 正確性を犠牲にした過度な擬人化
- 子供っぽすぎるdesign
- 初心者向けという理由だけで既存機能を削除すること

GlassboxはAIについて学ぶための教科書ではなく、AIの仕組みを観察、操作するための実験装置として設計する。

### 17.9 改修手順

Ultra-Beginner First UXに関する改修は、原則として次の段階で進める。

#### Phase 1 — Audit

Glassbox I、II、IIIそれぞれについて、次を確認する。

- 初心者が最初に何を見るか
- 最初に何を要求されるか
- どこで専門用語が出るか
- どこで説明文を読まなければ進めないか
- 操作と結果の因果関係が見えるか
- 認知負荷が高い箇所
- 初心者が誤解しそうな箇所
- 既に非常に良い箇所

特に「AI知識ゼロの人が、説明文を飛ばして触った場合」という視点で評価する。

#### Phase 2 — Prioritize

発見した問題をCritical、High、Medium、Lowに分類する。

初心者理解への効果が大きく、既存設計への破壊が小さい変更を優先する。全面rewriteは原則として避ける。

#### Phase 3 — Implement

承認された範囲で、優先度の高い変更を実装する。必要に応じてUI、文言、初回導線、Auto Run、tooltip、状態表示、Animation、説明順序、Advanced表示などを変更できる。

意味の変化、構成変更、既存運用の変更、大規模改修、機能削除が必要な場合は、実行前に理由と案を示して確認を得る。

#### Phase 4 — Verify

次の想定ユーザーで再評価する。

> GitHubを知らない。Pythonを知らない。AIの仕組みを知らない。ChatGPTは使ったことがある。ニューラルネットワークという単語を聞いたことがある程度。長い説明を読むのは苦手。ただし、画面上の変化を見ることはできる。

このユーザーについて、次を確認する。

- 何をすればよいか分かるか
- 最初の意味のある現象を無理なく観察できるか
- なぜ結果が変わったのか分かるか
- 少なくとも一つ新しい概念を直感的に理解できるか
- 「次は何が起きるのか」と興味を持ち、次の操作や観察へ進みたくなるか
- 次に試せる操作が、現在観察した現象との関係を理解できる形で提示されているか

UI改修では自動testだけで完了とせず、実browserで初回起動、対象操作、Auto Run、narrow width、keyboard focus、Console warning / error、表示と実stateの一致を確認する。

### 17.10 Document同期

UIや設計思想に重要な変更が入った場合は、README、AGENTS.md、ROADMAP、その他の該当文書を必要に応じて同期する。ただし文書を増やすこと自体を目的にせず、既存文書への追記を優先する。

### 17.11 最終評価基準

技術者から見て説明が正しいことだけを評価基準にしない。

最も重要な評価基準は、AIについて何も知らない人が説明を読む前に触ってみて、「あ、こういう仕組みなのか」と一つでも自力で気付けるかである。

現状確認を先に行い、この方針と衝突している箇所を特定する。その後、既存の良い部分を残しながら段階的に改善する。
