# 「自動で見る」共通UX

Glassbox AI I / II / IIIの初心者向け入口に適用する共通仕様です。

## 目的

利用者が専門用語を理解していなくても、最初の1クリックで実際の計算を開始し、状態が変わった結果まで到達できるようにします。

説明の順序は次に統一します。

```text
スタート → 動く → 結果 → 何が変わったか → 現象の名前 → 数式と内部値
```

## 二つの入口

### 自動で見る（おすすめ）

- 初期設定のまま1クリックで開始できる。
- `ready → running → complete`の状態を表示する。
- 表示する結果は、各appが通常の詳細表示で使用する実計算と同じデータから取得する。
- 完了時は専門用語より先に、何が変わったかを平易な文章で示す。
- 二重実行中は主ボタンを無効化する。

Phase 1の完了単位は次のとおりです。

| App | 1クリックで完了する実計算 | 最初に示す結果 |
| --- | --- | --- |
| Glassbox AI I | 1回の順伝播、教師あり学習、更新後の再計算 | 正解の選ばれやすさの学習前後 |
| Glassbox AI II | 1 Training Stepと更新後の再計算 | 間違いの大きさの学習前後 |
| Glassbox AI III | 1 Episode、報酬集計、方策更新 | 累積報酬、行動回数、餌取得数 |

Glassbox AI IIIはPhase 2で、初心者向け主操作を10エピソードの連続表示へ拡張しました。各実移動で世界を描画し、エピソード完了ごとに履歴と報酬グラフを更新します。再生は一時停止・再開でき、表示速度を0.5 / 1 / 2 / 4倍から選べます。

Glassbox AI IはPhase 3で、初心者向け主操作を139スナップショットの連続表示へ拡張しました。入力から予測、正解との差、全勾配、39更新、再計算、比較までを順番に描画します。再生は一時停止・再開でき、表示速度を0.5 / 1 / 2 / 4倍から選べます。

Glassbox AI IIはPhase 4で、初心者向け主操作を38段階の連続表示へ拡張しました。開始時の16段階Forward、1 Training Stepの5段階、更新後の生成用16段階Forward、実際のToken選択を一本につなぎます。再生は一時停止・再開でき、表示速度を0.5 / 1 / 2 / 4倍から選べます。

その後の初心者testを受け、Phase 5では入口の学習目標をさらに絞りました。表面では`候補確率 → 1 Token選択 → 文末追加 → 次の予測`を既定5 Token繰り返し、内部16段階とTrainingは生成現象を見た後の詳細へ移します。

生成反復を理解した後の疑問を受け、Phase 6では`お手本を見る → 次の語を予測 → 外れた分を調整 → 同じ問いで再確認`を第二入口として追加しました。同じPromptの学習前と500回後を、候補Probabilityと生成文で左右比較します。

### 1ステップずつ詳しく見る

- 自動実行と別のモデルやダミーデータを作らない。
- 同じtimelineまたはTraceを先頭へ戻す。
- 既存の`次のステップ`または`Next`へfocusし、詳細計算へ入れる。
- 状態は`detail`として表示する。

## 共通DOM契約

各appの最初の操作領域は次を持ちます。

- `data-experience-entry`
- `data-experience-state="ready|running|complete|detail"`
- `data-experience-mode="auto"`
- `data-experience-mode="detail"`
- `data-flow-step="start|motion|result"`
- `aria-live="polite"`の結果表示

IDや内部adapter名は各app固有でも構いません。利用者向けのボタン名と状態の意味は共通にします。

## 数値と表示の原則

- 疑似animation、固定の成功結果、説明用ダミー値を使わない。
- 表示用の丸め値を次の計算へ使用しない。
- 異常値や実計算errorを成功表示で隠さない。
- 専門用語は現象を見せた後に付ける。
- 詳細表示のTrace、timeline、Parameter値と自動実行結果を一致させる。

## Phase 2 — Glassbox AI III

- `ReinforcementStepEngine`の`rl-transition`を実移動の表示境界にする。
- 表示境界の間にある観測、方策、抽選、Return、Gradient、39更新も同じタイムライン上で省略せず実行する。
- 10エピソードを有限回だけ実行し、進捗、直近イベント、単発報酬、累積報酬、餌、探索/活用を実値で更新する。
- 一時停止中はengineを進めず、再開時は同じエピソードの同じ位置から続ける。
- 単一エピソードの報酬改善を成功と断定せず、履歴の上下と目的達成を分けて表示する。

## Phase 3 — Glassbox AI I

- `StepEngine`の139個の完全snapshotを、先頭から比較まで一つずつ描画する。
- 推論終了時は同じengineへ既存の学習timelineを追加し、one-hot、損失、誤差、全勾配、39更新、更新後Forwardを省略しない。
- 初心者向けstatusでは、予測、正解との差、数字への差の伝播、数字の更新、前後比較の順に現象を示す。
- 一時停止中はengine indexを進めず、再開時は同じsnapshotから続ける。
- 詳細入口では、同じ学習timelineを`0 / 139`へ戻して既存の数式・実値表示へ接続する。

## Phase 4 — Glassbox AI II

- clone保存した16段階Forward Traceを、開始時の予測と更新後の生成で一段ずつ描画する。
- TrainerのForward、Loss、Backward、Gradient、SGD Updateを段階APIで実行し、一括`trainOneStep()`も同じAPIを使う。
- 生成時は既存のGreedy / Temperature設定とSeed付き乱数を使い、確率を作った同じTraceからTokenを選ぶ。
- 一時停止中は段階とParameterを変えず、再開時は同じ位置から続ける。
- 詳細入口では、開始時PromptとTraceを同時に復元し、既存16段階Forwardの`1 / 16`へ接続する。

## Phase 5 — Glassbox AI II 生成ループ優先

- 初心者向け表面は、モデルが実際に使う候補分布を横棒と実数値で表示する。
- Greedyは最大確率、TemperatureはSeed付き乱数が入った累積確率区間を選び、選択行を枠と記号でも強調する。
- 選んだTokenを文章の末尾へ追加し、新規Tokenを`NEW`ラベルで示してから、伸びた文で次の候補計算へ戻る。
- Context Lengthを超えた場合は、次回計算に残るToken列と範囲外へ出たTokenを分けて表示する。
- 表面の自動再生ではTrainingを実行しない。`Token → Embedding → Attention → Logits → Softmax → Loss → Gradient`を短い視覚導線として下に置き、既存Forward / Training画面へ接続する。
- 詳細入口では、最後の候補を計算したPromptとclone保存済みTraceを同時に復元する。

## Phase 6 — Glassbox AI II 文章らしさの学習前後

- 固定Prompt `the cat eats`と同梱12例文を使用し、現在の同一modelを500 Training Step更新する。学習済みmodelや表示専用の成功文は使わない。
- 学習前と現在の上位候補、選択Token、5 Token生成結果、期待Token `fish`のProbability、Corpus平均Lossを左右に並べる。
- 50 Stepごとの10 checkpointでTrainerを実際に進め、候補の揺れを履歴へ残す。途中で別Tokenが最大になる場合も成功風に補正しない。
- 一時停止中はParameterとTraining stepを変えず、再開時は同じmodel状態から続ける。待機時間は表示だけに作用する。
- 完了後の詳細入口は、学習後modelの固定Prompt Traceを`1 / 16`へ戻す。
- 極小Corpusの語順学習であり、意味理解の証明や大規模生成AIと同等の文章品質ではないことを観測結果と分けて明示する。

## 次の確認

I / II / IIIの時間方向表示と、IIの生成・学習前後の二段入口は完了しました。次は第三者初心者再testで、生成反復から学習による候補変化まで説明なしに追えるかを観察し、文言確定後にScreenshot / overview GIFを更新します。

## 2026-08-22 — Ultra-Beginner First UX Phase 1 Audit

### 監査目的

AI知識がなく、説明文を飛ばして画面を触る利用者を想定し、Glassbox I、II、IIIの現行入口とAuto Runを再評価した。

今回の中心的な評価基準は次のとおりである。

> 現在の現象を観察したあと、「次はどうなる？」という興味が生まれ、その疑問を確かめられる次の操作が、現在の現象との関係を理解できる形で提示されているか。

一律の完了時間は評価基準にしない。実装変更は行わず、現状の良い部分と衝突箇所を特定した。

### 実施範囲

- Series landingとI、II、IIIの初回表示を確認
- 説明を開かず、最初に見える主操作を実行
- Iの139 step教師あり自動再生を完了まで確認
- IIの5 Token生成と500回Training比較を完了まで確認
- IIIの10 episode連続表示を完了まで確認
- Auto Run中と完了後の状態、表示値、scroll着地点、次に見える操作を確認
- I、II、IIIのConsole warning / errorを確認
- rootで`npm run check`を実行

実browserの通常viewportは1280×720だった。360px幅へのviewport overrideは接続中のbrowserへ反映されず、接続可能な別browserもなかったため、狭幅の実操作は今回未確認である。CSSの620px / 980px breakpointは静的に確認したが、実表示確認の代替とは扱わない。Keyboardの直接focusとoutlineは確認できたが、Tab移動の連続操作はbrowser接続上で進まなかったため、focus順序の実操作確認も未完了である。

### Series landing

#### 初心者が最初に見るもの

- 「まず動かす。変化を見る。仕組みの名前はそのあとで」というシリーズ方針
- `Iを自動で見る（おすすめ）`という単一の目立つ入口
- `スタート → AIが動く → 結果 → 学習後の変化 → あとから詳しく`という流れ

#### 良い部分

- 説明を読む前にIへ進める主操作が初期viewport内にある。
- Series全体の価値を「実際の計算を動かして変化を見る」と短く示している。
- I、II、IIIを一つのmodelへ統合せず、学習信号の違いを維持している。

#### 衝突箇所

- 各course cardでは、Weight、softmax、Loss、Parameter、Token、Transformer、Attention、Logits、REINFORCE等が現象体験前にまとまって表示される。
- `約3分`、`約4分`という時間表示は、理解や好奇心より完了時間を先に意識させる可能性がある。
- 初心者が「見たい動き」を選ぶ見出しに対し、cardの比較材料は動きより技術構成が中心である。

### Glassbox AI I — 「学習とは何か」

#### 初心者が最初に見るもの

- 「5つの数字から予測を作り、1回の学習で結果が変わる」という短い目的
- 初期viewport内の`自動で見る`
- `予測、誤差、更新、結果の変化`という観察順

#### 最初に要求されること

設定変更や説明読解は不要で、主buttonを1回押せば開始できる。

#### 専門用語と認知負荷

- 主buttonに`139段階`、進捗に`139ステップ`が現れ、初心者には意味の分からない大きさが最初から強調される。
- headerの`フェーズ`、`ステップ`、`学習回数`はQuick Startより先に表示される。
- 詳細領域ではWeight、Loss、Gradient、39 Parameter等が全面表示される。下方にあるため初回操作の妨げにはならないが、Auto Runの完了地点がこの詳細領域内になる。

#### 操作と結果の因果

- Auto Run中は「予測を作っています」「正解との差を数字へ戻しています」等の短いstatusへ段階的に切り替わる。
- 実際の139 snapshotを通り、最終的に正解の選ばれやすさが`33.4614% → 38.1488%`へ変わることを確認できた。
- Pauseと速度変更があり、実計算結果を変えずに観察速度だけを変えられる。

#### 誤解しそうな箇所

- 教師あり学習の出力classが`行動A 前進`、`行動B 左折`、`行動C 右折`として表示されるため、Glassbox IIIの行動学習と混同しやすい。
- 完了後は比較領域へscrollし、初心者向けstatusと`1ステップずつ詳しく見る`がviewport外へ出る。
- 完了地点ではLoss、変更Parameter数、最大変更量等が同時に並び、最初に見るべき「間違いを見て少し直した」が詳細値に埋もれる。
- 完了後のviewportに見えるbuttonは前後step等のdisabled操作が中心で、「もう一度直すと予測はどう変わる？」を確かめる次の操作がない。

#### 既に非常に良い箇所

- 説明なしで開始できる。
- 予測、誤り、修正、再予測を一つの実計算timelineとして保持している。
- 学習前後を同じ入力で比較し、確率上昇とLoss低下を表示している。
- 詳細入口は同じtimelineを先頭へ戻すため、初心者表示と数学表示が別のdummy結果にならない。

### Glassbox AI II — 「予測とは何か」

#### 初心者が最初に見るもの

- 「次の1語を予測して選ぶ。その繰り返しで文章が伸びる」という目的
- 初期viewport内の`文章が伸びる様子を見る`
- 候補、選択、追加、もう一度という4段階loop

#### 最初に要求されること

設定変更や説明読解は不要で、主buttonを1回押せば開始できる。

#### 専門用語と認知負荷

- Quick Startより前のstatusにParameters、Vocabulary、Training stepが表示される。
- 主button、進捗、候補見出しにTokenが現象体験前から使われる。
- `<BOS>`、Context範囲、Transformerという名称も初心者向け観察面に現れるが、候補選択と文末追加の主現象はそれらを理解しなくても追える。

#### 操作と結果の因果

- 実際の候補Probabilityを横棒と数値で表示し、選ばれたTokenを枠と記号で強調する。
- 選択したTokenが文章末尾へ追加され、伸びた文章で候補が変わる過程を5回反復する。
- 実browserでは`the cat eats`が`the cat eats the dog cats milk the`へ伸び、最終候補も更新された。
- 次の学習観察では、同じ問いに対する実Training前後を左右比較し、`fish`が7.59%から19.81%、Corpus平均Lossが3.3856から1.2615へ変わった。
- 中間checkpointの揺れも成功表示へ補正せず、そのまま表示している。

#### 次の興味への接続

- 生成結果の直後に`では、どうして文章らしくなる？`という次の疑問が現れる。
- 次の操作`練習前と500回後を見比べる`は、いま見た不自然な生成が練習でどう変わるかを直接確かめる実験になっている。
- Training完了後も`さらに500回練習して比べる`が見え、練習量と候補変化の関係を続けて試せる。

#### 誤解しそうな箇所

- Tokenを「1語」と案内している一方、punctuationや`<UNK>`もTokenになるため、正式名称を導入する短い橋渡しがないと「Tokenは必ず単語」と誤解する可能性がある。
- 初期statusのParameters、Vocabulary、Training stepは、最初に必要な観察対象と誤認される可能性がある。

#### 既に非常に良い箇所

- 現象、次の疑問、次の実験が連続しており、今回採用した評価基準に最も近い。
- 候補確率、選択、追加、候補変化を同じ実Traceから表示する。
- 生成とTrainingを一度に見せず、二段階の初心者導線へ分けている。
- 極小Corpusの語順学習であり、意味理解の証明ではないことを明示している。

### Glassbox AI III — 「行動を学ぶとは何か」

#### 初心者が最初に見るもの

- 「小さな世界で行動し、結果を受けて選び方が変わる」という目的
- 初期viewport内の`自動で見る`
- `10回の試行錯誤`と、AIの移動と結果を連続表示する案内

#### 最初に要求されること

設定変更や説明読解は不要で、主buttonを1回押せば開始できる。

#### 専門用語と認知負荷

- 進捗にEpisodeが最初から表示される。
- Auto Runが表示する主panelでは、REINFORCE、方策、割引率、Temperature、学習率、探索／活用、割引Return、方策Gradient等が同時に現れる。
- 完了後の着地点は10行の履歴表であり、累積報酬、探索／活用、診断等を同時に読む必要がある。

#### 操作と結果の因果

- Auto Run中は世界、実際の移動、環境event、単発報酬、累積報酬を更新する。
- 実browserでは10 episodeを完了し、合計-2.8700点、餌2個という実結果とepisode別履歴を表示した。
- 餌取得episodeをgraph上で区別し、報酬だけ高くても目的達成とは断定しない。
- Pauseと速度変更があり、実計算結果を変えずに観察速度だけを変えられる。

#### 誤解しそうな箇所

- 移動、結果、報酬までは見えるが、その報酬が選び方をどう変え、次の同様な状況でどの行動が選ばれやすくなったかが初心者向け表面では比較されない。
- 10回の結果は上下に揺れ、最後のepisodeも餌0、累積報酬-0.47である。正確な実結果だが、初心者には「何を学んだのか」が読み取りにくい。
- 完了後は履歴graphと表へscrollし、初心者向けstatus、世界、次の操作がviewport外へ出る。
- 完了後のviewportにはbuttonがなく、「ごほうびの付け方を変えると、次の行動はどう変わる？」等を確かめる操作へ接続しない。
- 診断文の`顕著な異常パターンなし`や`局所最適候補`は、最初の因果理解より先に専門的評価を要求する。

#### 既に非常に良い箇所

- Grid Worldの移動と環境結果は視覚的で、三つの中で行動そのものを最も直感的に観察しやすい。
- Rewardの高低と餌取得を分け、表示上の成功へ実結果を補正しない。
- 詳細timelineでは観測、方策確率、抽選、遷移、報酬、Return、Gradient、39更新を同じepisodeから追跡できる。
- Iの教師ありtimelineとIIIの強化学習timelineを混同していない。

### 優先度

#### Critical

1. **Glassbox IIIの初心者向け完了体験で、中心概念が閉じていない。** 行動、結果、報酬までは見えるが、報酬を受けて次の行動選択がどう変わったかを比較できない。IIIの成功条件である「良かった／悪かったという結果を使って行動を変える」が、説明を読まずに到達できる表面へ出ていない。

#### High

1. **IとIIIのAuto Run完了後に、現在の観察と関係する次の操作がない。** どちらも詳細な比較または履歴へ着地し、初心者向けstatusと次の入口がviewport外になる。
2. **Glassbox Iの教師ありclass名が行動名になっている。** `前進／左折／右折`はIIIの行動学習との混同を招き、IとIIIの責務差を初心者に説明しにくい。
3. **IとIIIのLevel 1完了表示へLevel 3情報が集中する。** Loss、Parameter、探索／活用、診断等が主結果と同時に出て、最初に理解すべき因果を弱める。

#### Medium

1. Iの`139段階`、IIのTokenとmodel status、IIIのEpisode等、正式名称または内部単位が現象より先に出る。
2. Series landingのcourse cardが、見たい現象より技術用語と構成を比較材料にしている。
3. Series landingの所要時間表示が、好奇心と観察の流れより完了時間を先に意識させる可能性がある。
4. IとIIIの`結果を見たあとに`と書かれた詳細buttonは初回から押せるため、表示文と実際の利用可能状態が一致しない。
5. IIではTokenを先に使っており、「画面で選ばれた一まとまりをTokenと呼ぶ」というLevel 2への短い橋渡しが不足している。

#### Low

1. 英語のeyebrowや専門語の英日混在は、主操作の理解を止めないが、初心者向け情報階層の一貫性を弱める。
2. IとIIIの共通Quick Start文言は統一感がある一方、各module固有の次の疑問を十分に表していない。

### 改修前に維持するもの

- 外部API、学習済みmodel、ML libraryを使わない実計算
- Iの139 snapshot、IIのclone保存Trace、IIIのRL因果timeline
- Pause、再開、速度変更と実stateの一致
- IIの候補、選択、追加、再予測からTraining比較への連続導線
- IIIの報酬と餌取得を分ける正確な評価
- 専門用語と数式へ進める詳細観察機能

### Phase 2へ渡す最小変更候補

1. IIIの初心者向け表面に、同じまたは比較可能な観測に対する行動候補の学習前後を追加し、Rewardから次の選択変化までを閉じる。
2. IとIIIの完了地点に、直前の結果から生まれる短い問いと、それを確かめる一つの次操作を置く。
3. Iの初心者向け表示から行動class名を中立的な予測対象へ置き換える案を、保存形式と詳細表示を維持したdisplay-only変更として検討する。
4. IとIIIの完了表面では主要な一変化だけをLevel 1に残し、Loss、Parameter、診断、全履歴をLevel 2 / 3へ分ける。
5. IIは構造を維持し、現象後にTokenという名前を与える短いLevel 2表示と、初期statusの専門語抑制を検討する。

これらはPhase 1の所見であり、まだUI変更の決定ではない。特にIIIの比較方法とIのclass表示変更は、実計算との対応、比較可能性、既存test、表示互換性を確認してからPhase 2で具体案を決める。

## Phase 3 — Ultra-Beginner First UX 実装

Phase 1のAuditと承認済み具体案に基づき、初心者向け完了面を次の共通形へ変更した。

```text
観察した変化
↓
短い因果説明
↓
現象の正式名称
↓
いま見た現象と直接つながる次の問い
↓
その場で試せる次操作
```

### Glassbox AI I

- `139ステップ`をLevel 1の進捗から外し、`予測を作る → 間違いを手がかりに直す → 同じ問題をもう一度予測する`という意味単位で表示する。
- 完了面では正解classをdisplay-onlyの`答えA / B / C`として表示し、同じ問題の正解確率を学習前後で比較する。
- `もう1回直して比べる`は、現在の学習済みnetworkと同じ入力・正解から、同じ実計算timelineをもう一度開始する。
- `重み（Weight）`は結果を観察した後に導入し、Loss、Gradient、39 Parameterは詳細表示へ残す。

### Glassbox AI II

- 主button、進捗、候補見出しでは最初からTokenの理解を要求せず、`5回`、`次に続きそうな候補`、`今回選ばれたもの`と表示する。
- 最初の選択を観察した後、その選択単位を`Token（トークン）`と呼ぶこと、単語だけでなく記号の場合もあることをLevel 2で示す。
- Parameters、Vocabulary、Training stepは初期statusから設定details内へ移し、主現象より先に読ませない。
- 既存の候補Probability、選択、文末追加、再予測、500回Training比較は維持する。

### Glassbox AI III

- 10回の最初のepisodeにおける最初のworldとsensor入力をclone保存し、その時点の方策確率を記録する。
- 10回完了後、更新後networkへ保存済みの同じ入力を渡して方策確率を再計算する。異なる場面の成績比較や、単調な改善の主張には置き換えない。
- 完了面では餌、衝突、危険接触の実集計と、前進・左折・右折の学習前後Probabilityを表示する。
- `さらに10回試して比べる`は、現在のnetworkを新しい比較元として同じ実験を続ける。`1回分を止めながら見る`は既存の詳細timelineへ接続する。

### Series landing

- 所要時間を選択材料から外し、I / II / IIIを技術名ではなく`予測はどう上手になる？`、`文章はどう続く？`、`良い・悪い結果から行動はどう変わる？`という問いで選べるようにした。
- model構造と専門用語は各cardのdetailsへ分離し、初心者が読まなくても実験を開始できる。

## 2026-08-27 — Beginner Terminology Bridge Phase 1 Audit

### 監査目的

第三者初心者test 2で確認された「動きは分かる。一度説明されれば分かる。しかし専門用語が分からない」という課題を受け、現行I / II / IIIの専門用語が、画面上の実物、DOM、実Traceまたはmodel stateへどこまで接続しているかを監査した。

今回の評価単位は用語の定義文だけではない。次が一続きになっているかを確認した。

```text
用語
↓
初心者が既に観察した現象
↓
画面上の実物
↓
識別可能なDOM
↓
実際の値を供給するTrace / model state
↓
同じ箇所をもう一度観察する操作
```

UI、DOM、URL、Trace、数学処理は変更していない。

### 実施範囲

- I / II / IIIの`index.html`、描画code、step engine、Trace構築、Auto Runを静的確認
- 各appの初期表示、Auto Run中、完了後、詳細表示への復帰を実browserで確認
- 用語説明から対象へ直接移動するlink、button、`data-term`等の有無を確認
- 現在対象を示すclass、data属性、DOM IDと実値sourceを照合
- Console warning / errorを確認

実browserでは、三appともConsole warning / errorは0件だった。

### シリーズ共通結果

| 評価項目 | 現状 | 判定 |
|---|---|---|
| 現象を先に見せる入口 | I / II / IIIすべてに存在する | 良好 |
| 現象後の正式名称 | IはWeight、IIはToken、IIIはPolicyを導入する | 部分達成 |
| 用語の短い説明 | 各appに静的な用語補足がある | 部分達成 |
| 用語から該当実物への直接移動 | 三appとも該当link / button / `data-term`は0件 | 未達 |
| 対象DOMの識別 | Iのnode / connection / Parameter、IIのstage / Token / Attention cell / Parameter、IIIの因果rail / 経験行 / Parameterは識別可能 | 良好な基盤あり |
| 実値source | Iは完全snapshot、IIはclone保存Forward Trace、IIIはRL step snapshotとcloneした比較入力 | 良好 |
| 用語を知った後の同一箇所再観察 | 詳細先頭へ戻る導線はあるが、導入した用語の対象へは直接着地しない | 未達 |
| URL / focusによる直接表示 | `path`は粗い入口切替だけで、用語・stage・対象を指定できない | 未達 |

三appとも「説明に使える実値とDOM」は既に多く存在する。主な欠損は新しい計算や可視化ではなく、用語と既存対象を結ぶmetadata、直接移動、focus、highlightの接着層である。

### Glassbox AI I — 表示用語と実物・DOM・Trace

#### 対応表

| 用語 | 現在の実物 | DOM / 描画経路 | 実値source | 現状評価 |
|---|---|---|---|---|
| 入力 | 5本のslider、入力node I1〜I5 | `#input-controls`、`#network-svg [data-node="input-N"]` | `StepEngine.current.forward.inputs` | 対応済み |
| node | SVG上のI / H / O円 | `[data-node="input-N|hidden-N|output-N"]` | current snapshotの`forward` | 実物は明確だが用語補足なし |
| 重み / Weight | node間の線、太さ、符号、正確な値 | `.network-connection[data-parameter]`、`#connection-inspector`、`#parameter-{name}` | snapshotの`network`と`PARAMETER_SPECS` | 最も強い対応基盤あり |
| Bias | Parameter表のBias行、計算式 | `#parameter-b_H*` / `#parameter-b_O*`、`#formula-display` | `network.biasH / biasO`とsnapshotの`parameterInfo` | network図上の実物はない |
| 重み付き和 | H nodeの`z`と段階式 | `[data-node="hidden-N"]`、`#formula-display` | `forward.hiddenPreActivations` | stage中は対応する |
| Activation / tanh | H nodeの`a`と段階式 | `[data-node="hidden-N"]`、`#natural-explanation` | `forward.hiddenActivations` | stage中は対応する |
| logit | O nodeの`logit` | `[data-node="output-N"]` | `forward.logits` | 常時実値を確認可能 |
| softmax / Probability | O nodeの`p`と出力summary | `[data-node="output-N"]`、`#output-summary` | `forward.probabilities` | 実物は明確 |
| Loss / 誤差 | Loss stageの式と学習前後比較 | `#step-title`、`#formula-display`、`#comparison` | `training.loss / comparison` | 実値はあるが用語から直接行けない |
| Gradient | 現在の接続、Parameter行、式 | `.network-connection.active`、`#parameter-{name}.active`、`#formula-display` | `training.parameterInfo[name].gradient` | stage中の対応は非常に強い |
| Backpropagation | 出力側からH側へ戻る一連のstage | `#step-title`、active node / connection | 139 step snapshotの`active`とtraining値 | 単一の対象ではなく区間指定が必要 |
| Learning Rate | 設定inputと更新式 | `#learning-rate`、`#formula-display` | input値と`parameterInfo.learningRate` | 実物はあるが現象前から正式語で表示 |
| Parameter Update | 全39行のbefore / gradient / update / after | `#parameter-body` | snapshotの`parameterInfo` | 強い対応あり |
| Epoch | 静的用語補足だけ | `#glossary dt/dd` | 対応する現行timeline stateなし | 現行体験との対応なし |

#### 実browserで確認した状態

- 自動観察中の`中間ノードH2への積 3/5`では、`w_I3_H2`の接続線、H2 node、`parameter-w_I3_H2`行が同時にactiveになった。
- 同じ時点で実値を代入した`x3 × w_I3_H2`と、「重みは接続ごとの影響の向きと強さ」という自然言語説明が表示された。
- 完了後はWeightという名称を導入し、同じ入力の正解確率`33.4614% → 38.1488%`を表示した。
- `どの数字を変えたか見る`を押すとtimeline先頭の`計算開始前`へ戻り、focusは`#step-next`へ移る。Weight接続、更新stage、Parameter行へ直接着地せず、active connection / Parameterは0件だった。

#### 良い部分

- `active.connection`、`active.node`、`active.parameter`が一つのsnapshotにあり、数式、SVG、Parameter表を同じ実計算stageへ同期できている。
- Weight、Gradient、Updateは新しい可視化を作らなくても、既存DOMを対象指定するだけで用語から直接示せる。
- 接続線はkeyboard focus可能で、`data-parameter`と正確な値を持つ。

#### 欠損

- 静的用語補足の`dt / dd`には対象stage、DOM selector、初心者向け一言、再観察操作のmetadataがない。
- Weight導入後のbuttonは「詳細の先頭」へ戻るだけで、直前に名付けた線を示さない。
- BiasはParameter表と式には存在するが、network図上で「nodeへ最後に足す数字」として指せる実物がない。
- `Epoch`は現行単一sample体験の実物やTrace位置と対応せず、今回の「実装している現象だけを用語化する」方針に合わない。

### Glassbox AI II — 表示用語と実物・DOM・Trace

#### 対応表

| 用語 | 現在の実物 | DOM / 描画経路 | 実値source | 現状評価 |
|---|---|---|---|---|
| Token | 文中の分割単位、Token cell、ID / POS | `#beginner-sentence .beginner-token`、`#token-cells [data-token-index]` | `trace.tokens`とTokenizer | 初心者面と詳細面の両方に実物あり |
| Probability Distribution | 候補barと百分率 | `#beginner-candidates .candidate-row`、`#prediction-view` | `trace.logits`から`generationDistribution()`で算出 | 強い対応あり |
| Selection / Sampling | 選択orb、候補row、説明 | `#beginner-selected-token`、`.candidate-row.selected` | `selectGenerationToken()`の実選択 | 現象は明確、正式名称の橋渡しはない |
| Temperature | 設定inputとSampling時の分布 | `#temperature` | `generationDistribution(logits, temperature)` | 実値対応はあるが比較導線なし |
| Embedding | 8次元vector | pipeline stage 2〜4、`#stage-view` | `trace.tokenEmbeddings / positionEmbeddings / initialRepresentation` | Trace対応済み |
| Attention | matrix、選択cell、Q / K / weight分解 | `[data-attention-row][data-attention-col]`、`#attention-inspector` | `trace.heads[*]` | 最も強い詳細対応 |
| Causal Mask | Matrixの`MASK` cell | `#attention-matrix td.mask` | `trace.heads[*].maskedScores` | 実物と実値が一致 |
| Residual / MLP / LayerNorm | stage vectorと全体図node | `#stage-view`、`#architecture-view .architecture-node` | 対応するTrace tensor | stage対応あり |
| Logits / Softmax | stage 15 / 16のvector、候補確率 | `#pipeline .pipeline-step`、`#stage-view`、`#prediction-view` | `trace.logits / probabilities` | 実値対応済み |
| Loss | 位置別表、履歴graph、前後比較 | `#training-sample`、`#loss-chart`、`#training-comparison` | Trainerのbefore / after Traceとloss history | 実値対応済み |
| Gradient / SGD | Training flow、Parameter Inspector | `[data-training-phase]`、`#parameter-table [data-parameter]`、`#parameter-inspector` | autograd gradientと`trainer.lastUpdate` | 実値対応済み |

#### 実browserで確認した状態

- Auto Run中の候補表示では、実Traceの最終logitから作られた`dog 10.69%`、`<UNK> 8.53%`等がbarと数値で表示された。
- 同時に16 stage pipelineは`Probabilities`、全体図は`Softmax → Next Token`をcurrent表示した。
- 最初の選択後にToken名称が表示され、5回完了後も生成文、候補、実値sourceの注記を確認できた。
- `1ステップずつ詳しく見る`は同じ生成Traceを保持したまま`Tokenizer`へ戻り、`<BOS> → 0`等の実Token IDを表示した。

#### 良い部分

- 16 stageのkeyとclone保存Traceのfieldがほぼ一対一で、用語から正確なstageへ移動する基盤がある。
- Token cell、Attention cell、Parameter rowは個別のdata属性を持ち、対象を細かく指定できる。
- 初心者面の候補確率と詳細面のProbability stageが同じforward結果へ接続している。
- Tokenは現象後に名前を導入し、詳細復帰時にTokenizerとToken cellへ着地するため、三app中で用語と再観察の接続が最も進んでいる。

#### 欠損

- 用語補足はArchitecture tab内の静的listで、各stage、Attention tab、Parameter tabへ移動できない。
- 詳細導線は常にTokenizerへ戻るため、Token以外の用語を学んだ後に該当stageへ直接行けない。
- pipeline、tab、見出し、設定の多くが英語専門語のままで、AI名称以外を知らない利用者には「いま何が起きているか」より名称が先に見える。
- `Temperature`と`Sampling`は実装されているが用語補足に含まれず、値を変えると候補分布と選択がどう変わるかを同じ入力で比較する橋がない。
- Architectureのcurrent強調は一部の代表keyだけで、Token Embedding、Position Embedding、Raw Score、Mask、Attention Weight等の全16 stageとは完全一致しない。

### Glassbox AI III — 表示用語と実物・DOM・Trace

#### 対応表

| 用語 | 現在の実物 | DOM / 描画経路 | 実値source | 現状評価 |
|---|---|---|---|---|
| 観測 / 入力 | Grid Worldと5 sensor値 | `#rl-grid-board`、observation stageの`#rl-formula` | RL stepの`details.inputs / sensed` | 実値対応済み |
| Policy / 方策 | 3行動の確率bar、学習前後比較 | `#rl-policy-bars .rl-policy-row`、`#rl-beginner-policy-comparison` | `details.policy`、clone入力による`reference.before / after` | 初心者面の中心として良好 |
| Reward / 報酬 | 移動結果、単発値、累積値 | active経験行、`#rl-formula`、`#rl-cumulative-reward` | `experience.reward / cumulativeReward` | 因果が見える |
| Exploration / Exploitation | 選択区間、経験行、count | `#rl-formula`、`#rl-experience-body tr.active` | `samplePolicy()`とseed乱数 | 実値対応済み |
| Episode | 1回のworld履歴と履歴graph | `#rl-episode-number`、`#rl-history-chart` | `ReinforcementStepEngine.summary` | 実物はあるが初回から正式語が多い |
| Discounted Return | 逆向き計算と経験表G列 | `data-rl-axis="return"`、`#rl-formula`、経験表 | `step.returns` | 実値対応済み |
| Policy Gradient | Loss、output delta、平均勾配 | `data-rl-axis="gradient"`、`#rl-formula`、RL Parameter表 | `details.gradient / aggregateGradient` | 実値対応済み |
| REINFORCE | Returnから方策勾配、39更新までの区間 | 因果rail 5〜8、step title / formula | RL timeline全体 | 単一のDOM対象はなく区間指定が必要 |
| Parameter Update | active Parameter行とbefore / gradient / update / after | `#rl-parameter-{name}.active` | `step.parameterInfo`とstep network snapshot | 表では強く対応 |
| Weight | network接続線とRL Parameter行 | `.network-connection[data-parameter]`、`#rl-parameter-{name}` | network state | RL stage中のnetwork線activeとは未接続 |

#### 実browserで確認した状態

- Auto Run中の`rl-transition`では因果railの`遷移・報酬`と経験表の1行がactiveになり、環境event、単発報酬、累積報酬を同じstepから表示した。
- 同じ時点で3行動の実方策確率をbar表示した。
- 10回完了後は同じclone入力に対する方策を`前進 36.0% → 38.0%`等で比較し、その現象をPolicyと名付けた。
- `1回分を止めながら見る`はepisode先頭の`rl-ready`へ戻る。Policy barは空になり、Policy stageや比較対象へ直接着地しない。

#### 良い部分

- `stage`と`data-rl-axis`が観測、方策、抽選、遷移・報酬、Return、Gradient、Update、比較へ対応している。
- `active.experienceIndex`により、説明、式、世界、経験表を同じ時刻へ合わせられる。
- beginner比較は異なる場面を比較せず、保存済みの同じ入力を更新前後networkへ渡した実Probabilityである。
- Reward、餌取得、衝突、危険を別々に表示し、報酬増加を能力向上と断定しない。

#### 欠損

- canonicalなIIIの用語補足がIと同じ教師あり用語listである。Policy、Reward、Episode、Exploration、Return、REINFORCEがなく、代わりにLoss Function、Backpropagation、Epoch等が並ぶ。
- 初心者向けにPolicyを導入した直後の詳細buttonが、Policyの実物ではなく確率未表示の`rl-ready`へ戻る。
- RL timelineの`active`は経験行とParameter行には接続するが、network SVGのnode / connection強調へ接続しない。`rl-update`も`active.parameter`だけで、同名の接続線をactiveにしない。
- 因果railと設定欄に、方策確率、探索／活用、割引Return、方策Gradient、REINFORCE、温度、学習率が一度に現れる。現象後の段階開示になっていない。
- `rl-phase`は初心者向け名称ではなく`rl-transition`等の内部stage keyをそのまま表示する。

### 優先度

#### Critical

1. **三appとも、用語説明から該当実物へ直接戻る導線がない。** `Glassbox AIで見る`に相当するlink / button / term metadataは0件であり、採用済みの`現象 → 名前 → 実物と対応 → もう一度触る`loopが閉じていない。
2. **Glassbox AI IIIの用語補足がcanonicalな強化学習内容と一致していない。** Policy、Reward、Return、REINFORCE等を説明せず、教師あり学習用のLoss、Backpropagation、Epochを表示するため、初心者が現在の実物と用語を対応できない。

#### High

1. **IのWeight導入後とIIIのPolicy導入後が、名付けた対象ではなくtimeline先頭へ戻る。** 直前の理解を再観察で確かめられない。
2. **IIIのRL stageとnetwork図のactive状態が接続していない。** RL Parameter行は強調できるが、同じWeightの線や関連nodeを同時に示せない。
3. **IIはTrace対応が強い一方、Token以外の用語から16 stage、Attention cell、Parameterへ直接移動できない。** 静的用語補足と実物が分離している。
4. **詳細領域では用語名が自然言語より先に大量表示される。** 特にIIの英語pipeline / tab、IIIの設定と因果railは、対象利用者が用語を知らない前提と衝突する。

#### Medium

1. IのBiasは実値と式があるが、network図上で指せる表示対象がない。
2. IのEpochは現行単一sample体験に対応する実物がない。
3. IIのTemperature / Samplingは実装済みだが用語補足と同一入力比較がない。
4. IIのArchitecture current表示は16 stageすべてと一対一ではない。
5. 三appとも用語、初心者向け説明、stage、selector、focus先を一元管理するmetadataがなく、文言と対象の対応が複数fileへ分散している。

#### Low

1. `node / Node`、`Probability / 確率`、`Parameter / パラメータ`等の英日表記が画面間で揺れている。
2. DOM IDは十分に存在するが、SVG node groupなど一部の観察対象はfocus対象ではない。

### Phase 2へ渡す最小変更候補

全面的な用語page新設より先に、既存DOMとTraceを使う小さな接着層を優先する。

1. appごとに、主要用語の`term id / 短い説明 / stage / selector / focus先 / 詳細説明`を持つ小さなmetadataを定義する。
2. 現行の用語補足をbuttonまたはlink化し、`Glassbox AIで見る`で既存timelineを該当stageへ移動し、既存active classまたは専用の一時highlightを付ける。
3. Iは最初にWeightを対象とし、実際の更新済みsnapshot内の一接続、同名Parameter行、更新式を同時に示す。Epochは主要導線から外す候補とする。
4. IIはToken、Probability、Attentionを最初の三用語とし、それぞれTokenizer / Probabilities / Attention Weights stageへ接続する。現在の`ForwardStepEngine.index`とclone保存Traceをそのまま使う。
5. IIIは用語補足をcanonical RL用語へ同期し、Policy、Reward、Returnを最初の三用語とする。Policyは`rl-policy`、Rewardは`rl-transition`、Returnは`rl-return`の実stepへ接続する。
6. IIIの`rl-update`では`active.parameter`からWeightなら対応するnetwork connection、Biasなら対応nodeを導けるようにする。ただし表示sourceは既存RL step snapshotのままとする。
7. URL queryを追加する場合も、`term=weight`だけでdummy表示を作らず、通常操作で生成した同じTrace / timeline位置を再現してからfocusする。

これらは監査結果に基づく具体案であり、まだUI実装の決定ではない。Phase 2では、既存の用語補足を文脈内panelとして拡張する案を第一候補とし、独立pageやrouting追加が本当に必要かを比較してから提示する。

## 2026-08-27 — Priority A 実装結果

Phase 1 AuditのCritical / Highから、既存Traceを壊さず用語と実物を結ぶ変更を実装した。

- シリーズ共通：用語カードをLevel 1〜3に分け、短い説明、画面上の場所、「Glassbox AIで見る」を同じ文脈に配置した。
- I：Weightを含む主要用語から既存`StepEngine` snapshotへ移動し、数式、SVG接続、同名Parameter行を同時に強調する。現行Traceに実物がないEpochは主要用語から外した。
- II：Token、Probability、Attention等をclone保存済みForward Traceの該当stageへ接続した。未実行時に実値のないLoss / Gradientは空表示へ移動させず、「学習を1回試した後」と示す。
- III：教師あり用語集をRL専用のObservation、Policy、Reward、Sampling、Return、Policy Gradient、Learning Rateへ置き換えた。内部stage keyは初心者向けの動詞句で表示する。
- III：ネットワーク図と出力確率を、教師あり側のcurrent snapshotではなく表示中のRL step snapshotへ同期した。`rl-update`では同じParameterの表行とWeight接続またはBias nodeを同時にactive表示する。
- 数学処理：損失、確率、勾配、更新量、方策、報酬の算出式や値は変更していない。直接移動APIは既存の保存済みstep indexを選ぶだけで、説明専用のdummy値を作らない。

### 検証

- root `npm run check`: portal 11、I 44、II 24、III 48、合計127 test pass。
- 実browser：IのWeightは`w_I1_H1`更新stage、実更新式、active接続、active Parameter行へ着地した。
- 実browser：IIのToken / Probability / AttentionはTokenizer / Probabilities / Attention Softmaxへ着地し、保存済み候補確率とAttention matrixを表示した。
- 実browser：IIIのPolicy / Reward / Return / Learning Rateは対応するRL stageへ着地し、実方策、単発報酬、割引Return、更新式を表示した。Weight更新時はRL network接続とRL Parameter行が同時にactiveになった。
- 390×844で三appともdocument横overflowは0。Console warning / errorは0件。

## 2026-09-05 — 観察と用語を往復する導線

既存の「体験 → 観察 → 因果関係の理解 → 名前 → 詳細」を維持し、次を追加した。

- 三appに、初回体験・現在の計算・用語へ移動する短いnavigationを追加。用語を調べた後は、元のsnapshot位置と操作元のfocusへ戻れる。自動再生は勝手に再開しない。新しい実験へ切り替えた場合は古い復帰を適用しない。
- I / IIIでは現在の段階の用語を同じ計算欄で開ける。次の対象は、実timelineの次のstepから表示する。
- Iの通常画面は答えA / B / Cに統一。入力は5個の数字、正解は人が指定することを説明する。保存JSONの名前、出力index、計算値、legacy実装は変えない。
- IIの全16段階に、専門用語の知識を前提としない観察文と次の操作を追加。Loss / Gradientは未学習なら学習操作へ、学習済みなら実際の結果へ案内する。
- IIのParameter cellはマウスhoverだけに頼らず、クリック・Enterで更新前、制限前の勾配、実際の変更量、更新後を表示する。保存されたupdateを使い、未学習の変更量を捏造しない。後から学習率を変えても過去の更新式は変わらない。
- IIの再初期化 / JSON読込時は、古い学習比較と用語復帰を残さない。文章生成中に用語を調べた場合は、戻る操作で生成途中の文も復帰する。
- IIIのReward / Returnは、360px幅でも横長の行全体ではなく実際の数値cellへfocusする。
- 各appの最後に、今回の学習信号、解釈上の限界、次の実験への問いを追加。series linkはapp単独起動でも使える公開URLとした。外部script / API依存はない。

### 評価の範囲

この変更の検証対象は、実際に動く導線、実数値との一致、復帰、表示、keyboard操作である。初心者本人の理解度・学習効果を保証するものではない。第三者testでは説明せずに触ってもらい、(1)何を変えたか、(2)どの数字が変わったか、(3)その名前は何か、(4)次にどこを試すか、を本人の言葉と画面上の指示で確認する。全機能の暗記や一度での完全理解を要求しない。

### 2026-09-05 検証結果

- `npm run check`: Portal 11、I 44、II 25、III 48、合計128/128成功。I/IIIのcoverageコマンドも成功（計測対象の行coverageはI 92.75%、III 92.57%。UI全操作のcoverageを意味しない）。
- 実browserでIの12用語、IIの7用語、IIIの7用語について、対象focus/強調と元の場面への復帰を確認。IIの用語buttonはEnter往復後も元の用語へfocusが戻る。
- Iの1回学習：同じ問題の答えA確率33.4614% → 38.1488%。IIIの10回：同じ開始場面で前進36.0% → 38.0%、左折32.6% → 31.3%、右折31.3% → 30.7%。成功を演出せず、実結果を表示。
- IIの生成：`the cat eats the dog cats milk the`。500回学習後の固定問いは`the cat eats fish.`。fish確率7.59% → 19.81%、Corpus平均Loss3.3856 → 1.2615。生成と500回学習の一時停止・再開を確認。
- IIは16段階すべての観察文と次操作を確認。生成途中のToken参照から戻ると、生成文・停止状態・元buttonのfocusが復帰する。Attentionはfocus先cellとInspectorの行/列が一致する。
- IIの実更新表示：embeddings.token[0]は更新前-0.1665、制限前勾配0.3214、実変更量-0.0040、更新後-0.1705。未学習の誘導、学習後のLoss/Gradient、再初期化後に古い比較が残らないことを確認。
- 360×800で三appのdocument横overflowは0。IIIはReturn実数値cellが画面内に入り、IIは更新式全体と復帰buttonを目視確認。I/IIIの隣の用語へのTab移動とoutlineも確認。
- Console warning/errorは検証tabで0件。初期の旧キャッシュ参照を検出し、最新ソースをno-storeの専用ローカルserverで再検証した。
- 第三者初心者本人の理解度、全Parameter/全matrix cellを含む全Tab経路、全支援技術の組合せは未評価。既存画像/GIFは以前の撮影時点のままであり、今回の画面差分の証拠とはしない。
