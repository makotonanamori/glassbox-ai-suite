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
