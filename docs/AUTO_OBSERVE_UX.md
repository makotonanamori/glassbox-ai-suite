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

## 後続Phase

Phase 1は共通入口と実計算への接続を担当します。時間方向の連続表示は次の順で追加します。

1. Glassbox AI III：連続Episodeと試行錯誤の変化
2. Glassbox AI I：教師あり学習の自動再生
3. Glassbox AI II：予測・学習・生成の自動再生

後続Phaseでも、このDOM契約と実値一致を維持します。
