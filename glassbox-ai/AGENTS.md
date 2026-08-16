# Glassbox AI I / III — Project Instructions

このprojectで作業する前に、次をすべて読むこと。

1. `../AGENTS.md` — workspace全体の安全・記録ルール
2. `../AGENTS-Glassbox.md` — Glassboxシリーズ共通方針
3. このファイル — `glassbox-ai`固有の実装境界

## 現在のscope

このfolderは単一のローカルbrowserアプリとして、次を含む。

- 5入力→4中間→3出力のニューラルネット
- tanh、softmax、cross entropy、全39 Parameterのbackpropagation / SGD
- 7×7 Grid World、安全教師、5 sensor
- 温度付き確率方策、探索/活用、Reward、割引Return、REINFORCE
- 教師あり139 step数学timelineとは独立したRL因果timeline
- Network state JSON、RL history JSON、操作log

第三弾は実装済みである。Q-learning、価値関数、Actor-Critic、経験再生へ無断で置き換えない。

## 変更境界

- `glassbox-ai-ii`のTransformer codeをこのprojectへ移さない。
- 教師あり139 stepとRL timelineを一つのtimelineへ統合しない。
- UI表示は実計算値または完全snapshotだけを使用する。
- 前後step移動でworld、network、部分updateのstateを一致させる。
- 既存39 Parameterの命名、shape、保存形式を変更する場合は互換性案を先に示す。
- 外部API、学習済みmodel、ML library、外部CDNを追加しない。

## 記録

変更理由、教材上の影響、検証結果、残る制限をIssue、PR、READMEのいずれかへ追跡可能に残す。

## 検証

```powershell
npm run check
npm run test:coverage
```

UI変更時は専用起動経路を使用する。

```powershell
.\start-glassbox-ai.cmd
```

実browserで対象操作、計算値、前後step、narrow width、Console warning/errorを確認する。数学変更では解析勾配と中央差分、step実行と一括実行の一致を検証する。

## 現在の優先事項

- 初見利用者向けQuick Startと教材導線
- 既存数学の正確性、再現性、accessibility
- Iの教師あり学習とIIIの環境報酬学習の違いを明示
- MIT Licenseと第三者素材境界の維持

大規模なfolder再編、RL分離、shared core新設は独立提案とし、承認前に開始しない。
