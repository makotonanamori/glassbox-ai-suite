# Glassbox AI Suite — Repository Instructions

このrepositoryでは、次の順序で指示を読む。

1. この`AGENTS.md`
2. `AGENTS-Glassbox.md`
3. 対象app直下の`AGENTS.md`

## Repository境界

- Glassbox AI I = `glassbox-ai` = 教師あり5→4→3ニューラルネットと139 step数学timeline。
- Glassbox AI II = `glassbox-ai-ii` = Decoder-only Transformer言語モデルと16 stage Forward Trace。
- Glassbox AI III = `glassbox-ai-iii` = Grid World、環境報酬、REINFORCE専用timeline。
- I / IIIに残る分離前の重複実装はlegacy compatibility layerである。通常UIでは隠し、新機能の正本にしない。
- 三つのappを一つのmodelやtimelineへ統合しない。
- Series landingはnavigationだけを担当し、表示用model計算を持たない。
- Series landingの順序と番号はI → II → IIIに固定する。
- 外部API、学習済みmodel、ML library、外部CDN、telemetryを追加しない。

## 変更と検証

- 既存数学、Trace、snapshot、serializationの互換性を保つ。
- 大規模なdirectory再編やshared core抽出は先に提案する。
- 変更理由と検証結果はIssue、PR、またはREADMEへ追跡可能に残す。
- 完了前にrootで`npm run check`を実行する。
- UI変更では実browser、360px幅、keyboard focus、Console warning/errorを確認する。

## 公開

- 公開rootはこのdirectoryだけに限定する。
- LicenseはrootのMIT Licenseを正本とする。
- 実在しない利用実績、release、security窓口、公開URLを記載しない。
