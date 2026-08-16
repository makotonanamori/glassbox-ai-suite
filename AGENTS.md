# Glassbox AI Suite — Repository Instructions

このrepositoryでは、次の順序で指示を読む。

1. この`AGENTS.md`
2. `AGENTS-Glassbox.md`
3. 対象app直下の`AGENTS.md`

## Repository境界

- `glassbox-ai`は教師ありニューラルネットと、互換維持用の従来Grid World / REINFORCE画面を含む。
- `glassbox-ai-ii`は独立したDecoder-only Transformerで、RLを含まない。
- `glassbox-ai-iii`は独立したGrid World / REINFORCE実験で、他appをruntime参照しない。
- 三つのappを一つのmodelやtimelineへ統合しない。
- Series landingはnavigationだけを担当し、表示用model計算を持たない。
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
