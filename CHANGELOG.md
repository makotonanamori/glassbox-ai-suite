# Changelog

Glassbox AI Suiteの利用者向け変更履歴です。

形式は[Keep a Changelog](https://keepachangelog.com/ja/1.1.0/)を参考にし、version番号は[Semantic Versioning](https://semver.org/lang/ja/)に従います。

> [!NOTE]
> `1.0.0`は`package.json`とGitHub Pagesで公開中の検証済みbaselineを表します。対応するGit tagとGitHub Releaseは、第三者初心者test後のfeedbackを確認してから作成する予定であり、現時点では未作成です。

## [Unreleased]

### Added

- 検証済み公開baselineと今後の変更を追跡する、このCHANGELOGを追加。
- `docs/AUTO_OBSERVE_UX.md`へ、`ready / running / complete / detail`状態、共通DOM契約、実値一致、後続animation Phaseを記録。
- Glassbox AI IIIへ、10エピソードの実移動を時間方向に表示する連続runner、一時停止・再開、0.5 / 1 / 2 / 4倍の速度切替、進捗表示を追加。
- Glassbox AI Iへ、既存139 snapshotを順番に描画する教師あり学習runner、一時停止・再開、0.5 / 1 / 2 / 4倍の速度切替、進捗表示を追加。
- Glassbox AI IIへ、予測16段階、実Training 5段階、生成16段階、Token選択をつなぐ38段階runner、一時停止・再開、0.5 / 1 / 2 / 4倍の速度切替を追加。

### Changed

- README冒頭の導線を「ブラウザですぐ試す」→「ローカルで実行する」→「開発・テストする」に再構成し、GitHub Pagesと各教材への直接linkを追加。
- Series landingを「まず動かす → 結果を見る」入口へ変更し、I / II / IIIすべてに`自動で見る（おすすめ）`と`1ステップずつ詳しく見る`の共通二段構造を追加。
- Iの1回教師あり学習、IIの1 Training Step、IIIの1 Episodeを、既存の実計算engineを使う1クリック観察へ接続。
- IIIの`自動で見る`と`10エピソードを自動で見る`を同じ連続runnerへ統一し、実環境遷移ごとの世界表示とエピソードごとの履歴更新へ変更。
- Iの`自動で見る`を、結果へ即時移動する操作から、入力・予測・誤差・全勾配・39更新・再計算・比較を実snapshotで連続表示する操作へ変更。
- IIの`自動で見る`を、1 Training Stepの即時比較から、同じForward Trace、段階Trainer、Generationを使う予測 → 学習 → 生成の連続表示へ変更。
- 初心者testを受け、IIの最初の自動再生を内部38段階中心から、候補確率 → 1 Token選択 → 文末追加 → 反復を大きく見せる生成ループ中心へ再構成。Transformer内部と学習は、その下の`1ステップずつ詳しく見る`へ分離。

第三者初心者testで得られたfeedbackに基づく変更は、実装時にこの章へ追加します。将来計画は[ROADMAP.md](./ROADMAP.md)を参照してください。

## [1.0.0] - 2026-08-16

公開・検証済みbaseline: [`6d0ad66`](https://github.com/makotonanamori/glassbox-ai-suite/tree/6d0ad66)

### Added

- Glassbox AI I：5入力 → 4中間 → 3出力networkの教師あり学習、cross entropy、誤差逆伝播、SGD、139 step数学timeline。
- Glassbox AI II：1層Decoder-only TransformerのToken、Embedding、Causal Attention、16 stage Forward Trace、next-token loss、autograd、SGD。
- Glassbox AI III：7×7環境、環境履歴、探索と活用、累積報酬、REINFORCE、強化学習専用timeline。
- I / II / IIIの独立起動、suite landing、GitHub Pages公開、Windows / Ubuntu CI。
- Seed再現、JSON round-trip、有限差分gradient check、数値安定性test。
- MIT License、貢献手順、Security Policy、Issue forms、Pull Request template。
- Roadmap、公開実画面のScreenshot、suite overview GIF。

### Changed

- canonicalな責務をI＝教師あり学習、II＝言語モデル、III＝強化学習として確定。
- Landingの学習順と学習信号をI → II → IIIへ統一。
- Glassbox AI IIIの公開headerをREINFORCEの方策更新に一致させた。
- Glassbox AI I / IIIのREADMEでcanonical implementationとlegacy compatibility referenceを明確に分離。

### Security

- GitHub Private vulnerability reportingを有効化。
- Security Policyへ対応範囲、報告項目、response process、trust boundariesを明記。

### Verification

- 自動test：Portal 10、Glassbox AI I 41、Glassbox AI II 16、Glassbox AI III 44、合計111件成功。
- GitHub Actions：Ubuntu / Windows CI成功。
- GitHub Pages：landing、I / II / III、Screenshot、GIFの公開応答を確認。

### Baseline commit history

- [`30bdcf0`](https://github.com/makotonanamori/glassbox-ai-suite/commit/30bdcf0) — GitHub Pagesでsuiteを公開。
- [`e87b43f`](https://github.com/makotonanamori/glassbox-ai-suite/commit/e87b43f) — GitHub Pages publishingを堅牢化。
- [`5ed4ce2`](https://github.com/makotonanamori/glassbox-ai-suite/commit/5ed4ce2) — I / II / IIIのcanonicalな責務を確定。
- [`2c2392f`](https://github.com/makotonanamori/glassbox-ai-suite/commit/2c2392f) — canonicalな学習信号表記を同期。
- [`6d0ad66`](https://github.com/makotonanamori/glassbox-ai-suite/commit/6d0ad66) — Roadmap、実画面media、metadata、Security導線を追加。

[Unreleased]: https://github.com/makotonanamori/glassbox-ai-suite/compare/6d0ad66...HEAD
[1.0.0]: https://github.com/makotonanamori/glassbox-ai-suite/tree/6d0ad66
