# Security Policy

Glassbox AI Suiteは、外部通信、ユーザーアカウント、クラウド保存、遠隔測定を持たない静的Webアプリです。小さな教材でも、Import、local hosting、GitHub Pagesの境界を明示して扱います。

## Supported versions

| 対象 | Security update |
| --- | --- |
| `main`および現在のGitHub Pages | 対応 |
| 過去のcommit、local copy、第三者改変版 | 個別対応なし |

## Report a vulnerability

[GitHub Private vulnerability reporting](https://github.com/makotonanamori/glassbox-ai-suite/security/advisories/new)を正式窓口として使用します。

悪用可能な詳細、未公開の攻撃手順、個人情報を公開Issueへ投稿しないでください。通常の表示・計算上の不具合はBug reportを使用できます。

報告には可能な範囲で次を含めてください。

- 影響するURL、application、commitまたはversion
- 再現条件と最小の再現手順
- 期待結果、実際の結果、考えられる影響
- browser、OS、local / GitHub Pagesの別
- 安全に共有できるLogや検証用ファイル

受領後は内容を非公開で確認し、影響範囲、修正、公開時期を報告者と調整します。初回応答は7日以内を目標としますが、保証されたSLAではありません。

## Trust boundaries

- Importするnetwork JSONとmodel JSONを信頼しない入力として検証します。
- JSON内の文字列をHTMLとして実行せず、textとして表示します。
- 外部CDN、外部AI API、remote analyticsを使用しません。
- Local launcherは`127.0.0.1`だけへbindします。
- 公開時は`glassbox-ai-suite`だけをWeb rootにし、親workspaceを公開しません。
- Screenshot / GIFは公開版の画面だけを含み、user dataやlocal pathを含めません。

## Out of scope

- 古いcopyまたは第三者が変更したhosting環境だけで起きる問題
- browser本体、GitHub、Python HTTP server自体の脆弱性
- 教材モデルの低精度や学習性能をSecurity vulnerabilityとする報告

これらに見えてもsuite固有の安全上の影響がある場合は、Private vulnerability reportingから相談できます。
