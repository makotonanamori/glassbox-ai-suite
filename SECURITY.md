# Security Policy

Glassbox AI Suiteは、外部通信、ユーザーアカウント、クラウド保存、遠隔測定を持たない静的Webアプリです。

## 対応対象

最新版の既定branchを対象とします。古いcopyや第三者改変版は個別サポート対象外です。

## 脆弱性の報告

公開repositoryではGitHub Private vulnerability reportingを有効にし、Security Advisoryを正式窓口として使用する予定です。窓口設定前は、悪用可能な詳細を公開Issueへ投稿しないでください。

報告には、影響するversion、再現条件、期待結果と実際の結果、考えられる影響を含めてください。

## 信頼境界

- Importするnetwork JSONとmodel JSONを信頼しない入力として検証します。
- JSON内の文字列をHTMLとして実行せず、textとして表示します。
- 外部CDN、外部API、remote analyticsを使用しません。
- Local launcherは`127.0.0.1`だけへbindします。
- 公開時は`glassbox-ai-suite`だけをWeb rootにし、親workspaceを公開しません。
