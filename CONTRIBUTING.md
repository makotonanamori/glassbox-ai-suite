# Glassbox AI Suiteへの貢献

Glassbox AIは、結果だけでなく「その数値がどの入力、Parameter、数式から来たか」を追えることを最優先にする教育用実験装置です。

## 開発環境

- Node.js 20以降
- Python 3
- 現行ブラウザ
- 外部packageのinstallは不要

```powershell
npm run check
.\start-glassbox-ai-suite.cmd
```

## 変更原則

1. ニューラルネット、Transformer、autograd、方策勾配は自前実装を維持する。
2. 外部AI API、学習済みモデル、MLライブラリ、外部CDNを追加しない。
3. UIは実計算値またはclone保存したTrace / snapshotだけを表示する。
4. 教師あり139ステップとRL環境・報酬タイムラインを統合しない。
5. Transformer UIは`model.forward()`のTraceを唯一の表示源とする。
6. 数学変更には手計算または有限差分による検証を追加する。
7. 観測値と、人間による意味解釈を区別する。

## Pull Requestチェックリスト

- [ ] 変更理由と教材上の狙いを説明した
- [ ] Model logicとUI logicを分離した
- [ ] 表示値が実計算または保存済みTrace由来である
- [ ] 数式変更に自動テストまたは数値検証がある
- [ ] `npm run check`が成功する
- [ ] 変更した導線を実ブラウザで確認した
- [ ] 360px幅とConsole warning/errorを確認した
- [ ] READMEまたは既知の制限を更新した

## Browser確認

Series landingから対象実験へ移動し、変更対象の操作を実際に完了してください。DOMの存在だけでなく、計算後の確率、Loss、報酬、Gradient、更新後値が期待する状態へ変わることを確認します。
