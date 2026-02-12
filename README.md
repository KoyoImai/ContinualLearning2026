# ContinualLearning2026
継続学習用のプログラムです．
表現学習を強化することで，破滅的忘却に対して頑健な特徴表現を獲得する継続学習手法（[Co2L](https://arxiv.org/pdf/2106.14413)，[CCLIS](https://arxiv.org/pdf/2403.04599)）を改善する方向性を見つけるために色々分析する予定．

# プログラムの全体像
学習・評価に使用するプログラムの全体像は以下の通りです．
各ディレクトリの詳細については，それぞれのディレクトリのREADMEを確認してください．
```
SSOCL/
├── buffer
├── configs           : 学習・評価の設定を記述する.yamlファイルの格納場所．
├── criterions
├── dataloaders
├── main.py
├── models
├── optimizers
├── trainers
└── utils.py          : その他のモジュールを実装するutilsファイル．
```

# 実行方法
## 学習
指定したデータセットで学習を行います．
学習の設定は`configs`ディレクトリの下に`.yaml`ファイルを追加，修正することで変更できます．
`.yaml`ファイルを追加して学習する場合は，実行時の`--config-path`と`--config-name`を修正してから実行してください．
- debug supcon:
    ```
    python main.py --config-path ./configs/default/supcon --config-name debug
    ```

- debug proto_supcon:
    ```
    python main.py --config-path ./configs/default/proto_supcon --config-name debug
    ```