# ContinualLearning2026
継続学習用のプログラムです．
表現学習を強化することで，破滅的忘却に対して頑健な特徴表現を獲得する継続学習手法（[Co2L](https://arxiv.org/pdf/2106.14413)，[CCLIS](https://arxiv.org/pdf/2403.04599)，[ProNC](https://openreview.net/pdf?id=E3bBZ02Qcc) など）を改善する方向性を見つけるために色々分析する予定です．

# 研究のモチベーション
[Co2L](https://arxiv.org/pdf/2106.14413)，[CCLIS](https://arxiv.org/pdf/2403.04599)などの既存研究は，[教師あり対照損失(SupCon)](https://arxiv.org/pdf/2004.11362)で獲得した特徴表現は，破滅的忘却に対して頑健であることから，継続学習に教師あり対照損失を導入することで性能向上を図った．
しかし，既存研究（[Co2L](https://arxiv.org/pdf/2106.14413)，[CCLIS](https://arxiv.org/pdf/2403.04599)，[ProNC](https://openreview.net/pdf?id=E3bBZ02Qcc)）の問題点として，リプレイバッファにどのようなデータを保持するか，保持したデータをどのように活用するかについては十分に議論されていないという点が挙げられる．
これらの研究は，Asymmetric SupCon損失などの導入といった，損失関数を工夫することで，モデルが持つ特徴表現をより頑健にすることに注力しいている一方で，「リプレイバッファ内にどのデータを保持するのが有効化か」「リプレイバッファ内のデータをどのように活用するか」に対する議論はあまりされていない．
[CCLIS](https://arxiv.org/pdf/2403.04599)は，対照学習においてハードネガティブが重要であることに注目し，リプレイバッファにハードネガティブのサンプルを優先的に保持しているが，ここでも深い議論はされておらず，他指標でのサンプルの保持との比較実験はされていない．

そこで本研究では，「リプレイバッファ内にどのデータを保持するのが有効化か」「リプレイバッファ内のデータをどのように活用するか」に焦点を当て，表現学習の向上に有効なリプレイ戦略を調査する．


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
## デバッグ（訓練）
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

## デバッグ（線形分類による評価）
タスクを指定し評価を行います．
評価の設定は，学習に使用した`.yaml`ファイルの`linear`から設定できます．
- debug supcon:
    ```
    python main_linear.py --config-path ./configs/default/supcon --config-name debug
    ```

- debug proto_supcon:
    ```
    python main_linear.py --config-path ./configs/default/proto_supcon --config-name debug
    ```

## デバッグ（分析）


