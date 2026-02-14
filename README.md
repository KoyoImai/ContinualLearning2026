# ContinualLearning2026
継続学習用のプログラムです．
表現学習を強化することで，破滅的忘却に対して頑健な特徴表現を獲得する継続学習手法（[Co2L(ICCV2021)](https://arxiv.org/pdf/2106.14413)，[CCLIS(AAAI2024)](https://arxiv.org/pdf/2403.04599)，[ProNC(ICLR2026)](https://openreview.net/pdf?id=E3bBZ02Qcc) など）を改善する方向性を見つけるために色々分析する予定です．

# 研究のモチベーション
[Co2L(ICCV2021)](https://arxiv.org/pdf/2106.14413)，[CCLIS(AAAI2024)](https://arxiv.org/pdf/2403.04599)などの既存研究は，[教師あり対照損失(SupCon)](https://arxiv.org/pdf/2004.11362)で獲得した特徴表現は破滅的忘却に対して頑健であることから，継続学習に教師あり対照損失を導入することで性能向上を図った．
しかし，既存研究（[Co2L(ICCV2021)](https://arxiv.org/pdf/2106.14413)，[CCLIS(AAAI2024)](https://arxiv.org/pdf/2403.04599)，[ProNC(ICLR2026)](https://openreview.net/pdf?id=E3bBZ02Qcc)）の問題点として，リプレイバッファにどのようなデータを保持するか，保持したデータをどのように活用するかについては十分に議論されていないという点が挙げられる．
これらの研究は，Asymmetric SupCon損失などの導入といった，損失関数を工夫することで，モデルが持つ特徴表現をより頑健にすることに注力しいている一方で，「リプレイバッファ内にどのデータを保持するのが有効か」「リプレイバッファ内のデータをどのように活用するとよいか」に対する議論はされていない．
[CCLIS(AAAI2024)](https://arxiv.org/pdf/2403.04599)は，対照学習においてハードネガティブが重要であることに注目し，リプレイバッファにハードネガティブのサンプルを優先的に保持しているが，ここでも深い議論はされておらず，他指標でのサンプルの保持との比較実験はされていない．

そこで本研究では，「リプレイバッファ内にどのデータを保持するのが有効化か」「リプレイバッファ内のデータをどのように活用するか」に焦点を当て，表現学習の向上に有効なリプレイ戦略を調査する．

## 仮説
Q1：既存の研究では，モデルの忘却の多くは分類層で発生し，モデル内部の特徴表現では忘却が発生しにくいと述べられている([論文(CVPR2022)](https://arxiv.org/pdf/2203.13381))にも関わらず，破滅的忘却は依然として発生し，モデルの性能を低下させている．
この特徴空間における破滅的忘却の原因は何か？

A1：新しいタスクを学習することで，モデルの古いクラスの特徴分布と新しいクラスの特徴分布が重なることが原因．
継続学習では，過去に学習したタスクのデータは破棄し，新しいタスクのデータで学習を行う．
そのため，古いタスクに含まれるクラスと新しいタスクに含まれるクラス間を明示的に分離することが困難となるため，特徴空間上での忘却が発生するのではと考えられる．

Q2：古いクラスのデータの一部を学習に利用できるようリプレイバッファを導入し，過去タスクのデータを保持すれば忘却を抑制できるのではないか？

A2：


# プログラムの全体像
学習・評価に使用するプログラムの全体像は以下の通りです．
各ディレクトリの詳細については，それぞれのディレクトリのREADMEを確認してください．
```
ContinualLearning2026/
├── buffer              : 色々なバッファを実装
├── configs             : 学習・評価の設定を記述する.yamlファイルを格納したディレクトリ
├── criterions          : 色々な損失関数を実装
├── dataloaders         : データローダー関係を実装
├── main.py             : 学習を行うためのmainファイル
├── models              : 色々なモデルを実装
├── optimizers          : 色々なOptimizerを実装
├── trainers            : 手法毎の訓練や評価，分析処理を実装
└── utils.py            : その他のモジュールを実装
```

# 実行方法
## デバッグ（訓練）
指定したデータセットで学習を行います．
学習の設定は`configs`ディレクトリの下に`.yaml`ファイルを追加，修正することで変更できます．
`.yaml`ファイルを追加して学習する場合は，実行時の`--config-path`と`--config-name`を修正してから実行してください．
- debug supcon:
    supconの学習は以下で実行可能です．
    ```
    python main.py --config-path ./configs/default/supcon --config-name debug
    ```
    TensorBoardの可視化は以下で実行可能です．
    ```
    tensorboard --logdir /home/kouyou/ContinualLearning2026/logs/debug_77_2026_0212/cifar10/5_2/resnet18_mlp_supcon_ird1.0_sgd_random500/tb --port 6006 --host localhost
    ```
    ```
    tensorboard --logdir /home/kouyou/ContinualLearning2026/logs --port 6006 --host localhost
    ```
- debug proto_supcon:
    ```
    python main.py --config-path ./configs/default/proto_supcon --config-name debug
    ```
- debug cclis:
    ```
    python main.py --config-path ./configs/default/cclis --config-name debug
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
tensorboardを起動することで多少の分析はできますが，それ以外を見たい場合は以下を実行してください．



# 分析
## リプレイバッファに保存するデータ選択方法の変更実験
