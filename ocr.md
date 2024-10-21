# モデルのアーキテクチャを変更する例
学習部分のコード
```py
model = Sequential()
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=input_shape, kernel_regularizer=l2(0.01)))
model.add(BatchNormalization())
model.add(Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2(0.01)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu', kernel_regularizer=l2(0.01)))
model.add(BatchNormalization())
model.add(Conv2D(128, (3, 3), activation='relu', kernel_regularizer=l2(0.01)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))  # 初期層では小さなドロップアウト率
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))  # 最後の層では大きなドロップアウト率

model.add(Dense(num_classes, activation='softmax')) # 出力層


# 汎用性のあるモデルを構築するために、モデルのコンパイル時に損失関数、最適化関数、評価関数を指定
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    shear_range=0.2,
    horizontal_flip=True
)


# モデルの再訓練
model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.0005),  # 学習率
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# リデーション損失が5エポック連続で改善しない場合、トレーニングが自動的に停止
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# モデルの学習
model.fit(datagen.flow(x_train, y_train, batch_size=32),
          steps_per_epoch=len(x_train) // 32,
          epochs=20,
          validation_data=(x_test, y_test),
          callbacks=[early_stopping])
```

## CNN
```py
model.add(Conv2D(64, kernel_size=(2, 2), activation='relu', input_shape=input_shape))
model.add(Conv2D(128, (3, 3), activation='relu'))
```
1つ目の畳み込み層 (Conv2D(64, kernel_size=(2, 2), activation='relu'))

最初の層は、画像のピクセルレベルの基本的な特徴（エッジ、コーナーなど）を抽出します。小さなフィルタサイズ（ここでは2x2）を使うことで、非常に細かい特徴を捉えることができます。
2つ目の畳み込み層 (Conv2D(128, (3, 3), activation='relu'))

2つ目の層は、1つ目の層が抽出した特徴をさらに組み合わせて、より抽象的で高次の特徴を学習します。フィルタサイズを3x3にすることで、少し広範囲の特徴を捉えます。


activation='relu' : 非線形性を表現する

```py
model.add(MaxPooling2D(pool_size=(2, 2)))
```
この部分は画像のサイズを半分にしているため、小さい値になりすぎるとモデルをパッケージ化した際に、認識画像を読み込めなくなる

### l2正規化 : kernel_regularizer=l2(0.01)
モデルの損失関数にペナルティ項を追加して、重みが大きくなりすぎないようにする

![alt text](image.png)

\[
\text{Loss} = \text{Original Loss} + \lambda \sum_{i} w_i^2
\]

- Original Loss: 元々の損失関数（例えば、クロスエントロピー損失など）。
- wi: モデルの各重み。
- λ: 正則化係数（ハイパーパラメータ）。通常は0.01や0.001のように小さな値に設定。


## 内部共変量シフト(進行中の学習変化量)を正規化
```py
model.add(BatchNormalization())
```
各ミニバッチに対して、入力を平均が0、標準偏差が1になるように正規化

## 入力マップサイズの削減　
```py
model.add(MaxPooling2D(pool_size=(2, 2)))
```

```
<!-- 入力データ -->
1 3 2 4
5 6 7 8
3 2 1 0
9 8 5 4

<!-- 2×2のプーリングを適応後 -->
6 8
9 5

```

## ドロップアウト率
```py
model.add(Dropout(0.25))  # 初期層では小さなドロップアウト率
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))  # 最後の層では大きなドロップアウト率
```
次の層に渡されるニューロンを任意の割合で無効化
0.5 -> 50 %
過学習の防止

## 学習率
```py
model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.001),  # 学習率
              loss='categorical_crossentropy',
              metrics=['accuracy'])
```
小さい値から徐々に大きくしていく


## 学習率の減衰
訓練が進むにつれて学習率を徐々に減少させる「学習率スケジューリング」や「学習率減衰（decay）」を使用するのも効果的。
->初期の段階では大きくパラメータを更新し、最適解に近づくにつれて小さく調整する
```py
# 学習率を動的変更
def scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * 0.9
lr_scheduler = tf.keras.callbacks.LearningRateScheduler(scheduler)
```

- 最初の10エポックでは、学習率をそのまま使用
- 10エポック以降は、学習率を毎回0.9倍

これにより、学習が進むにつれてステップサイズを小さくし、細かい調整が可能。

## データの多様性増加・モデルの汎化能力を向上
トレーニング中の各データに対してランダムなパラメータを適応
```py
datagen = ImageDataGenerator(
    rotation_range=10,        # 画像を最大10度までランダムに回転
    width_shift_range=0.1,    # 画像を水平方向に最大10%までランダムにシフト
    height_shift_range=0.1,   # 画像を垂直方向に最大10%までランダムにシフト
    zoom_range=0.1            # 画像を最大10%までランダムにズーム
)
```


## 2024/9/12 追記
## 重みを追加
```py
# クラス重みを設定（少ないクラスに重みを付ける）
class_weight = {0: 1.0, 1: 0.5, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0, 6: 1.0, 7: 0.5, 8: 1.0, 9: 1.0}
```

1. 特定の数字の形状（
  例:「1」や「7」）が他の数字に比べて単純である場合、モデルはそのパターンをより早く学習しやすい
2. モデルのバイアス
  初期重みの影響: 
    初期段階で特定の数字の特徴が過度に強調されることがある。例えば、特定のフィルタが「1」や「7」に特に敏感な特徴を捉えやすい形で学習されている可能性を考慮する必要がある
  ドロップアウトやデータ拡張の影響:
    ドロップアウト層やデータ拡張の効果により、特定の数字が強調されることも考えらる。データの回転やシフトによって、特定のパターンが他の数字と重なりやすくなることも考慮する

