リポジトリの名前は顔が長方形のキャラクター+ocrにする


## Install
```py
pip isntall xxx
```

## Usage
```py
from xxx import xxx

# 画像読み込み処理
loaded_img = xxx.load(img)

# 認識率を取得
predictions = xxx.predict(loaded_img) # これは引数いる(どのデータで認識するかを分けたい)
predicted_class = xxx.predclass() # 内部的にデータ保持していればいいから引数いらない

# 認識結果を取得
predicted_digit = xxx.predclass()

# 認識率を取得
rec_rate = xxx.recrate()

image_path = 'path/to/img'
```

## Param: default
`load(str)`: read image
`loadbyte(byte)`: read byte image
`predict`: probability of each class
`setlabels(arr[str])`: recognition class (default = `['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']`)
`predclass`: most predicted label
`recrate`: most recognition rate

from PIL import Image
import numpy as np

# 1. 保存したモデルをロード

# 2. 画像の読み込みと前処理
def preprocess_image(image_path):
    # 画像を開く
    img = Image.open(image_path).convert('L')  # グレースケールに変換 ('L'モード)
    
    # モデルが期待するサイズにリサイズ（60x90に合わせてください）
    img = img.resize((90, 60))  # (横, 縦)の順
    
    # 画像をnumpy配列に変換
    img_array = np.array(img)
    
    # 正規化 (ピクセル値を0-1に変換)
    img_array = img_array.astype('float32') / 255.0
    
    # モデルが期待する形 (1, 60, 90, 1)に変換
    img_array = np.expand_dims(img_array, axis=0)  # バッチサイズの次元を追加
    img_array = np.expand_dims(img_array, axis=-1)  # チャンネル数の次元を追加（グレースケールなので1チャンネル）
    
    print(f"Preprocessed image shape: {img_array.shape}")  # 形状確認
    return img_array


# 3. ローカル画像のパスを指定して画像を前処理
image_path = 'a.jpg'  # 認識したいローカル画像のパス
preprocessed_image = preprocess_image(image_path)

# 4. モデルで推論を行う
predictions = model.predict(preprocessed_image)

# 5. 結果を表示（最も高い確率を持つクラスを取得）
predicted_class = np.argmax(predictions)
confidence = np.max(predictions)  # 認識率（確率）を取得

# 文字リスト（例として、数字0〜9の場合）
class_labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

# 認識した文字を表示
recognized_character = class_labels[predicted_class]
print(f"Recognized character: {recognized_character}")
print(f"Confidence: {confidence * 100:.2f}%")  # 認識率をパーセントで表示
