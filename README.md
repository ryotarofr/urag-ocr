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
- `load(str)`: read image
- `loadbyte(byte)`: read byte image
- `predict`: probability of each class
- `setlabels(arr[str])`: recognition class (default = `['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']`)
- `predclass`: most predicted label
- `recrate`: most recognition rate
