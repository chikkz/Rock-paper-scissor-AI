# rock-paper-scissor

An AI to play the Rock Paper Scissors game

## Requirements
- Python 3
- Keras
- Tensorflow
- OpenCV

## Set up instructions
1. Clone the repo
$ cd rock-paper-scissors
```

2. Install the dependencies
```sh
$ pip install -r requirements.txt
```

3. Gather Images for each gesture (rock, paper and scissors and None):
In this example, we gather 200 images for the "rock" gesture
```sh
$ python gather_images.py rock 200
```

4. Train the model
```sh
$ python train.py
```

5. Test the model on some images
```sh
$ python test.py <path_to_test_image>
```

6. Play the game with your computer!
```sh
$ python play.py
```
