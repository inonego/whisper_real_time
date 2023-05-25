# 실시간 Whisper 음성 인식
이 모듈은 실시간으로 음성을 텍스트로 변환해줍니다.

다만, 문장 단위 변환(말이 끊기는 지점으로 구분)이기 때문에, 실시간으로 단어가 입력되는 보여주는 결과를 원하신다면 원본 레포를 참고하시길 바랍니다. 
https://github.com/davabase/whisper_real_time

### 마이크 입력으로 사용하는 방법(예시)
``` python
from transcriber import Transcriber

# Set and load whisper model
transcriber = Transcriber(model = "large") 

# You can use your mic after transcriber.use_mic() is called
transcriber.use_mic() 

# msg를 출력하는 함수입니다.
def show(msg):
    print("출력 :" + msg)

# 콜백 함수를 집어넣으면, 나중에 음성이 텍스트로 변환되었을때의 결과값을 제공하여 실행시킵니다. 
transcriber.execute(show)
```

### 다른 입력으로 사용하는 방법(예시)
``` python
# 처리할 음성의 샘플레이트와 샘플 너비를 설정합니다.
transcriber.SAMPLE_RATE, transcriber.SAMPLE_WIDTH = SAMPLE_RATE, SAMPLE_WIDTH

# 주어진 PCM 데이터를 transcriber의 data_queue에 집어넣으면 됩니다.
def callback(data: bytes):
    transcriber.data_queue.put(data)

listen_to_something(callback)
```

# Real Time Whisper Transcription

![Demo gif](demo.gif)

This is a demo of real time speech to text with OpenAI's Whisper model. It works by constantly recording audio in a thread and concatenating the raw bytes over multiple recordings.

To install dependencies simply run
```
pip install -r requirements.txt
```
in an environment of your choosing.

Whisper also requires the command-line tool [`ffmpeg`](https://ffmpeg.org/) to be installed on your system, which is available from most package managers:

```
# on Ubuntu or Debian
sudo apt update && sudo apt install ffmpeg

# on Arch Linux
sudo pacman -S ffmpeg

# on MacOS using Homebrew (https://brew.sh/)
brew install ffmpeg

# on Windows using Chocolatey (https://chocolatey.org/)
choco install ffmpeg

# on Windows using Scoop (https://scoop.sh/)
scoop install ffmpeg
```

For more information on Whisper please see https://github.com/openai/whisper

The code in this repository is public domain.
