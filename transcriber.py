# Code from https://github.com/davabase/whisper_real_time
# The code in this repository is public domain.

import io
import whisper
import torch
import asyncio
import speech_recognition as sr
 
from sys import platform
from queue import Queue
from datetime import datetime, timedelta
from tempfile import NamedTemporaryFile

class Transcriber:
    def __init__(self, model = "medium", non_english = True, record_timeout = 2, phrase_timeout = 3):
        # Select model from
        # tiny/base/small/medium/large
        self.model = model
        # Check non_english for Korean or etc generation
        self.non_english = non_english

        # Timeout for speech endings
        self.record_timeout = record_timeout
        self.phrase_timeout = phrase_timeout

        self.SAMPLE_RATE = 48000
        self.SAMPLE_WIDTH = 2

        self.data_queue = Queue()

        # Load whisper model to trancribe
        self.load_model()

    def load_model(self):
        # Load / Download model
        model = self.model

        if model != "large" and not self.non_english:
            model = model + ".en"

        self.loaded_model = whisper.load_model(model)
  
        # Cue the user that we're ready to go.
        print("Model loaded.\n")
 
    def set_sample(self, rate, width):
        self.SAMPLE_RATE = rate
        self.SAMPLE_WIDTH = width
    
    async def execute(self):
        # Current raw audio bytes.
        sample = bytes() 
        # The last time a recording was retreived from the queue.
        phrase_time = None

        audio_path = NamedTemporaryFile().name 

        while True:
            try: 
                now = datetime.utcnow()

                # Pull raw recorded audio from the queue.
                if not self.data_queue.empty():
                    # This is the last time we received new audio data from the queue.
                    phrase_time = now

                    # Concatenate our current audio data with the latest audio data.
                    while not self.data_queue.empty(): sample += self.data_queue.get()

                # If enough time has passed between recordings, consider the phrase complete.
                # Clear the current working audio buffer to start over with the new data.
                if not sample == bytes() and (phrase_time and now - phrase_time > timedelta(seconds=self.phrase_timeout)):
                    # Use AudioData to convert the raw data to wav data.
                    audio_data = sr.AudioData(sample, self.SAMPLE_RATE, self.SAMPLE_WIDTH)
                    wav_data = io.BytesIO(audio_data.get_wav_data())

                    # Write wav data to the temporary file as bytes.
                    with open(audio_path, 'w+b') as f:
                        f.write(wav_data.read())

                    # Read the transcription.
                    result = self.loaded_model.transcribe(audio_path, fp16=torch.cuda.is_available())['text'].strip() 

                    # Return transcribed sentences.
                    #yield result
                    if not result.isspace(): print(result)

                    # Clear the previous sample.
                    sample = bytes()

                # Infinite loops are bad for processors, must sleep.
                await asyncio.sleep(0.25)
            except KeyboardInterrupt:
                break

    # Callback function for source_callback in execute
    def use_mic(self,  default_microphone = 'pulse', energy_threshold = 1000): 
        # We use SpeechRecognizer to record our audio because it has a nice feauture where it can detect when speech ends.
        recorder = sr.Recognizer()
        recorder.energy_threshold = energy_threshold
        # Definitely do this, dynamic energy compensation lowers the energy threshold dramtically to a point where the SpeechRecognizer never stops recording.
        recorder.dynamic_energy_threshold = False

        # Important for linux users.
        # Prevents permanent application hang and crash by using the wrong Microphone
        if 'linux' in platform:
            mic_name = default_microphone
            if not mic_name or mic_name == 'list':
                print("Available microphone devices are: ")
                for index, name in enumerate(sr.Microphone.list_microphone_names()):
                    print(f"Microphone with name \"{name}\" found")   
                return
            else:
                for index, name in enumerate(sr.Microphone.list_microphone_names()):
                    if mic_name in name:
                        source = sr.Microphone(sample_rate=16000, device_index=index)
                        break
        else:
            source = sr.Microphone(sample_rate=16000)
 
        self.set_sample(source.SAMPLE_RATE, source.SAMPLE_WIDTH)

        with source:
            recorder.adjust_for_ambient_noise(source)

        def record_callback(_, audio:sr.AudioData) -> None:
            """
            Threaded callback function to recieve audio data when recordings finish.
            audio: An AudioData containing the recorded bytes.
            """
            # Grab the raw bytes and push it into the thread safe queue.
            self.data_queue.put(audio.get_raw_data())

        # Create a background thread that will pass us raw audio bytes.
        # We could do this manually but SpeechRecognizer provides a nice helper.
        recorder.listen_in_background(source, record_callback, phrase_time_limit=self.record_timeout)

# Example

'''
transcriber = Transcriber(model = "large") 

transcriber.use_mic()

asyncio.run(transcriber.execute())
'''