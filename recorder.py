
import sys
import pyaudio
import wave


def run_recorder(outfile='recordings/speech_recording.wav'):

    FORMAT = pyaudio.paInt16
    CHANNEL = 1
    RATE = 44100
    CHUNK = 1024
    RECORD_TIME = 5
    WAV_OUT = outfile

    audio = pyaudio.PyAudio()

    stream = audio.open(format=FORMAT, channels=CHANNEL,
                        rate=RATE, input=True,
                        frames_per_buffer=CHUNK)

    print('[i] Microphone Opened...\n')

    frames = []
    time = range(0, int(RATE / CHUNK * RECORD_TIME))

    xi = 0
    for i in time:
        xi += 1
        z = (f'[i] Recording - {round((xi/len(time))*100,2)} [' + '.' * xi +']')
        data = stream.read(CHUNK)
        frames.append(data)
        sys.stdout.write('\r'+z)


    print('\n[i] Microphone Closed...')


    stream.stop_stream()
    stream.close()
    audio.terminate()
    
    waveFile = wave.open(WAV_OUT, 'wb')
    waveFile.setnchannels(CHANNEL)
    waveFile.setsampwidth(audio.get_sample_size(FORMAT))
    waveFile.setframerate(RATE)
    waveFile.writeframes(b''.join(frames))
    waveFile.close()


if __name__ == '__main__':

    run_recorder()