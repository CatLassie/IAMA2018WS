import librosa

def loadAudio():
    filePath = 'music_data/shortName.flac'
    y, sr = librosa.load(filePath) #sr=11025
    print(y, sr)


def main():
    print("IAMI executed as main")
    loadAudio()


if __name__ == "__main__":
    main()
