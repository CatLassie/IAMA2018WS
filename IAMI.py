import librosa

def loadAudio():
    filePath = 'music_data/shortName.flac'
    y, sr = librosa.load(filePath, sr=96000)
    print("sample values: ", y)
    print("sample #: ", len(y))
    print("sampling rate: ", sr)
    print()

    framesMatrix = librosa.util.frame(y, frame_length=2048, hop_length=1024)
    print('frame size:', len(framesMatrix))
    print('frame #', len(framesMatrix[0]))
    print()
    """for i, f in enumerate(framesMatrix):
        print(i, f[0])"""



def main():
    print("IAMI executed as main")
    loadAudio()

if __name__ == "__main__":
    main()
