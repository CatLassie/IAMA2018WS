import librosa
import numpy as np
import feature as f
import onset as o


def main():
    file_path = 'music_data/shortName.flac'
    y = f.load_audio(file_path)

    # COMPUTE SPECTROGRAM
    log_mel_spectrogram = f.compute_spectrogram(y, 2048, 1024, 40)

    # COMPUTE ONSET DETECTION FUNCTION
    # skip normalization
    # norm_spectrogram = o.normalize_frequencies(log_mel_spectrogram)
    odf = o.compute_odf(log_mel_spectrogram) #o.compute_odf(norm_spectrogram)

    # DETECT ONSETS
    peaks = o.apply_threshold(odf, 1500)
    print(odf)
    # print(peaks)
    for i, p in enumerate(peaks):
        if p > 0:
            print(i/100, '   ', p)




if __name__ == "__main__":
    main()
