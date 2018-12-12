import librosa
import numpy as np
import feature as f
import onset as o


def main():
    file_path = 'music_data/shortName.flac'
    y = f.load_audio(file_path)
    log_mel_spectrogram = f.compute_spectrogram(y, 2048, 1024, 40)
    norm_spectrogram = o.normalize_frequencies(log_mel_spectrogram)
    odf = o.compute_odf(norm_spectrogram)
    peaks = o.apply_threshold(odf, 0.5)
    print(odf)
    # print(peaks)
    for i, p in enumerate(peaks):
        if p > 0:
            print(i/100)




if __name__ == "__main__":
    main()
