#!/usr/bin/env python
# encoding: utf-8
"""
Intelligent Audio and Music Analysis Exercise 1

The goal of this exercise is to learn the basics in onset detection, beat
tracking and tempo estimation.

After completing this exercise you should have learned some music information
retrieval (MIR) basics and fostered your knowledge about these topics.

First of all, you need access to audio files and annotations. Download the .zip
file from TUWEL and unpack it. Please make sure that you do not use backward
slashes if working on Windows. Either use normal forward slashes in path names
or use the functions provided by the os.path module.

The data is split into several sub-folders, train and test. Each folder
contains audio files (in .flac format) as well as annotations.

Not all audio files have all kinds of annotations, thus depending on the task
only a subset of all files can be used for evaluation.

For development of the algorithms, you can use any software in whatever
programming language as long as you code the steps by yourself. Python is
recommended, as there are several popular audio frameworks available:

- madmom (https://github.com/CPJKU/madmom)
- librosa (https://github.com/librosa/librosa)

Of course I recommend madmom ;)

"""

import argparse
import os
import pickle

import numpy as np
import matplotlib.pyplot as plt

FPS = 100

"""

Audio pre-processing:
---------------------

Step 1: read in the audio signal

Step 2: split signal into overlapping frames of length 2048 samples and a
        frame rate of 100 fps

Step 3: for each frame compute the STFT

Step 4: discard phase information and keep only the magnitudes
  
Step 5: filter the magnitudes with a Mel filterbank (40 bands)

Step 6: apply logarithmic scaling (adding a constant for numerical stability)

You are not required to code all steps by yourself, i.e. you are allowed to use
the functionality of any audio framework as long as the individual steps are 
recognisable as such. E.g. it is considered ok to use existing functions to
load an audio file and split it into overlapping frames. 
  
"""


def pre_process(filename):
    """
    Pre-process the audio signal.

    Parameters
    ----------
    filename : str
        File to be processed.

    Returns
    -------
    spectrogram : numpy array
        Spectrogram.

    """
    # TODO: your changes here
    spectrogram = None
    return spectrogram


"""
Onset detection function:
-------------------------

For onset detection, the spectral flux should be used.

Step 1: compute the temporal difference  

Step 2: keep only the positive differences

Step 3: sum these differences, to obtain the onset detection function (ODF)

"""


def onset_detection_function(spectrogram):
    """
    Compute an onset detection function.

    Parameters
    ----------
    spectrogram : numpy array
        Spectrogram

    Returns
    -------
    odf : numpy array
        Onset detection function.

    """
    # TODO: your changes here
    odf = None
    return odf


"""
Onset detection:
----------------

To detect the onsets in the ODF, the following procedure should be applied:

Step 1: subtract a moving average from the ODF (optional, but recommended)

Step 2: discard all ODF values below a certain threshold 

Step 3: select local maxima as onset positions

Step 4: discard onsets too close together (recommended: within 30ms)

Step 5: evaluate onset detection performance (see note below)

Step 6: optimise the parameters to get the best performance on the training set
        parameters to be optimised: detection threshold, frame size (1024, 
        2048, 4096), number of filter bands (40, 80), constant to be added for
        logarithmic scaling (1e-10, 1e-5, 1). The values in parentheses are
        suggested variations, experiment as you like. Please be aware that
        parameters may very likely have mutual influences. A coarse 
        optimisation is enough. The main goal of this step is to understand 
        the impact of these variations rather than getting another 0.01%
        performance.
        
To evaluate the performance, you can use the 'evaluate' script which comes with
madmom. Please invoke it with `evaluate onsets $PATH` where $PATH points to the
data directory containing all detections and annotations. If these are in
separate folders, simply add a second path. Of course you can also use any
other means of evaluation, e.g. mir_eval.
        
"""


def detect_onsets(odf, threshold):
    """
    Detect the onsets in the onset detection function (ODF).

    Parameters
    ----------
    odf : numpy array
        Onset detection function.
    threshold : float
        Threshold for onset detection.

    Returns
    -------
    onsets : numpy array
        Detected onsets (in seconds).

    """
    # TODO: your changes here
    onsets = None
    return onsets


"""
Tempo estimation:
-----------------

To detect the tempo/periodicity the ODF, the following procedure should be
applied:

Step 1: compute the auto-correlation function (ACF) of the ODF

Step 2: select an appropriate peak of the ACF as the main periodicity

Step 3: compute the tempo (in bpm, beats per minute)

Step 4: evaluate tempo estimation performance (e.g. with `evaluate tempo *`)

Step 5: optimise the parameters to get the best performance on the training set
        parameters to be optimised: lag range for ACF computation (lower bound:
        40-80bpm, upper bound 140-220bpm), peak selection mechanism (e.g. 
        clustering of peaks, smoothing of lag values)

"""


def estimate_tempo(odf):
    """
    Determine dominant tempi by means of auto-correlation.

    Parameters
    ----------
    odf : numpy array
        Onset detection function.

    Returns
    -------
    tempo : float
        Detected tempo (in bpm).

    """
    # TODO: your changes here
    tempo = None
    return tempo


"""
Beat tracking:
--------------

To detect the beats in the ODF, the following procedure should be applied:

Step 1: determine the best possible offset for beat tracking given the tempo
        and select the first beat

Step 2: determine consecutive beats based on the tempo; allow some tempo 
        deviation between consecutive beats;

Step 3: continue until all beats are tracked

Step 4: evaluate beat tracking performance (e.g. with `evaluate beats -q *`)

Step 5: optimise the parameters to get the best performance on the training set
        parameters to be optimised: allowed tempo deviation (0-20%)

"""


def track_beats(odf, tempo):
    """
    Track the beats in the detection function, given a predetermined tempo.

    Parameters
    ----------
    odf : numpy array
        Onset detection function.
    tempo : float
        Dominant periodicity / tempo of the piece.

    Returns
    -------
    beats : numpy array
        Detected beats (in seconds).

    """
    # TODO: your changes here
    beats = None
    return beats


"""
Machine learning approach:
--------------------------

A simple machine learning approach should be investigated. The question to 
be answered is: can a simple neural network improve the onset detection
performance?

In order to answer this question, the hand-crafted ODF computation should be
replaced by a multiplayer perceptron (MLP).

Step 1: use sklearn to design an MLPRegressor with default parameters

Step 2: use the same features as in the audio pre-processing section as inputs

Step 3: as targets, use the annotated onset positions of the training set and
        assign each target frame a value of 5 (this has been found to put the
        output in a similar value range in order to reuse the code to detect
        the onsets, still the threshold and other parameters most certainly
        need some adaption).

Step 4: train the model and evaluate performance on the training and test sets

Step 5: add the temporal differences as additional features and train a new
        model; compare the performance of this model with the one of step 4

Step 6: train another model using early stopping (use only 90% of the data for 
        training and 10% to evaluate on unseen data) to prevent overfitting;
        compare the performance of this model with the one of steps 4 and 5
        
"""


def train(audio_files, annotation_files, model_file):
    """
    Train an MLP on the data.

    Parameters
    ----------
    audio_files : list
        List of input files.
    annotation_files : list
        List of corresponding annotation files.
    model_file : str
        Save the trained model to this file.

    """
    from sklearn.neural_network import MLPRegressor
    # TODO: your changes here
    # define network
    pass


"""
Final evaluation:
-----------------

Fix all adjustable parameters for onset/tempo/beat detection for your 
submission. The submitted system will be evaluated with unknown test sets.

To receive a grade, your algorithm must be able to:
- process an input audio file given in WAV format (44.1kHz, 16bit, mono)
- detect the onsets, tempo and beats in the audio signal and write them to
  three separate files, ending with .onsets.txt, .bpm.txt and .beats.txt 
  respectively
- write these files to the directory the algorithm is called from or a
  configurable output dir
- if your system can operate on multiple files, the output files should be 
  named as the input file (without the .wav extension) with the suffixes 
  given above
  
Additionally, add a one page PDF report detailing the challenges you 
encountered and the observations you made. For all tasks involving comparative
steps or steps requiring to optimise some parameters, please add a table 
comparing the different parameters or outline in your own words how different
settings affected the performance.

Please also add feedback that you have. 

The best overall approach wins honour and chocolate. Good luck!
 
"""


def main():
    """IAMA Exercise 1."""

    # define parser
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter, description='''
    Submission template for exercise 1 IAMA 2018W.

    ''')
    # verbose
    p.add_argument('files', nargs='+', help='files to be processed')
    p.add_argument('-v', dest='verbose', action='count', default=0,
                   help='increase verbosity level')
    p.add_argument('-o', dest='output_dir', default=None,
                   help='output directory [default=%(default)s]')
    p.add_argument('--model', action='store', type=str, help='model file',
                   default=None)
    # other arguments
    p.add_argument('-t', dest='threshold', action='store', type=float,
                   default=0.9,
                   help='onset threshold [default=%(''default).2f]')
    # MLP training
    p.add_argument('--train', action='store_true', default=False,
                   help='train a MLP for onset detection')
    p.add_argument('--diffs', action='store_true', default=False,
                   help='add diffs to specs')
    p.add_argument('--early_stopping', action='store_true', default=False,
                   help='use early stopping')

    # parse arguments
    args = p.parse_args()

    # print arguments
    if args.verbose > 1:
        print(args)

    # make sure the directory exists
    if args.output_dir is not None:
        try:
            # create output directory
            os.mkdir(args.output_dir)
        except OSError:
            # directory exists already
            pass

    # extract certain files from all files given
    from madmom.utils import search_files, match_file
    audio_files = search_files(args.files, '.wav')
    audio_files += search_files(args.files, '.flac')
    onset_annotations = search_files(args.files, '.onsets')

    # load pre-trained model
    if args.model is not None and not args.train:
        with open(args.model, 'rb') as f:
            args.model = None  # load model

    # train model
    if args.train:
        # default model name
        if args.model is None:
            args.model = 'model.pkl'
        # do not iterate over files, use them all for training
        train(audio_files, onset_annotations, args.model)
    else:
        # process each file individually
        for input_file in audio_files:
            if args.verbose:
                print(input_file)
            # output file name
            if args.output_dir is not None:
                output_file = "%s/%s" % (args.output_dir,
                                         os.path.basename(input_file))
            else:
                output_file = input_file
            # strip off the extension
            output_file = os.path.splitext(output_file)[0]

            # pre-process audio file
            spec = pre_process(input_file)
            # compute ODF
            if args.model is not None:
                # use trained model to compute ODF
                odf = args.model.predict(spec)
            else:
                # use hand-crafted method
                odf = onset_detection_function(spec)
            # detect onsets
            onsets = detect_onsets(odf, threshold=args.threshold)
            # estimate tempo
            tempo = estimate_tempo(odf)
            # track beats
            beats = track_beats(odf, tempo)
            # write to output files
            from madmom.io import write_onsets, write_beats, write_tempo
            write_onsets(onsets, output_file + '.onsets.txt')
            # Note: write_tempo writes 3 values to file, only the first is of
            #       interest, the others (2nd tempo and relative strength) can
            #       be ignored.
            write_tempo(tempo, output_file + '.bpm.txt')
            write_beats(beats, output_file + '.beats.txt')


if __name__ == '__main__':
    main()
