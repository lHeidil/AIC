# Mind-Cloud-AIC
Automatic Speech Recognition (ASR) and Speaker Diarization System for Egyptian Arabic

## Dataset

The dataset comprises an extensive collection of 50,715 audio files for training and 2,199 audio files for validation. Each audio file is a single-channel 16-bit PCM WAV with a sample rate of 16,000 Hz. The corresponding transcripts are provided in the `train.csv` file with the following fields:

- **audio**: The name of the corresponding `.wav` file.
- **Transcription**: The words spoken by the reader.

## Preprocessing

We prepared a comprehensive vocabulary consisting of 38 characters, including variations and out-of-vocabulary (OOV) tokens. The following transformations were applied to the data:

- **Spectrogram Generation**: Spectrograms were generated using Short-Time Fourier Transform (STFT) with a frame length of 240 (15 ms), frame step of 120, and FFT length of 256.
- **Normalization**: Spectrograms were normalized.
- **Label Encoding**: Labels were split and encoded to match the prepared vocabulary.
- **Batching**: A batch size of 32 was used.

![Example of the Spectrogram](https://github.com/Yahia-Ibrahim/mind-cloud-AIC/assets/120991373/5f83164f-89a3-4375-8392-1ae68542d696)

## Speaker Diarization Model

Speaker diarization partitions audio into segments based on speaker identity. We used the `pyannote/speaker-diarization-3.1` model for this purpose. The diarization process involves:

- **Feature Extraction**: Acoustic features are extracted from the audio signal.
- **Speech Activity Detection**: Segments of speech are identified and separated from non-speech elements.
- **Speaker Change Detection**: Transitions between speakers are detected.
- **Speech Turn Representation**: Speaker turns are represented and grouped.
- **Clustering**: Speaker embeddings are clustered to segment the audio into speaker-specific chunks.

## ASR Model Architecture

### Input Layer
- Accepts spectrogram inputs of shape `(None, input_dim)`.

### Convolutional Layers
- **Conv1**: 96 filters, kernel size `[11, 41]`, strides `[2, 2]`, followed by ReLU activation.
- **Conv2**: 128 filters, kernel size `[11, 21]`, strides `[1, 2]`, followed by ReLU activation.

### Reshape Layer
- Reshapes the output from the convolutional layers to a 2D tensor for the RNN layers.

### RNN Layers
- **Bidirectional GRU Layers**: Five layers, each with 768 units, using `tanh` activation and `sigmoid` recurrent activation. Outputs from forward and backward GRU cells are concatenated.

### Dense Layer
- A fully connected layer with `2 * rnn_units` units followed by ReLU activation.

### Output Layer
- A dense layer with `output_dim + 1` units and a softmax activation function to predict character probabilities.

### Compilation

- **Optimizer**: Adam optimizer.
- **Loss Function**: Connectionist Temporal Classification (CTC) Loss.

## Segmentation

Audio is divided into smaller sub-files, each corresponding to a segment where a single speaker is talking. This is achieved using timestamps provided by the diarization model.

## Transcript Matching

Transcripts are generated using the ASR model. Timestamps from the diarization model are used to align transcripts with corresponding audio segments, ensuring accurate matching between spoken words and the speaker.

## Conclusion

The integration of ASR and speaker diarization models aimed to provide a structured and accurate breakdown of the audio content. However, while the speaker diarization effectively segmented the audio into speaker-specific segments, the ASR model's text results have not met the desired accuracy. The alignment of transcripts with audio segments showed promise but revealed areas for improvement in transcription quality. Ongoing efforts will focus on enhancing the ASR model’s performance and refining the overall system to better handle the complexities of Egyptian Arabic speech.

## References

1. Abbas Raza Ali, "Multi-Dialect Arabic Speech Recognition," Faculty of Science and Technology, Bournemouth University, Poole BH12 5BB, United Kingdom, email: abbas.raza.ali@gmail.com.
2. Mohamed Reda Bouadjenek and Ngoc Dung Huynh, "Automatic Speech Recognition using CTC," Date created: 2021/09/26, Last modified: 2021/09/26, Description: Training a CTC-based model for automatic speech recognition.
3. pyannote, "pyannote-audio," GitHub repository, 2024. [Online]. Available: [https://github.com/pyannote/pyannote-audio](https://github.com/pyannote/pyannote-audio).
