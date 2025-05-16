# Accelerating Bio-acoustic Detection: A Comparative Study of CNN Performance on CPU vs. GPU

This project investigates the performance gains of training a Convolutional Neural Network (CNN) on a GPU versus a CPU for the task of detecting bird calls, specifically those of the Northern Bobwhite (NOBO). We compare training times, throughput, latency, and memory usage across different batch sizes and epochs.

## Project overview
Bird calls classification isin wildlife monitoring and conservation. However, training deep learning models on large audio datasets is computationally expensive. This project explores how using a GPU instead of a CPU affects training performance for a CNN-based NOBO call detector. crucial 

## File structure
├── `CPU.py` # Spectrogram generation and CNN training on CPU \
├── `GPU.py` # Spectrogram generation and CNN training on GPU\
├── `README.md` \
├── `Requirements.txt`

## Environment Setup
Clone the repository

`git clone https://github.com/your-username/Bio-acoustic_Detection_CPU_vs_GPU.git`
`cd Bio-acoustic_Detection_CPU_vs_GPU` 

Create a virtual environment

`python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate`

Install dependencies

`pip install -r requirements.txt`
💡 The project requires Python 3.9+ and was tested on Apple Silicon (M4) with unified memory.


## Dataset
Northern Bob White calls \
Size: ~1.6GB \
Samples: 1394 labeled NOBO calls

### Preprocessing
Typically a bird call is about 1 second. To capture the surrounding noise as well 3 second calls have been considered.  

We start by reading WAV files and their corresponding annotation text files. For each audio file. A 3-second clip is extracted and using an overlapping window it is labelled as 1 or 0 based on whether they overlap with a NOBO call or not.


 ## Spectogram generation
 Using these spectrograms are generates using multiprocessors to process clips in parallel and generate log-mel spectrograms using librosa and scipy in CPU. Where as,  resampling and STFT are implemented in the GPU using OpenCL for speed and efficiency, and then applied a mel filter bank on the output.

 ## Model training
 A simple 3 layer CNN with batch normalization and dropout is trained both on CPU and GPU using TensorFlow which is tested across different batch sizes (16, 64, 256).

 ## Hardware Setup
 We used 40 core GPU, 10 core CPU (Apple silicon CPU vs GPU M4)

 ## Performance evaluation
After training the code was run both in CPU and GPU and the following metrics were recorded.
1. Training and inference time 
2. Calculated throughput (samples/sec) 
3. Latency
4. Memory usage


NOTE: The data belongs to the so and so lab at University of Maryland, College Park and hence cannot be publicly shared.
