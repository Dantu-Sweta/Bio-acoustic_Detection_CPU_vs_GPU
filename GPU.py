import os
import re
import glob
import time
import platform
import psutil
import numpy as np
import pandas as pd
import soundfile as sf
import librosa
import pyopencl as cl
import tensorflow as tf
import sklearn
from sklearn import metrics
from sklearn.metrics import classification_report
from scipy.signal import stft, resample, get_window
from multiprocessing import Pool
from tqdm import tqdm

base_path = '/Users/yashwanth/Library/CloudStorage/GoogleDrive-ypinnapu@umd.edu/Shared drives/Audiomoth/data/Selection Tables'
folder_name = 'Summer Calls/2022 - Revising for Bob white annotation together/'
relative_folder_path = os.path.join(base_path, folder_name)

folder = glob.glob(os.path.join(relative_folder_path, "**", "*"), recursive=True)
audio_files, selection_files = [], []

for file in folder:
    if file.lower().endswith('.wav'):
        audio_files.append(file)
    elif file.lower().endswith('.txt'):
        selection_files.append(file)

def extract_date(file_path):
    match = re.search(r'(\d{8})', os.path.basename(file_path))
    return int(match.group(1)) if match else -1

audio_files_sorted = sorted(audio_files, key=extract_date)
selection_files_sorted = sorted(selection_files, key=extract_date)

all_annots = []
for audio_file, sel_file in zip(audio_files_sorted, selection_files_sorted):
    df = pd.read_csv(sel_file, sep='\t')
    for _, row in df.iterrows():
        start = float(row['Begin Time (s)'])
        end = float(row['End Time (s)'])
        label = row.get('Species', row.get('Annotation', ''))
        is_nobo = 1 if label == 'NOBO' else 0
        all_annots.append([os.path.basename(audio_file), start, end, is_nobo])

annotations_df = pd.DataFrame(all_annots, columns=['file', 'start_time', 'end_time', 'NOBO'])
annotations_df.set_index(['file', 'start_time', 'end_time'], inplace=True)

all_windows = []
window_size, window_step, min_overlap = 3.0, 2.0, 0.7

for audio_file in audio_files_sorted:
    basename = os.path.basename(audio_file)
    file_annots = annotations_df.loc[basename] if basename in annotations_df.index.get_level_values(0) else pd.DataFrame()
    if isinstance(file_annots, pd.Series):
        file_annots = file_annots.to_frame().T
    duration = librosa.get_duration(path=audio_file)
    start = 0.0
    while start + window_size <= duration:
        end = start + window_size
        overlap = 0.0
        if not file_annots.empty:
            for idx, annot in file_annots.iterrows():
                a_start, a_end = idx[0], idx[1]
                o = max(0.0, min(end, a_end) - max(start, a_start))
                if annot['NOBO'] == 1 and o >= min_overlap:
                    overlap = o
                    break
        label = 1 if overlap >= min_overlap else 0
        all_windows.append([audio_file, start, end, label])
        start += window_step

labels_df = pd.DataFrame(all_windows, columns=['file', 'start_time', 'end_time', 'NOBO'])
labels_df.set_index(['file', 'start_time', 'end_time'], inplace=True)

nobo_1 = labels_df[labels_df['NOBO'] == 1]
nobo_0 = labels_df[labels_df['NOBO'] == 0].sample(n=len(nobo_1), random_state=42)
labels_df = pd.concat([nobo_0, nobo_1]).sample(frac=1, random_state=42)

train_valid, test_df = sklearn.model_selection.train_test_split(labels_df, test_size=0.2, random_state=1)
train_df, valid_df = sklearn.model_selection.train_test_split(train_valid, test_size=0.1, random_state=0)


# Constants
target_sr = 22050
clip_duration_sec = 3.0
n_fft = 512
hop_length = 256
n_mels = 128
fmin = 1000
fmax = 4000
output_dir = "spectrograms_opencl"
os.makedirs(output_dir, exist_ok=True)

# OpenCL setup
ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

kernel_code = """
__kernel void resample_linear(
    __global const float *input,
    __global float *output,
    const int input_len,
    const float resample_ratio)
{
    int i = get_global_id(0);
    float pos = i * resample_ratio;
    int idx = (int)floor(pos);
    float frac = pos - idx;

    if (idx + 1 < input_len)
        output[i] = (1.0f - frac) * input[idx] + frac * input[idx + 1];
    else
        output[i] = input[input_len - 1];
}

__kernel void stft_magnitude(
    __global const float *audio,
    __global const float *window,
    __global float2 *stft_output,
    const int n_fft,
    const int hop_length,
    const int num_frames)
{
    int frame = get_global_id(0);
    if (frame >= num_frames) return;

    int start = frame * hop_length;
    for (int k = 0; k < n_fft; ++k) {
        float real = 0.0f;
        float imag = 0.0f;
        for (int n = 0; n < n_fft; ++n) {
            int sample_idx = start + n;
            float sample = (sample_idx < 132300) ? audio[sample_idx] * window[n] : 0.0f;
            float angle = -2.0f * 3.14159265359f * k * n / n_fft;
            real += sample * cos(angle);
            imag += sample * sin(angle);
        }
        int idx = frame * n_fft + k;
        stft_output[idx] = (float2)(real, imag);
    }
}
"""

prg = cl.Program(ctx, kernel_code).build()

def hz_to_mel(f): return 2595 * np.log10(1 + f / 700)
def mel_to_hz(m): return 700 * (10**(m / 2595) - 1)

def mel_filterbank(sr, n_fft, n_mels, fmin, fmax):
    mel_min = hz_to_mel(fmin)
    mel_max = hz_to_mel(fmax)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    bins = np.floor((n_fft + 1) * hz_points / sr).astype(int)

    fbanks = np.zeros((n_mels, n_fft // 2 + 1))
    for i in range(1, n_mels + 1):
        fbanks[i - 1, bins[i - 1]:bins[i]] = np.linspace(0, 1, max(1, bins[i] - bins[i - 1]))
        fbanks[i - 1, bins[i]:bins[i + 1]] = np.linspace(1, 0, max(1, bins[i + 1] - bins[i]))
    return fbanks.astype(np.float32)

mel_fb = mel_filterbank(target_sr, n_fft, n_mels, fmin, fmax)

def generate_and_save_spectrograms(df, split):
    os.makedirs(os.path.join(output_dir, split), exist_ok=True)
    success_count = 0
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss
    start_time = time.time()

    for (file_path, start, end), row in tqdm(df.iterrows(), total=len(df), desc=f"{split}"):
        label = row['NOBO']
        try:
            y, sr = sf.read(file_path)
            if y.ndim > 1:
                y = y[:, 0]

            resample_ratio = sr / target_sr
            target_len = int(len(y) / resample_ratio)
            y = y.astype(np.float32)

            input_buf = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=y)
            output_buf = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, size=target_len * 4)

            prg.resample_linear(queue, (target_len,), None,
                                input_buf, output_buf,
                                np.int32(len(y)), np.float32(resample_ratio))

            resampled = np.empty(target_len, dtype=np.float32)
            cl.enqueue_copy(queue, resampled, output_buf)

            start_sample = int(start * target_sr)
            clip = resampled[start_sample:start_sample + int(target_sr * clip_duration_sec)]
            if len(clip) < int(target_sr * clip_duration_sec):
                clip = np.pad(clip, (0, int(target_sr * clip_duration_sec) - len(clip)))

            num_frames = 1 + (len(clip) - n_fft) // hop_length
            hann_win = get_window("hann", n_fft, fftbins=True).astype(np.float32)

            audio_buf = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=clip)
            window_buf = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=hann_win)
            stft_buf = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, size=n_fft * num_frames * 8)

            prg.stft_magnitude(queue, (num_frames,), None,
                               audio_buf, window_buf, stft_buf,
                               np.int32(n_fft), np.int32(hop_length), np.int32(num_frames))

            stft_result = np.empty((num_frames, n_fft), dtype=np.complex64)
            cl.enqueue_copy(queue, stft_result.view(np.float32), stft_buf)

            magnitude = np.abs(stft_result[:, :n_fft // 2 + 1].T)
            mel_spec = np.dot(mel_fb, magnitude)
            log_mel = np.log(mel_spec + 1e-6)

            fname = os.path.basename(file_path).replace(".wav", f"_t{int(start)}_label{label}.npy")
            save_path = os.path.join(output_dir, split, fname)
            np.save(save_path, log_mel.astype(np.float32))

            success_count += 1

        except Exception as e:
            print(f"[ERROR] {file_path} @ {start}-{end} -> {e}")

    end_time = time.time()
    mem_after = process.memory_info().rss
    total_time = end_time - start_time
    memory_used_mb = (mem_after - mem_before) / (1024 ** 2)
    latency = total_time / max(success_count, 1)
    throughput = success_count / total_time

    print(f"\n{split.upper()} Spectrogram Generation Metrics (OpenCL on Mac Silicon):")
    print(f"  Total Time        : {total_time:.2f} seconds")
    print(f"  Spectrograms Saved: {success_count}")
    print(f"  Throughput        : {throughput:.2f} samples/second")
    print(f"  Latency           : {latency*1000:.2f} ms/sample")
    print(f"  Unified Memory Used : {memory_used_mb:.2f} MB")

generate_and_save_spectrograms(train_df, "train")
generate_and_save_spectrograms(valid_df, "valid")
generate_and_save_spectrograms(test_df, "test")



def load_data(folder):
    x, y = [], []
    for path in sorted(glob.glob(os.path.join(folder, "*.npy"))):
        if "label" not in path:
            continue
        try:
            arr = np.load(path).astype(np.float32)
            label = int(path.split("label")[-1].replace(".npy", ""))
            arr = arr[..., np.newaxis]
            x.append(arr)
            y.append(label)
        except:
            continue
    x = np.stack(x)
    y = np.array(y, dtype=np.float32)
    return x, y

x_train, y_train = load_data("spectrograms_opencl/train")
x_valid, y_valid = load_data("spectrograms_opencl/valid")
x_test, y_test = load_data("spectrograms_opencl/test")

mean = np.mean(x_train)
std = np.std(x_train) + 1e-6
x_train = (x_train - mean) / std
x_valid = (x_valid - mean) / std
x_test = (x_test - mean) / std

def build_model(input_shape):
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

def run_inference(model, x_test, batch_size):
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss
    start_time = time.time()

    y_pred_probs = model.predict(x_test, batch_size=batch_size)
    y_pred = (y_pred_probs > 0.5).astype(int).flatten()

    end_time = time.time()
    mem_after = process.memory_info().rss

    inference_time = end_time - start_time
    memory_used_mb = (mem_after - mem_before) / (1024 ** 2)
    throughput = len(x_test) / inference_time
    latency = inference_time / len(x_test)

    print(f"\nInference Metrics (Batch Size: {batch_size})")
    print(f"  Inference Time      : {inference_time:.2f} seconds")
    print(f"  Throughput          : {throughput:.2f} samples/second")
    print(f"  Latency             : {latency * 1000:.2f} ms/sample")
    print(f"  Unified Memory Used : {memory_used_mb:.2f} MB")

# Loop over different batch sizes
for batch_size in [16, 64, 256]:
    print(f"\n--- Training with Batch Size: {batch_size} ---")
    with tf.device('/GPU:0'):
        model = build_model(input_shape=x_train.shape[1:])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss
        start_time = time.time()

        model.fit(
            x_train, y_train,
            validation_data=(x_valid, y_valid),
            epochs=20,
            batch_size=batch_size,
            verbose=0
        )

        end_time = time.time()
        mem_after = process.memory_info().rss

        training_time = end_time - start_time
        memory_used_mb = (mem_after - mem_before) / (1024**2)
        total_samples = x_train.shape[0]
        throughput = total_samples / training_time
        latency = training_time / total_samples

        print(f"\nTraining Metrics (Batch Size: {batch_size})")
        print(f"  Training Time       : {training_time:.2f} seconds")
        print(f"  Throughput          : {throughput:.2f} samples/second")
        print(f"  Latency             : {latency * 1000:.2f} ms/sample")
        print(f"  Unified Memory Used : {memory_used_mb:.2f} MB")

    run_inference(model, x_test, batch_size=batch_size)
