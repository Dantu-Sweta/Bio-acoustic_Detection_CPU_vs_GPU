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
from sklearn.model_selection import train_test_split
from tqdm import tqdm

relative_folder_path = "Your_Path"

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



# Parameters
target_sr = 22050
clip_duration_sec = 3.0
n_fft = 512
hop_length = 256
n_mels = 128
fmin = 1000
fmax = 4000
output_dir = "spectrograms_parallel_cpu"

# Mel filter bank calculation
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

# Single row processing
def process_row(args):
    file_path, start, end, label, split = args
    try:
        y, sr = sf.read(file_path)
        if y.ndim > 1:
            y = y[:, 0]
        resample_ratio = target_sr / sr
        target_len = int(len(y) * resample_ratio)
        y_resampled = resample(y, target_len)

        start_sample = int(start * target_sr)
        clip_len = int(target_sr * clip_duration_sec)
        clip = y_resampled[start_sample:start_sample + clip_len]
        if len(clip) < clip_len:
            clip = np.pad(clip, (0, clip_len - len(clip)))

        f, t, Zxx = stft(clip, fs=target_sr, nperseg=n_fft, noverlap=n_fft - hop_length, window='hann')
        magnitude = np.abs(Zxx[:n_fft // 2 + 1, :])
        mel_spec = np.dot(mel_fb, magnitude)
        log_mel = np.log(mel_spec + 1e-6)

        os.makedirs(os.path.join(output_dir, split), exist_ok=True)
        fname = os.path.basename(file_path).replace(".wav", f"_t{int(start)}_label{label}.npy")
        save_path = os.path.join(output_dir, split, fname)
        np.save(save_path, log_mel.astype(np.float32))
        return True
    except Exception as e:
        return f"[ERROR] {file_path} @ {start}-{end} -> {e}"

# Parallel execution with metrics
def generate_and_save_spectrograms_parallel(df, split):
    task_list = [(file_path, start, end, row['NOBO'], split) for (file_path, start, end), row in df.iterrows()]
    
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss
    start_time = time.time()

    with Pool(processes=10) as pool:
        results = list(tqdm(pool.imap_unordered(process_row, task_list), total=len(task_list), desc=f"Processing {split}"))

    end_time = time.time()
    mem_after = process.memory_info().rss

    success_count = sum(1 for r in results if r is True)
    fail_count = len(results) - success_count
    total_time = end_time - start_time
    throughput = success_count / total_time
    latency = total_time / success_count
    mem_used_mb = (mem_after - mem_before) / (1024**2)

    print(f"\n[{split.upper()}] Spectrograms saved: {success_count} | Failed: {fail_count}")
    print(f"Performance Metrics (CPU, Parallel):")
    print(f"  Execution Time     : {total_time:.2f} seconds")
    print(f"  Throughput         : {throughput:.2f} clips/second")
    print(f"  Latency            : {latency*1000:.2f} ms/clip")
    print(f"  Memory Used        : {mem_used_mb:.2f} MB")

# Example usage
if __name__ == "__main__":
    generate_and_save_spectrograms_parallel(train_df, "train")
    generate_and_save_spectrograms_parallel(valid_df, "valid")
    generate_and_save_spectrograms_parallel(test_df, "test")


import os
import time
import psutil
import numpy as np
import tensorflow as tf
import glob
from sklearn.metrics import classification_report

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
        except Exception:
            continue
    x = np.stack(x)
    y = np.array(y, dtype=np.float32)
    return x, y

x_train, y_train = load_data("spectrograms_parallel_cpu/train")
x_valid, y_valid = load_data("spectrograms_parallel_cpu/valid")
x_test, y_test = load_data("spectrograms_parallel_cpu/test")

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

tf.config.set_visible_devices([], 'GPU')
tf.config.threading.set_intra_op_parallelism_threads(10)
tf.config.threading.set_inter_op_parallelism_threads(10)

batch_sizes = [16, 64, 256]

results = []

for batch_size in batch_sizes:
    print(f"\n--- Training with Batch Size: {batch_size} ---")
    with tf.device('/CPU:0'):
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
            verbose=0,
            callbacks=[tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)]
        )

        end_time = time.time()
        mem_after = process.memory_info().rss

        training_time = end_time - start_time
        memory_used_mb = (mem_after - mem_before) / (1024 ** 2)
        total_samples = x_train.shape[0]
        throughput = total_samples / training_time
        latency = training_time / total_samples

        print(f"\nTraining Metrics (CPU with 10 Threads, Batch Size: {batch_size})")
        print(f"  Training Time       : {training_time:.2f} seconds")
        print(f"  Throughput          : {throughput:.2f} samples/second")
        print(f"  Latency             : {latency*1000:.2f} ms/sample")
        print(f"  Memory Used         : {memory_used_mb:.2f} MB")

        # Inference
        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss
        start_time = time.time()

        y_pred_probs = model.predict(x_test, batch_size=batch_size, verbose=0)
        y_pred = (y_pred_probs > 0.5).astype(int).flatten()

        end_time = time.time()
        mem_after = process.memory_info().rss

        inference_time = end_time - start_time
        memory_used_mb = (mem_after - mem_before) / (1024 ** 2)
        throughput = len(x_test) / inference_time
        latency = inference_time / len(x_test)

        print(f"\nInference Metrics (CPU with 10 Threads, Batch Size: {batch_size})")
        print(f"  Inference Time      : {inference_time:.2f} seconds")
        print(f"  Throughput          : {throughput:.2f} samples/second")
        print(f"  Latency             : {latency*1000:.2f} ms/sample")
        print(f"  Memory Used         : {memory_used_mb:.2f} MB")

