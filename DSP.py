import numpy as np
import time
from scipy.fft import fft

# Configuration
SAMPLE_RATE = 8000
CHUNK_SIZE = 256
FILTER_ORDER = 32
CUTOFF_FREQ = 1000.0
VAD_THRESHOLD = 0.02
BUFFER_SIZE = 4
SIMD_WIDTH = 4

# Fixed-point Q15 format
Q15_SCALE = 32768.0
Q15_MAX = 32767
Q15_MIN = -32768

def float_to_q15(x):
    return np.clip(np.round(x * Q15_SCALE), Q15_MIN, Q15_MAX).astype(np.int16)

def q15_to_float(x):
    return x.astype(np.float32) / Q15_SCALE

def q15_multiply(a, b):
    result = (a.astype(np.int32) * b.astype(np.int32)) >> 15
    return np.clip(result, Q15_MIN, Q15_MAX).astype(np.int16)

class RingBuffer:
    def __init__(self, size, chunk_size):
        self.size = size
        self.chunk_size = chunk_size
        self.buffer = np.zeros((size, chunk_size), dtype=np.float32)
        self.write_ptr = 0
        self.read_ptr = 0
        self.count = 0

    def write(self, data):
        if self.count < self.size:
            self.buffer[self.write_ptr, :] = data
            self.write_ptr = (self.write_ptr + 1) & (self.size - 1)
            self.count += 1
            return True
        return False

    def read(self):
        if self.count > 0:
            data = self.buffer[self.read_ptr, :]
            self.read_ptr = (self.read_ptr + 1) & (self.size - 1)
            self.count -= 1
            return data
        return None

class OptimizedDSP:
    def __init__(self):
        # Pre-compute Hann window
        self.window_lut = np.zeros(CHUNK_SIZE, dtype=np.float32)
        for i in range(CHUNK_SIZE):
            self.window_lut[i] = 0.5 * (1.0 - np.cos(2.0 * np.pi * i / (CHUNK_SIZE - 1)))
        self.window_lut_q15 = float_to_q15(self.window_lut)

        # Pre-compute FIR coefficients (windowed sinc)
        self.fir_coeffs = np.zeros(FILTER_ORDER, dtype=np.float32)
        cutoff_norm = CUTOFF_FREQ / (SAMPLE_RATE / 2)
        for i in range(FILTER_ORDER):
            if i == (FILTER_ORDER - 1) // 2:
                self.fir_coeffs[i] = cutoff_norm
            else:
                x = np.pi * cutoff_norm * (i - (FILTER_ORDER - 1) / 2)
                self.fir_coeffs[i] = cutoff_norm * np.sin(x) / x
            # Apply Hamming window
            self.fir_coeffs[i] *= 0.54 - 0.46 * np.cos(2.0 * np.pi * i / (FILTER_ORDER - 1))

        self.fir_coeffs_q15 = float_to_q15(self.fir_coeffs)

        # Initialize state
        self.filter_delay_line_q15 = np.zeros(FILTER_ORDER, dtype=np.int16)
        self.dc_accumulator_q15 = np.array([0], dtype=np.int16) # Initialize as a NumPy array
        self.dc_alpha_q15 = float_to_q15(0.995)

    def preprocess(self, chunk):
        """Combined DC removal + windowing in single loop"""
        chunk_q15 = float_to_q15(chunk)
        one_minus_alpha_q15 = float_to_q15(0.005)

        for i in range(len(chunk_q15)):
            # DC removal
            dc_input = q15_multiply(one_minus_alpha_q15, chunk_q15[i])
            self.dc_accumulator_q15 = q15_multiply(self.dc_alpha_q15, self.dc_accumulator_q15) + dc_input
            self.dc_accumulator_q15 = self.dc_accumulator_q15.astype(np.int16) # Ensure it remains int16
            # Remove DC and apply window
            dc_removed = chunk_q15[i] - self.dc_accumulator_q15
            chunk_q15[i] = q15_multiply(dc_removed, self.window_lut_q15[i])

        return q15_to_float(chunk_q15)

    def simd_fir_filter(self, input_data):
        """SIMD-style FIR filtering with manual multiply-accumulate"""
        output = np.zeros_like(input_data)
        input_q15 = float_to_q15(input_data)

        for i in range(len(input_q15)):
            # Shift delay line
            for j in range(FILTER_ORDER - 1, 0, -1):
                self.filter_delay_line_q15[j] = self.filter_delay_line_q15[j-1]
            self.filter_delay_line_q15[0] = input_q15[i]

            # SIMD multiply-accumulate (4 taps at once)
            accumulator = 0
            simd_groups = FILTER_ORDER // SIMD_WIDTH

            for group in range(simd_groups):
                base_idx = group * SIMD_WIDTH
                accumulator += int(self.filter_delay_line_q15[base_idx]) * int(self.fir_coeffs_q15[base_idx])
                accumulator += int(self.filter_delay_line_q15[base_idx+1]) * int(self.fir_coeffs_q15[base_idx+1])
                accumulator += int(self.filter_delay_line_q15[base_idx+2]) * int(self.fir_coeffs_q15[base_idx+2])
                accumulator += int(self.filter_delay_line_q15[base_idx+3]) * int(self.fir_coeffs_q15[base_idx+3])

            # Handle remaining taps
            for j in range(simd_groups * SIMD_WIDTH, FILTER_ORDER):
                accumulator += int(self.filter_delay_line_q15[j]) * int(self.fir_coeffs_q15[j])

            output_q15 = np.clip(accumulator >> 15, Q15_MIN, Q15_MAX)
            output[i] = q15_to_float(np.array([output_q15], dtype=np.int16))[0]

        return output

    def compute_magnitude(self, data):
        """Manual FFT magnitude computation"""
        fft_result = fft(data)
        n_pos = len(data) // 2
        magnitude = np.zeros(n_pos, dtype=np.float32)

        for i in range(n_pos):
            real_part = fft_result[i].real
            imag_part = fft_result[i].imag
            magnitude[i] = np.sqrt(real_part * real_part + imag_part * imag_part)

        return magnitude

    def analyze(self, filtered, magnitude):
        """Branchless post-processing"""
        # Manual energy calculation
        energy = sum(x * x for x in filtered) / len(filtered)
        voice_detected = energy > VAD_THRESHOLD

        # Manual peak detection
        max_magnitude = max(magnitude)
        peak_idx = magnitude.argmax()
        dominant_freq = peak_idx * SAMPLE_RATE / (2 * len(magnitude))

        # Lightweight event classification
        if voice_detected:
            event_type = "low_freq" if dominant_freq < 300 else ("speech" if dominant_freq < 1000 else "high_freq")
        else:
            event_type = "noise"

        return {
            'voice_detected': voice_detected,
            'energy': energy,
            'dominant_freq': dominant_freq,
            'event_type': event_type
        }

def generate_test_signal():
    """Generate minimal test signal"""
    duration = 1.0
    t = np.linspace(0, duration, int(duration * SAMPLE_RATE), endpoint=False)
    signal = (0.5 * np.sin(2 * np.pi * 440 * t) +
              0.3 * np.sin(2 * np.pi * 880 * t) +
              0.1 * np.random.randn(len(t)))
    return signal + 0.1  # Add DC offset

def run_dsp_pipeline():
    """Minimal DSP pipeline execution"""
    dsp = OptimizedDSP()
    ring_buffer = RingBuffer(BUFFER_SIZE, CHUNK_SIZE)

    # Generate and chunk test signal
    test_signal = generate_test_signal()
    chunks = [test_signal[i:i+CHUNK_SIZE] for i in range(0, len(test_signal)-CHUNK_SIZE+1, CHUNK_SIZE)]

    results = []
    processing_times = []

    for chunk in chunks:
        if not ring_buffer.write(chunk):
            continue

        buffer_chunk = ring_buffer.read()
        if buffer_chunk is not None:
            start_time = time.time()

            # Pipeline stages
            preprocessed = dsp.preprocess(buffer_chunk)
            filtered = dsp.simd_fir_filter(preprocessed)
            magnitude = dsp.compute_magnitude(filtered)
            analysis = dsp.analyze(filtered, magnitude)

            processing_time = time.time() - start_time
            processing_times.append(processing_time)

            results.append(analysis)

    # Minimal performance report
    avg_time_ms = np.mean(processing_times) * 1000
    chunk_duration_ms = CHUNK_SIZE / SAMPLE_RATE * 1000
    real_time_factor = chunk_duration_ms / avg_time_ms

    print(f"Processed {len(results)} chunks")
    print(f"Avg processing: {avg_time_ms:.2f}ms (limit: {chunk_duration_ms:.2f}ms)")
    print(f"Real-time factor: {real_time_factor:.1f}x")
    print(f"Status: {'REAL-TIME' if real_time_factor > 1 else 'NOT REAL-TIME'}")
    print(f"Voice events: {sum(1 for r in results if r['voice_detected'])}")

    return results

if __name__ == "__main__":
    results = run_dsp_pipeline()
    print("DSP pipeline complete")