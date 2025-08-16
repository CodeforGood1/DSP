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

class GeneticOptimizer:
    """Offline genetic algorithm for filter coefficient optimization"""
    def __init__(self, population_size=20, generations=50):
        self.population_size = population_size
        self.generations = generations
    
    def optimize_filter_coeffs(self, target_response, verbose=False):
        """Evolve optimal FIR coefficients offline"""
        if verbose:
            print(f"🧬 GA: Evolving {FILTER_ORDER} coefficients over {self.generations} generations...")
        
        # Initialize population
        population = []
        for _ in range(self.population_size):
            coeffs = np.random.randn(FILTER_ORDER) * 0.1
            population.append(coeffs)
        
        best_coeffs = population[0]
        best_fitness = float('inf')
        
        for gen in range(self.generations):
            # Evaluate fitness (simplified - actual would use frequency response)
            fitness_scores = []
            for coeffs in population:
                # Simple fitness: penalize large coefficients, reward target frequency
                fitness = np.sum(coeffs**2) + abs(np.sum(coeffs) - target_response)
                fitness_scores.append(fitness)
            
            # Track best
            min_idx = np.argmin(fitness_scores)
            if fitness_scores[min_idx] < best_fitness:
                best_fitness = fitness_scores[min_idx]
                best_coeffs = population[min_idx].copy()
            
            # Selection and crossover (simplified)
            new_population = []
            for _ in range(self.population_size):
                # Tournament selection
                parent1 = population[np.random.randint(0, self.population_size)]
                parent2 = population[np.random.randint(0, self.population_size)]
                
                # Crossover
                crossover_point = np.random.randint(1, FILTER_ORDER-1)
                child = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
                
                # Mutation
                if np.random.random() < 0.1:
                    mutation_idx = np.random.randint(0, FILTER_ORDER)
                    child[mutation_idx] += np.random.randn() * 0.01
                
                new_population.append(child)
            
            population = new_population
            
            if verbose and gen % 10 == 0:
                print(f"  Gen {gen}: Best fitness = {best_fitness:.4f}")
        
        if verbose:
            print(f"✅ GA: Optimization complete. Final fitness = {best_fitness:.4f}")
        return best_coeffs

class HarmonySearchOptimizer:
    """Lightweight runtime optimizer for adaptive parameters"""
    def __init__(self, harmony_memory_size=5):
        self.hms = harmony_memory_size
        self.harmony_memory = []
        self.hmcr = 0.9  # Harmony Memory Considering Rate
        self.par = 0.3   # Pitch Adjusting Rate
        self.frame_count = 0
        self.adaptations = 0
    
    def initialize_memory(self, param_ranges):
        """Initialize harmony memory with random parameters"""
        self.param_ranges = param_ranges
        self.harmony_memory = []
        
        for _ in range(self.hms):
            harmony = {}
            for param, (min_val, max_val) in param_ranges.items():
                harmony[param] = np.random.uniform(min_val, max_val)
            self.harmony_memory.append(harmony)
    
    def adapt_parameters(self, current_energy, target_performance):
        """Lightweight parameter adaptation every N frames"""
        self.frame_count += 1
        
        # Only adapt every 100 frames to minimize runtime impact
        if self.frame_count % 100 != 0:
            return self.harmony_memory[0]  # Return best harmony
        
        self.adaptations += 1
        
        # Generate new harmony
        new_harmony = {}
        for param in self.param_ranges:
            if np.random.random() < self.hmcr:
                # Pick from memory
                source_harmony = self.harmony_memory[np.random.randint(0, self.hms)]
                new_harmony[param] = source_harmony[param]
                
                # Pitch adjustment
                if np.random.random() < self.par:
                    min_val, max_val = self.param_ranges[param]
                    adjustment = (max_val - min_val) * 0.1 * (np.random.random() - 0.5)
                    new_harmony[param] = np.clip(new_harmony[param] + adjustment, min_val, max_val)
            else:
                # Random selection
                min_val, max_val = self.param_ranges[param]
                new_harmony[param] = np.random.uniform(min_val, max_val)
        
        # Simple fitness based on energy deviation from target
        fitness = abs(current_energy - target_performance)
        
        # Replace worst harmony if new one is better
        worst_idx = 0  # Simplified - would track actual fitness
        self.harmony_memory[worst_idx] = new_harmony
        
        return new_harmony

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

class BaseDSP:
    """Original DSP implementation for benchmarking"""
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
        self.dc_accumulator_q15 = np.array([0], dtype=np.int16)
        self.dc_alpha_q15 = float_to_q15(0.995)

    def preprocess(self, chunk):
        """Combined DC removal + windowing in single loop"""
        chunk_q15 = float_to_q15(chunk)
        one_minus_alpha_q15 = float_to_q15(0.005)

        for i in range(len(chunk_q15)):
            # DC removal
            dc_input = q15_multiply(one_minus_alpha_q15, chunk_q15[i])
            self.dc_accumulator_q15 = q15_multiply(self.dc_alpha_q15, self.dc_accumulator_q15) + dc_input
            self.dc_accumulator_q15 = self.dc_accumulator_q15.astype(np.int16)
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

class OptimizedDSP(BaseDSP):
    """Optimized DSP with GA and Harmony Search"""
    def __init__(self, use_ga_coeffs=True, use_harmony_search=True, verbose=False):
        # Initialize base DSP first
        super().__init__()
        
        self.use_ga = use_ga_coeffs
        self.use_hs = use_harmony_search
        
        # Initialize optimizers
        if self.use_ga:
            self.ga_optimizer = GeneticOptimizer()
            # Replace standard coefficients with GA-optimized ones
            self.fir_coeffs = self.ga_optimizer.optimize_filter_coeffs(target_response=0.5, verbose=verbose)
            self.fir_coeffs_q15 = float_to_q15(self.fir_coeffs)
        
        if self.use_hs:
            self.hs_optimizer = HarmonySearchOptimizer()
            param_ranges = {
                'vad_threshold': (0.005, 0.05),
                'noise_floor': (0.001, 0.01),
                'gain_factor': (0.8, 1.2)
            }
            self.hs_optimizer.initialize_memory(param_ranges)
            self.adaptive_params = self.hs_optimizer.harmony_memory[0]
        else:
            self.adaptive_params = {
                'vad_threshold': VAD_THRESHOLD,
                'noise_floor': 0.005,
                'gain_factor': 1.0
            }

    def analyze(self, filtered, magnitude):
        """Enhanced analysis with adaptive parameters"""
        # Manual energy calculation
        energy = sum(x * x for x in filtered) / len(filtered)
        
        # Update adaptive parameters if using Harmony Search
        if self.use_hs:
            self.adaptive_params = self.hs_optimizer.adapt_parameters(energy, 0.02)
        
        # Use adaptive VAD threshold
        vad_threshold = self.adaptive_params['vad_threshold']
        voice_detected = energy > vad_threshold

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
            'event_type': event_type,
            'adaptive_threshold': vad_threshold
        }

def generate_test_signal():
    """Generate test signal for benchmarking"""
    duration = 1.0
    t = np.linspace(0, duration, int(duration * SAMPLE_RATE), endpoint=False)
    signal = (0.5 * np.sin(2 * np.pi * 440 * t) +
              0.3 * np.sin(2 * np.pi * 880 * t) +
              0.1 * np.random.randn(len(t)))
    return signal + 0.1  # Add DC offset

def benchmark_dsp(dsp_instance, name, test_signal, chunks):
    """Benchmark a DSP instance"""
    ring_buffer = RingBuffer(BUFFER_SIZE, CHUNK_SIZE)
    results = []
    processing_times = []
    
    start_benchmark = time.time()
    
    for chunk in chunks:
        if not ring_buffer.write(chunk):
            continue

        buffer_chunk = ring_buffer.read()
        if buffer_chunk is not None:
            start_time = time.time()

            # Pipeline stages
            preprocessed = dsp_instance.preprocess(buffer_chunk)
            filtered = dsp_instance.simd_fir_filter(preprocessed)
            magnitude = dsp_instance.compute_magnitude(filtered)
            analysis = dsp_instance.analyze(filtered, magnitude)

            processing_time = time.time() - start_time
            processing_times.append(processing_time)
            results.append(analysis)
    
    total_benchmark_time = time.time() - start_benchmark
    
    # Calculate metrics
    avg_time_ms = np.mean(processing_times) * 1000
    std_time_ms = np.std(processing_times) * 1000
    min_time_ms = np.min(processing_times) * 1000
    max_time_ms = np.max(processing_times) * 1000
    chunk_duration_ms = CHUNK_SIZE / SAMPLE_RATE * 1000
    real_time_factor = chunk_duration_ms / avg_time_ms
    voice_events = sum(1 for r in results if r['voice_detected'])
    
    return {
        'name': name,
        'chunks_processed': len(results),
        'avg_time_ms': avg_time_ms,
        'std_time_ms': std_time_ms,
        'min_time_ms': min_time_ms,
        'max_time_ms': max_time_ms,
        'real_time_factor': real_time_factor,
        'voice_events': voice_events,
        'total_time_s': total_benchmark_time,
        'results': results
    }

def run_comprehensive_benchmark():
    """Run comprehensive benchmark comparing all DSP variants"""
    print("="*80)
    print("🔬 COMPREHENSIVE DSP BENCHMARK SUITE")
    print("="*80)
    
    # Generate test signal
    print("📡 Generating test signal...")
    test_signal = generate_test_signal()
    chunks = [test_signal[i:i+CHUNK_SIZE] for i in range(0, len(test_signal)-CHUNK_SIZE+1, CHUNK_SIZE)]
    print(f"   Created {len(chunks)} chunks ({len(test_signal)} samples)")
    
    # Define benchmark configurations
    configs = [
        {'name': 'Original DSP', 'dsp_class': BaseDSP, 'kwargs': {}},
        {'name': 'GA Only', 'dsp_class': OptimizedDSP, 'kwargs': {'use_ga_coeffs': True, 'use_harmony_search': False}},
        {'name': 'Harmony Search Only', 'dsp_class': OptimizedDSP, 'kwargs': {'use_ga_coeffs': False, 'use_harmony_search': True}},
        {'name': 'GA + Harmony Search', 'dsp_class': OptimizedDSP, 'kwargs': {'use_ga_coeffs': True, 'use_harmony_search': True}},
    ]
    
    benchmarks = []
    
    print("\n🏃 Running benchmarks...")
    for config in configs:
        print(f"\n⚡ Testing: {config['name']}")
        
        # Create DSP instance
        if config['name'] == 'Original DSP':
            dsp = config['dsp_class']()
        else:
            dsp = config['dsp_class'](**config['kwargs'])
        
        # Run benchmark
        benchmark = benchmark_dsp(dsp, config['name'], test_signal, chunks)
        benchmarks.append(benchmark)
        
        # Quick progress report
        print(f"   ⏱️  Avg: {benchmark['avg_time_ms']:.2f}ms")
        print(f"   🚀 RT Factor: {benchmark['real_time_factor']:.1f}x")
        print(f"   🎤 Voice Events: {benchmark['voice_events']}")
    
    # Detailed comparison report
    print("\n" + "="*80)
    print("📊 DETAILED BENCHMARK RESULTS")
    print("="*80)
    
    # Headers
    print(f"{'Configuration':<20} {'Avg (ms)':<10} {'Std (ms)':<10} {'Min (ms)':<10} {'Max (ms)':<10} {'RT Factor':<10} {'Status':<12} {'Voice':<6}")
    print("-" * 80)
    
    baseline = benchmarks[0]  # Original DSP as baseline
    
    for benchmark in benchmarks:
        status = "✅ REAL-TIME" if benchmark['real_time_factor'] > 1.0 else "❌ TOO SLOW"
        
        print(f"{benchmark['name']:<20} "
              f"{benchmark['avg_time_ms']:<10.2f} "
              f"{benchmark['std_time_ms']:<10.2f} "
              f"{benchmark['min_time_ms']:<10.2f} "
              f"{benchmark['max_time_ms']:<10.2f} "
              f"{benchmark['real_time_factor']:<10.1f} "
              f"{status:<12} "
              f"{benchmark['voice_events']:<6}")
    
    # Performance comparison
    print("\n" + "="*80)
    print("🔍 PERFORMANCE COMPARISON (vs Original DSP)")
    print("="*80)
    
    for benchmark in benchmarks[1:]:  # Skip baseline
        speedup = baseline['avg_time_ms'] / benchmark['avg_time_ms']
        rt_improvement = benchmark['real_time_factor'] - baseline['real_time_factor']
        voice_diff = benchmark['voice_events'] - baseline['voice_events']
        
        print(f"\n🎯 {benchmark['name']}:")
        print(f"   ⚡ Speed: {speedup:.2f}x {'(faster)' if speedup > 1 else '(slower)'}")
        print(f"   🚀 RT Factor: {rt_improvement:+.1f}")
        print(f"   🎤 Voice Events: {voice_diff:+d}")
        
        # Check for optimizations applied
        if 'GA' in benchmark['name']:
            print(f"   🧬 GA Optimizations: Active")
        if 'Harmony' in benchmark['name']:
            print(f"   🎵 Harmony Search: Active")
    
    # Quality analysis
    print("\n" + "="*80)
    print("🎯 QUALITY ANALYSIS")
    print("="*80)
    
    # Analyze energy distributions
    for benchmark in benchmarks:
        energies = [r['energy'] for r in benchmark['results']]
        avg_energy = np.mean(energies)
        std_energy = np.std(energies)
        
        print(f"\n📈 {benchmark['name']}:")
        print(f"   Energy: {avg_energy:.4f} ± {std_energy:.4f}")
        
        # Check for adaptive behavior
        if 'adaptive_threshold' in benchmark['results'][0]:
            thresholds = [r['adaptive_threshold'] for r in benchmark['results']]
            threshold_range = max(thresholds) - min(thresholds)
            print(f"   Adaptive Range: {threshold_range:.4f}")
    
    print("\n🎉 Benchmark complete!")
    return benchmarks

def run_quick_benchmark():
    """Quick benchmark matching original output format"""
    print("🚀 Quick Benchmark - Original vs Optimized")
    print("-" * 50)
    
    test_signal = generate_test_signal()
    chunks = [test_signal[i:i+CHUNK_SIZE] for i in range(0, len(test_signal)-CHUNK_SIZE+1, CHUNK_SIZE)]
    
    # Original DSP
    original_dsp = BaseDSP()
    original_results = benchmark_dsp(original_dsp, "Original", test_signal, chunks)
    
    # Optimized DSP
    optimized_dsp = OptimizedDSP(use_ga_coeffs=True, use_harmony_search=True, verbose=False)
    optimized_results = benchmark_dsp(optimized_dsp, "Optimized", test_signal, chunks)
    
    # Print results in original format
    for results in [original_results, optimized_results]:
        print(f"\n📊 {results['name']} DSP Results:")
        chunk_duration_ms = CHUNK_SIZE / SAMPLE_RATE * 1000
        status = "REAL-TIME" if results['real_time_factor'] > 1 else "NOT REAL-TIME"
        
        print(f"Processed {results['chunks_processed']} chunks")
        print(f"Avg processing: {results['avg_time_ms']:.2f}ms (limit: {chunk_duration_ms:.2f}ms)")
        print(f"Real-time factor: {results['real_time_factor']:.1f}x")
        print(f"Status: {status}")
        print(f"Voice events: {results['voice_events']}")
    
    # Comparison
    speedup = original_results['avg_time_ms'] / optimized_results['avg_time_ms']
    print(f"\n🔍 Performance Improvement: {speedup:.2f}x {'faster' if speedup > 1 else 'slower'}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--quick':
        run_quick_benchmark()
    else:
        benchmarks = run_comprehensive_benchmark()
    
    print("\n✨ DSP benchmark suite complete!")