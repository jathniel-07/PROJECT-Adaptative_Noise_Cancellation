import numpy as np
import scipy.signal as signal
from scipy.io import wavfile
import os
from tqdm import tqdm
import soundfile as sf

class ANCDataGenerator:
    def __init__(self, sample_rate=48000, num_samples=50000, duration=2.0):
        """
        Initialize ANC Data Generator
        
        Parameters:
        - sample_rate: Audio sampling rate (Hz)
        - num_samples: Total number of samples to generate
        - duration: Duration of each audio clip in seconds
        """
        self.sample_rate = sample_rate
        self.num_samples = num_samples
        self.duration = duration
        self.samples_per_clip = int(sample_rate * duration)
        
        # Create output directory
        os.makedirs('anc_training_data', exist_ok=True)
        os.makedirs('anc_training_data/noise', exist_ok=True)
        os.makedirs('anc_training_data/primary', exist_ok=True)
        os.makedirs('anc_training_data/error', exist_ok=True)
        os.makedirs('anc_training_data/reference', exist_ok=True)
    
    def generate_white_noise(self, duration):
        """Generate white noise with flat spectral density"""
        n_samples = int(self.sample_rate * duration)
        return np.random.randn(n_samples).astype(np.float32)
    
    def generate_pink_noise(self, duration):
        """Generate pink noise using Voss-McCartney algorithm"""
        n_samples = int(self.sample_rate * duration)
        
        # Simplified pink noise generation
        noise = self.generate_white_noise(duration)
        
        # Apply filter to convert white to pink
        b, a = signal.butter(5, 1000/(self.sample_rate/2), 'high')
        pink_noise = signal.filtfilt(b, a, noise)
        
        return pink_noise.astype(np.float32)
    
    def generate_brownian_noise(self, duration):
        """Generate Brownian (red) noise"""
        n_samples = int(self.sample_rate * duration)
        
        # Generate using random walk
        white_noise = self.generate_white_noise(duration)
        brown_noise = np.cumsum(white_noise)
        
        # Normalize and remove DC offset
        brown_noise = brown_noise - np.mean(brown_noise)
        brown_noise = brown_noise / np.max(np.abs(brown_noise))
        
        return brown_noise.astype(np.float32)
    
    def generate_purple_noise(self, duration):
        """Generate purple noise (high-frequency emphasis)"""
        n_samples = int(self.sample_rate * duration)
        
        white_noise = self.generate_white_noise(duration)
        
        # Differentiate to get high-pass effect
        purple_noise = np.diff(white_noise)
        purple_noise = np.append(purple_noise, 0)
        
        return purple_noise.astype(np.float32)
    
    def generate_modulated_noise(self, duration, freq_range=(50, 1000), modulation_freq=5):
        """Generate noise with frequency modulation for realistic ANC"""
        n_samples = int(self.sample_rate * duration)
        
        # Base noise
        base_noise = self.generate_white_noise(duration)
        
        # Amplitude modulation
        t = np.linspace(0, duration, n_samples)
        mod_signal = 0.5 + 0.5 * np.sin(2 * np.pi * modulation_freq * t)
        
        modulated_noise = base_noise * mod_signal
        
        return modulated_noise.astype(np.float32)
    
    def generate_periodic_noise(self, duration, frequency=50):
        """Generate periodic/tonal noise (like engine hum)"""
        n_samples = int(self.sample_rate * duration)
        t = np.linspace(0, duration, n_samples)
        
        # Fundamental + harmonics
        fundamental = 0.5 * np.sin(2 * np.pi * frequency * t)
        harmonic1 = 0.3 * np.sin(2 * np.pi * (frequency * 2) * t)
        harmonic2 = 0.2 * np.sin(2 * np.pi * (frequency * 3) * t)
        
        # Add some noise floor
        noise_floor = 0.1 * self.generate_white_noise(duration)
        
        periodic_noise = fundamental + harmonic1 + harmonic2 + noise_floor
        
        return periodic_noise.astype(np.float32)
    
    def generate_impulse_noise(self, duration):
        """Generate impulse/short burst noise"""
        n_samples = int(self.sample_rate * duration)
        
        t = np.linspace(0, duration, n_samples)
        impulse_response = np.exp(-abs(t - 0.5) * 100)
        
        # Random impulses
        impulse_train = np.zeros(n_samples)
        num_impulses = np.random.randint(5, 20)
        for _ in range(num_impulses):
            impulse_pos = np.random.randint(0, n_samples // 2)
            impulse_train[impulse_pos] += np.random.uniform(-1, 1)
        
        # Convolve with impulse response
        convolved = signal.convolve(impulse_train, impulse_response, mode='same')
        
        return (convolved / np.max(np.abs(convolved))).astype(np.float32)
    
    def add_noise_characteristics(self, noise, snr_range=(10, 30)):
        """Add realistic noise characteristics to signal"""
        # Apply low-pass filter (simulating acoustic filtering)
        b, a = signal.butter(2, 5000/(self.sample_rate/2), 'low')
        filtered_noise = signal.filtfilt(b, a, noise)
        
        # Add some compression
        filtered_noise = np.tanh(filtered_noise)
        
        return filtered_noise
    
    def create_primary_signal(self, noise_type='mixed'):
        """Create primary signal (noise to be cancelled)"""
        if noise_type == 'white':
            primary = self.generate_white_noise(self.duration)
        elif noise_type == 'pink':
            primary = self.generate_pink_noise(self.duration)
        elif noise_type == 'brownian':
            primary = self.generate_brownian_noise(self.duration)
        elif noise_type == 'periodic':
            freq = np.random.randint(50, 200)
            primary = self.generate_periodic_noise(self.duration, frequency=freq)
        elif noise_type == 'impulse':
            primary = self.generate_impulse_noise(self.duration)
        elif noise_type == 'modulated':
            freq = np.random.randint(1, 10)
            primary = self.generate_modulated_noise(self.duration, modulation_freq=freq)
        elif noise_type == 'mixed':
            # Mix multiple noise types
            n1 = self.generate_white_noise(self.duration) * 0.3
            n2 = self.generate_periodic_noise(self.duration, frequency=np.random.randint(50, 200)) * 0.4
            n3 = self.generate_modulated_noise(self.duration) * 0.3
            primary = n1 + n2 + n3
        else:
            primary = self.generate_white_noise(self.duration)
        
        # Normalize
        primary = primary / np.max(np.abs(primary))
        
        return primary
    
    def create_reference_signal(self, primary_signal):
        """Create reference signal (simulating microphone input before noise path)"""
        # Apply delay and filter to simulate acoustic path
        delay_samples = int(self.sample_rate * 0.001)  # 1ms delay
        
        reference = np.zeros(len(primary_signal))
        
        if delay_samples < len(primary_signal):
            reference[delay_samples:] = primary_signal[:-delay_samples]
        else:
            reference = primary_signal.copy()
        
        # Add some filter variation (simulating path dynamics)
        b, a = signal.butter(2, 4000/(self.sample_rate/2), 'low')
        reference = signal.filtfilt(b, a, reference)
        
        return reference
    
    def create_error_signal(self, primary_signal, cancellation_factor=0.7):
        """Create error signal (residual after partial cancellation)"""
        # Simulate partial noise cancellation
        cancelled_part = primary_signal * cancellation_factor
        
        # Add some residual noise
        residual = self.generate_white_noise(self.duration) * 0.2
        
        error = primary_signal - cancelled_part + residual
        
        return error
    
    def normalize_audio(self, audio):
        """Normalize audio to avoid clipping"""
        max_val = np.max(np.abs(audio))
        if max_val > 1.0:
            audio = audio / max_val
        return audio
    
    def save_to_wav(self, audio, filename, sample_rate=None):
        """Save audio to WAV file"""
        if sample_rate is None:
            sample_rate = self.sample_rate
        
        # Convert to appropriate format for saving
        audio_int16 = (audio * 32767).astype(np.int16)
        
        sf.write(filename, audio_int16, sample_rate)
    
    def generate_dataset(self):
        """Generate complete training dataset"""
        print(f"Generating {self.num_samples} ANC training samples...")
        print(f"Sample rate: {self.sample_rate} Hz")
        print(f"Duration per sample: {self.duration} seconds")
        
        noise_types = ['white', 'pink', 'brownian', 'periodic', 'impulse', 
                      'modulated', 'mixed']
        
        progress_bar = tqdm(total=self.num_samples)
        
        for i in range(self.num_samples):
            # Select noise type (cycling through types)
            noise_type = noise_types[i % len(noise_types)]
            
            # Generate primary signal
            primary = self.create_primary_signal(noise_type=noise_type)
            
            # Create reference signal (from microphone before cancellation path)
            reference = self.create_reference_signal(primary)
            
            # Create error signal (residual after ANC attempt)
            cancellation_factor = np.random.uniform(0.5, 0.85)
            error = self.create_error_signal(primary, cancellation_factor)
            
            # Normalize all signals
            primary = self.normalize_audio(primary)
            reference = self.normalize_audio(reference)
            error = self.normalize_audio(error)
            
            # Save to files
            prefix = f"sample_{i:06d}"
            
            save_path_primary = os.path.join('anc_training_data/primary', 
                                            f'{prefix}.wav')
            save_path_reference = os.path.join('anc_training_data/reference',
                                              f'{prefix}.wav')
            save_path_error = os.path.join('anc_training_data/error',
                                          f'{prefix}.wav')
            
            self.save_to_wav(primary, save_path_primary)
            self.save_to_wav(reference, save_path_reference)
            self.save_to_wav(error, save_path_error)
            
            # Save combined data as numpy for ML training
            np.save(f'anc_training_data/{prefix}_data.npy', {
                'primary': primary,
                'reference': reference,
                'error': error,
                'noise_type': noise_type
            })
            
            progress_bar.update(1)
        
        progress_bar.close()
        print("\nDataset generation complete!")
        print(f"Data saved to: anc_training_data/")
    
    def generate_numpy_array(self):
        """Generate dataset as numpy arrays for direct ML model training"""
        print("Generating numpy array format...")
        
        data_samples = []
        
        noise_types = ['white', 'pink', 'brownian', 'periodic', 'impulse', 
                      'modulated', 'mixed']
        
        for i in tqdm(range(self.num_samples)):
            noise_type = noise_types[i % len(noise_types)]
            
            primary = self.create_primary_signal(noise_type=noise_type)
            reference = self.create_reference_signal(primary)
            error = self.create_error_signal(primary, np.random.uniform(0.5, 0.85))
            
            # Normalize
            primary = self.normalize_audio(primary)
            reference = self.normalize_audio(reference)
            error = self.normalize_audio(error)
            
            data_samples.append({
                'primary': primary,
                'reference': reference,
                'error': error,
                'noise_type': noise_type
            })
        
        return data_samples
    
    def save_to_hdf5(self, samples):
        """Save to HDF5 format for efficient storage"""
        import h5py
        
        # Stack arrays
        primary_array = np.array([s['primary'] for s in samples])
        reference_array = np.array([s['reference'] for s in samples])
        error_array = np.array([s['error'] for s in samples])
        
        with h5py.File('anc_training_data/anc_dataset.h5', 'w') as f:
            f.create_dataset('primary', data=primary_array)
            f.create_dataset('reference', data=reference_array)
            f.create_dataset('error', data=error_array)
            f.create_dataset('noise_types', data=np.array([s['noise_type'] for s in samples]))
            
            # Add metadata
            attrs = f.attrs
            attrs['sample_rate'] = self.sample_rate
            attrs['num_samples'] = len(samples)
            attrs['samples_per_clip'] = self.samples_per_clip
            attrs['duration'] = self.duration
        
        print("Saved HDF5 dataset to anc_training_data/anc_dataset.h5")


def main():
    """Main execution function"""
    
    # Initialize generator
    generator = ANCDataGenerator(
        sample_rate=48000,      # Standard audio sampling rate
        num_samples=50000,      # Number of samples to generate
        duration=2.0           # Duration in seconds per sample
    )
    
    # Generate dataset (choose one method)
    
    # Method 1: Save as individual WAV files
    print("="*60)
    generator.generate_dataset()
    print("="*60)
    
    # Method 2: Generate and save as numpy array for direct ML training
    print("\nGenerating numpy arrays...")
    samples = generator.generate_numpy_array()
    print(f"Generated {len(samples)} samples in memory")
    
    # Optional: Save to HDF5 for efficient storage
    print("\nSaving to HDF5 format...")
    generator.save_to_hdf5(samples)
    
    print("\nTraining data generation complete!")
    print("="*60)


if __name__ == "__main__":
    main()