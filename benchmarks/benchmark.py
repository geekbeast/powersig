from abc import ABC, abstractmethod
import csv
import os
from typing import Any, Dict, Optional

from benchmarks.util import track_peak_memory
from cupy.cuda.memory import OutOfMemoryError
from benchmarks.configuration import CPU_MEMORY, CUPY_MEMORY, DURATION, GPU_MEMORY, SIGNATURE_KERNEL, RUN_ID

class Benchmark(ABC):
    def __init__(self, filename: str, csv_fields: list[str], backend: str, name: str, debug: bool = False):
        """
        Initialize the benchmark class.
        
        Args:
            filename: Path to the CSV file where results will be stored
            csv_fields: List of field names for the CSV file
            backend_name: Name of the backend being benchmarked
            name: Name of this benchmark instance
            debug: Whether to enable debug output
        """
        self.filename = filename
        self.csv_fields = csv_fields
        self.backend = backend
        self.name = name
        self.debug = debug
        self.is_ready = False
    
    def setup_writer(self) -> None:
        """Setup the CSV writer and create file if it doesn't exist."""
        file_exists = os.path.isfile(self.filename)
        
        self.file = open(self.filename, 'a', newline='')
        self.writer = csv.DictWriter(self.file, fieldnames=self.csv_fields)
        
        if not file_exists:
            self.writer.writeheader()

        self.is_ready = True
    
    @abstractmethod
    def setup(self) -> None:
        """
        Setup any necessary resources or configurations for the benchmark.
        This should be called before running any benchmarks.
        """
        pass
    
    @abstractmethod
    def before_run(self, data: Any, stats: Dict[str, Any]) -> Any:
        """
        Prepare the data/objects needed for a single benchmark run.
        This should return an object that can be used directly for benchmarking.
        
        Args:
            data: The input data to prepare for benchmarking. Expected shape is (1, length, dim)
                 where length is the length of the time series and dim is its dimension.
            stats: Dictionary to store benchmark statistics. Can be used to record
                  implementation-specific parameters.
            
        Returns:
            Any: Object that will be used for benchmarking
        """
        pass
    
    @abstractmethod
    def compute_signature_kernel(self, data: Any) -> float:
        """
        Compute the signature kernel on the prepared data.
        
        Args:
            data: The object returned by before_run(). Expected shape is (1, length, dim)
                 where length is the length of the time series and dim is its dimension.
            
        Returns:
            float: The computed signature kernel value
        """
        pass
    
    def benchmark(self, data: Any, run_id: int, params: Optional[Dict[str, Any]] = None) -> None:
        """
        Run the actual benchmark on the prepared data.
        This method handles timing and memory tracking.
        
        Args:
            data: The input data to benchmark. Expected shape is (1, length, dim)
                 where length is the length of the time series and dim is its dimension.
            run_id: Identifier for this particular benchmark run (required)
            params: Optional dictionary of additional parameters to include in stats
        """
        # Make sure that the writer is setup for saving benchmark results
        if not self.is_ready:
            self.setup_writer()

        # Initialize stats with basic info
        stats = {
            "length": data.shape[1],
            RUN_ID: run_id
        }
        
        # Merge in any additional parameters
        if params is not None:
            stats.update(params)
        
        try:
            # Prepare the data with stats dictionary
            prepared_data = self.before_run(data, stats)
            
            # Run the benchmark with memory tracking
            with track_peak_memory(self.backend, stats, data.device):
                result = self.compute_signature_kernel(prepared_data)
                stats[SIGNATURE_KERNEL] = result
                
            # Print debug information if enabled
            if self.debug:
                print(f"\nBenchmark Debug Info:")
                print(f"Benchmark: {self.name}")
                print(f"Backend: {self.backend}")
                print(f"Result: {result}")
                print(f"Duration: {stats.get(DURATION, 'N/A')} seconds")
                print(f"Memory Usage:")
                print(f"  GPU Memory: {stats.get(GPU_MEMORY, 'N/A')} MB")
                print(f"  CPU Memory: {stats.get(CPU_MEMORY, 'N/A')} MB")
                print(f"  CuPy Memory: {stats.get(CUPY_MEMORY, 'N/A')} MB")
                
            # Write results if benchmark succeeded
            self.write_results(stats)
            
        except OutOfMemoryError as e:
            print(f"Benchmark {self.name} on backend {self.backend} out of memory for time series of length {data.shape[1]}: {str(e)}")
            print(f"Input data shape: {data.shape}")
            print(f"Prepared data shape: {prepared_data.shape if hasattr(prepared_data, 'shape') else 'No shape'}")

        except Exception as e:
            print(f"Benchmark {self.name} on backend {self.backend} failed with error: {str(e)}")
            print(f"Input data shape: {data.shape}")
            print(f"Prepared data shape: {prepared_data.shape if hasattr(prepared_data, 'shape') else 'No shape'}")

    
    def write_results(self, stats: Dict[str, float]) -> None:
        """
        Write benchmark results to the CSV file.
        
        Args:
            stats: Dictionary containing benchmark results
        """
        self.writer.writerow(stats)
        self.file.flush()
    
    def cleanup(self) -> None:
        """Cleanup any resources used by the benchmark."""
        if hasattr(self, 'file'):
            self.file.close() 