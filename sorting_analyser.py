import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
import tracemalloc
import random
import statistics
from typing import List, Tuple, Dict, Any, Callable
from functools import wraps
import io

# Page configuration
st.set_page_config(
    page_title="Sorting Analyser",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

class PerformanceProfiler:
    """Context manager for tracking performance metrics of sorting algorithms."""
    
    def __init__(self):
        self.comparisons = 0
        self.swaps = 0
        self.assignments = 0
        self.start_time = None
        self.end_time = None
        self.peak_memory = 0
        
    def __enter__(self):
        """Start profiling."""
        tracemalloc.start()
        self.start_time = time.perf_counter()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop profiling and calculate metrics."""
        self.end_time = time.perf_counter()
        current, peak = tracemalloc.get_traced_memory()
        self.peak_memory = peak
        tracemalloc.stop()
        
    def compare(self) -> bool:
        """Increment comparison counter."""
        self.comparisons += 1
        return True
        
    def swap(self):
        """Increment swap counter."""
        self.swaps += 1
        
    def assign(self):
        """Increment assignment counter."""
        self.assignments += 1
        
    def get_runtime(self) -> float:
        """Get runtime in milliseconds."""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time) * 1000
        return 0
        
    def get_memory_mb(self) -> float:
        """Get peak memory usage in MB."""
        return self.peak_memory / (1024 * 1024)

class SortingAlgorithms:
    """Collection of sorting algorithms with performance instrumentation."""
    
    @staticmethod
    def bubble_sort(arr: List[int], profiler: PerformanceProfiler) -> List[int]:
        """
        Bubble Sort implementation with performance tracking.
        Time Complexity: O(n²), Space Complexity: O(1)
        """
        arr = arr.copy()
        n = len(arr)
        
        for i in range(n):
            swapped = False
            for j in range(0, n - i - 1):
                if profiler.compare() and arr[j] > arr[j + 1]:
                    arr[j], arr[j + 1] = arr[j + 1], arr[j]
                    profiler.swap()
                    swapped = True
            if not swapped:
                break
                
        return arr
    
    @staticmethod
    def selection_sort(arr: List[int], profiler: PerformanceProfiler) -> List[int]:
        """
        Selection Sort implementation with performance tracking.
        Time Complexity: O(n²), Space Complexity: O(1)
        """
        arr = arr.copy()
        n = len(arr)
        
        for i in range(n):
            min_idx = i
            for j in range(i + 1, n):
                if profiler.compare() and arr[j] < arr[min_idx]:
                    min_idx = j
            
            if min_idx != i:
                arr[i], arr[min_idx] = arr[min_idx], arr[i]
                profiler.swap()
                
        return arr
    
    @staticmethod
    def insertion_sort(arr: List[int], profiler: PerformanceProfiler) -> List[int]:
        """
        Insertion Sort implementation with performance tracking.
        Time Complexity: O(n²), Space Complexity: O(1)
        """
        arr = arr.copy()
        
        for i in range(1, len(arr)):
            key = arr[i]
            profiler.assign()
            j = i - 1
            
            while j >= 0 and profiler.compare() and arr[j] > key:
                arr[j + 1] = arr[j]
                profiler.assign()
                j -= 1
                
            arr[j + 1] = key
            profiler.assign()
            
        return arr
    
    @staticmethod
    def merge_sort(arr: List[int], profiler: PerformanceProfiler) -> List[int]:
        """
        Merge Sort implementation with performance tracking.
        Time Complexity: O(n log n), Space Complexity: O(n)
        """
        def merge(left: List[int], right: List[int]) -> List[int]:
            result = []
            i = j = 0
            
            while i < len(left) and j < len(right):
                if profiler.compare() and left[i] <= right[j]:
                    result.append(left[i])
                    profiler.assign()
                    i += 1
                else:
                    result.append(right[j])
                    profiler.assign()
                    j += 1
                    
            result.extend(left[i:])
            result.extend(right[j:])
            profiler.assignments += len(left[i:]) + len(right[j:])
            
            return result
        
        def merge_sort_helper(arr: List[int]) -> List[int]:
            if len(arr) <= 1:
                return arr
                
            mid = len(arr) // 2
            left = merge_sort_helper(arr[:mid])
            right = merge_sort_helper(arr[mid:])
            
            return merge(left, right)
        
        return merge_sort_helper(arr.copy())
    
    @staticmethod
    def quick_sort(arr: List[int], profiler: PerformanceProfiler) -> List[int]:
        """
        Quick Sort implementation with performance tracking.
        Time Complexity: O(n log n) average, O(n²) worst, Space Complexity: O(log n)
        """
        def partition(arr: List[int], low: int, high: int) -> int:
            pivot = arr[high]
            i = low - 1
            
            for j in range(low, high):
                if profiler.compare() and arr[j] <= pivot:
                    i += 1
                    arr[i], arr[j] = arr[j], arr[i]
                    profiler.swap()
                    
            arr[i + 1], arr[high] = arr[high], arr[i + 1]
            profiler.swap()
            return i + 1
        
        def quick_sort_helper(arr: List[int], low: int, high: int):
            if low < high:
                pi = partition(arr, low, high)
                quick_sort_helper(arr, low, pi - 1)
                quick_sort_helper(arr, pi + 1, high)
        
        arr = arr.copy()
        quick_sort_helper(arr, 0, len(arr) - 1)
        return arr
    
    @staticmethod
    def heap_sort(arr: List[int], profiler: PerformanceProfiler) -> List[int]:
        """
        Heap Sort implementation with performance tracking.
        Time Complexity: O(n log n), Space Complexity: O(1)
        """
        def heapify(arr: List[int], n: int, i: int):
            largest = i
            left = 2 * i + 1
            right = 2 * i + 2
            
            if left < n and profiler.compare() and arr[left] > arr[largest]:
                largest = left
                
            if right < n and profiler.compare() and arr[right] > arr[largest]:
                largest = right
                
            if largest != i:
                arr[i], arr[largest] = arr[largest], arr[i]
                profiler.swap()
                heapify(arr, n, largest)
        
        arr = arr.copy()
        n = len(arr)
        
        # Build max heap
        for i in range(n // 2 - 1, -1, -1):
            heapify(arr, n, i)
            
        # Extract elements from heap
        for i in range(n - 1, 0, -1):
            arr[0], arr[i] = arr[i], arr[0]
            profiler.swap()
            heapify(arr, i, 0)
            
        return arr

class InputGenerator:
    """Generates different types of input arrays for testing."""
    
    @staticmethod
    def generate_random(size: int, min_val: int = 1, max_val: int = 1000) -> List[int]:
        """Generate random array."""
        return [random.randint(min_val, max_val) for _ in range(size)]
    
    @staticmethod
    def generate_sorted(size: int, min_val: int = 1, max_val: int = 1000) -> List[int]:
        """Generate sorted array."""
        return sorted([random.randint(min_val, max_val) for _ in range(size)])
    
    @staticmethod
    def generate_reverse_sorted(size: int, min_val: int = 1, max_val: int = 1000) -> List[int]:
        """Generate reverse sorted array."""
        return sorted([random.randint(min_val, max_val) for _ in range(size)], reverse=True)
    
    @staticmethod
    def generate_nearly_sorted(size: int, shuffle_percent: float = 0.1, min_val: int = 1, max_val: int = 1000) -> List[int]:
        """Generate nearly sorted array with small percentage shuffled."""
        arr = sorted([random.randint(min_val, max_val) for _ in range(size)])
        shuffle_count = int(size * shuffle_percent)
        
        for _ in range(shuffle_count):
            i, j = random.randint(0, size - 1), random.randint(0, size - 1)
            arr[i], arr[j] = arr[j], arr[i]
            
        return arr

class BenchmarkRunner:
    """Handles benchmarking of sorting algorithms."""
    
    def __init__(self):
        self.algorithms = {
            'Bubble Sort': SortingAlgorithms.bubble_sort,
            'Selection Sort': SortingAlgorithms.selection_sort,
            'Insertion Sort': SortingAlgorithms.insertion_sort,
            'Merge Sort': SortingAlgorithms.merge_sort,
            'Quick Sort': SortingAlgorithms.quick_sort,
            'Heap Sort': SortingAlgorithms.heap_sort
        }
        
    def run_benchmark(self, algorithm_name: str, input_array: List[int], repetitions: int = 1) -> Dict[str, Any]:
        """Run benchmark for a specific algorithm."""
        if algorithm_name not in self.algorithms:
            raise ValueError(f"Algorithm {algorithm_name} not found")
            
        algorithm = self.algorithms[algorithm_name]
        results = {
            'algorithm': algorithm_name,
            'input_size': len(input_array),
            'repetitions': repetitions,
            'runtimes': [],
            'comparisons': [],
            'swaps': [],
            'assignments': [],
            'peak_memory': []
        }
        
        for _ in range(repetitions):
            with PerformanceProfiler() as profiler:
                sorted_array = algorithm(input_array, profiler)
                
                results['runtimes'].append(profiler.get_runtime())
                results['comparisons'].append(profiler.comparisons)
                results['swaps'].append(profiler.swaps)
                results['assignments'].append(profiler.assignments)
                results['peak_memory'].append(profiler.get_memory_mb())
        
        # Calculate statistics
        results['avg_runtime'] = statistics.mean(results['runtimes'])
        results['min_runtime'] = min(results['runtimes'])
        results['max_runtime'] = max(results['runtimes'])
        results['std_runtime'] = statistics.stdev(results['runtimes']) if repetitions > 1 else 0
        
        results['avg_comparisons'] = statistics.mean(results['comparisons'])
        results['avg_swaps'] = statistics.mean(results['swaps'])
        results['avg_assignments'] = statistics.mean(results['assignments'])
        results['avg_peak_memory'] = statistics.mean(results['peak_memory'])
        
        return results

class Visualizer:
    """Handles creation of performance visualization charts."""
    
    @staticmethod
    def create_runtime_chart(df: pd.DataFrame) -> go.Figure:
        """Create runtime vs input size chart."""
        fig = go.Figure()
        
        for algorithm in df['algorithm'].unique():
            algo_data = df[df['algorithm'] == algorithm]
            fig.add_trace(go.Scatter(
                x=algo_data['input_size'],
                y=algo_data['avg_runtime'],
                mode='lines+markers',
                name=algorithm,
                line=dict(width=2),
                marker=dict(size=8)
            ))
        
        fig.update_layout(
            title='Runtime vs Input Size',
            xaxis_title='Input Size',
            yaxis_title='Runtime (ms)',
            hovermode='x unified',
            template='plotly_white'
        )
        
        return fig
    
    @staticmethod
    def create_comparisons_chart(df: pd.DataFrame) -> go.Figure:
        """Create comparisons vs input size chart."""
        fig = go.Figure()
        
        for algorithm in df['algorithm'].unique():
            algo_data = df[df['algorithm'] == algorithm]
            fig.add_trace(go.Scatter(
                x=algo_data['input_size'],
                y=algo_data['avg_comparisons'],
                mode='lines+markers',
                name=algorithm,
                line=dict(width=2),
                marker=dict(size=8)
            ))
        
        fig.update_layout(
            title='Comparisons vs Input Size',
            xaxis_title='Input Size',
            yaxis_title='Number of Comparisons',
            hovermode='x unified',
            template='plotly_white'
        )
        
        return fig
    
    @staticmethod
    def create_swaps_chart(df: pd.DataFrame) -> go.Figure:
        """Create swaps/assignments vs input size chart."""
        fig = go.Figure()
        
        for algorithm in df['algorithm'].unique():
            algo_data = df[df['algorithm'] == algorithm]
            swaps_assignments = algo_data['avg_swaps'] + algo_data['avg_assignments']
            fig.add_trace(go.Scatter(
                x=algo_data['input_size'],
                y=swaps_assignments,
                mode='lines+markers',
                name=algorithm,
                line=dict(width=2),
                marker=dict(size=8)
            ))
        
        fig.update_layout(
            title='Swaps + Assignments vs Input Size',
            xaxis_title='Input Size',
            yaxis_title='Number of Swaps + Assignments',
            hovermode='x unified',
            template='plotly_white'
        )
        
        return fig
    
    @staticmethod
    def create_memory_chart(df: pd.DataFrame) -> go.Figure:
        """Create memory usage vs input size chart."""
        fig = go.Figure()
        
        for algorithm in df['algorithm'].unique():
            algo_data = df[df['algorithm'] == algorithm]
            fig.add_trace(go.Scatter(
                x=algo_data['input_size'],
                y=algo_data['avg_peak_memory'],
                mode='lines+markers',
                name=algorithm,
                line=dict(width=2),
                marker=dict(size=8)
            ))
        
        fig.update_layout(
            title='Peak Memory Usage vs Input Size',
            xaxis_title='Input Size',
            yaxis_title='Peak Memory (MB)',
            hovermode='x unified',
            template='plotly_white'
        )
        
        return fig

def main():
    """Main Streamlit application."""
    
    # Title and description
    st.title("📊 Sorting Algorithm Analyser")
    st.markdown("""
    This application allows you to visualize, compare, and analyze the time and space complexities 
    of various sorting algorithms. Select algorithms, configure input parameters, and view detailed 
    performance metrics with interactive charts.
    """)
    
    # Initialize session state
    if 'results_df' not in st.session_state:
        st.session_state.results_df = pd.DataFrame()
    
    # Sidebar configuration
    st.sidebar.header("🔧 Configuration")
    
    # Algorithm selection
    st.sidebar.subheader("Select Algorithms")
    available_algorithms = ['Bubble Sort', 'Selection Sort', 'Insertion Sort', 'Merge Sort', 'Quick Sort', 'Heap Sort']
    selected_algorithms = st.sidebar.multiselect(
        "Choose algorithms to benchmark:",
        available_algorithms,
        default=['Bubble Sort', 'Merge Sort', 'Quick Sort']
    )
    
    # Input configuration
    st.sidebar.subheader("Input Configuration")
    input_size = st.sidebar.slider(
        "Input Size",
        min_value=10,
        max_value=10000,
        value=1000,
        step=10
    )
    
    # Warning for large input sizes
    if input_size > 5000:
        st.sidebar.warning("⚠️ Large input sizes may cause performance issues with O(n²) algorithms!")
    
    input_type = st.sidebar.selectbox(
        "Input Type",
        ["Random", "Sorted", "Reverse Sorted", "Nearly Sorted"]
    )
    
    repetitions = st.sidebar.slider(
        "Number of Repetitions (for averaging)",
        min_value=1,
        max_value=10,
        value=3
    )
    
    # Generate input array
    generator = InputGenerator()
    if input_type == "Random":
        input_array = generator.generate_random(input_size)
    elif input_type == "Sorted":
        input_array = generator.generate_sorted(input_size)
    elif input_type == "Reverse Sorted":
        input_array = generator.generate_reverse_sorted(input_size)
    else:  # Nearly Sorted
        input_array = generator.generate_nearly_sorted(input_size)
    
    # Run analysis button
    if st.sidebar.button("🚀 Run Analysis", type="primary"):
        if not selected_algorithms:
            st.sidebar.error("Please select at least one algorithm!")
        else:
            # Progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            benchmark_runner = BenchmarkRunner()
            results = []
            
            for i, algorithm in enumerate(selected_algorithms):
                status_text.text(f"Running {algorithm}...")
                
                with st.spinner(f"Benchmarking {algorithm}..."):
                    result = benchmark_runner.run_benchmark(algorithm, input_array, repetitions)
                    results.append(result)
                
                progress_bar.progress((i + 1) / len(selected_algorithms))
            
            # Convert results to DataFrame
            df_data = []
            for result in results:
                df_data.append({
                    'algorithm': result['algorithm'],
                    'input_size': result['input_size'],
                    'avg_runtime': result['avg_runtime'],
                    'min_runtime': result['min_runtime'],
                    'max_runtime': result['max_runtime'],
                    'std_runtime': result['std_runtime'],
                    'avg_comparisons': result['avg_comparisons'],
                    'avg_swaps': result['avg_swaps'],
                    'avg_assignments': result['avg_assignments'],
                    'avg_peak_memory': result['avg_peak_memory']
                })
            
            st.session_state.results_df = pd.DataFrame(df_data)
            status_text.success("Analysis complete!")
            progress_bar.empty()
    
    # Clear results button
    if st.sidebar.button("🗑️ Clear Results"):
        st.session_state.results_df = pd.DataFrame()
        st.sidebar.success("Results cleared!")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col2:
        st.subheader("📋 Input Preview")
        if len(input_array) > 0:
            st.write(f"**Type:** {input_type}")
            st.write(f"**Size:** {len(input_array)} elements")
            st.write(f"**First 10 elements:** {input_array[:10]}")
            st.write(f"**Last 10 elements:** {input_array[-10:]}")
    
    with col1:
        if not st.session_state.results_df.empty:
            st.subheader("📊 Performance Results")
            
            # Results table
            st.dataframe(
                st.session_state.results_df.round(4),
                use_container_width=True,
                hide_index=True
            )
            
            # Export button
            csv_buffer = io.StringIO()
            st.session_state.results_df.to_csv(csv_buffer, index=False)
            st.download_button(
                label="📥 Download Results as CSV",
                data=csv_buffer.getvalue(),
                file_name=f"sorting_analysis_{input_type}_{input_size}.csv",
                mime="text/csv"
            )
    
    # Visualizations
    if not st.session_state.results_df.empty:
        st.subheader("📈 Performance Visualizations")
        
        # Create tabs for different charts
        tab1, tab2, tab3, tab4 = st.tabs(["Runtime", "Comparisons", "Swaps/Assignments", "Memory Usage"])
        
        visualizer = Visualizer()
        
        with tab1:
            fig = visualizer.create_runtime_chart(st.session_state.results_df)
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            fig = visualizer.create_comparisons_chart(st.session_state.results_df)
            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            fig = visualizer.create_swaps_chart(st.session_state.results_df)
            st.plotly_chart(fig, use_container_width=True)
        
        with tab4:
            fig = visualizer.create_memory_chart(st.session_state.results_df)
            st.plotly_chart(fig, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown("Built with ❤️ using Streamlit")

if __name__ == "__main__":
    main()
