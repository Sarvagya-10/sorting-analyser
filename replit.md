# Sorting Analyser - Streamlit Application

## Overview

This is a Python Streamlit application that provides a comprehensive analysis and visualization tool for sorting algorithms. The app allows users to compare different sorting algorithms by measuring their performance metrics including execution time, memory usage, comparisons, and swaps across various input types and sizes.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Framework**: Streamlit - chosen for its simplicity in creating interactive web applications with minimal code
- **Layout**: Wide layout with sidebar configuration for better user experience
- **Components**: 
  - Sidebar for user controls and settings
  - Main area for results display and visualizations
  - Interactive charts and data tables

### Backend Architecture
- **Core Logic**: Python-based sorting algorithm implementations
- **Performance Monitoring**: Custom `PerformanceProfiler` context manager for tracking metrics
- **Data Processing**: Pandas DataFrames for storing and analyzing results
- **Visualization Engine**: Dual approach using both Matplotlib and Plotly for different chart types

### Key Design Decisions
1. **Context Manager Pattern**: Used for performance profiling to ensure proper resource cleanup
2. **Modular Algorithm Structure**: Each sorting algorithm is implemented as a separate function with consistent interfaces
3. **Real-time Metrics**: Algorithms are instrumented to track comparisons, swaps, and memory usage during execution
4. **Flexible Input Generation**: Multiple input types (random, sorted, reverse sorted, nearly sorted) for comprehensive testing

## Key Components

### 1. Performance Profiling System
- **PerformanceProfiler Class**: Context manager that tracks:
  - Execution time using `time.perf_counter()`
  - Memory usage using `tracemalloc`
  - Algorithm-specific metrics (comparisons, swaps, assignments)
- **Instrumentation**: Methods to increment counters during algorithm execution

### 2. Sorting Algorithm Implementations
- **Algorithms Included**: Bubble Sort, Selection Sort, Insertion Sort, Merge Sort, Quick Sort, Heap Sort
- **Performance Tracking**: Each algorithm integrated with the profiler
- **Consistent Interface**: All algorithms follow the same input/output pattern

### 3. Input Generation System
- **Types**: Random, sorted, reverse sorted, nearly sorted arrays
- **Configuration**: User-defined size (10-10,000 elements) and value ranges
- **Flexibility**: Parameterized generation for different test scenarios

### 4. Benchmarking Framework
- **Multi-algorithm Support**: Run multiple algorithms simultaneously
- **Statistical Analysis**: Multiple runs with averaging, min/max, standard deviation
- **Data Storage**: Results stored in Pandas DataFrames for easy manipulation

### 5. Visualization System
- **Chart Types**:
  - Runtime vs. Input Size
  - Comparisons vs. Input Size  
  - Swaps/Assignments vs. Input Size
  - Peak Memory Usage vs. Input Size
- **Libraries**: Plotly for interactive charts, Matplotlib for static visualizations
- **Features**: Tooltips, legends, theoretical Big O curve overlays

## Data Flow

1. **User Input**: User selects algorithms, input size, type, and repetitions via sidebar
2. **Input Generation**: System generates test arrays based on user specifications
3. **Algorithm Execution**: Selected algorithms run with performance profiling
4. **Data Collection**: Metrics collected and stored in DataFrames
5. **Statistical Analysis**: Results processed for averages, min/max, standard deviation
6. **Visualization**: Charts generated from processed data
7. **Display**: Results presented in tables and interactive charts

## External Dependencies

### Core Libraries
- **streamlit**: Web application framework
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **matplotlib**: Static plotting
- **plotly**: Interactive visualizations
- **time**: Performance timing
- **tracemalloc**: Memory profiling
- **random**: Random number generation
- **statistics**: Statistical calculations

### Standard Library
- **typing**: Type hints for better code documentation
- **functools**: Decorator utilities
- **io**: Input/output operations

## Deployment Strategy

### Development Environment
- **Platform**: Replit-compatible Python environment
- **Dependencies**: All dependencies are standard Python packages available via pip
- **Configuration**: Streamlit app configuration set for wide layout and expanded sidebar

### Production Considerations
- **Scalability**: Input size limited to 10,000 elements to prevent performance issues
- **Memory Management**: Proper cleanup using context managers
- **Error Handling**: Robust error handling for edge cases in sorting algorithms
- **Performance**: Efficient implementations with minimal overhead

### Running the Application
- **Command**: `streamlit run sorting_analyser.py`
- **Port**: Default Streamlit port (8501)
- **Browser**: Automatic browser opening for user interface

## Technical Implementation Notes

### Algorithm Instrumentation
- Each sorting algorithm is wrapped with performance tracking
- Comparisons and swaps are counted using the profiler's methods
- Memory usage is tracked at peak execution points

### Data Structure Choice
- Pandas DataFrames chosen for result storage due to:
  - Easy statistical operations
  - Seamless integration with visualization libraries
  - Tabular display capabilities in Streamlit

### Visualization Strategy
- Plotly preferred for interactive charts with tooltips and zoom
- Matplotlib used for simpler static visualizations
- Consistent color schemes and styling across all charts

This architecture provides a comprehensive, user-friendly tool for analyzing sorting algorithm performance while maintaining clean, maintainable code structure.