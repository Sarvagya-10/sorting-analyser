# Sorting Analyser

## Overview
Sorting Analyser is an interactive Streamlit application that allows users to visualize, compare, and analyze the time and space complexities of various sorting algorithms. The application provides detailed performance metrics with interactive charts for different sorting algorithms.

## Features
- **Multiple Sorting Algorithms**: Bubble Sort, Selection Sort, Insertion Sort, Merge Sort, Quick Sort, and Heap Sort
- **Customizable Input**: Configure input size and type (Random, Sorted, Reverse Sorted, Nearly Sorted)
- **Performance Metrics**: Runtime, Comparisons, Swaps/Assignments, and Memory Usage
- **Interactive Visualizations**: Compare algorithm performance with interactive charts
- **Data Export**: Download analysis results as CSV

## How to Run
1. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```
2. Run the Streamlit application:
   ```
   streamlit run sorting_analyser.py
   ```

## Technologies Used
- Python
- Streamlit
- Pandas
- NumPy
- Matplotlib
- Plotly

## Implementation Details
The application includes:
- Performance profiling for each algorithm
- Various input generators for different test cases
- Benchmarking with repetitions for statistical significance
- Interactive data visualization

## License
MIT