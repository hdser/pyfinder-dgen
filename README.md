# PyFinder-DGen: PyFinder Data Generator

<img src="img/pyfinderdgen.webp" alt="pyfinder" style="height:350px;width:900px;">


A network generator for creating synthetic trust networks with various topological patterns. This tool is designed to generate realistic network structures for testing and simulation purposes.

## Features

- Multiple network topology types
- Configurable reciprocal trust relationships
- Parallel processing for high performance
- Token balance generation with customizable distributions
- Comprehensive network analytics and visualization
- Docker support for containerized execution

## Network Types

### 1. Scale-Free Network
- **Implementation**: Preferential attachment model using power-law degree distribution
- **Use Case**: Mimics real-world social networks and financial systems
- **Key Parameters**:
  - `hub_nodes`: Number of high-degree nodes
  - `hub_connection_ratio`: Fraction of connections going to hubs
  - `token_concentration`: Degree of token wealth concentration

```json
{
    "network_type": "SCALE_FREE",
    "network_pattern": {
        "hub_nodes": 50,
        "hub_connection_ratio": 0.2,
        "token_concentration": 0.6
    }
}
```

### 2. Community Structure
- **Implementation**: Dense intra-community and sparse inter-community connections
- **Use Case**: Models networks with distinct groups or clusters
- **Key Parameters**:
  - `num_communities`: Number of distinct communities
  - `community_density`: Connection density within communities
  - `inter_community_density`: Connection density between communities

```json
{
    "network_type": "COMMUNITY",
    "network_pattern": {
        "num_communities": 5,
        "community_density": 0.7,
        "inter_community_density": 0.1
    }
}
```

### 3. Core-Periphery Network
- **Implementation**: Dense core with sparse peripheral connections
- **Use Case**: Financial networks with central institutions
- **Key Parameters**:
  - `core_size_ratio`: Size of the core relative to network
  - `token_concentration`: Concentration of tokens in core

```json
{
    "network_type": "CORE_PERIPHERY",
    "network_pattern": {
        "core_size_ratio": 0.1,
        "token_concentration": 0.8
    }
}
```

### 4. Bottleneck Network
- **Implementation**: Communities connected through bottleneck nodes
- **Use Case**: Networks with gatekeepers or intermediaries
- **Key Parameters**:
  - `num_bottlenecks`: Number of bottleneck nodes
  - `bottleneck_connections`: Connections per bottleneck

```json
{
    "network_type": "BOTTLENECK",
    "network_pattern": {
        "num_bottlenecks": 3,
        "bottleneck_connections": 5
    }
}
```

## Configuration Parameters

### Global Parameters
```json
{
    "output_dir": "data_output",
    "num_addresses": 100000,
    "chunk_size": 10000,
    "n_jobs": 8,
    "avg_trust_connections": 5.0,
    "avg_tokens_per_user": 5.0
}
```

### Network Pattern Parameters
```json
{
    "min_tokens_per_address": 3,
    "target_tokens_per_address": 5,
    "reciprocal_trust_ratio": 0.3
}
```

## Network Testing Framework

### Overview
The generator includes a comprehensive testing framework that analyzes the generated networks for specific patterns and characteristics. These tests help validate that the generated networks exhibit the expected properties and structural patterns.

### Enabling Tests
Tests can be enabled in two ways:

1. **Via Configuration File**:
```json
{
    "output_dir": "data_scale_free",
    "network_type": "SCALE_FREE",
    "run_tests": true,  // Enable tests for this configuration
    "network_pattern": {
        "hub_nodes": 50,
        "hub_connection_ratio": 0.2
    }
}
```

2. **Via Command Line**:
```bash
python data_generator.py --config_file config.json --run_tests
```

### Test Cases
The framework generates several types of test cases to validate different network characteristics:

1. **Long Path Tests**
   - Identifies paths requiring more than 3 hops
   - Validates network connectivity and path length distribution
   - Useful for testing information flow across the network

2. **Bottleneck Tests**
   - Identifies critical nodes that bridge different network components
   - Validates the presence of expected network bottlenecks
   - Important for understanding network resilience

3. **Community Tests**
   - **Intra-Community Flow**: Tests connections within communities
   - **Inter-Community Flow**: Tests connections between different communities
   - Validates community structure and connectivity patterns

4. **High Balance Tests**
   - Tests flows between nodes with high token balances
   - Validates token distribution and economic network structure

### Test Output
Tests generate a detailed report (`test_cases_report.txt`) containing:

```
Network Test Cases Report
========================
Network Type: scale_free
Number of Addresses: 100000
Average Trust Connections: 15.0

Test Case: Long Path
===================
Description: Path requiring 4 hops
Source: 0x123...
Target: 0x456...
Expected Properties:
  min_path_length: 3

[Additional test cases...]
```

### Understanding Test Results

1. **Long Path Results**
   - Look for paths > 3 hops
   - Higher numbers indicate more complex network structure
   - Useful for estimating network diameter

2. **Bottleneck Analysis**
   - Identifies critical network junctions
   - Important for:
     - Network resilience assessment
     - Identifying potential congestion points
     - Understanding network centralization

3. **Community Structure**
   - Intra-community density should be higher than inter-community
   - Validates proper community formation
   - Helps understand network modularity

4. **Balance Distribution**
   - Validates token concentration patterns
   - Ensures economic network properties
   - Tests wealth distribution characteristics

### Example Test Configuration

```json
{
    "output_dir": "data_community",
    "num_addresses": 100000,
    "network_type": "COMMUNITY",
    "run_tests": true,
    "network_pattern": {
        "num_communities": 5,
        "community_density": 0.7,
        "inter_community_density": 0.1
    }
}
```


## Installation & Usage

### Using Python Directly

1. Clone the repository:
```bash
git clone https://github.com/yourusername/pyfinder-network-generator.git
cd pyfinder-network-generator
```

2. Install requirements:
```bash
pip install -r requirements.txt
```

3. Create a configuration file (e.g., `config.json`):
```json
[
    {
        "output_dir": "data_scale_free",
        "network_type": "SCALE_FREE",
        "num_addresses": 100000,
        "network_pattern": {
            "hub_nodes": 50,
            "hub_connection_ratio": 0.2
        }
    }
]
```

4. Run the generator:
```bash
python data_generator.py --config_file config.json --output_base_dir output
```

### Using Docker Compose

1. Build and run using docker-compose:
```bash
docker-compose up --build
```

2. The generated data will be available in the `output` directory.

## Output Files

The generator produces the following files for each network:

- `data-trust.csv`: Trust relationships
  ```
  truster,trustee
  0x123...,0x456...
  ```

- `data-balance.csv`: Token balances
  ```
  account,tokenAddress,demurragedTotalBalance
  0x123...,0x789...,1000000000000000
  ```

- `validation_report.txt`: Network metrics and statistics

- `network_analysis.png`: Visualizations including:
  - Trust relationship degree distribution
  - Token balance distribution
  - Tokens per account distribution
  - Reciprocal trust ratio evolution

## Performance Considerations

The generator implements several optimizations:

1. **Parallel Processing**
   - Uses multiprocessing for edge generation
   - Configurable number of worker processes
   - Chunk-based processing for memory efficiency

2. **Memory Optimization**
   - Efficient data structures (sets for edge tracking)
   - Batch processing for large networks
   - Streaming file writes for large outputs

3. **Computational Optimization**
   - Numpy operations for numerical computations
   - Optimized graph metrics without full graph construction
   - Efficient random number generation

## Example Configurations

### Large Scale-Free Network
```json
{
    "output_dir": "data_large_scale_free",
    "num_addresses": 1000000,
    "chunk_size": 100000,
    "n_jobs": 8,
    "avg_trust_connections": 15.0,
    "avg_tokens_per_user": 10.0,
    "network_type": "SCALE_FREE",
    "network_pattern": {
        "hub_nodes": 50,
        "hub_connection_ratio": 0.2,
        "token_concentration": 0.6,
        "min_tokens_per_address": 5,
        "target_tokens_per_address": 10,
        "reciprocal_trust_ratio": 0.3
    }
}
```

### Dense Community Network
```json
{
    "output_dir": "data_community",
    "num_addresses": 100000,
    "chunk_size": 10000,
    "n_jobs": 8,
    "avg_trust_connections": 8.0,
    "avg_tokens_per_user": 6.0,
    "network_type": "COMMUNITY",
    "network_pattern": {
        "num_communities": 5,
        "community_density": 0.7,
        "inter_community_density": 0.1,
        "min_tokens_per_address": 3,
        "target_tokens_per_address": 6,
        "reciprocal_trust_ratio": 0.4
    }
}
```

## Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
