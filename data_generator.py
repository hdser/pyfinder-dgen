import os
import random
import numpy as np
import pandas as pd
from typing import List, Tuple, Generator, Dict, Optional
from enum import Enum
from dataclasses import dataclass
import math
from multiprocessing import Pool, cpu_count
from functools import partial
import sys
from tqdm import tqdm
import argparse
import json

class NetworkType(Enum):
    SCALE_FREE = "scale_free"
    RANDOM = "random"
    SMALL_WORLD = "small_world"
    HIERARCHICAL = "hierarchical"
    CORE_PERIPHERY = "core_periphery"
    COMMUNITY = "community"
    BOTTLENECK = "bottleneck"
    STAR = "star"


@dataclass
class NetworkPattern:
    """Configure specific network patterns."""
    num_communities: int = 3
    community_density: float = 0.7
    inter_community_density: float = 0.1
    num_bottlenecks: int = 2
    bottleneck_connections: int = 5
    core_size_ratio: float = 0.1
    token_concentration: float = 0.8
    hub_nodes: int = 5
    hub_connection_ratio: float = 0.3
    min_tokens_per_address: int = 3
    target_tokens_per_address: int = 50


class NetworkValidator:
    """Validate and analyze generated networks."""

    @staticmethod
    def validate_trust_relationships(trust_df: pd.DataFrame) -> Dict:
        """Validate trust relationships and return metrics."""
        metrics = {
            "total_relationships": len(trust_df),
            "unique_trusters": len(trust_df['truster'].unique()),
            "unique_trustees": len(trust_df['trustee'].unique()),
            "self_trusts": len(trust_df[trust_df['truster'] == trust_df['trustee']]),
            "reciprocal_trusts": len(set(zip(trust_df['truster'], trust_df['trustee'])) &
                                   set(zip(trust_df['trustee'], trust_df['truster'])))
        }

        # Due to large size, we avoid building the entire graph
        # You can add additional validation as needed

        return metrics

    @staticmethod
    def validate_token_balances(balance_df: pd.DataFrame) -> Dict:
        """Validate token balances and return metrics."""
        metrics = {
            "total_balance_records": len(balance_df),
            "unique_accounts": len(balance_df['account'].unique()),
            "unique_tokens": len(balance_df['tokenAddress'].unique()),
            "avg_tokens_per_account": len(balance_df) / len(balance_df['account'].unique()),
            "max_tokens_per_account": balance_df['account'].value_counts().max(),
            "min_balance": float(min(balance_df['demurragedTotalBalance'].astype(float))),
            "max_balance": float(max(balance_df['demurragedTotalBalance'].astype(float))),
            "total_balance": float(sum(balance_df['demurragedTotalBalance'].astype(float)))
        }

        # Token concentration analysis
        token_holdings = balance_df.groupby('account')['tokenAddress'].count()
        metrics["token_concentration_gini"] = NetworkValidator._gini_coefficient(token_holdings.values)

        return metrics

    @staticmethod
    def _gini_coefficient(values: np.array) -> float:
        """Calculate Gini coefficient for distribution analysis."""
        sorted_values = np.sort(values)
        n = len(values)
        index = np.arange(1, n + 1)
        return ((2 * index - n - 1) * sorted_values).sum() / (n * sorted_values.sum())

    @staticmethod
    def generate_report(trust_df: pd.DataFrame, balance_df: pd.DataFrame, output_dir: str):
        """Generate comprehensive validation report."""
        trust_metrics = NetworkValidator.validate_trust_relationships(trust_df)
        balance_metrics = NetworkValidator.validate_token_balances(balance_df)

        # Save metrics
        with open(os.path.join(output_dir, "validation_report.txt"), "w") as f:
            f.write("Trust Network Metrics:\n")
            f.write("=====================\n")
            for key, value in trust_metrics.items():
                f.write(f"{key}: {value}\n")

            f.write("\nToken Balance Metrics:\n")
            f.write("=====================\n")
            for key, value in balance_metrics.items():
                f.write(f"{key}: {value}\n")


class PyFinderDataGenerator:
    def __init__(self,
                 num_addresses: int = 1000000,
                 avg_trust_connections: float = 5.0,
                 avg_tokens_per_user: float = 5.0,
                 min_balance: int = 100,
                 max_balance: int = 1000000,
                 network_type: NetworkType = NetworkType.SCALE_FREE,
                 network_pattern: Optional[NetworkPattern] = None,
                 chunk_size: int = 100000,
                 n_jobs: int = None,
                 seed: int = None):
        """
        Initialize the PyFinder data generator.

        Args:
            num_addresses: Number of addresses/tokens in the network
            avg_trust_connections: Average number of trust connections per address
            avg_tokens_per_user: Average number of tokens each address holds
            min_balance: Minimum token balance (will be converted to demurraged format)
            max_balance: Maximum token balance (will be converted to demurraged format)
            network_type: Type of network structure to generate
            network_pattern: Additional network pattern configuration
            chunk_size: Size of chunks for processing and writing
            n_jobs: Number of parallel jobs to use
            seed: Random seed for reproducibility
        """
        self.num_addresses = num_addresses
        self.avg_trust_connections = avg_trust_connections
        self.avg_tokens_per_user = avg_tokens_per_user
        self.min_balance = min_balance
        self.max_balance = max_balance
        self.network_type = network_type
        self.network_pattern = network_pattern or NetworkPattern()
        self.chunk_size = chunk_size
        self.n_jobs = n_jobs or cpu_count()
        self.seed = seed
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)
        self.addresses = self._generate_addresses()
        self.total_edges = 0

    def _generate_addresses(self) -> List[str]:
        """
        Generate unique Ethereum-style addresses.

        Returns:
            List of unique hex addresses with '0x' prefix
        """
        print("Generating addresses...")
        addresses = []
        hex_chars = '0123456789abcdef'
        address_set = set()
        num_addresses = self.num_addresses

        for _ in tqdm(range(num_addresses), desc="Addresses"):
            addr = '0x' + ''.join(random.choices(hex_chars, k=40))
            while addr in address_set:
                addr = '0x' + ''.join(random.choices(hex_chars, k=40))
            addresses.append(addr)
            address_set.add(addr)

        return addresses

    def _generate_trust_edges_chunk(self, address_chunk: List[str]) -> List[Tuple[str, str]]:
        """Generate trust edges for a chunk of addresses."""
        edges = []
        if self.network_type == NetworkType.SCALE_FREE:
            for addr in address_chunk:
                degree = int(np.random.power(2.1) * self.avg_trust_connections)
                degree = max(1, degree)
                targets = np.random.choice(
                    [a for a in self.addresses if a != addr],
                    size=min(degree, len(self.addresses) - 1),
                    replace=False
                )
                edges.extend((addr, target) for target in targets)

        elif self.network_type == NetworkType.RANDOM:
            p = self.avg_trust_connections / self.num_addresses
            for source in address_chunk:
                num_targets = np.random.binomial(len(self.addresses) - 1, p)
                num_targets = max(1, num_targets)
                targets = np.random.choice(
                    [addr for addr in self.addresses if addr != source],
                    size=min(num_targets, len(self.addresses) - 1),
                    replace=False
                )
                edges.extend((source, target) for target in targets)

        elif self.network_type == NetworkType.COMMUNITY:
            # Precompute community assignments if not already done
            if not hasattr(self, 'address_to_community'):
                self.address_to_community = {}
                self.communities = {}
                nodes_per_community = self.num_addresses // self.network_pattern.num_communities
                for i in range(self.network_pattern.num_communities):
                    start_idx = i * nodes_per_community
                    end_idx = start_idx + nodes_per_community
                    community_addresses = self.addresses[start_idx:end_idx]
                    self.communities[i] = community_addresses
                    for addr in community_addresses:
                        self.address_to_community[addr] = i

            for u in address_chunk:
                i = self.address_to_community[u]
                community = self.communities[i]

                # Within-community connections
                for v in community:
                    if v != u and random.random() < self.network_pattern.community_density:
                        edges.append((u, v))

                # Inter-community connections
                other_nodes = [
                    addr for addr in self.addresses if self.address_to_community[addr] != i
                ]
                num_external = int(
                    len(other_nodes) * self.network_pattern.inter_community_density
                )
                if num_external > 0:
                    external_targets = random.sample(
                        other_nodes, min(num_external, len(other_nodes))
                    )
                    edges.extend((u, v) for v in external_targets)

        else:
            for source in address_chunk:
                num_connections = random.randint(
                    max(1, int(self.avg_trust_connections * 0.5)),
                    int(self.avg_trust_connections * 1.5)
                )
                targets = random.sample(
                    [addr for addr in self.addresses if addr != source],
                    min(num_connections, self.num_addresses - 1)
                )
                edges.extend((source, target) for target in targets)

        return edges

    def _generate_trust_edges_parallel(self) -> Generator[Tuple[str, str], None, None]:
        """Generate trust edges using parallel processing."""
        # Split addresses into chunks for parallel processing
        chunk_size = math.ceil(len(self.addresses) / (self.n_jobs * 4))  # 4 chunks per process
        address_chunks = [
            self.addresses[i:i+chunk_size]
            for i in range(0, len(self.addresses), chunk_size)
        ]

        print(f"Generating trust edges using {self.n_jobs} processes...")
        with Pool(processes=self.n_jobs) as pool:
            for chunk_edges in tqdm(pool.imap_unordered(self._generate_trust_edges_chunk, address_chunks),
                                    total=len(address_chunks), desc="Trust Edges Chunks"):
                for edge in chunk_edges:
                    yield edge

    def generate_trust_edges(self, output_file: str):
        """
        Generate trust edges and write to CSV file.
        """
        print("Generating trust edges...")
        with open(output_file, 'w') as f:
            f.write('truster,trustee\n')  # Write header

        total_edges = 0
        chunk_edges = []
        for edge in self._generate_trust_edges_parallel():
            chunk_edges.append(edge)
            if len(chunk_edges) >= self.chunk_size:
                df_chunk = pd.DataFrame(chunk_edges, columns=['truster', 'trustee'])
                df_chunk.to_csv(output_file, mode='a', header=False, index=False)
                total_edges += len(chunk_edges)
                print(f"  Written {total_edges} trust relationships...")
                chunk_edges = []

        if chunk_edges:
            df_chunk = pd.DataFrame(chunk_edges, columns=['truster', 'trustee'])
            df_chunk.to_csv(output_file, mode='a', header=False, index=False)
            total_edges += len(chunk_edges)
            print(f"  Written {total_edges} trust relationships...")

        self.total_edges = total_edges
        print(f"Total edges generated: {total_edges}")

    def _generate_token_balances_chunk(self, data: Tuple[List[str], int]) -> List[Dict]:
        """Generate token balances for a chunk of addresses."""
        addresses, chunk_id = data
        balances = []

        # Generate personal token balances
        for addr in addresses:
            balance = np.random.randint(
                max(1, int(self.max_balance * 0.7)),
                max(2, self.max_balance)
            )
            demurraged_balance = str(balance) + "0" * 15
            balances.append({
                'account': addr,
                'tokenAddress': addr,
                'demurragedTotalBalance': demurraged_balance
            })

        # Pre-calculate parameters
        mu = np.log(max(1, self.min_balance) + (self.max_balance - max(1, self.min_balance)) / 4)
        sigma = 0.5

        # Generate other token balances
        for address in addresses:
            num_tokens = int(np.clip(
                np.random.normal(
                    self.network_pattern.target_tokens_per_address,
                    self.network_pattern.target_tokens_per_address * 0.2
                ),
                self.network_pattern.min_tokens_per_address,
                self.avg_tokens_per_user
            ))

            if num_tokens > 0:
                token_sources = random.sample(
                    [addr for addr in self.addresses if addr != address],
                    min(num_tokens, len(self.addresses) - 1)
                )

                token_balances = np.exp(np.random.normal(mu, sigma, size=len(token_sources)))
                token_balances = np.clip(
                    token_balances,
                    max(1, self.min_balance),
                    max(2, int(self.max_balance * 0.5))
                ).astype(int)

                for token, balance in zip(token_sources, token_balances):
                    demurraged_balance = str(balance) + "0" * 15
                    balances.append({
                        'account': address,
                        'tokenAddress': token,
                        'demurragedTotalBalance': demurraged_balance
                    })

        return balances

    def generate_token_balances(self, output_file: str):
        """
        Generate token balances and write to CSV file.
        """
        print("Generating token balances...")
        with open(output_file, 'w') as f:
            f.write('account,tokenAddress,demurragedTotalBalance\n')  # Write header

        # Split addresses into chunks
        chunk_size = math.ceil(len(self.addresses) / (self.n_jobs * 4))
        address_chunks = [
            (self.addresses[i:i+chunk_size], idx)
            for idx, i in enumerate(range(0, len(self.addresses), chunk_size))
        ]

        total_balances = 0
        with Pool(processes=self.n_jobs) as pool:
            for chunk_balances in tqdm(pool.imap_unordered(self._generate_token_balances_chunk, address_chunks),
                                       total=len(address_chunks), desc="Token Balances Chunks"):
                df_chunk = pd.DataFrame(chunk_balances)
                df_chunk.to_csv(output_file, mode='a', header=False, index=False)
                total_balances += len(chunk_balances)
                print(f"  Written {total_balances} token balances...")

        print(f"Total token balances generated: {total_balances}")

    def save_to_csv(self, output_dir: str = "data"):
        """Generate and save the data to CSV files."""
        os.makedirs(output_dir, exist_ok=True)
        trust_edges_file = os.path.join(output_dir, "data-trust.csv")
        token_balances_file = os.path.join(output_dir, "data-balance.csv")

        # Generate trust edges
        self.generate_trust_edges(trust_edges_file)

        # Generate token balances
        self.generate_token_balances(token_balances_file)

        # Generate validation report
        print("\nGenerating validation report...")
        validator = NetworkValidator()

        # Read files in chunks for validation
        trust_df = pd.read_csv(trust_edges_file)
        balance_df = pd.read_csv(token_balances_file)
        validator.generate_report(trust_df, balance_df, output_dir)

        print("Data generation completed.")


def generate_example_networks():
    """Generate example networks with different characteristics."""

    # Test network with bottlenecks
    bottleneck_pattern = NetworkPattern(
        num_bottlenecks=3,
        bottleneck_connections=5,
        min_tokens_per_address=3,
        target_tokens_per_address=5
    )

    bottleneck_network = PyFinderDataGenerator(
        num_addresses=1000000,
        chunk_size=100000,
        n_jobs=8,
        avg_trust_connections=5.0,
        avg_tokens_per_user=5.0,
        network_type=NetworkType.BOTTLENECK,
        network_pattern=bottleneck_pattern
    )
    bottleneck_network.save_to_csv("data_bottleneck")

    # Community structure network
    community_pattern = NetworkPattern(
        num_communities=5,
        community_density=0.7,
        inter_community_density=0.1,
        min_tokens_per_address=3,
        target_tokens_per_address=6
    )

    community_network = PyFinderDataGenerator(
        num_addresses=1000000,
        chunk_size=100000,
        n_jobs=8,
        avg_trust_connections=8.0,
        avg_tokens_per_user=6.0,
        network_type=NetworkType.COMMUNITY,
        network_pattern=community_pattern
    )
    community_network.save_to_csv("data_community")

    # Core-periphery network
    core_pattern = NetworkPattern(
        core_size_ratio=0.1,
        token_concentration=0.8,
        min_tokens_per_address=4,
        target_tokens_per_address=8
    )

    core_network = PyFinderDataGenerator(
        num_addresses=1000000,
        chunk_size=100000,
        n_jobs=8,
        avg_trust_connections=10.0,
        avg_tokens_per_user=8.0,
        network_type=NetworkType.CORE_PERIPHERY,
        network_pattern=core_pattern
    )
    core_network.save_to_csv("data_core_periphery")

    # Large scale-free network
    large_pattern = NetworkPattern(
        hub_nodes=50,
        hub_connection_ratio=0.2,
        token_concentration=0.6,
        min_tokens_per_address=5,
        target_tokens_per_address=10
    )

    large_network = PyFinderDataGenerator(
        num_addresses=1000000,
        chunk_size=100000,
        n_jobs=8,
        avg_trust_connections=15.0,
        avg_tokens_per_user=10.0,
        network_type=NetworkType.SCALE_FREE,
        network_pattern=large_pattern
    )
    large_network.save_to_csv("data_large_scale_free")


def generate_networks(configs, output_base_dir="data"):
    """Generate example networks with different characteristics."""
    for config in configs:
        # Extract network pattern parameters
        pattern_params = config.get('network_pattern', {})
        network_pattern = NetworkPattern(**pattern_params)

        # Extract other generator parameters
        generator_params = {
            'num_addresses': config.get('num_addresses', 1000000),
            'avg_trust_connections': config.get('avg_trust_connections', 5.0),
            'avg_tokens_per_user': config.get('avg_tokens_per_user', 5.0),
            'min_balance': config.get('min_balance', 100),
            'max_balance': config.get('max_balance', 1000000),
            'network_type': NetworkType[config.get('network_type', 'SCALE_FREE')],
            'network_pattern': network_pattern,
            'chunk_size': config.get('chunk_size', 100000),
            'n_jobs': config.get('n_jobs', None),
            'seed': config.get('seed', None)
        }

        generator = PyFinderDataGenerator(**generator_params)

        # Determine output directory
        output_dir = os.path.join(output_base_dir, config.get('output_dir', 'data'))

        print(f"Generating network: {config.get('output_dir', 'data')}")

        # Generate data
        generator.save_to_csv(output_dir=output_dir)

def main():
    parser = argparse.ArgumentParser(description='PyFinder Data Generator')
    parser.add_argument('--config_file', type=str, help='Path to configuration file (JSON)')
    parser.add_argument('--output_base_dir', type=str, default='data', help='Base output directory')
    args = parser.parse_args()

    if args.config_file:
        with open(args.config_file, 'r') as f:
            configs = json.load(f)
        generate_networks(configs, output_base_dir=args.output_base_dir)
    else:
        # If no config file is provided, you can run a default or exit
        print("Please provide a configuration file using --config_file")
        exit(1)

if __name__ == "__main__":
    main()
