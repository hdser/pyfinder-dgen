import os
import random
import numpy as np
import pandas as pd
from typing import List, Tuple, Generator, Dict, Optional, Set
from enum import Enum
from dataclasses import dataclass
import math
from multiprocessing import Pool, cpu_count
from functools import partial
import sys
from tqdm import tqdm
import argparse
import json
import matplotlib.pyplot as plt
from collections import Counter
from scipy import stats
import networkx as nx
from community import community_louvain

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
class TestCase:
    """Class to store test case information."""
    name: str
    source: str
    target: str
    description: str
    expected_properties: dict


class NetworkTestGenerator:
    """Generate test cases for network analysis."""
    
    def __init__(self, trust_df: pd.DataFrame, balance_df: pd.DataFrame):
        """Initialize with trust and balance data."""
        self.trust_df = trust_df
        self.balance_df = balance_df
        self.G = self._build_network()
        self.communities = None
        self.high_balance_nodes = self._get_high_balance_nodes()
    
    def _build_network(self) -> nx.DiGraph:
        """Build network from trust relationships efficiently."""
        print("Building network for test case generation...")
        G = nx.DiGraph()
        # Add edges in batch for better performance
        edges = list(zip(self.trust_df['truster'], self.trust_df['trustee']))
        G.add_edges_from(edges)
        return G
    
    def _get_high_balance_nodes(self, top_n: int = 20) -> List[str]:
        """Get nodes with highest total token balances."""
        try:
            self.balance_df['demurragedTotalBalance'] = self.balance_df['demurragedTotalBalance'].astype(float)
            return list(self.balance_df.groupby('account')['demurragedTotalBalance']
                       .sum()
                       .nlargest(top_n)
                       .index)
        except Exception as e:
            print(f"Warning: Could not identify high balance nodes: {str(e)}")
            return []
    
    def _detect_communities(self) -> None:
        """Detect communities using Louvain method."""
        if self.communities is None:
            try:
                print("Detecting communities...")
                G_undirected = self.G.to_undirected()
                self.communities = community_louvain.best_partition(G_undirected)
            except Exception as e:
                print(f"Warning: Could not detect communities: {str(e)}")
                self.communities = {}
    
    def _find_path_nodes(self, source: str, target: str, max_length: int = 5) -> Optional[List[str]]:
        """Find path between nodes with length limit."""
        try:
            path = nx.shortest_path(self.G, source, target)
            if len(path) <= max_length:
                return path
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            pass
        return None

    def generate_long_path_case(self) -> Optional[TestCase]:
        """Generate test case for paths longer than 3 hops."""
        try:
            connected_components = list(nx.strongly_connected_components(self.G))
            if not connected_components:
                return None
            
            largest_component = max(connected_components, key=len)
            nodes = list(largest_component)
            
            # Sample nodes for efficiency
            sample_size = min(100, len(nodes))
            sample_nodes = random.sample(nodes, sample_size)
            
            for _ in range(50):  # Try 50 times to find a suitable pair
                source, target = random.sample(sample_nodes, 2)
                try:
                    path_length = nx.shortest_path_length(self.G, source, target)
                    if path_length > 3:
                        return TestCase(
                            name="Long Path",
                            source=source,
                            target=target,
                            description=f"Path requiring {path_length} hops",
                            expected_properties={"min_path_length": 3}
                        )
                except nx.NetworkXNoPath:
                    continue
                    
        except Exception as e:
            print(f"Warning: Could not generate long path test case: {str(e)}")
        return None

    def generate_bottleneck_case(self) -> Optional[TestCase]:
        """Generate test case through network bottlenecks."""
        try:
            G_undirected = self.G.to_undirected()
            articulation_points = list(nx.articulation_points(G_undirected))
            
            if not articulation_points:
                return None
                
            bottleneck = random.choice(articulation_points)
            G_temp = G_undirected.copy()
            G_temp.remove_node(bottleneck)
            components = list(nx.connected_components(G_temp))
            
            if components:
                comp = random.choice(components)
                if len(comp) > 1:
                    source = random.choice(list(comp))
                    possible_sinks = [n for n in self.G.nodes() if n not in comp and n != bottleneck]
                    
                    if possible_sinks:
                        sink = random.choice(possible_sinks)
                        return TestCase(
                            name="Bottleneck Flow",
                            source=source,
                            target=sink,
                            description=f"Flow through bottleneck node {bottleneck}",
                            expected_properties={"bottleneck_node": bottleneck}
                        )
                        
        except Exception as e:
            print(f"Warning: Could not generate bottleneck test case: {str(e)}")
        return None

    def generate_community_cases(self) -> List[TestCase]:
        """Generate intra and inter community test cases."""
        cases = []
        self._detect_communities()
        
        if not self.communities:
            return cases

        try:
            # Intra-community case
            comm_sizes = {}
            for node, comm_id in self.communities.items():
                comm_sizes[comm_id] = comm_sizes.get(comm_id, 0) + 1
            
            large_communities = [comm_id for comm_id, size in comm_sizes.items() if size >= 5]
            
            if large_communities:
                comm_id = random.choice(large_communities)
                comm_nodes = [n for n, c in self.communities.items() if c == comm_id]
                
                for _ in range(10):
                    source, target = random.sample(comm_nodes, 2)
                    if nx.has_path(self.G, source, target):
                        cases.append(TestCase(
                            name="Intra-Community Flow",
                            source=source,
                            target=target,
                            description=f"Flow within community {comm_id}",
                            expected_properties={"community_id": comm_id}
                        ))
                        break

            # Inter-community case
            if len(set(self.communities.values())) >= 2:
                comm_ids = list(set(self.communities.values()))
                for _ in range(10):
                    c1, c2 = random.sample(comm_ids, 2)
                    nodes1 = [n for n, c in self.communities.items() if c == c1]
                    nodes2 = [n for n, c in self.communities.items() if c == c2]
                    
                    if nodes1 and nodes2:
                        source = random.choice(nodes1)
                        target = random.choice(nodes2)
                        if nx.has_path(self.G, source, target):
                            cases.append(TestCase(
                                name="Inter-Community Flow",
                                source=source,
                                target=target,
                                description=f"Flow between communities {c1} and {c2}",
                                expected_properties={
                                    "source_community": c1,
                                    "target_community": c2
                                }
                            ))
                            break
                            
        except Exception as e:
            print(f"Warning: Could not generate community test cases: {str(e)}")
        
        return cases

    def generate_high_balance_case(self) -> Optional[TestCase]:
        """Generate test case between high-balance nodes."""
        if not self.high_balance_nodes or len(self.high_balance_nodes) < 2:
            return None
            
        try:
            for _ in range(10):
                source, target = random.sample(self.high_balance_nodes, 2)
                if nx.has_path(self.G, source, target):
                    return TestCase(
                        name="High Balance Flow",
                        source=source,
                        target=target,
                        description="Flow between high-balance nodes",
                        expected_properties={
                            "min_balance_source": float(self.balance_df[
                                self.balance_df['account'] == source
                            ]['demurragedTotalBalance'].sum()),
                            "min_balance_target": float(self.balance_df[
                                self.balance_df['account'] == target
                            ]['demurragedTotalBalance'].sum())
                        }
                    )
        except Exception as e:
            print(f"Warning: Could not generate high balance test case: {str(e)}")
        return None

    def generate_test_cases(self, num_cases: int = 5) -> List[TestCase]:
        """Generate a diverse set of test cases."""
        test_cases = []
        generators = [
            self.generate_long_path_case,
            self.generate_bottleneck_case,
            self.generate_high_balance_case
        ]
        
        # Add community cases separately since they return a list
        community_cases = self.generate_community_cases()
        if community_cases:
            test_cases.extend(community_cases[:2])  # Add up to 2 community cases
        
        # Generate other cases
        for generator in generators:
            case = generator()
            if case:
                test_cases.append(case)
        
        # Ensure we don't exceed requested number of cases
        return test_cases[:num_cases]


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
    reciprocal_trust_ratio: float = 0.3


class NetworkValidator:
    """Validate and analyze generated networks."""

    @staticmethod
    def validate_trust_relationships(trust_df: pd.DataFrame) -> Dict:
        """Validate trust relationships and return metrics using optimized methods."""
        # Convert to sets for faster lookups
        trust_pairs = set(zip(trust_df['truster'], trust_df['trustee']))
        unique_trusters = set(trust_df['truster'])
        unique_trustees = set(trust_df['trustee'])
        
        # Efficiently calculate reciprocal trusts
        reciprocal_count = sum(1 for (a, b) in trust_pairs if (b, a) in trust_pairs) // 2
        
        # Calculate self-trusts efficiently
        self_trusts = sum(1 for a, b in trust_pairs if a == b)
        
        metrics = {
            "total_relationships": len(trust_pairs),
            "unique_trusters": len(unique_trusters),
            "unique_trustees": len(unique_trustees),
            "self_trusts": self_trusts,
            "reciprocal_trusts": reciprocal_count,
            "reciprocal_trust_ratio": reciprocal_count / (len(trust_pairs) / 2) if trust_pairs else 0
        }
        
        return metrics

    @staticmethod
    def validate_token_balances(balance_df: pd.DataFrame) -> Dict:
        """Validate token balances and return metrics using optimized methods."""
        # Convert to numpy arrays for faster computation
        balances = balance_df['demurragedTotalBalance'].astype(float).values
        accounts = balance_df['account'].values
        
        # Use numpy for efficient calculations
        unique_accounts, account_counts = np.unique(accounts, return_counts=True)
        
        metrics = {
            "total_balance_records": len(balance_df),
            "unique_accounts": len(unique_accounts),
            "unique_tokens": len(balance_df['tokenAddress'].unique()),
            "avg_tokens_per_account": len(balance_df) / len(unique_accounts),
            "max_tokens_per_account": account_counts.max(),
            "min_balance": float(np.min(balances)),
            "max_balance": float(np.max(balances)),
            "total_balance": float(np.sum(balances)),
            "balance_std": float(np.std(balances)),
            "balance_mean": float(np.mean(balances))
        }
        
        # Calculate token concentration using optimized method
        metrics["token_concentration_gini"] = NetworkValidator._gini_coefficient_optimized(account_counts)
        
        return metrics

    @staticmethod
    def _gini_coefficient_optimized(array: np.ndarray) -> float:
        """Calculate Gini coefficient using numpy operations."""
        array = np.sort(array)
        index = np.arange(1, len(array) + 1)
        return (np.sum((2 * index - len(array) - 1) * array)) / (len(array) * np.sum(array))

    @staticmethod
    def generate_report(trust_df: pd.DataFrame, balance_df: pd.DataFrame, output_dir: str):
        """Generate comprehensive validation report with optimized visualizations."""
        trust_metrics = NetworkValidator.validate_trust_relationships(trust_df)
        balance_metrics = NetworkValidator.validate_token_balances(balance_df)

        plt.figure(figsize=(15, 10))

        # Trust relationship degree distribution (optimized)
        truster_degrees = trust_df['truster'].value_counts().values
        plt.subplot(2, 2, 1)
        plt.hist(truster_degrees, bins='auto', density=True)
        plt.title("Trust Relationship Out-Degree Distribution")
        plt.xlabel("Out-Degree")
        plt.ylabel("Density")

        # Token balance distribution
        balances = balance_df['demurragedTotalBalance'].astype(float).values
        plt.subplot(2, 2, 2)
        plt.hist(np.log10(balances), bins='auto', density=True)
        plt.title("Token Balance Distribution (log10)")
        plt.xlabel("Log10(Balance)")
        plt.ylabel("Density")

        # Tokens per account distribution
        tokens_per_account = balance_df['account'].value_counts().values
        plt.subplot(2, 2, 3)
        plt.hist(tokens_per_account, bins='auto', density=True)
        plt.title("Tokens per Account Distribution")
        plt.xlabel("Number of Tokens")
        plt.ylabel("Density")

        # Reciprocal trust ratio over time
        plt.subplot(2, 2, 4)
        reciprocal_pairs = set()
        all_pairs = set()
        ratios = []
        chunk_size = len(trust_df) // 10
        for i in range(0, len(trust_df), chunk_size):
            chunk = trust_df.iloc[i:i+chunk_size]
            chunk_pairs = set(zip(chunk['truster'], chunk['trustee']))
            all_pairs.update(chunk_pairs)
            reciprocal_pairs.update((a, b) for a, b in chunk_pairs if (b, a) in all_pairs)
            ratios.append(len(reciprocal_pairs) / (len(all_pairs) / 2) if all_pairs else 0)
        
        plt.plot(range(len(ratios)), ratios)
        plt.title("Reciprocal Trust Ratio Evolution")
        plt.xlabel("Network Growth (10% chunks)")
        plt.ylabel("Reciprocal Trust Ratio")

        # Save visualizations
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "network_analysis.png"), dpi=300, bbox_inches='tight')
        plt.close()

        # Save metrics report
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
                 seed: int = None,
                 run_tests: bool = False):  # Added run_tests parameter
        """Initialize generator with optional test flag."""
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
        self.run_tests = run_tests  # Store test flag
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)
        self.addresses = self._generate_addresses()
        self.total_edges = 0
        self.existing_edges: Set[Tuple[str, str]] = set()


    def _generate_addresses(self) -> List[str]:
        """Generate unique Ethereum-style addresses using numpy for speed."""
        print("Generating addresses...")
        hex_chars = np.array(list('0123456789abcdef'))
        addresses = []
        address_set = set()

        # Generate addresses in batches for efficiency
        batch_size = min(100000, self.num_addresses)
        remaining = self.num_addresses

        with tqdm(total=self.num_addresses, desc="Addresses") as pbar:
            while remaining > 0:
                current_batch = min(batch_size, remaining)
                
                # Generate random addresses in batch
                batch_addresses = ['0x' + ''.join(np.random.choice(hex_chars, size=40)) 
                                 for _ in range(current_batch)]
                
                # Filter duplicates
                new_addresses = [addr for addr in batch_addresses if addr not in address_set]
                addresses.extend(new_addresses)
                address_set.update(new_addresses)
                
                pbar.update(len(new_addresses))
                remaining = self.num_addresses - len(addresses)

        return addresses[:self.num_addresses]


    def _generate_trust_edges_chunk(self, data: Tuple[List[str], int]) -> List[Tuple[str, str]]:
        """Generate trust edges for a chunk with reciprocal trust control."""
        address_chunk, chunk_id = data  # Unpack only address chunk and chunk ID
        edges = []
        local_edges = set()

        for addr in address_chunk:
            targets = self._select_targets(addr, self.existing_edges | local_edges)
            
            # Add regular edges
            new_edges = [(addr, target) for target in targets]
            edges.extend(new_edges)
            local_edges.update(new_edges)
            
            # Add reciprocal edges based on probability
            for target in targets:
                if random.random() < self.network_pattern.reciprocal_trust_ratio:
                    edges.append((target, addr))
                    local_edges.add((target, addr))

        return edges
    
    def _select_targets(self, source: str, existing_edges: Set[Tuple[str, str]]) -> List[str]:
        """Select target addresses for trust relationships based on network type."""
        available_targets = [addr for addr in self.addresses 
                           if addr != source and (source, addr) not in existing_edges]
        
        if not available_targets:
            return []

        if self.network_type == NetworkType.SCALE_FREE:
            num_targets = int(np.random.power(2.1) * self.avg_trust_connections)
        else:
            num_targets = random.randint(
                max(1, int(self.avg_trust_connections * 0.5)),
                int(self.avg_trust_connections * 1.5)
            )

        return random.sample(available_targets, min(num_targets, len(available_targets)))


    def _generate_trust_edges_parallel(self) -> Generator[Tuple[str, str], None, None]:
        """Generate trust edges using parallel processing."""
        # Split addresses into chunks for parallel processing
        chunk_size = math.ceil(len(self.addresses) / (self.n_jobs * 4))  # 4 chunks per process
        address_chunks = [
            (self.addresses[i:i+chunk_size], i // chunk_size)  # Include chunk index
            for i in range(0, len(self.addresses), chunk_size)
        ]

        print(f"Generating trust edges using {self.n_jobs} processes...")
        with Pool(processes=self.n_jobs) as pool:
            for chunk_edges in tqdm(pool.imap_unordered(self._generate_trust_edges_chunk, address_chunks),
                                  total=len(address_chunks), desc="Trust Edges Chunks"):
                for edge in chunk_edges:
                    yield edge

    def generate_trust_edges(self, output_file: str):
        """Generate trust edges and write to CSV file."""
        print("Generating trust edges...")
        with open(output_file, 'w') as f:
            f.write('truster,trustee\n')  # Write header

        total_edges = 0
        chunk_edges = []
        
        # Initialize shared edge tracking set
        self.existing_edges = set()
        
        for edge in self._generate_trust_edges_parallel():
            chunk_edges.append(edge)
            if len(chunk_edges) >= self.chunk_size:
                # Update existing edges set
                self.existing_edges.update(chunk_edges)
                
                # Write chunk to file
                df_chunk = pd.DataFrame(chunk_edges, columns=['truster', 'trustee'])
                df_chunk.to_csv(output_file, mode='a', header=False, index=False)
                total_edges += len(chunk_edges)
                print(f"  Written {total_edges} trust relationships...")
                chunk_edges = []

        if chunk_edges:
            # Handle remaining edges
            self.existing_edges.update(chunk_edges)
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

    def run_network_tests(self, trust_df: pd.DataFrame, balance_df: pd.DataFrame, output_dir: str):
        """Run network tests and generate test report."""
        print("\nRunning network tests...")
        test_generator = NetworkTestGenerator(trust_df, balance_df)
        test_cases = test_generator.generate_test_cases()
        
        # Write test cases to report
        test_report_path = os.path.join(output_dir, "test_cases_report.txt")
        with open(test_report_path, "w") as f:
            f.write(f"Network Test Cases Report\n")
            f.write(f"========================\n")
            f.write(f"Network Type: {self.network_type.value}\n")
            f.write(f"Number of Addresses: {self.num_addresses}\n")
            f.write(f"Average Trust Connections: {self.avg_trust_connections}\n\n")
            
            for case in test_cases:
                f.write(f"\nTest Case: {case.name}\n")
                f.write(f"{'=' * (len(case.name) + 10)}\n")
                f.write(f"Description: {case.description}\n")
                f.write(f"Source: {case.source}\n")
                f.write(f"Target: {case.target}\n")
                f.write("Expected Properties:\n")
                for k, v in case.expected_properties.items():
                    f.write(f"  {k}: {v}\n")
                f.write("\n")
        
        print(f"Test cases report generated: {test_report_path}")

    def save_to_csv(self, output_dir: str = "data"):
        """Generate and save the data to CSV files with optional testing."""
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

        # Run network tests if enabled
        if self.run_tests:
            self.run_network_tests(trust_df, balance_df, output_dir)

        print("Data generation completed.")



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
            'seed': config.get('seed', None),
            'run_tests': config.get('run_tests', False)  # Add run_tests parameter
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
    parser.add_argument('--run_tests', action='store_true', help='Run network tests')  # Add run_tests argument
    args = parser.parse_args()

    if args.config_file:
        with open(args.config_file, 'r') as f:
            configs = json.load(f)
            
        # If run_tests is specified via command line, override config settings
        if args.run_tests:
            for config in configs:
                config['run_tests'] = True
                
        generate_networks(configs, output_base_dir=args.output_base_dir)
    else:
        print("Please provide a configuration file using --config_file")
        exit(1)


if __name__ == "__main__":
    main()