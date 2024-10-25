import networkx as nx
import pandas as pd
import random
from typing import Tuple, List, Dict, Optional, Generator
import numpy as np
import os
from enum import Enum
from dataclasses import dataclass
import community 
import matplotlib.pyplot as plt
from collections import defaultdict
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from itertools import chain
import math


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
    def __init__(self,
                num_communities: int = 3,
                community_density: float = 0.7,
                inter_community_density: float = 0.1,
                num_bottlenecks: int = 2,
                bottleneck_connections: int = 5,
                core_size_ratio: float = 0.1,
                token_concentration: float = 0.8,
                hub_nodes: int = 5,
                hub_connection_ratio: float = 0.3,
                min_tokens_per_address: int = 3,  
                target_tokens_per_address: int = 50):  
        self.num_communities = num_communities
        self.community_density = community_density
        self.inter_community_density = inter_community_density
        self.num_bottlenecks = num_bottlenecks
        self.bottleneck_connections = bottleneck_connections
        self.core_size_ratio = core_size_ratio
        self.token_concentration = token_concentration
        self.hub_nodes = hub_nodes
        self.hub_connection_ratio = hub_connection_ratio
        self.min_tokens_per_address = min_tokens_per_address
        self.target_tokens_per_address = target_tokens_per_address

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
        
        # Create network and calculate metrics
        G = nx.from_pandas_edgelist(trust_df, 'truster', 'trustee', create_using=nx.DiGraph())
        metrics.update({
            "avg_out_degree": np.mean([d for _, d in G.out_degree()]),
            "max_out_degree": max([d for _, d in G.out_degree()]),
            "density": nx.density(G),
            "strongly_connected_components": nx.number_strongly_connected_components(G),
            "weakly_connected_components": nx.number_weakly_connected_components(G)
        })
        
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
        """Generate comprehensive validation report with visualizations."""
        trust_metrics = NetworkValidator.validate_trust_relationships(trust_df)
        balance_metrics = NetworkValidator.validate_token_balances(balance_df)
        
        # Create visualizations
        plt.figure(figsize=(15, 10))
        
        # Trust network degree distribution
        G = nx.from_pandas_edgelist(trust_df, 'truster', 'trustee', create_using=nx.DiGraph())
        degrees = [d for _, d in G.out_degree()]
        plt.subplot(2, 2, 1)
        plt.hist(degrees, bins=50)
        plt.title("Trust Relationship Degree Distribution")
        plt.xlabel("Degree")
        plt.ylabel("Count")
        
        # Token balance distribution
        plt.subplot(2, 2, 2)
        balances = balance_df['demurragedTotalBalance'].astype(float)
        plt.hist(np.log10(balances), bins=50)
        plt.title("Token Balance Distribution (log10)")
        plt.xlabel("Log10(Balance)")
        plt.ylabel("Count")
        
        # Tokens per account distribution
        plt.subplot(2, 2, 3)
        tokens_per_account = balance_df['account'].value_counts()
        plt.hist(tokens_per_account, bins=30)
        plt.title("Tokens per Account Distribution")
        plt.xlabel("Number of Tokens")
        plt.ylabel("Number of Accounts")
        
        # Save report
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "network_analysis.png"))
        plt.close()
        
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
                 num_addresses: int = 1000,
                 avg_trust_connections: float = 5.0,
                 avg_tokens_per_user: float = 5.0, 
                 min_balance: int = 100,
                 max_balance: int = 1000000,
                 network_type: NetworkType = NetworkType.SCALE_FREE,
                 network_pattern: Optional[NetworkPattern] = None,
                 chunk_size: int = 10000,
                 n_jobs: int = None):
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
            seed: Random seed for reproducibility
        """
        self.chunk_size = chunk_size
        self.n_jobs = n_jobs or max(1, mp.cpu_count() - 1)
        self.chunk_size = chunk_size
#        if seed is not None:
#            random.seed(seed)
#            np.random.seed(seed)
            
        self.num_addresses = num_addresses
        self.avg_trust_connections = avg_trust_connections
        self.avg_tokens_per_user = avg_tokens_per_user
        self.min_balance = min_balance
        self.max_balance = max_balance
        self.network_type = network_type
        self.network_pattern = network_pattern or NetworkPattern()
        
        # Generate addresses
        self.addresses = self._generate_addresses()
        self.tokens = self.addresses
        
    def _generate_addresses(self, count: int = None) -> List[str]:
        """
        Generate unique Ethereum-style addresses.
        
        Args:
            count: Number of addresses to generate. If None, uses self.num_addresses
            
        Returns:
            List of unique hex addresses with '0x' prefix
        """
        if count is None:
            count = self.num_addresses
            
        addresses = set()  # Use set for efficient uniqueness checking
        
        while len(addresses) < count:
            # Generate a new address
            new_addr = '0x' + ''.join(random.choices('0123456789abcdef', k=40))
            addresses.add(new_addr)
            
        return sorted(list(addresses)) 
    
    @staticmethod
    def _is_valid_eth_address(address: str) -> bool:
        """
        Validate Ethereum address format.
        
        Args:
            address: String to validate
            
        Returns:
            bool: True if address is valid format
        """
        if not isinstance(address, str):
            return False
        if not address.startswith('0x'):
            return False
        if len(address) != 42:  # 0x + 40 hex chars
            return False
        try:
            # Check if the rest is valid hex
            int(address[2:], 16)
            return True
        except ValueError:
            return False

    def _create_community_structure(self) -> nx.DiGraph:
        """Create a network with clear community structure."""
        G = nx.DiGraph()
        nodes_per_community = self.num_addresses // self.network_pattern.num_communities
        
        # Create communities
        for i in range(self.network_pattern.num_communities):
            community_nodes = self.addresses[i*nodes_per_community:(i+1)*nodes_per_community]
            
            # Add dense connections within community
            for u in community_nodes:
                for v in community_nodes:
                    if u != v and random.random() < self.network_pattern.community_density:
                        G.add_edge(u, v)
            
            # Add sparse connections between communities
            other_nodes = [n for n in self.addresses if n not in community_nodes]
            for u in community_nodes:
                for v in random.sample(other_nodes, 
                                     k=int(len(other_nodes) * self.network_pattern.inter_community_density)):
                    G.add_edge(u, v)
        
        return G

    def _create_bottleneck_structure(self) -> nx.DiGraph:
        """Create a network with bottleneck nodes."""
        G = nx.DiGraph()
        
        # Split nodes into regions
        region_size = self.num_addresses // (self.network_pattern.num_bottlenecks + 1)
        regions = []
        
        for i in range(self.network_pattern.num_bottlenecks + 1):
            start_idx = i * region_size
            end_idx = start_idx + region_size
            regions.append(self.addresses[start_idx:end_idx])
        # Create connections within regions
        for region in regions:
            for u in region:
                connections = random.sample(
                    [v for v in region if v != u],
                    k=min(len(region)-1, self.network_pattern.bottleneck_connections)
                )
                for v in connections:
                    G.add_edge(u, v)
        # Add bottleneck nodes connecting regions
        bottlenecks = random.sample(self.addresses, self.network_pattern.num_bottlenecks)
        for i in range(len(regions)-1):
            bottleneck = bottlenecks[i]
            # Connect bottleneck to both regions
            for region in [regions[i], regions[i+1]]:
                connections = random.sample(
                    region,
                    k=min(len(region), self.network_pattern.bottleneck_connections)
                )
                for v in connections:
                    G.add_edge(bottleneck, v)
                    G.add_edge(v, bottleneck)
        
        return G

    def _create_hierarchical_structure(self) -> nx.DiGraph:
        """Create a hierarchical network structure."""
        G = nx.DiGraph()
        levels = int(np.log2(self.num_addresses))
        nodes_per_level = [min(2**i, self.num_addresses//(2**(levels-i-1))) for i in range(levels)]
        
        current_idx = 0
        level_start_indices = [current_idx]
        
        for level_size in nodes_per_level:
            current_idx += level_size
            level_start_indices.append(current_idx)
        
        # Add edges between levels
        for level in range(levels-1):
            start_idx = level_start_indices[level]
            next_start_idx = level_start_indices[level+1]
            
            for i in range(start_idx, level_start_indices[level+1]):
                if i >= len(self.addresses):
                    break
                    
                # Connect to multiple nodes in next level
                num_children = random.randint(2, 4)
                possible_children = range(next_start_idx, level_start_indices[level+2])
                children = random.sample(
                    [self.addresses[j] for j in possible_children if j < len(self.addresses)],
                    min(num_children, len(possible_children))
                )
                
                for child in children:
                    G.add_edge(self.addresses[i], child)
        
        return G

    def _create_core_periphery_structure(self) -> nx.DiGraph:
        """Create a core-periphery network structure."""
        G = nx.DiGraph()
        
        # Define core and periphery
        core_size = int(self.num_addresses * self.network_pattern.core_size_ratio)
        core_nodes = self.addresses[:core_size]
        periphery_nodes = self.addresses[core_size:]
        
        # Add dense connections in core
        for u in core_nodes:
            for v in core_nodes:
                if u != v and random.random() < 0.7:
                    G.add_edge(u, v)
        
        # Connect periphery to core
        for node in periphery_nodes:
            # Connect to random core nodes
            num_connections = random.randint(1, 3)
            core_connections = random.sample(core_nodes, num_connections)
            for core_node in core_connections:
                G.add_edge(node, core_node)
                if random.random() < 0.3:  # Some reciprocal connections
                    G.add_edge(core_node, node)
        
        return G

    def _create_star_structure(self) -> nx.DiGraph:
        """Create a star-like network with multiple hubs."""
        G = nx.DiGraph()
        
        # Select hub nodes
        hub_nodes = self.addresses[:self.network_pattern.hub_nodes]
        regular_nodes = self.addresses[self.network_pattern.hub_nodes:]
        
        # Calculate connections per hub
        connections_per_hub = int(len(regular_nodes) * self.network_pattern.hub_connection_ratio)
        
        # Connect hubs to regular nodes
        for hub in hub_nodes:
            # Select random nodes to connect to
            connections = random.sample(regular_nodes, connections_per_hub)
            for node in connections:
                G.add_edge(hub, node)
                if random.random() < 0.3:  # Some reciprocal connections
                    G.add_edge(node, hub)
        
        return G
    

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
        with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
            for chunk_edges in executor.map(self._generate_trust_edges_chunk, address_chunks):
                for edge in chunk_edges:
                    yield edge

    def _generate_token_balances_chunk(self, data: Tuple[List[str], nx.DiGraph, int]) -> List[Dict]:
        """Generate token balances for a chunk of addresses."""
        addresses, G, chunk_id = data
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
            trusted_addresses = list(G.successors(address))
            if not trusted_addresses:
                continue

            num_tokens = int(np.clip(
                np.random.normal(
                    self.network_pattern.target_tokens_per_address,
                    self.network_pattern.target_tokens_per_address * 0.2
                ),
                self.network_pattern.min_tokens_per_address,
                len(trusted_addresses)
            ))

            if num_tokens > 0:
                token_sources = np.random.choice(
                    trusted_addresses,
                    size=min(num_tokens, len(trusted_addresses)),
                    replace=False
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

    def _generate_token_balances_parallel(self, G: nx.DiGraph) -> Generator[Dict, None, None]:
        """Generate token balances using parallel processing."""
        chunk_size = math.ceil(len(self.addresses) / (self.n_jobs * 4))
        address_chunks = [
            self.addresses[i:i+chunk_size]
            for i in range(0, len(self.addresses), chunk_size)
        ]
        
        # Prepare data for parallel processing
        chunk_data = [
            (chunk, G, i) for i, chunk in enumerate(address_chunks)
        ]
        
        print(f"Generating token balances using {self.n_jobs} processes...")
        with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
            for chunk_balances in executor.map(self._generate_token_balances_chunk, chunk_data):
                for balance in chunk_balances:
                    yield balance

    def _create_trust_network(self) -> nx.DiGraph:
        """Create the trust network using the specified network type with optimizations."""
        try:
            G = nx.DiGraph()
            # Pre-add all nodes to avoid repeated node additions
            G.add_nodes_from(self.addresses)
            
            if self.network_type == NetworkType.SCALE_FREE:
                # Use numpy for faster array operations
                degrees = np.random.power(2.1, self.num_addresses)  # Power law distribution
                degrees = np.maximum(1, (degrees * self.avg_trust_connections).astype(int))
                
                for i, addr in enumerate(self.addresses):
                    targets = np.random.choice(
                        self.addresses[:i] if i > 0 else self.addresses[1:],
                        size=min(degrees[i], i if i > 0 else self.num_addresses-1),
                        replace=False
                    )
                    G.add_edges_from(zip([addr] * len(targets), targets))
                
            elif self.network_type == NetworkType.RANDOM:
                # Use numpy for faster random generation
                p = self.avg_trust_connections / self.num_addresses
                edges = []
                for i, source in enumerate(self.addresses):
                    # Generate random connections in batches
                    mask = np.random.random(self.num_addresses) < p
                    mask[i] = False  # No self-loops
                    targets = np.array(self.addresses)[mask]
                    if len(targets) > 0:
                        edges.extend(zip([source] * len(targets), targets))
                G.add_edges_from(edges)
                
            elif self.network_type == NetworkType.SMALL_WORLD:
                k = max(2, int(self.avg_trust_connections))
                p = 0.1
                # Create base ring lattice
                n = len(self.addresses)
                edges = []
                for i in range(n):
                    for j in range(1, k // 2 + 1):
                        edges.append((self.addresses[i], self.addresses[(i + j) % n]))
                        edges.append((self.addresses[i], self.addresses[(i - j) % n]))
                
                # Rewire edges
                for u, v in edges:
                    if np.random.random() < p:
                        # Generate new target that maintains connectivity
                        possible_targets = [w for w in self.addresses if w != u and w != v]
                        new_target = np.random.choice(possible_targets)
                        G.add_edge(u, new_target)
                    else:
                        G.add_edge(u, v)
                
            elif self.network_type == NetworkType.COMMUNITY:
                # Optimize community structure generation
                nodes_per_community = self.num_addresses // self.network_pattern.num_communities
                edges = []
                
                for i in range(self.network_pattern.num_communities):
                    start_idx = i * nodes_per_community
                    end_idx = start_idx + nodes_per_community
                    community = self.addresses[start_idx:end_idx]
                    
                    # Within community connections (vectorized)
                    comm_size = len(community)
                    conn_matrix = np.random.random((comm_size, comm_size)) < self.network_pattern.community_density
                    np.fill_diagonal(conn_matrix, False)  # No self-loops
                    
                    source_idx, target_idx = np.where(conn_matrix)
                    edges.extend(zip(
                        [community[idx] for idx in source_idx],
                        [community[idx] for idx in target_idx]
                    ))
                    
                    # Inter-community connections (batched)
                    other_nodes = self.addresses[:start_idx] + self.addresses[end_idx:]
                    num_external = int(len(other_nodes) * self.network_pattern.inter_community_density)
                    if num_external > 0:
                        for node in community:
                            targets = np.random.choice(other_nodes, size=num_external, replace=False)
                            edges.extend(zip([node] * len(targets), targets))
                
                G.add_edges_from(edges)
                
            elif self.network_type == NetworkType.BOTTLENECK:
                # Optimize bottleneck structure generation
                region_size = self.num_addresses // (self.network_pattern.num_bottlenecks + 1)
                edges = []
                
                # Split into regions
                regions = [
                    self.addresses[i:i+region_size]
                    for i in range(0, self.num_addresses, region_size)
                ]
                
                # Create connections within regions efficiently
                for region in regions:
                    for node in region:
                        num_connections = min(
                            len(region) - 1,
                            self.network_pattern.bottleneck_connections
                        )
                        if num_connections > 0:
                            targets = np.random.choice(
                                [n for n in region if n != node],
                                size=num_connections,
                                replace=False
                            )
                            edges.extend(zip([node] * len(targets), targets))
                
                # Add bottleneck connections
                bottlenecks = np.random.choice(
                    self.addresses,
                    size=self.network_pattern.num_bottlenecks,
                    replace=False
                )
                
                for i, bottleneck in enumerate(bottlenecks):
                    if i < len(regions) - 1:
                        for region in [regions[i], regions[i+1]]:
                            targets = np.random.choice(
                                region,
                                size=min(len(region), self.network_pattern.bottleneck_connections),
                                replace=False
                            )
                            # Add bidirectional connections
                            edges.extend(zip([bottleneck] * len(targets), targets))
                            edges.extend(zip(targets, [bottleneck] * len(targets)))
                
                G.add_edges_from(edges)
                
            elif self.network_type == NetworkType.HIERARCHICAL:
                levels = int(np.log2(self.num_addresses))
                nodes_per_level = [
                    min(2**i, self.num_addresses//(2**(levels-i-1)))
                    for i in range(levels)
                ]
                
                current_idx = 0
                level_start_indices = [current_idx]
                edges = []
                
                for level_size in nodes_per_level:
                    current_idx += level_size
                    level_start_indices.append(current_idx)
                
                for level in range(levels-1):
                    start_idx = level_start_indices[level]
                    next_start_idx = level_start_indices[level+1]
                    
                    for i in range(start_idx, level_start_indices[level+1]):
                        if i >= len(self.addresses):
                            break
                        
                        num_children = np.random.randint(2, 5)
                        possible_children = range(next_start_idx, level_start_indices[level+2])
                        
                        children = np.random.choice(
                            [self.addresses[j] for j in possible_children if j < len(self.addresses)],
                            size=min(num_children, len(possible_children)),
                            replace=False
                        )
                        
                        edges.extend(zip([self.addresses[i]] * len(children), children))
                
                G.add_edges_from(edges)
                
            elif self.network_type == NetworkType.CORE_PERIPHERY:
                # Optimize core-periphery structure
                core_size = int(self.num_addresses * self.network_pattern.core_size_ratio)
                core_nodes = self.addresses[:core_size]
                periphery_nodes = self.addresses[core_size:]
                edges = []
                
                # Dense core connections
                for u in core_nodes:
                    targets = [v for v in core_nodes if v != u and np.random.random() < 0.7]
                    if targets:
                        edges.extend(zip([u] * len(targets), targets))
                
                # Periphery to core connections
                for node in periphery_nodes:
                    num_connections = np.random.randint(1, 4)
                    core_connections = np.random.choice(core_nodes, size=num_connections, replace=False)
                    edges.extend(zip([node] * len(core_connections), core_connections))
                    
                    # Some reciprocal connections
                    reciprocal = [c for c in core_connections if np.random.random() < 0.3]
                    if reciprocal:
                        edges.extend(zip(reciprocal, [node] * len(reciprocal)))
                
                G.add_edges_from(edges)
            
            else:
                # Fallback to simple random network
                p = 0.01
                edges = []
                batch_size = 1000
                
                for i in range(0, self.num_addresses, batch_size):
                    batch_sources = self.addresses[i:i+batch_size]
                    for source in batch_sources:
                        mask = np.random.random(self.num_addresses) < p
                        mask[i] = False  # No self-loops
                        targets = np.array(self.addresses)[mask]
                        if len(targets) > 0:
                            edges.extend(zip([source] * len(targets), targets))
                
                G.add_edges_from(edges)
            
            return G
            
        except Exception as e:
            print(f"Error creating network: {str(e)}")
            # Fallback to simple random network
            G = nx.DiGraph()
            G.add_nodes_from(self.addresses)
            
            # Generate random edges in batches
            edges = []
            p = 0.01
            batch_size = 1000
            
            for i in range(0, self.num_addresses, batch_size):
                batch_sources = self.addresses[i:i+batch_size]
                for source in batch_sources:
                    mask = np.random.random(self.num_addresses) < p
                    mask[i] = False  # No self-loops
                    targets = np.array(self.addresses)[mask]
                    if len(targets) > 0:
                        edges.extend(zip([source] * len(targets), targets))
            
            G.add_edges_from(edges)
            return G
    
    def _generate_trust_edges(self) -> Generator[Tuple[str, str], None, None]:
        """Generate trust edges in a memory-efficient way using generators."""
        if self.network_type == NetworkType.SCALE_FREE:
            # Process in chunks for scale-free network
            degrees = np.random.power(2.1, self.num_addresses)
            degrees = np.maximum(1, (degrees * self.avg_trust_connections).astype(int))
            
            for i, addr in enumerate(self.addresses):
                targets = np.random.choice(
                    self.addresses[:i] if i > 0 else self.addresses[1:],
                    size=min(degrees[i], i if i > 0 else self.num_addresses-1),
                    replace=False
                )
                for target in targets:
                    yield (addr, target)
                    
        elif self.network_type == NetworkType.RANDOM:
            p = self.avg_trust_connections / self.num_addresses
            # Process in chunks
            chunk_size = min(1000, self.num_addresses)
            for i in range(0, self.num_addresses, chunk_size):
                chunk_addresses = self.addresses[i:i+chunk_size]
                for source in chunk_addresses:
                    # Use numpy for efficient random generation
                    targets = np.random.choice(
                        [addr for addr in self.addresses if addr != source],
                        size=int(p * chunk_size),
                        replace=False,
                        p=None
                    )
                    for target in targets:
                        yield (source, target)
        
        elif self.network_type == NetworkType.COMMUNITY:
            nodes_per_community = self.num_addresses // self.network_pattern.num_communities
            for i in range(self.network_pattern.num_communities):
                start_idx = i * nodes_per_community
                end_idx = start_idx + nodes_per_community
                community = self.addresses[start_idx:end_idx]
                
                # Within community connections
                for u in community:
                    targets = [v for v in community if v != u and random.random() < self.network_pattern.community_density]
                    for v in targets:
                        yield (u, v)
                
                # Inter-community connections
                other_nodes = self.addresses[:start_idx] + self.addresses[end_idx:]
                num_external = int(len(other_nodes) * self.network_pattern.inter_community_density)
                if num_external > 0:
                    for node in community:
                        targets = random.sample(other_nodes, num_external)
                        for target in targets:
                            yield (node, target)
        
        else:
            # For other network types, create edges in chunks
            chunk_size = min(1000, self.num_addresses)
            for i in range(0, self.num_addresses, chunk_size):
                chunk_start = i
                chunk_end = min(i + chunk_size, self.num_addresses)
                chunk_addresses = self.addresses[chunk_start:chunk_end]
                
                for source in chunk_addresses:
                    num_connections = random.randint(
                        max(1, int(self.avg_trust_connections * 0.5)),
                        int(self.avg_trust_connections * 1.5)
                    )
                    targets = random.sample(
                        [addr for addr in self.addresses if addr != source],
                        min(num_connections, self.num_addresses - 1)
                    )
                    for target in targets:
                        yield (source, target)

    def _generate_token_balances(self) -> Generator[Dict, None, None]:
        """
        Generate token balance records as a memory-efficient generator.
        Each address has its own token and can hold tokens of addresses it trusts.
        """
        # Build trust network first
        G = nx.DiGraph()
        print("Building trust network...")
        trust_edges = list(self._generate_trust_edges())
        G.add_edges_from(trust_edges)
        print(f"Trust network built with {len(trust_edges)} edges")

        # Generate all personal tokens first (everyone has their own token)
        print("Generating personal token balances...")
        for addr in self.addresses:
            # Personal token balance is higher than other token balances
            balance = np.random.randint(
                max(1, int(self.max_balance * 0.7)),
                max(2, self.max_balance)
            )
            # Safe demurraged calculation using string concatenation
            demurraged_balance = str(balance) + "0" * 15

            yield {
                'account': addr,
                'tokenAddress': addr,  # Personal token
                'demurragedTotalBalance': demurraged_balance
            }

        # Pre-calculate log-normal parameters for other token balances
        mu = np.log(max(1, self.min_balance) + (self.max_balance - max(1, self.min_balance)) / 4)
        sigma = 0.5

        # Process addresses in chunks for token holdings
        print("Generating trusted token balances...")
        for i in range(0, self.num_addresses, self.chunk_size):
            batch_addresses = self.addresses[i:i+self.chunk_size]
            print(f"Processing addresses {i} to {min(i + self.chunk_size, self.num_addresses)}...")

            for address in batch_addresses:
                # Get addresses this account trusts (can hold their tokens)
                trusted_addresses = list(G.successors(address))
                if not trusted_addresses:
                    continue

                # Determine number of tokens this address will hold
                # Using normal distribution clipped to reasonable bounds
                num_tokens = int(np.clip(
                    np.random.normal(
                        self.network_pattern.target_tokens_per_address,
                        self.network_pattern.target_tokens_per_address * 0.2
                    ),
                    self.network_pattern.min_tokens_per_address,
                    len(trusted_addresses)
                ))

                if num_tokens > 0:
                    # Select which tokens to hold
                    token_sources = np.random.choice(
                        trusted_addresses,
                        size=min(num_tokens, len(trusted_addresses)),
                        replace=False
                    )

                    # Generate balances for selected tokens
                    token_balances = np.exp(np.random.normal(mu, sigma, size=len(token_sources)))
                    token_balances = np.clip(
                        token_balances,
                        max(1, self.min_balance),
                        max(2, int(self.max_balance * 0.5))  # Other token balances are lower than personal token
                    ).astype(int)

                    # Yield balance records for each token
                    for token, balance in zip(token_sources, token_balances):
                        # Safe demurraged calculation
                        demurraged_balance = str(balance) + "0" * 15

                        yield {
                            'account': address,
                            'tokenAddress': token,
                            'demurragedTotalBalance': demurraged_balance
                        }

            # Optional: Add some random trusted token relationships for diversity
            if random.random() < 0.1:  # 10% chance for each address
                for address in batch_addresses:
                    # Randomly select additional tokens to hold
                    potential_tokens = [
                        addr for addr in self.addresses
                        if addr != address and addr not in G.successors(address)
                    ]
                    if potential_tokens:
                        num_extra = random.randint(1, 3)  # 1-3 extra tokens
                        extra_tokens = random.sample(
                            potential_tokens,
                            min(num_extra, len(potential_tokens))
                        )

                        for token in extra_tokens:
                            balance = np.random.randint(
                                max(1, self.min_balance),
                                max(2, int(self.max_balance * 0.3))  # Lower balance for random tokens
                            )
                            demurraged_balance = str(balance) + "0" * 15

                            yield {
                                'account': address,
                                'tokenAddress': token,
                                'demurragedTotalBalance': demurraged_balance
                            }

        print("Token balance generation completed")

    def validate_balances(self, balance_df: pd.DataFrame) -> None:
        """Validate that all balances are positive and properly formatted."""
        # Convert to numeric for validation
        balance_df['demurragedTotalBalance'] = pd.to_numeric(
            balance_df['demurragedTotalBalance'],
            errors='coerce'
        )
        
        # Check for any non-positive balances
        non_positive = balance_df[balance_df['demurragedTotalBalance'] <= 0]
        if len(non_positive) > 0:
            print(f"Warning: Found {len(non_positive)} non-positive balances. Fixing...")
            balance_df.loc[
                balance_df['demurragedTotalBalance'] <= 0,
                'demurragedTotalBalance'
            ] = str(max(1, self.min_balance) * (10 ** 15))
        
        # Convert back to string format
        balance_df['demurragedTotalBalance'] = balance_df['demurragedTotalBalance'].astype(str)
        
        return balance_df
    
    def generate_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Generate trust networks and token balances with optimized validation."""
        # Validate addresses (use set operations for efficiency)
        address_set = set(self.addresses)
        if len(self.addresses) != len(address_set):
            raise ValueError("Duplicate addresses detected")
            
        # Batch validate addresses
        if not all(addr.startswith('0x') and len(addr) == 42 for addr in self.addresses):
            raise ValueError("Invalid address format detected")
        
        # Create trust network
        G = self._create_trust_network()
        if G is None:
            raise ValueError("Failed to create trust network")
        
        # Create DataFrames directly from edge list
        trust_df = pd.DataFrame(G.edges(), columns=['truster', 'trustee'])
        self.trust_df = trust_df
        
        # Generate token balances
        balance_records = self._generate_token_balances()
        balance_df = pd.DataFrame(balance_records)
        
        # Validate and fix balances
        balance_df = self.validate_balances(balance_df)
        
        self.balance_records = balance_records
        
        # Log statistics
        print(f"Generated {len(set(self.addresses))} unique addresses")
        print(f"Generated network with {len(trust_df)} trust relationships")
        print(f"Generated {len(balance_df)} token balance records")
        
        # Token balance statistics
        balance_values = pd.to_numeric(balance_df['demurragedTotalBalance'], errors='coerce')
        print("\nBalance distribution:")
        print(f"  Min balance: {balance_values.min():.0f}")
        print(f"  Max balance: {balance_values.max():.0f}")
        print(f"  Mean balance: {balance_values.mean():.0f}")
        
        # Token holding statistics
        tokens_per_address = balance_df.groupby('account').size()
        print("\nToken holdings distribution:")
        print(f"  Min tokens per address: {tokens_per_address.min()}")
        print(f"  Max tokens per address: {tokens_per_address.max()}")
        print(f"  Mean tokens per address: {tokens_per_address.mean():.2f}")
        
        return trust_df, balance_df

    def generate_test_cases(self) -> List[Tuple[str, str, int]]:
        """Generate interesting test cases for flow analysis with better error handling."""
        G = nx.DiGraph()
        for trust in self.trust_df.itertuples():
            G.add_edge(trust.truster, trust.trustee)
        
        test_cases = []
        
        try:
            # Case 1: Maximum flow between distant nodes
            connected_components = list(nx.strongly_connected_components(G))
            if connected_components:
                largest_component = max(connected_components, key=len)
                
                # Find nodes with paths longer than 3 within the largest component
                distant_pairs = []
                nodes_list = list(largest_component)
                sample_size = min(100, len(nodes_list))
                sample_nodes = random.sample(nodes_list, sample_size)
                
                for u in sample_nodes:
                    for v in sample_nodes:
                        if u != v:
                            try:
                                if nx.shortest_path_length(G, u, v) > 3:
                                    distant_pairs.append((u, v))
                            except nx.NetworkXNoPath:
                                continue
                
                if distant_pairs:
                    test_cases.append(("Long Path", random.choice(distant_pairs)))
            
            # Case 2: Flow through bottlenecks
            G_undirected = G.to_undirected()
            try:
                articulation_points = list(nx.articulation_points(G_undirected))
                if articulation_points:
                    bottleneck = random.choice(articulation_points)
                    G_temp = G_undirected.copy()
                    G_temp.remove_node(bottleneck)
                    components = list(nx.connected_components(G_temp))
                    if components:
                        comp = random.choice(components)
                        if len(comp) > 1:
                            source = random.choice(list(comp))
                            possible_sinks = [n for n in G.nodes() if n not in comp and n != bottleneck]
                            if possible_sinks:
                                sink = random.choice(possible_sinks)
                                test_cases.append(("Bottleneck", (source, sink)))
            except Exception as e:
                print(f"Warning: Could not generate bottleneck test cases: {str(e)}")
            
            # Case 3: Flow within densely connected community
            try:
                communities = community.best_partition(G_undirected)
                for comm_id in set(communities.values()):
                    comm_nodes = [n for n, c in communities.items() if c == comm_id]
                    if len(comm_nodes) >= 2:
                        candidates = []
                        for _ in range(10):  # Try 10 times to find connected pairs
                            source, sink = random.sample(comm_nodes, 2)
                            if nx.has_path(G, source, sink):
                                candidates.append((source, sink))
                        if candidates:
                            test_cases.append(("Intra-Community", random.choice(candidates)))
                            break
            except Exception as e:
                print(f"Warning: Could not generate community test cases: {str(e)}")
            
            # Case 4: Cross-community flow
            try:
                comm_ids = list(set(communities.values()))
                if len(comm_ids) >= 2:
                    valid_pairs = []
                    for _ in range(10):  # Try 10 times to find valid pairs
                        c1, c2 = random.sample(comm_ids, 2)
                        nodes1 = [n for n, c in communities.items() if c == c1]
                        nodes2 = [n for n, c in communities.items() if c == c2]
                        if nodes1 and nodes2:
                            source = random.choice(nodes1)
                            sink = random.choice(nodes2)
                            if nx.has_path(G, source, sink):
                                valid_pairs.append((source, sink))
                    if valid_pairs:
                        test_cases.append(("Inter-Community", random.choice(valid_pairs)))
            except Exception as e:
                print(f"Warning: Could not generate cross-community test cases: {str(e)}")
            
            # Case 5: Maximum token concentration path
            try:
                balances_df = pd.DataFrame(self.balance_records)
                balances_df['demurragedTotalBalance'] = balances_df['demurragedTotalBalance'].astype(float)
                high_balance_nodes = balances_df.groupby('account')['demurragedTotalBalance'].sum().nlargest(20).index
                
                valid_pairs = []
                for _ in range(10):  # Try 10 times to find valid pairs
                    if len(high_balance_nodes) >= 2:
                        source, sink = random.sample(list(high_balance_nodes), 2)
                        if nx.has_path(G, source, sink):
                            valid_pairs.append((source, sink))
                
                if valid_pairs:
                    test_cases.append(("High Balance", random.choice(valid_pairs)))
            except Exception as e:
                print(f"Warning: Could not generate high balance test cases: {str(e)}")
        
        except Exception as e:
            print(f"Warning: Error generating test cases: {str(e)}")
        
        return test_cases

    def save_to_csv(self, output_dir: str = "data"):
        """Save data to CSV files using parallel processing."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate and save trust relationships
        print("Generating trust relationships...")
        trust_path = os.path.join(output_dir, "data-trust.csv")
        with open(trust_path, 'w') as f:
            f.write('truster,trustee\n')
        
        chunk = []
        total_edges = 0
        for edge in self._generate_trust_edges_parallel():
            chunk.append(edge)
            if len(chunk) >= self.chunk_size:
                pd.DataFrame(chunk, columns=['truster', 'trustee']).to_csv(
                    trust_path, 
                    mode='a', 
                    header=False, 
                    index=False
                )
                total_edges += len(chunk)
                print(f"  Written {total_edges:,} trust relationships...")
                chunk = []
        
        if chunk:
            pd.DataFrame(chunk, columns=['truster', 'trustee']).to_csv(
                trust_path, 
                mode='a', 
                header=False, 
                index=False
            )
            total_edges += len(chunk)
        
        print(f"Generated {total_edges:,} trust relationships")
        
        # Build trust network for token generation
        print("Building trust network...")
        G = nx.DiGraph()
        trust_df = pd.read_csv(trust_path)
        G.add_edges_from(zip(trust_df['truster'], trust_df['trustee']))
        
        # Generate and save token balances
        print("\nGenerating token balances...")
        balance_path = os.path.join(output_dir, "data-balance.csv")
        with open(balance_path, 'w') as f:
            f.write('account,tokenAddress,demurragedTotalBalance\n')
        
        chunk = []
        total_balances = 0
        for balance in self._generate_token_balances_parallel(G):
            chunk.append(balance)
            if len(chunk) >= self.chunk_size:
                pd.DataFrame(chunk).to_csv(
                    balance_path,
                    mode='a',
                    header=False,
                    index=False
                )
                total_balances += len(chunk)
                print(f"  Written {total_balances:,} token balances...")
                chunk = []
        
        if chunk:
            pd.DataFrame(chunk).to_csv(
                balance_path,
                mode='a',
                header=False,
                index=False
            )
            total_balances += len(chunk)
        
        print(f"Generated {total_balances:,} token balances")
        
        # Generate validation report
        print("\nGenerating validation report...")
        validator = NetworkValidator()
        trust_df = pd.read_csv(trust_path)
        balance_df = pd.read_csv(balance_path)
        validator.generate_report(trust_df, balance_df, output_dir)
        
        print("Done!")

    def save_to_csv2(self, output_dir: str = "data"):
        """Save data to CSV files in chunks to manage memory."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save trust relationships first
        print("Generating trust relationships...")
        trust_path = os.path.join(output_dir, "data-trust.csv")
        
        # Write header
        with open(trust_path, 'w') as f:
            f.write('truster,trustee\n')
        
        # Write edges in chunks
        chunk = []
        total_edges = 0
        for edge in self._generate_trust_edges():
            chunk.append(edge)
            if len(chunk) >= self.chunk_size:
                pd.DataFrame(chunk, columns=['truster', 'trustee']).to_csv(
                    trust_path, 
                    mode='a', 
                    header=False, 
                    index=False
                )
                total_edges += len(chunk)
                print(f"  Written {total_edges} trust relationships...")
                chunk = []
        
        if chunk:
            pd.DataFrame(chunk, columns=['truster', 'trustee']).to_csv(
                trust_path, 
                mode='a', 
                header=False, 
                index=False
            )
            total_edges += len(chunk)
        
        print(f"Generated {total_edges} trust relationships.")
        
        # Generate and save token balances
        print("\nGenerating token balances...")
        balance_path = os.path.join(output_dir, "data-balance.csv")
        
        # Write header
        with open(balance_path, 'w') as f:
            f.write('account,tokenAddress,demurragedTotalBalance\n')
        
        # Process balances in chunks
        chunk = []
        total_balances = 0
        for balance in self._generate_token_balances():
            chunk.append(balance)
            if len(chunk) >= self.chunk_size:
                pd.DataFrame(chunk).to_csv(
                    balance_path,
                    mode='a',
                    header=False,
                    index=False
                )
                total_balances += len(chunk)
                print(f"  Written {total_balances} token balances...")
                chunk = []
        
        if chunk:
            pd.DataFrame(chunk).to_csv(
                balance_path,
                mode='a',
                header=False,
                index=False
            )
            total_balances += len(chunk)
        
        print(f"Generated {total_balances} token balances.")
        
        # Generate validation report
        print("\nGenerating validation report...")
        validator = NetworkValidator()
        
        # Read files in chunks for validation
        trust_df = pd.read_csv(trust_path)
        balance_df = pd.read_csv(balance_path)
        validator.generate_report(trust_df, balance_df, output_dir)
        
        print("Done!")

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
        chunk_size=10000, 
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
        chunk_size=10000, 
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
        chunk_size=10000, 
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
        chunk_size=10000, 
        n_jobs=8,
        avg_trust_connections=15.0,
        avg_tokens_per_user=10.0,
        network_type=NetworkType.SCALE_FREE,
        network_pattern=large_pattern
    )
    large_network.save_to_csv("data_large_scale_free")

if __name__ == "__main__":
    # Generate example networks
    generate_example_networks()