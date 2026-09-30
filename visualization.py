from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType
from pyspark.sql.functions import monotonically_increasing_id, col, desc
from graphframes import GraphFrame
from pyvis.network import Network
import json

def main():
    # Initialize Spark session with GraphFrames package
    print("Initializing Spark session...")
    spark = SparkSession.builder \
        .appName("LargeTrustGraphVisualization") \
        .config("spark.jars.packages", "graphframes:graphframes:0.8.2-spark3.0-s_2.12") \
        .getOrCreate()

    # Check if GraphFrames is loaded
    try:
        from graphframes import GraphFrame
        print("GraphFrames successfully loaded.")
    except ImportError:
        print("GraphFrames not available. Ensure Spark is configured correctly.")
        spark.stop()
        return

    # Define schema and load data
    schema = StructType([
        StructField("truster", StringType(), True),
        StructField("trustee", StringType(), True)
    ])

    trust_data_path = "output/data_bottleneck_short/data-trust.csv"
    print(f"Loading data from {trust_data_path}...")
    df = spark.read.csv(trust_data_path, schema=schema, header=True)

    # Create vertices and edges
    print("Extracting unique nodes...")
    nodes_truster = df.select(col("truster").alias("node")).distinct()
    nodes_trustee = df.select(col("trustee").alias("node")).distinct()
    nodes = nodes_truster.union(nodes_trustee).distinct()
    vertices = nodes.withColumn("id", monotonically_increasing_id())
    node_to_id = vertices.select("node", "id")
    
    df_truster = node_to_id.withColumnRenamed("node", "truster").withColumnRenamed("id", "src")
    df_trustee = node_to_id.withColumnRenamed("node", "trustee").withColumnRenamed("id", "dst")
    
    edges = df.join(df_truster, on="truster", how="inner") \
              .join(df_trustee, on="trustee", how="inner") \
              .select("src", "dst")

    # Create GraphFrame and run enhanced community detection
    print("Creating GraphFrame...")
    g = GraphFrame(vertices, edges)

    print("Running Label Propagation for community detection...")
    clusters = g.labelPropagation(maxIter=15)
    clusters = clusters.withColumnRenamed("label", "cluster")

    # Calculate both in and out degrees
    in_degrees = g.inDegrees if g.inDegrees is not None else spark.createDataFrame([], "id long, inDegree double")
    out_degrees = g.outDegrees if g.outDegrees is not None else spark.createDataFrame([], "id long, outDegree double")
    
    # Join degrees and calculate total degree
    degrees = out_degrees.join(in_degrees, "id", "outer").fillna(0)
    degrees = degrees.withColumn(
        "total_degree",
        col("inDegree") + col("outDegree")
    )

    # Run PageRank with adjusted parameters
    print("Running PageRank for node importance...")
    results = g.pageRank(resetProbability=0.15, maxIter=20)

    # Combine all metrics
    metrics = (results.vertices
              .join(clusters, "id")
              .join(degrees, "id", "outer")
              .join(node_to_id, "id")
              .fillna(0))

    # Select top nodes based on total degree and PageRank
    top_n = 100000
    print(f"Selecting top {top_n} nodes...")
    top_nodes = (metrics
                .withColumn("score", col("pagerank") * col("total_degree"))
                .orderBy(desc("score"))
                .limit(top_n))

    # Collect node and edge data
    top_node_ids = set(top_nodes.select("id").rdd.flatMap(lambda x: x).collect())
    filtered_edges = edges.filter(
        (col("src").isin(top_node_ids)) & 
        (col("dst").isin(top_node_ids))
    ).collect()
    
    filtered_nodes = top_nodes.collect()

    # Create cluster size mapping
    cluster_sizes = {}
    for node in filtered_nodes:
        cluster = node["cluster"]
        cluster_sizes[cluster] = cluster_sizes.get(cluster, 0) + 1

    # Sort clusters by size and assign colors accordingly
    sorted_clusters = sorted(cluster_sizes.items(), key=lambda x: x[1], reverse=True)
    cluster_mapping = {cluster: idx for idx, (cluster, _) in enumerate(sorted_clusters)}

    # Enhanced color palette with greater contrast
    predefined_colors = [
        "#E41A1C",  # Red
        "#377EB8",  # Blue
        "#4DAF4A",  # Green
        "#984EA3",  # Purple
        "#FF7F00",  # Orange
        "#FFFF33",  # Yellow
        "#A65628",  # Brown
        "#F781BF",  # Pink
        "#999999",  # Gray
        "#66C2A5",  # Teal
        "#FC8D62",  # Coral
        "#8DA0CB",  # Light Blue
        "#E78AC3",  # Light Pink
        "#A6D854",  # Light Green
        "#FFD92F"   # Gold
    ]

    # Initialize PyVis with updated physics configuration
    print("Initializing PyVis network...")
    net = Network(
        height="900px",
        width="1600px",
        directed=True,
        bgcolor="#ffffff"
    )

    # Updated physics options for better cluster separation
    options = {
        "physics": {
            "enabled": True,
            "solver": "forceAtlas2Based",
            "forceAtlas2Based": {
                "gravitationalConstant": -50,
                "centralGravity": 0.01,
                "springLength": 20,
                "springConstant": 0.8,
                "damping": 0.4,
                "avoidOverlap": 1.0
            },
            "minVelocity": 0.75,
            "maxVelocity": 50,
            "timestep": 0.35,
            "stabilization": {
                "enabled": True,
                "iterations": 1000,
                "updateInterval": 100,
                "fit": True
            }
        },
        "layout": {
            "randomSeed": 42,
            "improvedLayout": True,
            "clusterThreshold": 0
        },
        "nodes": {
            "font": {
                "size": 16,
                "face": "arial",
                "color": "#000000"
            },
            "fixed": {
                "x": False,
                "y": False
            }
        },
        "edges": {
            "smooth": False,
            "color": {
                "inherit": False
            }
        },
        "interaction": {
            "dragNodes": True,
            "hover": True,
            "zoomView": True,
            "navigationButtons": True
        }
    }

    net.set_options(json.dumps(options))

    # Add nodes with modified mass for better cluster separation
    print("Adding nodes...")
    for row in filtered_nodes:
        cluster = row["cluster"]
        cluster_idx = cluster_mapping[cluster]
        degree = float(row["total_degree"])
        pagerank = float(row["pagerank"])
        
        # Adjust size for better visibility
        size = min(50, max(20, 10 + (degree ** 0.5) * 3))
        
        color = predefined_colors[cluster_idx % len(predefined_colors)]
        
        # Calculate mass based on connections to affect layout
        mass = 1 + (degree ** 0.5)  # Non-linear scaling for better distribution
        
        net.add_node(
            row["id"],
            label=row["node"],
            color=color,
            size=size,
            title=f"Cluster: {cluster_idx}\nDegree: {int(degree)}\nPageRank: {pagerank:.4f}",
            group=cluster_idx,
            mass=mass,
            font={'size': 16, 'color': '#000000'}
        )

    # Add edges with cluster-based styling
    print("Adding edges...")
    edge_count = 0
    for edge in filtered_edges:
        src_id = edge["src"]
        dst_id = edge["dst"]
        
        # Get cluster information for source and destination
        src_node = next((n for n in filtered_nodes if n["id"] == src_id), None)
        dst_node = next((n for n in filtered_nodes if n["id"] == dst_id), None)
        
        if src_node and dst_node:
            same_cluster = src_node["cluster"] == dst_node["cluster"]
            
            net.add_edge(
                src_id,
                dst_id,
                color={
                    "color": "#666666",
                    "opacity": 0.4 if same_cluster else 0.1
                },
                width=2 if same_cluster else 0.5,
                physics=True  # Enable physics for initial layout
            )
            edge_count += 1
                
            # Add progress update
            if edge_count % 1000 == 0:
                print(f"Added {edge_count} edges...")

    # Save visualization
    output_file = "natural_clustered_trust_graph.html"
    print(f"Generating visualization and saving to {output_file}...")
    net.show(output_file)
    
    print("Stopping Spark session...")
    spark.stop()
    print("Script completed successfully.")

if __name__ == "__main__":
    main()
