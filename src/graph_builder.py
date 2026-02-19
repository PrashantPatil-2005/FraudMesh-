import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from src.config import (
    TARGET_COL, CARD_COL, MERCHANT_NODE_COL, AMOUNT_COL,
    PLOTS_DIR, MAX_SUBGRAPH_NODES
)

def build_graph(train_df, card_node_to_idx, merchant_node_to_idx):
    """
    Build a bipartite graph of cards and merchants from training data.
    
    Args:
        train_df: Training DataFrame with card and merchant columns
        card_node_to_idx: Mapping of card IDs to indices
        merchant_node_to_idx: Mapping of merchant IDs to indices
        
    Returns:
        nx.Graph: NetworkX graph object
    """
    print("\n" + "=" * 80)
    print("BUILDING TRANSACTION GRAPH")
    print("=" * 80)
    
    # ========================================================================
    # STEP 1: Create Empty Graph
    # ========================================================================
    print("\n[Step 1/6] Creating empty graph...")
    G = nx.Graph()
    print("  Graph type: Undirected bipartite graph")
    
    # ========================================================================
    # STEP 2: Add Card Nodes
    # ========================================================================
    print("\n[Step 2/6] Adding card nodes...")
    
    # Calculate card-level statistics
    card_stats = train_df.groupby(CARD_COL).agg({
        TARGET_COL: ['mean', 'count'],
        AMOUNT_COL: 'mean'
    }).reset_index()
    card_stats.columns = ['card_id', 'fraud_rate', 'transaction_count', 'avg_amount']
    
    # Add nodes
    for _, row in card_stats.iterrows():
        card_id = row['card_id']
        if pd.isna(card_id):
            continue
            
        node_name = f"card_{int(card_id)}"
        G.add_node(
            node_name,
            node_type='card',
            node_idx=card_node_to_idx.get(card_id, -1),
            fraud_rate=float(row['fraud_rate']),
            transaction_count=int(row['transaction_count']),
            avg_amount=float(row['avg_amount'])
        )
    
    card_node_count = len([n for n in G.nodes() if G.nodes[n]['node_type'] == 'card'])
    print(f"  Added {card_node_count:,} card nodes")
    
    # ========================================================================
    # STEP 3: Add Merchant Nodes
    # ========================================================================
    print("\n[Step 3/6] Adding merchant nodes...")
    
    # Calculate merchant-level statistics
    merchant_stats = train_df.groupby(MERCHANT_NODE_COL).agg({
        TARGET_COL: ['mean', 'count'],
        AMOUNT_COL: 'mean'
    }).reset_index()
    merchant_stats.columns = ['merchant_id', 'fraud_rate', 'transaction_count', 'avg_amount']
    
    # Add nodes
    for _, row in merchant_stats.iterrows():
        merchant_id = row['merchant_id']
        if pd.isna(merchant_id):
            continue
            
        # Use hash to create shorter node names if merchant_id is very long
        node_name = f"merchant_{hash(merchant_id) % 1000000}"
        G.add_node(
            node_name,
            node_type='merchant',
            node_idx=merchant_node_to_idx.get(merchant_id, -1),
            fraud_rate=float(row['fraud_rate']),
            transaction_count=int(row['transaction_count']),
            avg_amount=float(row['avg_amount'])
        )
    
    merchant_node_count = len([n for n in G.nodes() if G.nodes[n]['node_type'] == 'merchant'])
    print(f"  Added {merchant_node_count:,} merchant nodes")
    
    # ========================================================================
    # STEP 4: Add Edges
    # ========================================================================
    print("\n[Step 4/6] Adding edges...")
    
    # Group transactions by card-merchant pairs
    edge_stats = train_df.groupby([CARD_COL, MERCHANT_NODE_COL]).agg({
        TARGET_COL: ['sum', 'count'],
        AMOUNT_COL: ['sum', 'mean']
    }).reset_index()
    edge_stats.columns = ['card_id', 'merchant_id', 'fraud_count', 
                          'transaction_count', 'total_amount', 'avg_amount']
    
    edges_added = 0
    for _, row in edge_stats.iterrows():
        card_id = row['card_id']
        merchant_id = row['merchant_id']
        
        if pd.isna(card_id) or pd.isna(merchant_id):
            continue
        
        card_node_name = f"card_{int(card_id)}"
        merchant_node_name = f"merchant_{hash(merchant_id) % 1000000}"
        
        # Only add edge if both nodes exist
        if card_node_name in G.nodes() and merchant_node_name in G.nodes():
            G.add_edge(
                card_node_name,
                merchant_node_name,
                transaction_count=int(row['transaction_count']),
                fraud_count=int(row['fraud_count']),
                total_amount=float(row['total_amount']),
                avg_amount=float(row['avg_amount'])
            )
            edges_added += 1
    
    print(f"  Added {edges_added:,} edges")
    
    # ========================================================================
    # STEP 5: Print Graph Statistics
    # ========================================================================
    print("\n[Step 5/6] Graph statistics:")
    
    print(f"  Total nodes: {G.number_of_nodes():,}")
    print(f"    Card nodes: {card_node_count:,}")
    print(f"    Merchant nodes: {merchant_node_count:,}")
    print(f"  Total edges: {G.number_of_edges():,}")
    
    # Degree statistics
    card_degrees = [G.degree(n) for n in G.nodes() if G.nodes[n]['node_type'] == 'card']
    merchant_degrees = [G.degree(n) for n in G.nodes() if G.nodes[n]['node_type'] == 'merchant']
    
    if card_degrees:
        print(f"  Average card degree: {np.mean(card_degrees):.2f}")
        print(f"  Max card degree: {max(card_degrees)}")
    
    if merchant_degrees:
        print(f"  Average merchant degree: {np.mean(merchant_degrees):.2f}")
        print(f"  Max merchant degree: {max(merchant_degrees)}")
    
    # Connected components
    num_components = nx.number_connected_components(G)
    largest_cc = max(nx.connected_components(G), key=len)
    
    print(f"  Connected components: {num_components}")
    print(f"  Largest component size: {len(largest_cc):,} nodes")
    
    # ========================================================================
    # STEP 6: Visualize Fraud Cluster Subgraph
    # ========================================================================
    print("\n[Step 6/6] Creating fraud cluster visualization...")
    
    # Find a card with at least one fraud transaction
    fraud_cards = [n for n in G.nodes() 
                   if G.nodes[n]['node_type'] == 'card' 
                   and G.nodes[n]['fraud_rate'] > 0]
    
    if fraud_cards:
        # Pick the first fraud card
        seed_card = fraud_cards[0]
        
        # Get neighbors (merchants)
        merchants = list(G.neighbors(seed_card))
        
        # Get neighbors of those merchants (other cards) - depth 2
        subgraph_nodes = {seed_card}
        subgraph_nodes.update(merchants)
        
        for merchant in merchants[:10]:  # Limit to avoid too many nodes
            subgraph_nodes.update(list(G.neighbors(merchant))[:5])
        
        # Limit total nodes
        subgraph_nodes = list(subgraph_nodes)[:MAX_SUBGRAPH_NODES]
        
        # Create subgraph
        subgraph = G.subgraph(subgraph_nodes)
        
        # Plot
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Layout
        pos = nx.spring_layout(subgraph, k=2, iterations=50, seed=42)
        
        # Separate nodes by type
        card_nodes = [n for n in subgraph.nodes() if subgraph.nodes[n]['node_type'] == 'card']
        merchant_nodes = [n for n in subgraph.nodes() if subgraph.nodes[n]['node_type'] == 'merchant']
        
        # Draw card nodes
        card_colors = ['red' if subgraph.nodes[n]['fraud_rate'] > 0 else 'lightblue' 
                       for n in card_nodes]
        nx.draw_networkx_nodes(subgraph, pos, nodelist=card_nodes, 
                              node_color=card_colors, node_size=300, 
                              node_shape='o', ax=ax, label='Cards')
        
        # Draw merchant nodes
        merchant_colors = ['orange' if subgraph.nodes[n]['fraud_rate'] > 0 else 'lightgreen' 
                          for n in merchant_nodes]
        nx.draw_networkx_nodes(subgraph, pos, nodelist=merchant_nodes,
                              node_color=merchant_colors, node_size=500,
                              node_shape='s', ax=ax, label='Merchants')
        
        # Draw edges
        fraud_edges = [(u, v) for u, v in subgraph.edges() 
                      if subgraph[u][v]['fraud_count'] > 0]
        normal_edges = [(u, v) for u, v in subgraph.edges() 
                       if subgraph[u][v]['fraud_count'] == 0]
        
        nx.draw_networkx_edges(subgraph, pos, edgelist=normal_edges,
                              edge_color='gray', alpha=0.3, ax=ax)
        nx.draw_networkx_edges(subgraph, pos, edgelist=fraud_edges,
                              edge_color='red', width=2, alpha=0.6, ax=ax)
        
        ax.set_title('Sample Fraud Cluster Subgraph\n' + 
                    'Red cards/edges = fraud detected | Blue/Green = no fraud',
                    fontsize=14, fontweight='bold')
        ax.legend(loc='upper right')
        ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / 'sample_subgraph.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  Subgraph nodes: {len(subgraph_nodes)}")
        print(f"  Fraud edges: {len(fraud_edges)}")
        print(f"✓ Saved: sample_subgraph.png")
    else:
        print("  No fraud cards found for visualization")
    
    print("\n✓ Graph construction complete")
    return G
