import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import random
import os
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
import copy
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import time
import json
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
dir = './dataset'

class GNN_Encoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_dim, out_channels, dropout=0.2):
        super(GNN_Encoder, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.conv4 = GCNConv(hidden_dim, out_channels)
        self.dropout = dropout
        
    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv3(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv4(x, edge_index)
        return x

class ProjectionHead(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(ProjectionHead, self).__init__()
        self.projection = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )
        
    def forward(self, x):
        return F.normalize(self.projection(x), dim=1)

class SupervisedContrastiveLearningPubMed:
    def __init__(self, temperature=0.1, lr=0.001, epochs=100, device='cuda', 
                 proj_dim=128, batch_size=32, new_node_weight=2.0, original_node_lr_factor=1,
                 lr_factor=0.5, lr_patience=10, lr_min=1e-6, lr_threshold=0.01,
                 early_stop_patience=15, early_stop_threshold=0.995, 
                 rag_enhancement_interval=20, num_anchor_nodes=20, 
                 same_category_samples=3, different_category_samples=7,
                 openai_api_key="sk-3eVc2TVDNenkKyQUF9lUXan3Ec6a5bQeQZ8WAZYxSQV1m54K",
                 openai_base_url="https://api.chatanywhere.org/v1"):
        self.temperature = temperature
        self.lr = lr
        self.epochs = epochs
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.proj_dim = proj_dim
        self.batch_size = batch_size
        self.new_node_weight = new_node_weight
        self.original_node_lr_factor = original_node_lr_factor
        
        self.lr_factor = lr_factor
        self.lr_patience = lr_patience
        self.lr_min = lr_min
        self.lr_threshold = lr_threshold
        
        self.early_stop_patience = early_stop_patience
        self.early_stop_threshold = early_stop_threshold
        
        self.rag_enhancement_interval = rag_enhancement_interval
        self.num_anchor_nodes = num_anchor_nodes
        self.same_category_samples = same_category_samples
        self.different_category_samples = different_category_samples
        
        self.openai_client = OpenAI(
            api_key=openai_api_key,
            base_url=openai_base_url
        )
        
        self.embedding_model = None
        self.embeddings_cache = None
        self.raw_texts = None
        self.new_node_texts = None
        self.paper_labels = None
        self.category_to_papers = None
        
        self.feedback_dir = "rag_feedback"
        os.makedirs(self.feedback_dir, exist_ok=True)
        
        self.rag_enhanced_view = None
        self.current_rag_epoch = -1

    def load_data(self, seed=42):
        """Load original PubMed data and embeddings for newly generated nodes."""
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        
        path = dir + '/PubMed_orig/data'
        n_nodes = 19717
        
        data_Y = [None] * n_nodes
        data_pubid = [None] * n_nodes
        paper_to_index = {}
        
        with open(path + '/Pubmed-Diabetes.NODE.paper.tab', 'r') as node_file:
            node_file.readline()
            node_file.readline()
            
            for i, line in enumerate(node_file.readlines()):
                items = line.strip().split('\t')
                paper_id = items[0]
                data_pubid[i] = paper_id
                paper_to_index[paper_id] = i
                
                label = int(items[1].split('=')[-1]) - 1  
                data_Y[i] = label
        
        data_Y = np.array(data_Y)
        
        data_edges = []
        with open(path + '/Pubmed-Diabetes.DIRECTED.cites.tab', 'r') as edge_file:
            edge_file.readline()
            edge_file.readline()
            
            for i, line in enumerate(edge_file.readlines()):
                items = line.strip().split('\t')
                tail = items[1].split(':')[-1]
                head = items[3].split(':')[-1]
                
                if head != tail:
                    data_edges.append((paper_to_index[head], paper_to_index[tail]))
                    data_edges.append((paper_to_index[tail], paper_to_index[head]))
        
        data_edges = np.unique(data_edges, axis=0)
        edge_index = torch.tensor(data_edges.transpose(), dtype=torch.long)
        
        text_embed = np.load(dir + '/pubmed_process/pubmed_text_embed.npy')
        new_embed = np.load(dir + '/pubmed_process/pubmed_new_sample_weighted.npy')
        
        N = len(data_Y)
        M = new_embed.shape[0]
        C = np.max(data_Y) + 1
        
        new_y = []
        cycle_index = 0
        for _ in range(M):
            category = cycle_index % C
            new_y.append(category)
            cycle_index += 1
            
        labels = np.concatenate([data_Y, new_y])
        
        print(f"Original node count: {N}")
        print(f"New node count: {M}")
        print(f"Total labels: {len(labels)}")
        print(f"Original node embeddings shape: {text_embed.shape}")
        print(f"New node embeddings shape: {new_embed.shape}")
        print(f"New node label distribution: {np.bincount(new_y)}")
        print(f"PubMed categories: {['Diabetes Mellitus, Experimental', 'Diabetes Mellitus Type 1', 'Diabetes Mellitus Type 2']}")
        
        return text_embed, new_embed, edge_index, labels, N, M 

    def init_rag_components(self):
        """Initialize RAG components."""
        print("🔧 Initializing RAG components...")
        
        if self.embedding_model is None:
            print("   📦 Loading SentenceTransformer model...")
            model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_big")
            self.embedding_model = SentenceTransformer(model_path)
            print(f"   ✅ Embedding model loaded: {model_path}")
        
        if self.raw_texts is None:
            print("   📄 Loading raw text data...")
            base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset", "pubmed_process")
            with open(os.path.join(base_path, "raw_text.txt"), "r", encoding="utf-8", errors="replace") as f:
                self.raw_texts = f.readlines()
                self.raw_texts = [text.strip() for text in self.raw_texts]
            print(f"   ✅ Loaded {len(self.raw_texts)} raw texts")
        
        if self.new_node_texts is None:
            print("   📄 Loading new node text data...")
            base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset", "pubmed_process")
            try:
                with open(os.path.join(base_path, "pubmed_new_sample_93.txt"), "r", encoding="utf-8", errors="replace") as f:
                    self.new_node_texts = f.readlines()
                    self.new_node_texts = [text.strip() for text in self.new_node_texts]
                print(f"   ✅ Loaded {len(self.new_node_texts)} new node texts")
            except FileNotFoundError:
                print("   ⚠️  pubmed_new_sample_93.txt not found, using placeholder texts")
                self.new_node_texts = []
        
        if self.paper_labels is None:
            print("   🏷️  Loading paper labels...")
            labels_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                     "dataset", "pubmed_process", "paper_labels.npy")
            try:
                self.paper_labels = np.load(labels_file)
                print(f"   ✅ Loaded {len(self.paper_labels)} labels from file")
            except FileNotFoundError:
                print("   ⚠️  paper_labels.npy not found, creating from PubMed dataset...")
                pubmed_orig_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                            "dataset", "PubMed_orig", "data")
                node_file_path = os.path.join(pubmed_orig_path, "Pubmed-Diabetes.NODE.paper.tab")
                
                paper_labels = []
                with open(node_file_path, 'r') as node_file:
                    node_file.readline()
                    node_file.readline()
                    
                    for i, line in enumerate(node_file.readlines()):
                        items = line.strip().split('\t')
                        label = int(items[1].split('=')[-1]) - 1
                        paper_labels.append(label)
                
                self.paper_labels = np.array(paper_labels)
                np.save(labels_file, self.paper_labels)
                print(f"   ✅ Created and saved {len(self.paper_labels)} labels")
        
        if self.category_to_papers is None:
            print("   📊 Creating category mapping...")
            categories = ['Diabetes Mellitus, Experimental', 'Diabetes Mellitus Type 1', 'Diabetes Mellitus Type 2']
            self.category_to_papers = {}
            total_papers = len(self.paper_labels)
            for i, category in enumerate(categories):
                self.category_to_papers[i] = [j for j in range(total_papers) if self.paper_labels[j] == i]
                print(f"      - {category}: {len(self.category_to_papers[i])} papers")
        
        print("   ✅ RAG components initialized!") 

    def init_embeddings_cache(self, embeddings):
        """Initialize embeddings cache."""
        print("🔍 Initializing embeddings cache...")
        
        d = embeddings.shape[1]
        print(f"   📏 Embedding dimension: {d}")
        print(f"   📊 Number of vectors: {embeddings.shape[0]}")
        
        embeddings_normalized = normalize(embeddings, norm='l2', axis=1)
        print("   📐 L2 normalization complete")
        
        print(f"   ✅ Embeddings cache created with {embeddings_normalized.shape[0]} vectors of dimension {d}")
        
        return embeddings_normalized
    
    def retrieve_similar_documents(self, query_embed, k=10):
        """Retrieve similar documents (using cosine similarity)."""
        query_embed_normalized = normalize(query_embed.reshape(1, -1), norm='l2', axis=1)
        
        similarities = cosine_similarity(query_embed_normalized, self.embeddings_cache)[0]
        
        top_k_indices = np.argsort(similarities)[::-1][:k]
        top_k_similarities = similarities[top_k_indices]
        
        return top_k_similarities.reshape(1, -1), top_k_indices.reshape(1, -1)
    
    def get_text_embedding(self, text):
        """Get text embedding using the local model."""
        try:
            embedding = self.embedding_model.encode(text, convert_to_numpy=True)
            embedding = embedding.astype(np.float32)
            return embedding
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return None
    
    def call_llm_for_text_modification(self, anchor_text, anchor_category, similar_texts):
        """Call LLM to modify the anchor text - PubMed diabetes research version."""
        categories = ['Diabetes Mellitus, Experimental', 'Diabetes Mellitus Type 1', 'Diabetes Mellitus Type 2']
        
        prompt = f"""You are tasked with modifying a diabetes research paper text to make it more distinctive while maintaining its core category characteristics.

ANCHOR PAPER (Category: {categories[anchor_category]}):
{anchor_text}

SIMILAR PAPERS FOR REFERENCE:
{chr(10).join([f"- {text}" for text in similar_texts])}

TASK: Modify the anchor paper to:
1. Keep it clearly in the {categories[anchor_category]} category
2. Make it more distinctive from the similar papers
3. Maintain the same format: Title: ... Abstract: ... Keywords: ...
4. Preserve key medical concepts and clinical methodologies but vary the specific approach or research focus
5. Use different clinical terminology while staying within the same diabetes research domain

RULES:
- Output ONLY the modified paper in the exact same format
- Do not add explanations or meta-commentary  
- Ensure the modification makes medical sense
- Keep the core clinical contribution but vary the presentation
- Focus on diabetes-specific medical terminology and research approaches

Modified paper:"""

        try:
            response = self.openai_client.chat.completions.create(
                model='gpt-4o-mini',
                messages=[{"role": "user", "content": prompt}],
                max_tokens=800,
                temperature=0.7
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error calling LLM for text modification: {e}")
            return anchor_text
    
    def call_llm_for_edge_analysis(self, anchor_text, similar_texts, anchor_category):
        """Call LLM to analyze edge relationships - PubMed diabetes research version."""
        categories = ['Diabetes Mellitus, Experimental', 'Diabetes Mellitus Type 1', 'Diabetes Mellitus Type 2']
        
        similar_texts_with_indices = [f"Paper {i+1}: {text}" for i, text in enumerate(similar_texts)]
        
        prompt = f"""You are analyzing diabetes research paper relationships for a graph neural network.

ANCHOR PAPER (Category: {categories[anchor_category]}):
{anchor_text}

CANDIDATE PAPERS:
{chr(10).join(similar_texts_with_indices)}

TASK: Determine which papers should have edges (connections) to the anchor paper based on:
1. Clinical methodology similarity (similar experimental approaches, clinical trial designs, biomarker analysis)
2. Shared diabetes research domains or therapeutic targets
3. Medical conceptual relationships (related pathophysiology, treatment mechanisms, patient populations)
4. Citation worthiness in diabetes research (would these papers likely cite each other?)

For each paper, output ONLY "CONNECT" or "REMOVE" followed by the paper number.

IMPORTANT: 
- Papers in the same diabetes category should generally connect if they share clinical methodologies
- Papers in different diabetes categories should only connect if there's strong clinical/methodological overlap
- Be selective - not all papers need to connect
- Consider diabetes-specific research connections

Format your response as:
Paper 1: CONNECT/REMOVE
Paper 2: CONNECT/REMOVE
...
Paper {len(similar_texts)}: CONNECT/REMOVE"""

        try:
            response = self.openai_client.chat.completions.create(
                model='gpt-4o-mini',
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.3
            )
            
            response_text = response.choices[0].message.content.strip()
            decisions = []
            
            for line in response_text.split('\n'):
                if 'Paper' in line and (':' in line):
                    if 'CONNECT' in line.upper():
                        decisions.append(True)
                    elif 'REMOVE' in line.upper():
                        decisions.append(False)
                    else:
                        decisions.append(True)
            
            while len(decisions) < len(similar_texts):
                decisions.append(True)
            
            return decisions[:len(similar_texts)]
            
        except Exception as e:
            print(f"Error calling LLM for edge analysis: {e}")
            return [True] * len(similar_texts)

    def save_rag_feedback(self, epoch, anchor_idx, original_text, modified_text, 
                         similar_texts, edge_decisions, anchor_category):
        """Save feedback for the RAG enhancement process."""
        categories = ['Diabetes Mellitus, Experimental', 'Diabetes Mellitus Type 1', 'Diabetes Mellitus Type 2']
        
        feedback = {
            'epoch': epoch,
            'anchor_node_index': anchor_idx,
            'anchor_category': categories[anchor_category],
            'original_text': original_text,
            'modified_text': modified_text,
            'similar_texts': similar_texts,
            'edge_decisions': edge_decisions,
            'timestamp': time.time()
        }
        
        filename = os.path.join(self.feedback_dir, f"rag_feedback_epoch_{epoch}_anchor_{anchor_idx}.json")
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(feedback, f, ensure_ascii=False, indent=2)
        
        print(f"RAG feedback saved to {filename}")
    
    def create_graph_augmentations(self, x, edge_index, use_rag_enhanced_view=False, 
                                  drop_edge_ratio=0.1, drop_feature_ratio=0.1):
        """Create graph augmentation views - optimized version."""
        num_edges = edge_index.size(1)
        num_drop = int(num_edges * drop_edge_ratio)
        keep_indices = torch.randperm(num_edges)[num_drop:]
        edge_index_1 = edge_index[:, keep_indices]
        
        x_1 = x.clone()
        feature_dim = x.size(1)
        for i in range(x.size(0)):
            drop_indices = torch.randperm(feature_dim)[:int(feature_dim * drop_feature_ratio)]
            mask = torch.ones_like(x[i], dtype=torch.bool)
            mask[drop_indices] = False
            x_1[i] = x[i] * mask
            
        if use_rag_enhanced_view and self.rag_enhanced_view is not None:
            x_2, edge_index_2 = self.rag_enhanced_view
            x_2 = x_2.detach().clone()
            edge_index_2 = edge_index_2.clone()
        else:
            keep_indices_2 = torch.randperm(num_edges)[num_drop:]
            edge_index_2 = edge_index[:, keep_indices_2]
            
            x_2 = x.clone()
            for i in range(x.size(0)):
                drop_indices = torch.randperm(feature_dim)[:int(feature_dim * drop_feature_ratio)]
                mask = torch.ones_like(x[i], dtype=torch.bool)
                mask[drop_indices] = False
                x_2[i] = x[i] * mask
        
        return (x_1, edge_index_1), (x_2, edge_index_2)
    
    def get_batch_by_labels(self, indices, labels, batch_size):
        """Build a batch by labels ensuring enough samples from each class."""
        unique_labels = np.unique(labels)
        num_classes = len(unique_labels)
        
        class_indices = [[] for _ in range(num_classes)]
        for idx in indices:
            class_indices[labels[idx]].append(idx)
        
        samples_per_class = max(1, batch_size // (num_classes * 2))
        batch_indices = []
        
        for class_idx in class_indices:
            if len(class_idx) >= samples_per_class:
                selected = random.sample(class_idx, samples_per_class)
                batch_indices.extend(selected)
        
        remaining = batch_size - len(batch_indices)
        if remaining > 0:
            remaining_candidates = [idx for idx in indices if idx not in batch_indices]
            if remaining_candidates:
                additional = random.sample(remaining_candidates, min(remaining, len(remaining_candidates)))
                batch_indices.extend(additional)
        
        return batch_indices
    
    def supcon_loss(self, features, labels, mask=None):
        """Implement supervised contrastive loss (SupCon) - matching Eq. 2 in the paper."""
        labels = labels.contiguous().view(-1, 1)
        
        labels_augmented = torch.cat([labels, labels], dim=0)
        
        sim_matrix = torch.mm(features, features.t()) / self.temperature
        
        mask_self = ~torch.eye(sim_matrix.shape[0], dtype=torch.bool, device=sim_matrix.device)
        
        labels_equal = labels_augmented.eq(labels_augmented.t()).float()
        
        loss = 0.0
        num_valid_anchors = 0
        n_samples = features.shape[0]
        
        for i in range(n_samples):
            pos_indices = torch.where(labels_equal[i] * mask_self[i] > 0)[0]
            
            if len(pos_indices) == 0:
                continue
            
            neg_indices = torch.where(mask_self[i])[0]
            
            denominator = torch.exp(sim_matrix[i, neg_indices]).sum()
            
            anchor_loss = 0.0
            for p_idx in pos_indices:
                numerator = torch.exp(sim_matrix[i, p_idx])
                anchor_loss += -torch.log(numerator / denominator)
            
            anchor_loss = anchor_loss / len(pos_indices)
            
            if mask is not None:
                node_weight = self.new_node_weight if mask[i % (n_samples//2)] == 1 else 1.0
                anchor_loss = anchor_loss * node_weight
            
            loss += anchor_loss
            num_valid_anchors += 1
        
        if num_valid_anchors > 0:
            loss = loss / num_valid_anchors
        
        return loss 

    def perform_rag_enhancement(self, x, edge_index, epoch, labels, N, M):
        """Perform RAG enhancement - PubMed version."""
        print(f"\n{'='*80}")
        print(f"🚀 EPOCH {epoch}: Starting RAG enhancement (PubMed)")
        print(f"{'='*80}")
        print(f"📊 Current training state:")
        print(f"   - Original nodes: {N}")
        print(f"   - New nodes: {M}")
        print(f"   - Total nodes: {N + M}")
        print(f"   - Number of edges: {edge_index.size(1)}")
        print(f"   - Anchor nodes: {self.num_anchor_nodes}")
        print(f"   - Same-category samples: {self.same_category_samples}")
        print(f"   - Different-category samples: {self.different_category_samples}")
        
        if self.embeddings_cache is None:
            print(f"\n🔧 Initializing RAG components...")
            self.init_rag_components()
            current_embeddings = x.detach().cpu().numpy()
            self.embeddings_cache = self.init_embeddings_cache(current_embeddings)
            print(f"✅ RAG components initialization complete")
        else:
            print(f"\n🔄 Updating embeddings cache...")
            current_embeddings = x.detach().cpu().numpy()
            self.embeddings_cache = self.init_embeddings_cache(current_embeddings)
            print(f"✅ Embeddings cache updated")
        
        x_2 = x.clone()
        edge_index_2 = edge_index.clone()
        
        all_node_indices = list(range(x.size(0)))
        anchor_indices = random.sample(all_node_indices, 
                                     min(self.num_anchor_nodes, len(all_node_indices)))
        
        print(f"\n🎯 Selecting anchor nodes:")
        print(f"   - Selected {len(anchor_indices)} anchors from {len(all_node_indices)} nodes")
        print(f"   - Anchor indices: {anchor_indices}")
        
        successful_modifications = 0
        successful_edge_analyses = 0
        llm_text_calls = 0
        llm_edge_calls = 0
        total_edges_removed = 0
        
        categories = ['Diabetes Mellitus, Experimental', 'Diabetes Mellitus Type 1', 'Diabetes Mellitus Type 2']
        
        print(f"\n📝 Processing anchor nodes...")
        for i, anchor_idx in enumerate(anchor_indices, 1):
            print(f"\n📌 Processing anchor {i}/{len(anchor_indices)} (node index: {anchor_idx})")
            try:
                if anchor_idx < N:
                    anchor_text = self.raw_texts[anchor_idx]
                    anchor_category = self.paper_labels[anchor_idx]
                    node_type = "Original node"
                else:
                    new_node_idx = anchor_idx - N
                    anchor_category = labels[anchor_idx].item()
                    
                    if self.new_node_texts and new_node_idx < len(self.new_node_texts):
                        anchor_text = self.new_node_texts[new_node_idx]
                        node_type = "New node (real text)"
                    else:
                        anchor_text = f"Title: Generated Paper in {categories[anchor_category]}  Abstract: This is a generated diabetes research paper focusing on {categories[anchor_category]} methodologies and clinical approaches.  Keywords: diabetes, {categories[anchor_category].lower()}, clinical research, methodology"
                        node_type = "New node (placeholder text)"
                
                print(f"   🏷️  Node type: {node_type}")
                print(f"   📂 Category: {categories[anchor_category]} (index: {anchor_category})")
                print(f"   📄 Original text length: {len(anchor_text)} characters")
                
                print(f"   🔍 Retrieving similar documents...")
                anchor_embed = self.embeddings_cache[anchor_idx]
                similarities, similar_indices = self.retrieve_similar_documents(
                    anchor_embed, k=100
                )
                
                same_category_candidates = []
                different_category_candidates = []
                
                for j, sim_idx in enumerate(similar_indices[0]):
                    if sim_idx == anchor_idx:
                        continue
                    
                    if sim_idx < N:
                        if sim_idx >= len(self.raw_texts):
                            continue
                        sim_text = self.raw_texts[sim_idx]
                        sim_category = self.paper_labels[sim_idx]
                    else:
                        new_sim_idx = sim_idx - N
                        if self.new_node_texts and new_sim_idx < len(self.new_node_texts):
                            sim_text = self.new_node_texts[new_sim_idx]
                        else:
                            continue
                        sim_category = labels[sim_idx].item()
                    
                    similarity_score = similarities[0][j]
                    
                    if sim_category == anchor_category:
                        same_category_candidates.append((sim_idx, similarity_score, sim_text, sim_category))
                    else:
                        different_category_candidates.append((sim_idx, similarity_score, sim_text, sim_category))
                
                same_category_candidates.sort(key=lambda x: x[1], reverse=True)
                different_category_candidates.sort(key=lambda x: x[1], reverse=True)
                
                selected_same_category = same_category_candidates[:self.same_category_samples]
                selected_different_category = different_category_candidates[:self.different_category_samples]
                
                all_selected_samples = selected_same_category + selected_different_category
                
                same_category_texts = [item[2] for item in selected_same_category]
                different_category_texts = [item[2] for item in selected_different_category]
                all_similar_texts = same_category_texts + different_category_texts
                similar_texts_info = [(item[0], item[2], item[3]) for item in all_selected_samples]
                
                print(f"   📋 Retrieval results:")
                print(f"      - Candidate same-category samples: {len(same_category_candidates)}")
                print(f"      - Candidate different-category samples: {len(different_category_candidates)}")
                print(f"      - Selected same-category samples: {len(selected_same_category)}")
                print(f"      - Selected different-category samples: {len(selected_different_category)}")
                print(f"      - Total selected samples: {len(all_similar_texts)}")
                
                if selected_same_category:
                    same_scores = [f"{item[1]:.4f}" for item in selected_same_category]
                    print(f"      - Same-category similarity: {same_scores}")
                if selected_different_category:
                    diff_scores = [f"{item[1]:.4f}" for item in selected_different_category]
                    print(f"      - Different-category similarity: {diff_scores}")
                
                if len(all_similar_texts) > 0:
                    print(f"   🤖 Calling LLM for text modification...")
                    llm_text_calls += 1
                    modified_text = self.call_llm_for_text_modification(
                        anchor_text, anchor_category, all_similar_texts
                    )
                    
                    if modified_text and modified_text != anchor_text:
                        print(f"   ✅ Text modification successful")
                        print(f"      - Modified text length: {len(modified_text)} characters")
                        print(f"      - Change magnitude: {abs(len(modified_text) - len(anchor_text))} characters")
                        
                        print(f"   🔢 Generating new embedding...")
                        modified_embed = self.get_text_embedding(modified_text)
                        if modified_embed is not None:
                            x_2[anchor_idx] = torch.tensor(modified_embed, 
                                                         dtype=torch.float32, device=self.device)
                            successful_modifications += 1
                            print(f"   ✅ Embedding replaced successfully")
                        else:
                            print(f"   ❌ Failed to generate embedding, keeping original")
                    else:
                        print(f"   ⚠️  Text modification failed or no change, keeping original text")
                    
                    print(f"   🤖 Calling LLM for edge analysis...")
                    llm_edge_calls += 1
                    edge_decisions = self.call_llm_for_edge_analysis(
                        anchor_text, all_similar_texts, anchor_category
                    )
                    
                    print(f"   📊 LLM edge analysis results:")
                    connect_count = sum(edge_decisions)
                    remove_count = len(edge_decisions) - connect_count
                    print(f"      - LLM suggests keep connections: {connect_count}")
                    print(f"      - LLM suggests remove connections: {remove_count}")
                    
                    edges_to_remove = []
                    for k, (sim_idx, _, _) in enumerate(similar_texts_info):
                        if k < len(edge_decisions) and not edge_decisions[k]:
                            edge_mask = ((edge_index_2[0] == anchor_idx) & (edge_index_2[1] == sim_idx)) | \
                                      ((edge_index_2[0] == sim_idx) & (edge_index_2[1] == anchor_idx))
                            edges_to_remove.extend(torch.where(edge_mask)[0].tolist())
                    
                    if edges_to_remove:
                        keep_edges = torch.ones(edge_index_2.size(1), dtype=torch.bool)
                        keep_edges[edges_to_remove] = False
                        edge_index_2 = edge_index_2[:, keep_edges]
                        total_edges_removed += len(edges_to_remove)
                        successful_edge_analyses += 1
                        print(f"   ✅ Actually removed {len(edges_to_remove)} edges")
                    else:
                        print(f"   ℹ️  No edges to remove (target edges not present in graph)")
                    
                    self.save_rag_feedback(
                        epoch, anchor_idx, anchor_text, modified_text,
                        all_similar_texts, edge_decisions, anchor_category
                    )
                    print(f"   💾 Feedback saved")
                    
                else:
                    print(f"   ⚠️  Not enough similar samples found, skipping this anchor")
                    
            except Exception as e:
                print(f"   ❌ Error processing anchor {anchor_idx}: {e}")
                continue
        
        self.rag_enhanced_view = (x_2, edge_index_2)
        self.current_rag_epoch = epoch
        
        print(f"\n{'='*80}")
        print(f"✅ RAG enhancement complete! Summary:")
        print(f"{'='*80}")
        print(f"📈 Processing statistics:")
        print(f"   - Anchors processed: {len(anchor_indices)}")
        print(f"   - Successful text modifications: {successful_modifications}")
        print(f"   - Successful edge analyses: {successful_edge_analyses}")
        print(f"   - LLM text modification calls: {llm_text_calls}")
        print(f"   - LLM edge analysis calls: {llm_edge_calls}")
        print(f"   - Total LLM calls: {llm_text_calls + llm_edge_calls}")
        print(f"   - Edges removed: {total_edges_removed}")
        print(f"   - Remaining edges: {edge_index_2.size(1)}")
        print(f"📁 Feedback files saved to: {self.feedback_dir}/")
        print(f"⏰ This RAG enhancement will be used for all batches in epoch {epoch}")
        print(f"⏭️  Next RAG enhancement will run at epoch {epoch + self.rag_enhancement_interval}")
        print(f"{'='*80}") 

    def train(self):
        """Train the supervised contrastive learning model and optimize all node embeddings - PubMed version."""
        text_embed, new_embed, edge_index, labels, N, M = self.load_data()
        
        all_embeds = np.concatenate([text_embed, new_embed])
        feature_dim = all_embeds.shape[1]
        
        new_edges = []
        sim_threshold = 0.7
        for i in range(M):
            new_idx = N + i
            similarities = np.dot(new_embed[i], text_embed.T)
            connections = np.where(similarities > sim_threshold)[0]
            for j in connections:
                new_edges.append([new_idx, j])
                new_edges.append([j, new_idx])
        
        if new_edges:
            new_edge_index = torch.tensor(new_edges, dtype=torch.long).t()
            edge_index = torch.cat([edge_index, new_edge_index], dim=1)
        
        x_original = torch.nn.Parameter(torch.tensor(text_embed, dtype=torch.float).to(self.device))
        x_new = torch.nn.Parameter(torch.tensor(new_embed, dtype=torch.float).to(self.device))
        
        edge_index = edge_index.to(self.device)
        y = torch.tensor(labels, dtype=torch.long).to(self.device)
        
        encoder = GNN_Encoder(feature_dim, 256, feature_dim, dropout=0.2).to(self.device)
        projector = ProjectionHead(feature_dim, 128, self.proj_dim).to(self.device)
        
        optimizer = torch.optim.Adam([
            {'params': encoder.parameters(), 'lr': self.lr},
            {'params': projector.parameters(), 'lr': self.lr},
            {'params': [x_original], 'lr': self.lr * self.original_node_lr_factor},
            {'params': [x_new], 'lr': self.lr}
        ])
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min',
            factor=self.lr_factor,
            patience=self.lr_patience,
            threshold=self.lr_threshold,
            threshold_mode='rel',
            min_lr=self.lr_min,
            verbose=True
        )
        
        all_indices = list(range(N + M))
        
        node_type_mask = torch.zeros(N + M, dtype=torch.long, device=self.device)
        node_type_mask[N:] = 1
        
        best_loss = float('inf')
        patience_counter = 0
        
        print("Starting supervised contrastive training (update all node embeddings) - PubMed version...")
        print(f"🔄 RAG enhancement frequency: every {self.rag_enhancement_interval} epochs")
        print(f"📅 RAG enhancement planned epochs: {[i for i in range(self.rag_enhancement_interval, self.epochs, self.rag_enhancement_interval)]}")
        print(f"🧬 Dataset: PubMed diabetes research (3 classes)")
        print(f"{'='*60}")
        
        for epoch in range(self.epochs):
            encoder.train()
            projector.train()
            epoch_loss = 0
            num_batches = 0
            
            use_rag_enhanced = False
            if epoch % self.rag_enhancement_interval == 0:
                print(f"\n🎯 Epoch {epoch}: RAG enhancement trigger (epoch % {self.rag_enhancement_interval} == 0)")
                x_all = torch.cat([x_original, x_new], dim=0)
                self.perform_rag_enhancement(x_all, edge_index, epoch, labels, N, M)
                use_rag_enhanced = True
            else:
                if self.rag_enhanced_view is not None and self.current_rag_epoch >= 0:
                    use_rag_enhanced = True
                    if (epoch + 1) % 10 == 0:
                        print(f"\n📝 Epoch {epoch}: Continue using RAG-enhanced view (from epoch {self.current_rag_epoch})")
                        next_rag_epoch = ((epoch // self.rag_enhancement_interval) + 1) * self.rag_enhancement_interval
                        if next_rag_epoch <= self.epochs:
                            print(f"⏳ Next RAG enhancement at epoch {next_rag_epoch}")
                else:
                    use_rag_enhanced = False
                    next_rag_epoch = ((epoch // self.rag_enhancement_interval) + 1) * self.rag_enhancement_interval
                    if next_rag_epoch <= self.epochs:
                        if epoch == 1:
                            print(f"\n📝 Epoch {epoch}: Regular training mode (no RAG-enhanced view)")
                            print(f"⏳ Next RAG enhancement at epoch {next_rag_epoch}")
                        elif (epoch + 1) % 10 == 0:
                            print(f"\n📝 Epoch {epoch}: Regular training mode (no RAG-enhanced view)")
                            print(f"⏳ Next RAG enhancement at epoch {next_rag_epoch}")
            
            random.shuffle(all_indices)
            
            for i in range(0, len(all_indices), self.batch_size):
                batch_indices = self.get_batch_by_labels(
                    all_indices[i:i+self.batch_size], 
                    labels, 
                    min(self.batch_size, len(all_indices)-i)
                )
                
                if len(batch_indices) < 4:
                    continue
                
                optimizer.zero_grad()
                
                x_all = torch.cat([x_original, x_new], dim=0)
                
                (x_view1, edge_index_view1), (x_view2, edge_index_view2) = self.create_graph_augmentations(
                    x_all, edge_index, use_rag_enhanced_view=use_rag_enhanced
                )
                
                batch_tensor = torch.tensor(batch_indices, dtype=torch.long, device=self.device)
                
                h1 = encoder(x_view1, edge_index_view1)
                z1 = projector(h1[batch_tensor])
                
                h2 = encoder(x_view2, edge_index_view2)
                z2 = projector(h2[batch_tensor])
                
                z = torch.cat([z1, z2], dim=0)
                
                batch_labels = y[batch_tensor]
                
                batch_mask = node_type_mask[batch_tensor]
                
                loss = self.supcon_loss(z, batch_labels, batch_mask)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(projector.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(x_original, max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(x_new, max_norm=1.0)
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
                
                del z, h1, h2, z1, z2, loss, x_all
                torch.cuda.empty_cache()
            
            avg_loss = epoch_loss / num_batches if num_batches > 0 else float('inf')
            
            scheduler.step(avg_loss)
            
            if (epoch + 1) % 5 == 0 or epoch == 0:
                current_lr = optimizer.param_groups[0]['lr']
                original_lr = optimizer.param_groups[2]['lr']
                new_lr = optimizer.param_groups[3]['lr']
                
                next_rag_epoch = ((epoch // self.rag_enhancement_interval) + 1) * self.rag_enhancement_interval
                epochs_to_rag = next_rag_epoch - epoch if next_rag_epoch <= self.epochs else "N/A"
                
                print(f"\n📊 Epoch {epoch+1}/{self.epochs} training progress (PubMed):")
                print(f"   💰 Average loss: {avg_loss:.4f}")
                print(f"   📈 Learning rate - encoder: {current_lr:.6f}, original nodes: {original_lr:.6f}, new nodes: {new_lr:.6f}")
                if use_rag_enhanced:
                    if epoch % self.rag_enhancement_interval == 0:
                        print(f"   🎯 Current status: using newly created RAG-enhanced view")
                    else:
                        print(f"   🎯 Current status: continuing to use RAG-enhanced view (from epoch {self.current_rag_epoch})")
                else:
                    print(f"   📝 Current status: standard random augmentations")
                if epochs_to_rag != "N/A":
                    print(f"   ⏰ Epochs until next RAG enhancement: {epochs_to_rag}")
            
            if avg_loss < best_loss * self.early_stop_threshold:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= self.early_stop_patience:
                print(f"Early stopping triggered after {epoch+1} epochs. No improvement for {self.early_stop_patience} epochs.")
                break
        
        optimized_original_embed = x_original.detach().cpu().numpy()
        optimized_new_embed = x_new.detach().cpu().numpy()
        optimized_all_embed = np.concatenate([optimized_original_embed, optimized_new_embed], axis=0)
        
        return optimized_all_embed, optimized_original_embed, optimized_new_embed, text_embed, new_embed
    
    def save_optimized_embeddings(self, optimized_all_embed, optimized_original_embed, optimized_new_embed, 
                                 original_text_embed, original_new_embed, 
                                 all_filename='all_nodes_optimized_supcon_pubmed.npy',
                                 original_filename='original_nodes_optimized_supcon_pubmed.npy',
                                 new_filename='new_nodes_optimized_supcon_pubmed.npy'):
        """Save optimized node embeddings - PubMed version."""
        os.makedirs(os.path.dirname(dir + '/pubmed_process/' + all_filename), exist_ok=True)
        
        np.save(dir + '/pubmed_process/' + all_filename, optimized_all_embed)
        np.save(dir + '/pubmed_process/' + original_filename, optimized_original_embed)
        np.save(dir + '/pubmed_process/' + new_filename, optimized_new_embed)
        
        print(f"Original text embeddings shape: {original_text_embed.shape}")
        print(f"Original new-node embeddings shape: {original_new_embed.shape}")
        print(f"Optimized all-node embeddings shape: {optimized_all_embed.shape}")
        print(f"Optimized original-node embeddings shape: {optimized_original_embed.shape}")
        print(f"Optimized new-node embeddings shape: {optimized_new_embed.shape}")
        print(f"Optimized all-node embeddings saved to: {dir}/pubmed_process/{all_filename}")
        print(f"Optimized original-node embeddings saved to: {dir}/pubmed_process/{original_filename}")
        print(f"Optimized new-node embeddings saved to: {dir}/pubmed_process/{new_filename}")

if __name__ == "__main__":
    params = {
        'temperature': 0.07,
        'lr': 0.0003,
        'epochs': 50,
        'device': 'cuda',
        'proj_dim': 128,
        'batch_size': 128,
        'new_node_weight': 2.0,
        'original_node_lr_factor': 0.1,
        'lr_factor': 0.5,
        'lr_patience': 5,
        'lr_min': 1e-6,
        'lr_threshold': 0.01,
        'early_stop_patience': 15,
        'early_stop_threshold': 0.995,
        'rag_enhancement_interval': 5,
        'num_anchor_nodes': 15,
        'same_category_samples': 3,
        'different_category_samples': 7,
        'openai_api_key': "sk-3eVc2TVDNenkKyQUF9lUXan3Ec6a5bQeQZ8WAZYxSQV1m54K",
        'openai_base_url': "https://api.chatanywhere.org/v1"
    }
    
    supcon = SupervisedContrastiveLearningPubMed(**params)
    
    optimized_all_embed, optimized_original_embed, optimized_new_embed, original_text_embed, original_new_embed = supcon.train()
    
    supcon.save_optimized_embeddings(
        optimized_all_embed, 
        optimized_original_embed,
        optimized_new_embed,
        original_text_embed,
        original_new_embed,
        all_filename='all_nodes_optimized_supcon_rag_pubmed.npy',
        original_filename='original_nodes_optimized_supcon_rag_pubmed.npy',
        new_filename='new_nodes_optimized_supcon_rag_pubmed.npy'
    ) 