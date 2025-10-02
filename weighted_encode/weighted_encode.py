import numpy as np
import os
import re
from sentence_transformers import SentenceTransformer

def init_embedding_model():
    """Initialize the BERT model with local weights"""
    model_path = "/root/LLM4NG-main(label sparsity)/model_big"
    print(f"Loading model from: {model_path}")
    embedding_model = SentenceTransformer(model_path)
    return embedding_model

def extract_keywords_and_content(input_file):
    """Extract keywords and content separately while maintaining order"""
    papers = []
    keywords = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
                
            # Split by Keywords:
            parts = line.split("Keywords:")
            if len(parts) != 2:
                print(f"Warning: Line does not contain Keywords section")
                continue
                
            paper_content = parts[0].strip()
            paper_keywords = parts[1].strip()
            
            # Remove any trailing dots or spaces from keywords
            paper_keywords = paper_keywords.rstrip('.')
            
            # Store the content and keywords separately
            papers.append(paper_content)
            keywords.append(paper_keywords)
    
    return papers, keywords

def save_separated_files(papers, keywords, base_name):
    """Save papers and keywords to separate files"""
    # Save papers without keywords
    with open(f"{base_name}_content.txt", 'w', encoding='utf-8') as f:
        for paper in papers:
            f.write(f"{paper}\n")
    
    # Save keywords
    with open(f"{base_name}_keywords.txt", 'w', encoding='utf-8') as f:
        for kw in keywords:
            f.write(f"{kw}\n")

def encode_with_weights(model, papers, keywords, content_weight=1.0, keyword_weight=2.0):
    """Encode papers and keywords with different weights and combine them"""
    # Verify matching lengths
    assert len(papers) == len(keywords), f"Mismatch in number of papers ({len(papers)}) and keywords ({len(keywords)})"
    
    print("Encoding papers...")
    paper_embeddings = model.encode(papers, convert_to_numpy=True)
    
    print("Encoding keywords...")
    keyword_embeddings = model.encode(keywords, convert_to_numpy=True)
    
    # Verify embedding dimensions match
    assert paper_embeddings.shape == keyword_embeddings.shape, \
        f"Embedding dimensions mismatch: papers {paper_embeddings.shape} vs keywords {keyword_embeddings.shape}"
    
    # Normalize the weights so they sum to 1
    total_weight = content_weight + keyword_weight
    content_weight = content_weight / total_weight
    keyword_weight = keyword_weight / total_weight
    
    print(f"Combining embeddings with weights - Content: {content_weight:.2f}, Keywords: {keyword_weight:.2f}")
    
    # Combine embeddings with weights
    combined_embeddings = (content_weight * paper_embeddings + 
                         keyword_weight * keyword_embeddings)
    
    return combined_embeddings

def main():
    input_file = "pubmed_new_sample_500.txt"
    output_base = "pubmed_new_sample_500"
    
    # Extract keywords and content
    print("Extracting keywords and content...")
    papers, keywords = extract_keywords_and_content(input_file)
    print(f"Processed {len(papers)} papers")
    
    # Print first few pairs to verify correspondence
    print("\nVerifying content-keyword correspondence (first 2 pairs):")
    for i in range(min(2, len(papers))):
        print(f"\nPair {i+1}:")
        print(f"Content: {papers[i][:100]}...")
        print(f"Keywords: {keywords[i]}")
    
    # Save separated files
    print("\nSaving separated files...")
    save_separated_files(papers, keywords, output_base)
    
    # Initialize model
    print("Initializing model...")
    model = init_embedding_model()
    
    # Encode with weights and combine
    print("Encoding and combining with weights...")
    combined_embeddings = encode_with_weights(model, papers, keywords)
    
    # Save the combined embeddings
    output_file = f"{output_base}_weighted.npy"
    np.save(output_file, combined_embeddings)
    print(f"Saved weighted embeddings to {output_file}")
    
    # Also save individual embeddings for reference
    np.save(f"{output_base}_content.npy", model.encode(papers))
    np.save(f"{output_base}_keywords.npy", model.encode(keywords))
    
    print("\nProcessing completed:")
    print(f"- Content shape: {combined_embeddings.shape}")
    print(f"- Number of papers processed: {len(papers)}")
    print(f"- Files saved: {output_base}_content.txt, {output_base}_keywords.txt")
    print(f"- Embeddings saved: {output_base}_weighted.npy")

if __name__ == "__main__":
    main() 