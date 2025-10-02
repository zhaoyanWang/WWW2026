from openai import OpenAI
import numpy as np
import faiss
import torch
import os
from sklearn.metrics.pairwise import cosine_similarity
import random
from sentence_transformers import SentenceTransformer
import time
import re

# API key should be handled more securely in a production environment
client = OpenAI(
    api_key="sk-3eVc2TVDNenkKyQUF9lUXan3Ec6a5bQeQZ8WAZYxSQV1m54K",
    base_url="https://api.chatanywhere.org/v1"
)

# PubMed Categories - diabetes-related research categories
categories = [
    "Diabetes Mellitus, Experimental",
    "Diabetes Mellitus Type 1", 
    "Diabetes Mellitus Type 2"
]

# Initialize the sentence transformer model for embeddings
embedding_model = None

def init_embedding_model():
    global embedding_model
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_big")
    embedding_model = SentenceTransformer(model_path)
    print(f"Embedding model loaded from: {model_path}")
    return embedding_model

# Load PubMed dataset embeddings
def load_data():
    base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset", "pubmed_process")
    pubmed_embeds = np.load(os.path.join(base_path, "pubmed_text_embed.npy"))
    
    # Load label embeddings if available
    label_embeds_file = os.path.join(base_path, "label_text.npy")
    if os.path.exists(label_embeds_file):
        label_embeds = np.load(label_embeds_file)
    else:
        # Create label embeddings from label text if not available
        with open(os.path.join(base_path, "label_text.txt"), "r", encoding="utf-8") as f:
            label_texts = [line.strip() for line in f.readlines() if line.strip()]
        
        # Initialize embedding model if not done
        if embedding_model is None:
            init_embedding_model()
        
        # Create embeddings for each label
        label_embeds = []
        for label_text in label_texts:
            embed = embedding_model.encode(label_text, convert_to_numpy=True)
            label_embeds.append(embed)
        label_embeds = np.array(label_embeds, dtype=np.float32)
        
        # Save for future use
        np.save(label_embeds_file, label_embeds)
        print(f"Created and saved label embeddings to: {label_embeds_file}")
    
    # Load text content
    print(f"Loading raw text data from: {os.path.join(base_path, 'raw_text.txt')}")
    with open(os.path.join(base_path, "raw_text.txt"), "r", encoding="utf-8", errors="replace") as f:
        raw_texts = f.readlines()
    
    labels_file = os.path.join(base_path, "paper_labels.npy")
    
    try:
        # Try to load existing labels file
        paper_labels = np.load(labels_file)
        print(f"Loaded existing paper labels from: {labels_file}")
    except FileNotFoundError:
        print(f"Paper labels file not found, creating from original PubMed dataset...")
        
        pubmed_orig_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset", "PubMed_orig", "data")
        
        # Read node data from PubMed dataset
        node_file_path = os.path.join(pubmed_orig_path, "Pubmed-Diabetes.NODE.paper.tab")
        
        paper_labels = []
        paper_to_index = {}
        
        with open(node_file_path, 'r') as node_file:
            node_file.readline()  # Skip header
            node_file.readline()  # Skip second line
            
            for i, line in enumerate(node_file.readlines()):
                items = line.strip().split('\t')
                paper_id = items[0]
                paper_to_index[paper_id] = i
                
                # Extract label (convert from 1-3 to 0-2)
                label = int(items[1].split('=')[-1]) - 1
                paper_labels.append(label)
        
        paper_labels = np.array(paper_labels)
        
        np.save(labels_file, paper_labels)
        print(f"Created and saved paper labels to: {labels_file}, {paper_labels.shape}")
        
    
    print(f"Data loaded: {len(pubmed_embeds)} embeddings, {len(label_embeds)} labels, {len(raw_texts)} text documents")
    return pubmed_embeds, label_embeds, raw_texts, paper_labels

# # Initialize FAISS index
# def init_faiss_index(embeddings):
#     # Check if GPU is available
#     use_gpu = torch.cuda.is_available()
#     print(f"GPU available for FAISS: {use_gpu}")
    
#     # Get dimensions
#     d = embeddings.shape[1]
    
#     # Create index
#     index = faiss.IndexFlatIP(d)  # Inner product for cosine similarity on normalized vectors
#     print(index)
    
#     # Normalize embeddings for cosine similarity
#     embeddings_normalized = embeddings.copy()  # Make a copy to avoid modifying original
#     faiss.normalize_L2(embeddings_normalized)
    
#     # Use GPU if available
#     if use_gpu:
#         res = faiss.StandardGpuResources()
#         index = faiss.index_cpu_to_gpu(res, 0, index)
    
#     # Add vectors to index
#     index.add(embeddings_normalized)
#     print(f"FAISS index created with {index.ntotal} vectors of dimension {d}")
    
#     return index, use_gpu

# # Retrieve similar documents
# def retrieve_documents(index, query_embed, k=5):
#     # Normalize query embedding
#     query_embed_normalized = query_embed.copy()  # Make a copy to avoid modifying original
#     faiss.normalize_L2(query_embed_normalized)
    
#     # Search
#     D, I = index.search(query_embed_normalized, k)
    
#     return D, I

# Generate embedding for text using our local model
def get_embedding(text):
    global embedding_model
    if embedding_model is None:
        init_embedding_model()
    
    try:
        # Get embedding from the sentence transformer model
        embedding = embedding_model.encode(text, convert_to_numpy=True)
        embedding = embedding.astype(np.float32)
        return embedding
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return None

# Generate prompt with RAG - customized for PubMed diabetes research
def generate_rag_prompt(category, similar_docs_text):
    prompt = f"""Please generate a medical research paper belonging to the category [{category}], including a title, an abstract, and keywords. 

STEP 1: ANALYZE the example papers below to identify:
- The most common medical terms and research methodologies specific to [{category}]
- The typical clinical research problems addressed in [{category}] papers
- The distinctive medical approaches that make [{category}] different from other diabetes categories
- The key concepts, clinical methods, biomarkers, and therapeutic approaches used in this field

STEP 2: GENERATE a paper that:
- MUST use at least 15-20 of the key medical/clinical terms from [{category}] papers
- MUST address a research problem commonly found in [{category}] papers
- MUST employ clinical/medical methodologies characteristic of [{category}]
- MUST NOT drift into approaches typical of other diabetes categories

Example papers:
{similar_docs_text}

Ensure that your paper:
1. Uses the EXACT medical terminology from the example [{category}] papers
2. Addresses similar clinical research questions and problems to these [{category}] papers
3. Is highly specific to [{category}] and avoids generic approaches that could apply to any diabetes research
4. Maintains the same clinical/medical depth while offering original contributions
5. Strictly follows the format below:
Title: <Title>  Abstract: <Abstract>  Keywords: <Keywords>

Notes:
- For Keywords: Include 15-20 key medical/clinical terms that CLEARLY DEMONSTRATE this paper belongs to the [{category}] category, organized in the following categories:
  * Clinical Methodologies (5-7 terms)
  * Medical/Therapeutic Approaches (5-7 terms)
  * Biomarkers/Clinical Indicators (3-4 terms)
  * Research/Study Types (2-3 terms)
- Choose terms that are distinctive and characteristic of this specific diabetes category
- These keywords should help distinguish this paper from other diabetes research categories
- Your paper will be analyzed for similarity with [{category}] papers. Papers that don't sufficiently match the medical terminology, methodology, and clinical focus of this category will be rejected.

Important format instructions:
- Do not use any markdown formatting (no **, _, #, etc.)
- Use plain text only, with "Title:", "Abstract:", and "Keywords:" prefixes 
- The title, abstract, and keywords must each be on a single line
- There should be two spaces before "Abstract:" and before "Keywords:"
- Do not include any other elements such as sections or references

Correct format example:
Title: Sample Medical Paper Title  Abstract: This is the medical research abstract text.  Keywords: clinical_method1, therapeutic_approach1, biomarker1, study_type1, clinical_method2, therapeutic_approach2, biomarker2, clinical_method3, therapeutic_approach3, biomarker3, study_type2, clinical_method4, therapeutic_approach4, biomarker4, study_type3, clinical_method5, therapeutic_approach5, clinical_method6, therapeutic_approach6, clinical_method7
"""
    return prompt

# Use GPT API with stream
def gpt_api_stream(prompt):
    """Use streaming API with the enhanced RAG prompt"""
    messages = [
        {
            "role": "user",
            "content": prompt
        }
    ]
    try:
        # Add retry mechanism for API calls
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Non-streaming version for stability
                response = client.chat.completions.create(
                    model='gpt-4o-mini',
                    messages=messages,
                    max_tokens=1000
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"API call failed (attempt {attempt+1}/{max_retries}): {e}")
                    time.sleep(2)  # Wait before retry
                else:
                    raise
        
    except Exception as e:
        print(f"Error generating response after {max_retries} attempts: {e}")
        return None

# Generate feedback based on similarity analysis
def generate_feedback(current_embed, pubmed_embeds, raw_texts, category_indices, other_indices, 
                     generated_embeddings=None, generated_papers=None):
    # Compute cosine similarities with all papers
    similarities = cosine_similarity(current_embed.reshape(1, -1), pubmed_embeds).flatten()
    
    # Get similar papers from same category
    same_cat_similarities = [(i, similarities[i]) for i in category_indices]
    same_cat_similarities.sort(key=lambda x: x[1], reverse=True)
    
    all_categories = list(range(len(categories)))
    
    # Find current category index
    current_category_idx = None
    for cat_idx, indices in category_to_papers.items():
        if any(idx in category_indices for idx in indices):
            current_category_idx = cat_idx
            break
    # Output current category name
    if current_category_idx is not None:
        print(f"Current category identified as: {categories[current_category_idx]}")
    else:
        print("WARNING: Could not identify current category")
    
    other_cat_similarities_by_category = {}
    for cat_idx in all_categories:
        if cat_idx == current_category_idx:
            continue
        
        cat_papers = category_to_papers[cat_idx]
        if not cat_papers:
            continue
            
        cat_similarities = [(i, similarities[i]) for i in cat_papers]
        cat_similarities.sort(key=lambda x: x[1], reverse=True)
        
        other_cat_similarities_by_category[cat_idx] = cat_similarities[:5]
        print(f"Found {len(cat_similarities[:5])} most similar papers for category {categories[cat_idx]}")
    
    other_cat_similarities = [(i, similarities[i]) for i in other_indices]
    other_cat_similarities.sort(key=lambda x: x[1], reverse=True)
    top_other_cat = other_cat_similarities[:10]
    
    # Create feedback
    feedback = "Based on similarity analysis with existing papers:\n\n"
    
    # Feedback for same category papers
    feedback += "Similar papers in the same category:\n"
    for i, sim in same_cat_similarities[:5]:  # Show top 5 in summary, but use all 10 for warnings
        feedback += f"- Similarity: {sim:.4f} - {raw_texts[i]}\n"
    
    # Warning for too high similarity within same category
    if any(sim > 0.85 for _, sim in same_cat_similarities[:10]):  # Threshold for concern within same category
        feedback += "\nWARNING: Your paper is too similar to existing papers in the SAME category. Need more distinctiveness:\n"
        for i, sim in same_cat_similarities[:10]:
            if sim > 0.85:  # Higher threshold for same category
                feedback += f"- Similarity: {sim:.4f} - {raw_texts[i]}\n"
    
    # Check if similarity is too low with papers in the same category
    avg_similarity_same_cat = sum(sim for _, sim in same_cat_similarities[:5]) / 5
    if avg_similarity_same_cat < 0.6:  # Threshold for concern about low similarity
        feedback += "\nWARNING: Your paper has very low similarity to papers in the SAME category. It may not align properly with this category:\n"
        feedback += f"- Average similarity with top 5 papers: {avg_similarity_same_cat:.4f}\n"
        feedback += "- Your paper needs to be more closely aligned with the typical medical themes, clinical methodologies, and terminology of this category.\n"
        feedback += "Here are representative papers from this category that you should more closely align with:\n"
        for i, sim in same_cat_similarities[:5]:
            feedback += f"- Example paper: {raw_texts[i]}\n"
    
    # Feedback for other category papers (if too similar)
    if any(sim > 0.3 for _, sim in top_other_cat):  # Threshold for concern with other categories
        feedback += "\nWARNING: Your paper is too similar to papers in OTHER diabetes categories:\n"
        
        for cat_idx, cat_similarities in other_cat_similarities_by_category.items():
            if any(sim > 0.3 for _, sim in cat_similarities):
                feedback += f"\nSimilar papers in {categories[cat_idx]} category:\n"
                for i, sim in cat_similarities:
                    if sim > 0.3:
                        feedback += f"- Similarity: {sim:.4f} - {raw_texts[i]}\n"
    
    # Compare with previously generated papers
    if generated_embeddings is not None and len(generated_embeddings) > 0:
        # Calculate similarities with all previously generated papers
        prev_similarities = cosine_similarity(
            current_embed.reshape(1, -1), 
            np.array(generated_embeddings)
        ).flatten()
        
        # Create (index, similarity) pairs for all previously generated papers
        prev_similarities_pairs = [(j, sim) for j, sim in enumerate(prev_similarities)]
        
        # Sort by similarity (highest first)
        prev_similarities_pairs.sort(key=lambda x: x[1], reverse=True)
        
        # Select top 10 most similar previously generated papers
        top_prev = prev_similarities_pairs[:10]
        
        # If there are previously generated papers with high similarity
        if any(sim > 0.7 for _, sim in top_prev):
            feedback += "\nWARNING: Your paper is too similar to previously GENERATED papers:\n"
            for j, sim in top_prev:
                if sim > 0.7:  # Only show papers with similarity > 0.7
                    prev_paper = generated_papers[j]
                    # Show the full paper instead of just the title or a truncated version
                    feedback += f"- Similarity: {sim:.4f} - {prev_paper}\n"
    
    return feedback

# Add a function to save feedback to file
def save_feedback(feedback, index, category, output_dir="feedback"):
    """Save feedback to a text file for analysis"""
    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create filename with index and category
    filename = f"{output_dir}/feedback_{index+1:03d}_{category.replace(' ', '_').replace(',', '')}.txt"
    
    # Write feedback to file
    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"Feedback for paper #{index+1}, Category: {category}\n")
        f.write("="*80 + "\n\n")
        f.write(feedback)
    
    print(f"Saved feedback to {filename}")

# Add a function to save the improvement prompt
def save_improvement_prompt(prompt, index, category, output_dir="improvement_prompts"):
    """Save the complete improvement prompt to a text file for analysis"""
    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create filename with index and category
    filename = f"{output_dir}/improvement_prompt_{index+1:03d}_{category.replace(' ', '_').replace(',', '')}.txt"
    
    # Write prompt to file
    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"Improvement Prompt for paper #{index+1}, Category: {category}\n")
        f.write("="*80 + "\n\n")
        f.write(prompt)
    
    print(f"Saved improvement prompt to {filename}")

# Improve paper generation with feedback - modified to focus on specific improvements
def improve_paper_with_feedback(generated_paper, feedback, category, index=None):
    # Check what types of similarity issues exist
    same_cat_warning = "too similar to existing papers in the SAME category" in feedback
    other_cat_warning = "too similar to papers in OTHER" in feedback
    prev_gen_warning = "too similar to previously GENERATED papers" in feedback
    too_low_similarity_warning = "Your paper has very low similarity to papers in the SAME category" in feedback
    
    # Create more targeted instruction based on the type of issue
    targeted_instruction = ""
    if same_cat_warning:
        targeted_instruction += """For papers in the SAME diabetes category that are too similar:
- Keep the same general clinical approach but ADD NOVEL ELEMENTS
- Consider exploring related sub-areas or specialized clinical applications within this diabetes category
- Introduce innovative therapeutic variations or novel biomarker studies
- Focus on under-explored clinical aspects or emerging therapeutic directions in this field
- The paper should remain in the same diabetes category but be DISTINCT from existing work
"""

    if other_cat_warning:
        targeted_instruction += """For papers in OTHER diabetes categories that are too similar (SERIOUS ISSUE):
- Make SUBSTANTIAL CHANGES to ensure your paper clearly belongs to the correct diabetes category
- Completely rethink the clinical methodology, patient population, or therapeutic approach
- Eliminate terminology, methods, or concepts that are strongly associated with other diabetes categories
- Restructure the core clinical contribution to align with the target diabetes category's paradigms
- This requires major revision to align properly with the intended diabetes category
"""

    if prev_gen_warning:
        targeted_instruction += """For previously GENERATED papers that are too similar:
- Ensure your clinical contribution is unique compared to already generated papers
- Avoid repeating similar experimental setups, patient cohorts, or clinical conclusions
- Modify the specific clinical focus, therapeutic domain, or research approach
- Create a paper that explores a different clinical aspect of the diabetes field
"""

    if too_low_similarity_warning:
        targeted_instruction += """For papers with LOW SIMILARITY to the target diabetes category (SERIOUS ISSUE):
- Your paper is NOT SUFFICIENTLY ALIGNED with the clinical themes and approaches of this diabetes category
- EXTENSIVELY INCORPORATE medical terminology, concepts, and methodologies that are characteristic of this diabetes category
- Carefully FOLLOW THE CLINICAL EXAMPLES provided from this diabetes category
- Use similar clinical research problems, therapeutic approaches, and domain-specific medical language as the examples
- Focus on the CORE CLINICAL THEMES that appear frequently in the example papers
- The paper must remain unique, but should CLEARLY BELONG to this diabetes category in terms of its clinical approach and focus
"""

    prompt = f"""You have generated the following medical research paper:

{generated_paper}

However, there are concerns about its alignment with the target diabetes category. Here is the detailed similarity analysis:

{feedback}

Please REVISE this medical paper according to these SPECIFIC guidelines:

{targeted_instruction}

General requirements:
1. Maintain clear alignment with the [{category}] diabetes category
2. Use the examples of similar clinical papers as a guide for your revision
3. Keep the format: Title: <Title of the paper>  Abstract: <Abstract of the paper>  Keywords: <15-20 key medical/clinical terms>

For the Keywords section:
- Extract 15-20 key medical/clinical terms from your revised abstract that CLEARLY DEMONSTRATE this paper belongs to the [{category}] category
- Choose terms that are distinctive and characteristic of this specific diabetes category
- These keywords should help distinguish this paper from other diabetes categories
- Place the keywords immediately after the abstract, separated by two spaces

IMPORTANT FORMAT REQUIREMENTS:
- Do NOT use any markdown formatting (no **, no _, no #)
- Use plain text only with "Title:", "Abstract:", and "Keywords:" prefixes
- The title, abstract, and keywords MUST each be on a single line
- The words "Abstract:" and "Keywords:" should each have two spaces before them
- DO NOT include any other elements like sections or references

Your revised paper should be recognizably in the [{category}] category but with its own unique clinical identity and contribution."""

    # Save the complete improvement prompt if index is provided
    if index is not None:
        save_improvement_prompt(prompt, index, category)

    try:
        # Add retry mechanism
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model='gpt-4o-mini',
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=1000
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"API call failed (attempt {attempt+1}/{max_retries}): {e}")
                    time.sleep(2)  # Wait before retry
                else:
                    raise
    except Exception as e:
        print(f"Error improving paper after {max_retries} attempts: {e}")
        return generated_paper  # Return original if improvement fails

# Save generated embeddings for future use
def save_generated_papers(generated_papers, embeddings, output_file_base):
    # Save generated papers as text
    with open(f"{output_file_base}.txt", "w", encoding="utf-8") as f:
        for paper in generated_papers:
            # Apply complete cleaning process to ensure consistent formatting
            clean_paper = paper
            
            # Remove all Markdown formatting and other patterns
            clean_paper = clean_paper.replace("**Title:**", "Title:").replace("**Abstract:**", "Abstract:")
            clean_paper = clean_paper.replace("**Title**:", "Title:").replace("**Abstract**:", "Abstract:")
            clean_paper = clean_paper.replace("**Keywords:**", "Keywords:").replace("**Keywords**:", "Keywords:")
            clean_paper = clean_paper.replace("**", "").replace("*", "").replace("__", "").replace("_", "")
            clean_paper = clean_paper.replace("# ", "").replace("## ", "").replace("### ", "")
            
            # Remove any sections that might be included, but keep Keywords
            if "Introduction" in clean_paper:
                clean_paper = clean_paper.split("Introduction")[0]
            
            # Convert to single line format
            clean_paper = clean_paper.replace("\n", " ").strip()
            
            # Ensure proper spacing
            if "Title:" in clean_paper and "Abstract:" in clean_paper:
                # Extract title, abstract, and keywords if present
                title_part = ""
                abstract_part = ""
                keywords_part = ""
                
                if "Keywords:" in clean_paper:
                    # Format with keywords
                    parts = clean_paper.split("Keywords:")
                    abstract_and_title = parts[0]
                    keywords_part = parts[1].strip()
                    
                    if "Abstract:" in abstract_and_title:
                        title_part = abstract_and_title.split("Abstract:")[0].replace("Title:", "").strip()
                        abstract_part = abstract_and_title.split("Abstract:")[1].strip()
                        
                        # Reformat to match exact format with keywords
                        clean_paper = f"Title: {title_part}  Abstract: {abstract_part}  Keywords: {keywords_part}"
                else:
                    # Format without keywords
                    title_part = clean_paper.split("Abstract:")[0].replace("Title:", "").strip()
                    abstract_part = clean_paper.split("Abstract:")[1].strip()
                    
                    # Reformat to match exact format
                    clean_paper = f"Title: {title_part}  Abstract: {abstract_part}"
            else:
                # Try to extract using regex as a fallback
                title_match = re.search(r'(?:^|\s)(?:title:?\s*)([^.]+)', clean_paper, re.IGNORECASE)
                abstract_match = re.search(r'(?:abstract:?\s*)([^K]+)(?:Keywords:)?', clean_paper, re.IGNORECASE)
                keywords_match = re.search(r'(?:keywords:?\s*)(.+)$', clean_paper, re.IGNORECASE)
                
                if title_match and abstract_match:
                    title_part = title_match.group(1).strip()
                    abstract_part = abstract_match.group(1).strip()
                    
                    if keywords_match:
                        keywords_part = keywords_match.group(1).strip()
                        clean_paper = f"Title: {title_part}  Abstract: {abstract_part}  Keywords: {keywords_part}"
                    else:
                        clean_paper = f"Title: {title_part}  Abstract: {abstract_part}"
            
            f.write(f"{clean_paper}\n")
    
    # Save embeddings as numpy array
    embeddings_array = np.array(embeddings, dtype=np.float32)
    np.save(f"{output_file_base}.npy", embeddings_array)
    
    print(f"Saved {len(generated_papers)} generated papers and embeddings to {output_file_base}")

if __name__ == '__main__':
    total_calls = 93  # Adjusted for 3 categories: 31 papers per category for balanced generation
    category_count = len(categories)
    output_file = "pubmed_new_sample"
    save_checkpoints = False
    checkpoint_interval = 30
    
    # Load data
    print("Loading PubMed dataset...")
    pubmed_embeds, label_embeds, raw_texts, paper_labels = load_data()
    
    # Initialize embedding model
    init_embedding_model()
    
    # print("Initializing FAISS index...")
    # index, use_gpu = init_faiss_index(pubmed_embeds.copy())  # Copy to avoid modifying original
    
    # Get category indices for each paper in the dataset
    print("Organizing papers by diabetes category...")
    # Mapping from category indices to paper indices
    category_to_papers = {}
    
    total_papers = len(pubmed_embeds)
    for i, category in enumerate(categories):
        category_to_papers[i] = [j for j in range(total_papers) if paper_labels[j] == i]
        print(f"Category '{category}' has {len(category_to_papers[i])} papers")
    
    # Lists to store generated papers and their embeddings
    generated_papers = []
    generated_embeddings = []
    
    try:
        with open(f"{output_file}.txt", "w", encoding="utf-8") as file:
            for i in range(total_calls):
                # Select current category
                current_category = categories[i % category_count]
                category_index = categories.index(current_category)
                
                print(f"\n[{i+1}/{total_calls}] Generating paper for diabetes category: {current_category}")
                
                try:
                    # Get similar papers from the same category using the correct category mapping
                    category_indices = category_to_papers[category_index]
                    
                    # Prepare similar documents text for RAG by selecting random examples from the correct category
                    # This ensures we only use papers that are actually in the target category
                    sample_size = min(10, len(category_indices))
                    selected_indices = random.sample(category_indices, sample_size)
                    
                    similar_docs_text = ""
                    for idx in selected_indices:
                        if idx < len(raw_texts):
                            similar_docs_text += f"{raw_texts[idx]}\n\n"
                    
                    # Generate paper with RAG assistance
                    rag_prompt = generate_rag_prompt(current_category, similar_docs_text)
                    generated_paper = gpt_api_stream(rag_prompt)
                    
                    if generated_paper:
                        print("Medical paper generated, analyzing similarity...")
                        
                        # Get embedding for generated paper using local model
                        current_embed = get_embedding(generated_paper)
                        
                        if current_embed is None:
                            print("Failed to get embedding for generated paper, using fallback...")
                            # Fallback to random sampling if model fails
                            current_embed = pubmed_embeds[random.choice(category_indices)].reshape(1, -1)
                        
                        # Get indices of papers in the same and other categories
                        other_category_indices = []
                        for other_cat_idx in range(len(categories)):
                            if other_cat_idx != category_index:
                                other_category_indices.extend(category_to_papers[other_cat_idx])
                        
                        # Use ALL papers from other categories (no random sampling)
                        other_indices = other_category_indices
                        print(f"Comparing with {len(other_indices)} papers from other diabetes categories")
                        
                        # Generate feedback
                        feedback = generate_feedback(
                            current_embed, 
                            pubmed_embeds, 
                            raw_texts, 
                            category_indices, 
                            other_indices,
                            generated_embeddings,  # Pass previously generated embeddings
                            generated_papers       # Pass previously generated papers
                        )
                        
                        # Save feedback to file for analysis
                        save_feedback(feedback, i, current_category)
                        
                        # Determine the types of similarity issues
                        same_cat_warning = "too similar to existing papers in the SAME category" in feedback
                        other_cat_warning = "too similar to papers in OTHER" in feedback
                        prev_gen_warning = "too similar to previously GENERATED papers" in feedback
                        too_low_similarity_warning = "Your paper has very low similarity to papers in the SAME category" in feedback
                        needs_improvement = same_cat_warning or other_cat_warning or prev_gen_warning or too_low_similarity_warning
                        
                        # Improve paper with feedback if needed
                        improved = False
                        if needs_improvement:
                            # Create descriptive message about the issues
                            issue_types = []
                            if same_cat_warning:
                                issue_types.append("same category similarity")
                            if other_cat_warning:
                                issue_types.append("other category similarity")
                            if prev_gen_warning:
                                issue_types.append("similarity to previously generated papers")
                            if too_low_similarity_warning:
                                issue_types.append("low similarity to target category")
                            
                            issue_description = ", ".join(issue_types)
                            print(f"Improving paper with feedback (issues: {issue_description})...")
                            
                            # Only do one improvement round with more targeted guidance
                            improved_paper = improve_paper_with_feedback(generated_paper, feedback, current_category, index=i)
                            if improved_paper:
                                generated_paper = improved_paper
                                improved = True
                                
                                # Re-analyze the improved paper to log whether issues were resolved
                                improved_embed = get_embedding(improved_paper)
                                if improved_embed is not None:
                                    current_embed = improved_embed  # Update the embedding
                                    improved_feedback = generate_feedback(
                                        improved_embed, 
                                        pubmed_embeds, 
                                        raw_texts, 
                                        category_indices, 
                                        other_indices,
                                        generated_embeddings,
                                        generated_papers
                                    )
                                    
                                    # Save improved feedback to file for analysis
                                    save_feedback(improved_feedback, i, current_category, output_dir="feedback_improved")
                                    
                                    # Check if issues have been resolved
                                    same_cat_warning_after = "too similar to existing papers in the SAME category" in improved_feedback
                                    other_cat_warning_after = "too similar to papers in OTHER" in improved_feedback
                                    prev_gen_warning_after = "too similar to previously GENERATED papers" in improved_feedback
                                    too_low_similarity_warning_after = "Your paper has very low similarity to papers in the SAME category" in improved_feedback
                                    
                                    # Create status message
                                    unresolved_issues = []
                                    if same_cat_warning_after:
                                        unresolved_issues.append("same category similarity")
                                    if other_cat_warning_after:
                                        unresolved_issues.append("other category similarity")
                                    if prev_gen_warning_after:
                                        unresolved_issues.append("similarity to previously generated papers")
                                    if too_low_similarity_warning_after:
                                        unresolved_issues.append("low similarity to target category")
                                    
                                    if unresolved_issues:
                                        improvement_status = f"PARTIALLY SUCCESSFUL - unresolved issues: {', '.join(unresolved_issues)}"
                                    else:
                                        improvement_status = "SUCCESSFUL"
                                    
                                    print(f"Improvement analysis: {improvement_status}")
                            else:
                                print("Improvement failed")
                        
                        # Store the generated paper and its embedding
                        generated_papers.append(generated_paper)
                        generated_embeddings.append(current_embed)
                        
                        # Write to file with cleaning process
                        clean_paper = generated_paper
                        
                        # Remove all Markdown formatting and other patterns
                        clean_paper = clean_paper.replace("**Title:**", "Title:").replace("**Abstract:**", "Abstract:")
                        clean_paper = clean_paper.replace("**Title**:", "Title:").replace("**Abstract**:", "Abstract:")
                        clean_paper = clean_paper.replace("**Keywords:**", "Keywords:").replace("**Keywords**:", "Keywords:")
                        clean_paper = clean_paper.replace("**", "").replace("*", "").replace("__", "").replace("_", "")
                        clean_paper = clean_paper.replace("# ", "").replace("## ", "").replace("### ", "")
                        
                        # Remove any sections that might be included, but keep Keywords
                        if "Introduction" in clean_paper:
                            clean_paper = clean_paper.split("Introduction")[0]
                        
                        # Convert to single line format
                        clean_paper = clean_paper.replace("\n", " ").strip()
                        
                        # Ensure proper spacing and format
                        if "Title:" in clean_paper and "Abstract:" in clean_paper:
                            # Extract title, abstract, and keywords if present
                            title_part = ""
                            abstract_part = ""
                            keywords_part = ""
                            
                            if "Keywords:" in clean_paper:
                                # Format with keywords
                                parts = clean_paper.split("Keywords:")
                                abstract_and_title = parts[0]
                                keywords_part = parts[1].strip()
                                
                                if "Abstract:" in abstract_and_title:
                                    title_part = abstract_and_title.split("Abstract:")[0].replace("Title:", "").strip()
                                    abstract_part = abstract_and_title.split("Abstract:")[1].strip()
                                    
                                    # Reformat to match exact format with keywords
                                    clean_paper = f"Title: {title_part}  Abstract: {abstract_part}  Keywords: {keywords_part}"
                            else:
                                # Format without keywords
                                title_part = clean_paper.split("Abstract:")[0].replace("Title:", "").strip()
                                abstract_part = clean_paper.split("Abstract:")[1].strip()
                                
                                # Reformat to match exact format
                                clean_paper = f"Title: {title_part}  Abstract: {abstract_part}"
                        else:
                            # Try to extract using regex as a fallback
                            title_match = re.search(r'(?:^|\s)(?:title:?\s*)([^.]+)', clean_paper, re.IGNORECASE)
                            abstract_match = re.search(r'(?:abstract:?\s*)([^K]+)(?:Keywords:)?', clean_paper, re.IGNORECASE)
                            keywords_match = re.search(r'(?:keywords:?\s*)(.+)$', clean_paper, re.IGNORECASE)
                            
                            if title_match and abstract_match:
                                title_part = title_match.group(1).strip()
                                abstract_part = abstract_match.group(1).strip()
                                
                                if keywords_match:
                                    keywords_part = keywords_match.group(1).strip()
                                    clean_paper = f"Title: {title_part}  Abstract: {abstract_part}  Keywords: {keywords_part}"
                                else:
                                    clean_paper = f"Title: {title_part}  Abstract: {abstract_part}"
                        
                        file.write(f"{clean_paper}\n")
                        file.flush()
                        
                        # For console output
                        if "Title:" in clean_paper and "Abstract:" in clean_paper:
                            title_part = clean_paper.split("Abstract:")[0].replace("Title:", "").strip()
                            print(f"Saved {'improved ' if improved else ''}medical paper for diabetes category {current_category}")
                            print(f"Title: {title_part}")
                        else:
                            print(f"Saved paper (format may be irregular) for diabetes category {current_category}")
                    else:
                        print(f"Failed to generate response for diabetes category {current_category}.")
                except Exception as e:
                    print(f"Error processing paper {i+1}: {e}")
                    # Continue with next paper
                    continue
                
                # Periodically save progress
                if save_checkpoints and i > 0 and i % checkpoint_interval == 0:
                    print(f"\nSaving progress after {i+1} papers...")
                    save_generated_papers(generated_papers, generated_embeddings, f"{output_file}_checkpoint_{i+1}")
        
        # Save all generated papers and embeddings at the end
        save_generated_papers(generated_papers, generated_embeddings, output_file)
        
    except KeyboardInterrupt:
        print("\nProcess interrupted by user. Saving current progress...")
        # Save current progress if interrupted
        save_generated_papers(generated_papers, generated_embeddings, f"{output_file}_interrupted")
    except Exception as e:
        print(f"\nError in main process: {e}")
        # Save progress on error
        if generated_papers:
            save_generated_papers(generated_papers, generated_embeddings, f"{output_file}_error")
