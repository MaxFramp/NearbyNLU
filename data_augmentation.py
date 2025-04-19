import pandas as pd
import numpy as np
from transformers import pipeline
from googletrans import Translator
import random
from tqdm import tqdm
import os

# Initialize the paraphraser and translator
paraphraser = pipeline("text2text-generation", model="humarin/chatgpt_paraphraser_on_T5_base")
translator = Translator()

def paraphrase_text(text, num_paraphrases=2):
    """Generate paraphrases of the input text using T5 model."""
    paraphrases = []
    for _ in range(num_paraphrases):
        result = paraphraser(text, max_length=len(text) + 20, do_sample=True)[0]['generated_text']
        paraphrases.append(result)
    return paraphrases

def back_translate(text, num_translations=2):
    """Generate variations through back-translation."""
    variations = []
    languages = ['fr', 'de', 'es', 'it', 'nl']  # Intermediate languages
    
    for _ in range(num_translations):
        # Randomly select an intermediate language
        intermediate_lang = random.choice(languages)
        
        # Translate to intermediate language
        intermediate = translator.translate(text, dest=intermediate_lang).text
        
        # Translate back to English
        back_translated = translator.translate(intermediate, dest='en').text
        
        variations.append(back_translated)
    
    return variations

def augment_dataset(input_file, output_file, num_paraphrases=2, num_translations=2):
    """Augment the dataset using paraphrasing and back-translation."""
    # Read the original dataset
    df = pd.read_csv(input_file)
    
    # Create lists to store augmented data
    augmented_queries = []
    augmented_intents = []
    
    # Progress bar
    pbar = tqdm(total=len(df), desc="Augmenting dataset")
    
    # Process each row
    for _, row in df.iterrows():
        query = row['query']
        intent = row['intent']
        
        # Generate paraphrases
        paraphrases = paraphrase_text(query, num_paraphrases)
        
        # Generate back-translations
        translations = back_translate(query, num_translations)
        
        # Add original data
        augmented_queries.append(query)
        augmented_intents.append(intent)
        
        # Add paraphrases
        for paraphrase in paraphrases:
            augmented_queries.append(paraphrase)
            augmented_intents.append(intent)
        
        # Add back-translations
        for translation in translations:
            augmented_queries.append(translation)
            augmented_intents.append(intent)
        
        pbar.update(1)
    
    pbar.close()
    
    # Create new DataFrame with augmented data
    augmented_df = pd.DataFrame({
        'query': augmented_queries,
        'intent': augmented_intents
    })
    
    # Save augmented dataset
    augmented_df.to_csv(output_file, index=False)
    print(f"\nOriginal dataset size: {len(df)}")
    print(f"Augmented dataset size: {len(augmented_df)}")
    print(f"Augmented dataset saved to: {output_file}")

if __name__ == "__main__":
    input_file = "data/synthetic_gmaps_queries_hierarchical.csv"
    output_file = "data/synthetic_gmaps_queries_hierarchical_augmented.csv"
    
    augment_dataset(input_file, output_file) 