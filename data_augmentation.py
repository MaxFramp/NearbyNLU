import pandas as pd
import numpy as np
from transformers import pipeline
from googletrans import Translator
import random
from tqdm import tqdm
import os
import asyncio
import time
from tenacity import retry, stop_after_attempt, wait_exponential

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

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
async def translate_with_retry(text, dest_lang):
    """Translate text with retry logic and delay."""
    await asyncio.sleep(1)  # Add delay between requests
    return await translator.translate(text, dest=dest_lang)

async def back_translate(text, num_translations=2):
    """Generate variations through back-translation."""
    variations = []
    languages = ['fr', 'de', 'es', 'it', 'nl']  # Intermediate languages
    
    for _ in range(num_translations):
        try:
            # Randomly select an intermediate language
            intermediate_lang = random.choice(languages)
            
            # Translate to intermediate language
            intermediate = await translate_with_retry(text, intermediate_lang)
            
            # Add delay between translations
            # await asyncio.sleep(0.1)
            
            # Translate back to English
            back_translated = await translate_with_retry(intermediate.text, 'en')
            
            variations.append(back_translated.text)
        except Exception as e:
            print(f"Error during back-translation: {str(e)}")
            continue
    
    return variations

def save_progress(queries, intents, output_file):
    """Save current progress to CSV file."""
    df = pd.DataFrame({
        'sentence': queries,
        'label': intents
    })
    df.to_csv(output_file, index=False)
    print(f"\nProgress saved: {len(queries)} items processed")

async def augment_dataset(input_file, output_file, num_paraphrases=2, num_translations=2, save_interval=100):
    """Augment the dataset using paraphrasing and back-translation."""
    # Read the original dataset
    df = pd.read_csv(input_file)
    
    # Create lists to store augmented data
    augmented_queries = []
    augmented_intents = []
    
    # Progress bar
    pbar = tqdm(total=len(df), desc="Augmenting dataset")
    
    # Process each row
    for idx, row in df.iterrows():
        try:
            query = row['sentence']
            intent = row['label']
            
            # Generate paraphrases
            paraphrases = paraphrase_text(query, num_paraphrases)
            
            # Generate back-translations
            translations = await back_translate(query, num_translations)
            
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
            
            # Save progress periodically
            if (idx + 1) % save_interval == 0:
                save_progress(augmented_queries, augmented_intents, output_file)
            
            pbar.update(1)
            
        except Exception as e:
            print(f"\nError processing row {idx}: {str(e)}")
            # Save progress even if there's an error
            save_progress(augmented_queries, augmented_intents, output_file)
            continue
    
    pbar.close()
    
    # Final save
    save_progress(augmented_queries, augmented_intents, output_file)
    print(f"\nOriginal dataset size: {len(df)}")
    print(f"Augmented dataset size: {len(augmented_queries)}")
    print(f"Augmented dataset saved to: {output_file}")

if __name__ == "__main__":
    input_file = "data/full_natural_lifestyle_sentence_dataset.csv"
    output_file = "data/full_natural_lifestyle_sentence_dataset_augmented.csv"
    
    # Run the async function
    asyncio.run(augment_dataset(input_file, output_file)) 