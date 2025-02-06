import numpy as np
import faiss
import json
import subprocess
import argparse
from typing import List, Dict
from dataclasses import dataclass
from pathlib import Path
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor
from koboldapi import KoboldAPICore

@dataclass
class Config:
    """ Configuration for semantic merge operations """
    llama_path: Path
    model_path: Path
    kobold_url: str
    similarity_threshold: float = 0.65
    batch_size: int = 50

class SemanticMerger:
    def __init__(self, config: Config):
        self.config = config
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.core = KoboldAPICore("http://localhost:5001")
        
    async def generate_embeddings(self, tags: List[str]) -> np.ndarray:
        """ Generate embeddings using llama.cpp """
        embeddings_list = []

        for tag in tags:
            # Write batch to temp file
            #with open('input.txt', 'w', encoding='utf-8') as f:
             #   f.read()
            #print(batch)
             
            cmd = [
                str(self.config.llama_path),
                "-m", str(self.config.model_path),
                "--pooling", "mean",
                "--embd-normalize", "2",
                "--embd-output-format", "array",
                "--verbose",
                "-c", "2048",
                "-p", tag
            ]
            
            #print(f"Running command: {' '.join(str(x) for x in cmd)}")
            
            try:
                # Run embedding generation in thread pool to avoid blocking
                result = subprocess.run(
                        cmd, 
                        capture_output=True, 
                        text=True, 
                        check=True
                    )
                
                
                print(f"Raw output: {result.stdout[:200]}...")
                #print(f"Error output: {result.stderr[:200]}")
                
                # Parse the embeddings output - expecting array format
                try:
                    batch_embeddings = json.loads(result.stdout)
                    if isinstance(batch_embeddings, list):
                        embeddings_list.extend(batch_embeddings)
                    else:
                        print(f"Unexpected embedding format, got type: {type(batch_embeddings)}")
                        continue
                except json.JSONDecodeError as e:
                    print(f"JSON parse error: {e}")
                    continue
                    
            except subprocess.CalledProcessError as e:
                print(f"Process error: {e}")
                print(f"Error output: {e.stderr}")
                continue
                
        if not embeddings_list:
            raise RuntimeError("Failed to generate any valid embeddings")
            
        embeddings = np.array(embeddings_list).astype("float32")
        faiss.normalize_L2(embeddings)
        return embeddings

    def build_index(self, embeddings: np.ndarray) -> faiss.IndexFlatL2:
        """ Build FAISS index for similarity search """
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)
        return index

    def get_synonyms(self, tag: str, candidates: List[str]) -> List[str]:
        """ Get validated synonyms using KoboldAPI """
        prompt = """Your task is to identify ONLY exact synonyms and DIRECT PARENT categories for the input word. 
IMPORTANT: Never include specific types/subtypes of the input word!
DECISION RULES:
1. SYNONYMS: Include ONLY if:
   - Words mean EXACTLY the same thing (like "car" = "automobile")
   - Can substitute in ANY context with NO change in meaning
   
2. PARENT CATEGORIES: Include ONLY if:
   - Parent is MORE GENERAL than input word
   - Can say "X is a type of Y" but NOT "Y is a type of X"
   
CRITICAL RELATIONSHIP DIRECTION:
VALID "metal" -> "material" (VALID: metal is a type of material)
INVALID "metal" -> "brass" (INVALID: brass is a type of metal - wrong direction!)
INVALID "metal" -> "bronze" (INVALID: bronze is a type of metal - wrong direction!)

MORE EXAMPLES:
"dog" -> "animal" (VALID: parent category)
"dog" -> "poodle" (INVALID: poodle is a type of dog - wrong direction!)

"vehicle" -> "car" (INVALID: car is a type of vehicle - wrong direction!)
"vehicle" -> "transport" (VALID: parent category)

AUTOMATIC EXCLUSION:
- Any subtypes or specific varieties of the input word
- Related terms that aren't strictly synonyms
- Specific examples of the input category

EXAMPLE OUTPUTS:
Input: "metal"
Candidates: "brass, bronze, gold, material, substance"
Valid output: {"metal": ["material", "substance"]}  # only parent categories, NO subtypes

Input: "dog"
Candidates: "poodle, animal, pet, mammal, canine"
Valid output: {"dog": ["animal", "mammal"]}  # only parent categories

Only use the words provided in the candidates list. Do NOT add any new words.
Reply with EXACTLY: {
"word1": ["syn1", "syn2"], ...
"word2": ["syn3", "syn4"], ...
}"""
 

                    
        return (json.loads(self.core.wrap_and_generate(
            instruction=prompt,
            content=f'\nWord: "{tag}"\nCandidates: {", ".join(candidates)}\n'))).get(tag)
        
    async def process_tags(self, tags: List[str]) -> Dict[str, List[str]]:
        """ Process all tags to generate synonym mapping """
        print("Generating embeddings...")
        embeddings = await self.generate_embeddings(tags)
        
        print("Building search index...")
        index = self.build_index(embeddings)
        
        print("Processing tags...")
        print(f"Tags: {len(tags)}, Embeddings: {len(embeddings)}")
        
        tasks = []
        synonym_mapping = {}
                
        for idx, tag in enumerate(tags):
            query_embedding = embeddings[idx].reshape(1, -1)
            distances, indices = index.search(query_embedding, 11)
            candidates = [
                tags[i] for i, dist in zip(indices[0], distances[0])
                if tags[i] != tag and dist < self.config.similarity_threshold
            ][:10]
            
            if candidates:
                synonyms = self.get_synonyms(tag, candidates)
                if synonyms:
                    synonym_mapping[tag] = synonyms
                        
        return synonym_mapping

def main():
    parser = argparse.ArgumentParser(description='Semantic tag merger')
    parser.add_argument('--llama-path', type=Path, required=True, 
                       help='Path to llama.cpp embedding binary')
    parser.add_argument('--model-path', type=Path, required=True,
                       help='Path to GGUF model file')
    parser.add_argument('--kobold-url', type=str, default='http://localhost:5001',
                       help='KoboldCPP API URL')
    args = parser.parse_args()

    # Example tags - for testing
    tags_str = "car automobile vehicle sedan suv truck motorcycle bicycle bike coupe hatchback van minivan convertible roadster pickup lorry jeep electric vehicle EV hybrid gasoline diesel electric moped scooter skateboard bus train tram subway metro airplane aircraft helicopter boat ship yacht vessel off-road 4x4 4wd awd fwd rwd dune buggy atv go-kart racecar formula 1 rally car # * Colorsred blue green white black silver grey gray yellow purple orange pink brown beige gold cyan magenta lime teal indigo violet maroon navy olive turquoise lavender coral crimson aqua metallic pearlescent iridescent chromatic monochromatic Korean Japanese American German Italian Chinese British French Swedish Spanish Mexican Canadian Australian Indian Russian Brazilian European Asian African Latin American circuit electronics pcb printed circuit board motherboard cpu central processing unit gpu graphics processing unit ram random access memory ssd solid state drive hdd hard disk drive memory storage screen monitor display lcd liquid crystal display led light emitting diode oled organic light emitting diode touchscreen panel touch panel digitizer keyboard mouse trackpad laptop desktop tablet smartphone device gadget gizmo mobile phone cell phone smartwatch wearable headphones earphones earbuds speakers audio video camera lens sensor microchip semiconductor transistor capacitor resistor inductor diode integrated circuit IC SoC system on chip firmware software hardware driver operating system OS android iOS windows macOS linux unix programming coding algorithm data database cloud server cybersecurity artificial intelligence AI machine learning deep learning neural network robotics automation metal plastic glass wood steel stainless steel aluminum copper silicon carbon fiber fiberglass rubber leather fabric textile ceramic composite alloy titanium brass bronze polymer vinyl new used vintage modern retro classic contemporary antique old aged refurbished reconditioned second-hand pre-owned broken damaged faulty defective malfunctioning repaired restored maintained serviced upgraded modified customized personalized stock default original oem aftermarket indoor outdoor inside outside exterior interior street road highway freeway motorway garage workshop laboratory office home house apartment building factory warehouse store shop park nature city urban rural suburban clean dirty dusty filthy spotless pristine grimy soiled rusty corroded oxidized tarnished shiny glossy lustrous polished reflective matte dull flat textured rough smooth sleek soft hard rigid flexible delicate sturdy rugged durable fragile big small medium compact large tiny miniature micro nano huge massive enormous gigantic giant colossal bulky heavy light lightweight portable mini expensive cheap affordable budget luxury premium high-end low-end mid-range value bargain discount sale clearance pricey costly economical inexpensive fast slow quick rapid swift sluggish efficient inefficient powerful weak strong feeble robust sturdy reliable unreliable stable unstable consistent inconsistent durable fragile precise accurate inaccurate noisy quiet silent vibrant faded hot cold warm cool freezing boiling lukewarm temperate temperature climate weather humid dry wet damp moist rainy sunny cloudy foggy windy stormy seasonal tropical arctic desert professional amateur enthusiast beginner expert novice industrial commercial residential personal private public business enterprise domestic hobby diy do it yourself wired wireless cordless bluetooth wifi wi-fi cellular network internet connected online offline ethernet usb hdmi vga dvi displayport infrared rf radio frequency nfc near field communication  branded unbranded generic genuine fake counterfeit authentic replica original imitation knock-off trademark copyright patent artificial intelligence AI machine learning ML deep learning neural network neural net computer vision natural language processing NLP speech recognition voice recognition machine translation image recognition object detection pattern recognition data mining big data data science data analytics predictive analytics algorithm model training supervised learning unsupervised learning reinforcement learning transfer learning federated learning generative AI GAN generative adversarial network VAE variational autoencoder robotics automation autonomous self-driving smart intelligent cognitive computing expert system knowledge base inference engine chatbot virtual assistant conversational AI bot cloud computing edge computing quantum computing cybersecurity encryption blockchain decentralized IoT internet of things smart home smart city augmented reality AR virtual reality VR mixed reality MR extended reality XR algorithm model dataset training data test data validation data feature engineering dimensionality reduction clustering classification regression bias fairness explainable AI XAI interpretable AI GPU TPU tensor processing unit parallel processing distributed computing API application programming interface SDK software development kit open source proprietary cloud-based on-premise ethical AI responsible AI AI safety AI alignment handheld portable wearable digital analog automatic manual electric electronic mechanical hand-crafted mass-produced modular integrated standalone waterproof water-resistant dustproof shockproof eco-friendly sustainable recycled biodegradable minimalist ornate decorative functional ergonomic aesthetic utilitarian stylish trendy fashionable unusual unique rare common popular mainstream niche cutting-edge innovative state-of-the-art obsolete outdated manual"
    tags = tags_str.split(' ')
    tags = tags[0:10]
    config = Config(
        llama_path=args.llama_path,
        model_path=args.model_path,
        kobold_url=args.kobold_url
    )
    
    merger = SemanticMerger(config)
    
    # Run the merger
    try:
        synonym_mapping = asyncio.run(merger.process_tags(tags))
        print("Successfully processed tags")
        
        # Save results
        with open('synonym_mapping.json', 'w') as f:
            json.dump(synonym_mapping, f, indent=2)
            print("Saved results to synonym_mapping.json")
            
    except Exception as e:
        print(f"Error during processing: {e}")
        raise  # Re-raise to see full traceback

if __name__ == '__main__':
    main()
