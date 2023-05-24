# MusicLM: Generating Rap Beats from Text
The goal of this project is to implement an end-to-end MusicLM model to generate rap instrumentals from text prompts without vocals. MusicLM consists of MuLaN, a jointly trained text-audio embedding model, and three autoregressive Transformers: SemanticTransformer, CoarseTransformer, and FineTransformer.

MuLaN learns a shared embedding space for rap instrumental audio and text. The three Transformers then generate a sequence of discretized audio tokens conditioned on embeddings from MuLaN to produce rap instrumentals without vocals.

## Goal  
The goal is to implement an end-to-end MusicLM model to generate rap instrumentals from text prompts without vocals. We will gather as much rap instrumental data as possible, including mel spectrograms, MFCCs, and chroma features. 

We define a model architecture for MuLaN with dim=256, heads=4, and layers=6 based on model capacity and hardware limitations. We train MuLaN for 100-200 epochs, decreasing the learning rate by 0.1-0.5 if validation loss plateaus. We save checkpoint models frequently in case training needs to be stopped and restarted. 

MuLaNEmbedQuantizer quantizes the 256-dim MuLaN embeddings into 6 groups of 64 tokens each so the model can use discrete tokens. SemanticTransformer, CoarseTransformer, and FineTransformer are defined with audio_text_condition=True so they accept the quantized MuLaN embeddings.  

We define TransformerTrainer classes for each Transformer. The SemanticTransformer is tuned for 10+ epochs, trying learning rates of 1e-3 to 3e-3; dropouts of 0.1 to 0.3; and layers of 4 to 6.  The best model is trained for 300-500+ epochs. The same procedure is followed for the CoarseTransformer and FineTransformer.

We generate 25-50 samples for each of 30-50 new text prompts using the trained Transformers. Samples are evaluated by others to select some for demonstrating the model.  

In the demo, we show selected generated samples, discuss model strengths, weaknesses, limitations, and future directions. Plans may include:  
   - Increasing the model capacity and training more.  
   - Balancing the dataset by gathering more examples of certain genres/moods/tempos.  
   - Experimenting with different conditioning mechanisms beyond just the text prompt.  
   - Developing an interactive demo where users can generate samples from scratch.

## 1. Gather Data   
* Gather as much rap instrumental data as possible, including mel spectrograms, MFCCs, and chroma features. 
* Save data in TFRecords format for efficient loading.  

## 2. Define Model Architecture and Train MuLaN   
* Define a model architecture for MuLaN with dim=256, heads=4, and layers=6. 
* Train MuLaN, a jointly trained text-audio embedding model, for 100-200 epochs while decreasing the learning rate by 0.1-0.5 if validation loss plateaus.
* Save checkpoint models frequently in case training needs to be stopped and restarted.

## 3. Quantize MuLaN Embeddings and Define Transformers
* Define MuLaNEmbedQuantizer to quantize the 256-dim MuLaN embeddings into 6 groups of 64 tokens each.
* Define SemanticTransformer, CoarseTransformer, and FineTransformer with audio_text_condition=True so they accept the quantized MuLaN embeddings.

## 4. Tune and Train SemanticTransformer  
* Tune hyperparameters for SemanticTransformer for 10+ epochs. Try: 
   - Learning rates of 1e-3 to 3e-3  
   - Dropouts of 0.1 to 0.3  
   - Layers of 4 to 6
* Pick the best configuration and train the model for 300-500+ epochs. Save checkpoint models frequently.

## 5. Tune and Train CoarseTransformer and FineTransformer
* Follow the same tuning and training procedure for CoarseTransformer and FineTransformer.  
* Start with the hyperparameters from the best SemanticTransformer and adjust as needed. Save checkpoint models frequently.

## 6. Generate and Evaluate Samples  
* Generate 25-50 samples for each of 30-50 new text prompts using the trained Transformers.
* Have others evaluate the quality, how well the music matches the prompts, and coherence over time.  
* Select samples for demonstrating the model.

## 7. Demo and Determine Next Steps
* Discuss strengths, weaknesses, limitations, and future directions. Demo selected generated samples. 
* Plans may include:  
   - Increasing the model capacity and training more.   
   - Balancing the dataset by gathering more examples of certain genres/moods/tempos
