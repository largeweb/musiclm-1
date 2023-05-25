# MusicLM: Generating Rap Beats from Text
The goal of this project is to implement an end-to-end MusicLM model to generate rap instrumentals from text prompts. We will start with a simple design to test the concepts by generating 1000 audio clips, then discuss future considerations to improve the model.

## Simple Design
We will use ChatGPT to generate 1000 random captions for rap beats. For each caption, we will find a YouTube video with the audio, then use DeMUCS to remove the vocals. We will build an audio pipeline to extract mel spectrograms, MFCCs and chroma features. 

We will use:
- ChatGPT to generate 1000 random rap beat captions 
- YouTube API to find audio for each caption
- DeMUCS to remove vocals from YouTube audio
- Librosa to extract audio features
- PyTorch to build and train the model
- TensorFlow Datasets to handle the data

We define a simple model architecture for MuLaN and all Transformers with:
- dim=128 to balance complexity and training time 
- heads=2 based on hardware limitations  
- layers=3 to start simple

We train MuLaN, a jointly trained text-audio embedding model, for 50-100 epochs while decreasing the learning rate if validation loss plateaus. We save the model with the lowest validation loss.

MuLaNEmbedQuantizer quantizes the 128-dim MuLaN embeddings into 4 groups of 32 tokens each so the model can use discrete tokens. SemanticTransformer, CoarseTransformer, and FineTransformer are defined with audio_text_condition=True so they accept the quantized MuLaN embeddings.  

We tune hyperparameters for SemanticTransformer for 5-10 epochs. We train the best model for 50-100 epochs. The same procedure is followed for the CoarseTransformer and FineTransformer.

We generate 5-10 samples for each of 10-20 new captions using the trained Transformers. Samples are evaluated to select some for demonstrating the model.

In the demo, we show selected generated samples and discuss strengths, weaknesses, and future directions for improving the model.

## Future Considerations
To improve the model, we will:
- Gather more data, aiming for 10k-50k examples 
- Increase the model capacity with dim=256-512, heads=4-8, and layers=6 for MuLaN and Transformers. 
- Train MuLaN for 100-200 epochs, decreasing the learning rate by 0.1-0.5 if validation loss plateaus. Save checkpoint models frequently.
- Train Transformers for 300-500+ epochs. Save checkpoint models frequently.

We will add an adversarial loss for higher sample quality. We will experiment with conditioning on other signals like melody in addition to text.

We will develop an interactive demo for users to generate samples. Plans may include balancing the dataset and other techniques to improve diversity, quality and coherence.

In summary, this README provides a high-level end-to-end walkthrough for a simple MusicLM model to generate 1000 rap beats from text prompts, then discusses how to improve and build on that model for more advanced music generation.



# OLD FASHIONED (DETAILED) PLAN



# MusicLM: Generating Rap Beats from Text
The goal of this project is to implement an end-to-end MusicLM model to generate rap instrumentals from text prompts without vocals. MusicLM consists of MuLaN, a jointly trained text-audio embedding model, and three autoregressive Transformers: SemanticTransformer, CoarseTransformer, and FineTransformer.

MuLaN has already been trained. The three Transformers need to be tuned and trained to generate a sequence of discretized audio tokens conditioned on embeddings from MuLaN to produce rap instrumentals without vocals.

## Goal
The goal is to implement an end-to-end MusicLM model to generate rap instrumentals from text prompts without vocals. We will gather as much rap instrumental data as possible, including mel spectrograms, MFCCs, and chroma features.

MuLaNEmbedQuantizer quantizes the 256-dim MuLaN embeddings into 6 groups of 64 tokens each so the model can use discrete tokens.

## 1. Gather Data
* Gather as much rap instrumental data as possible, including mel spectrograms, MFCCs, and chroma features.
* Save data in TFRecords format for efficient loading.

## 2. Quantize MuLaN Embeddings
* Define MuLaNEmbedQuantizer to quantize the 256-dim MuLaN embeddings into 6 groups of 64 tokens each.

## 3. Tune and Train SemanticTransformer, CoarseTransformer and FineTransformer
* Tune hyperparameters for SemanticTransformer, CoarseTransformer and FineTransformer for 10+ epochs. Try:
   - Learning rates of 1e-3 to 3e-3
   - Dropouts of 0.1 to 0.3
* Pick the best configuration for each and train the models for 300-500+ epochs. Save checkpoint models frequently.

## 4. Generate and Evaluate Samples
* Generate 25-50 samples for each of 30-50 new text prompts using the trained Transformers.
* Have others evaluate the quality, how well the music matches the prompts, and coherence over time.
* Select samples for demonstrating the model.

## 5. Demo and Determine Next Steps
* Discuss strengths, weaknesses, limitations, and future directions. Demo selected generated samples.
* Plans may include:
   - Increasing the model capacity and training more.
   - Balancing the dataset by gathering more examples of certain genres/moods/tempos
   - Experimenting with different conditioning mechanisms beyond just the text prompt
   - Developing an interactive demo where users can generate samples from scratch.




## Next Steps:

1. Dataset Preparation: Collect a dataset of 1000 rap instrumental audio clips and their corresponding text captions from YouTube. Convert the audio clips into wav files and preprocess them as required.
2. MuLaN Model Training: Using the provided code in the MusicLM-Pytorch repository, adapt and train the MuLaN model on your dataset to create a joint embedding space for audio and text.
3. Conditioning AudioLM: After training the MuLaN model, use the `MuLaNEmbedQuantizer` class as instructed in the README.md file to generate conditioning embeddings for the semantic, coarse, and fine transformers of the AudioLM model.
4. Transforming AudioLM: Train the semantic, coarse, and fine transformers of AudioLM, and modify AudioLM to accept conditioning embeddings for music generation.
5. Integrating MusicLM: Create and use the MusicLM model with your trained AudioLM and MuLaNEmbedQuantizer to generate rap instrumental beats from text prompts.
6. Model Fine-tuning: Monitor the performance of MusicLM and fine-tune the models as required to improve the quality and consistency of generated music.
7. Project Documentation: Keep updating the README.md file and create necessary documentation to help the team understand and contribute to the project better. Include information about the dataset, the models, training procedures, any challenges faced, and instructions for future development.
8. Testing and Analysis: Perform extensive testing and evaluation of the generated music using appropriate metrics, and compare the results with existing baselines.
9. Collaboration and Code Reviews: Encourage communication, collaboration, and code reviews among the team members to improve code quality and ensure that the project runs smoothly.
10. Future Work: Identify areas of improvement, potential features and extensions, and incorporate them into the project roadmap. Implement them as necessary to enhance the capabilities of MusicLM.
