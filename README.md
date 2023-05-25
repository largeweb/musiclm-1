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
