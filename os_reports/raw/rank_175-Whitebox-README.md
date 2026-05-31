Data
Primary training data was provided by OpenADMET, including:
4,139 compounds with dose-response pEC50 values (PXR agonist assay)
2,647 compounds with counter-assay pEC50 values (PXR-null cell line)

Model
I fine-tuned CheMeleon, a graph neural network pre-trained on large-scale molecular data, on the 4,139 OpenADMET PXR training compounds. Training used a scaffold-based 5-fold cross-validation split to ensure generalization across chemical space. Test predictions were averaged across the five fold models. 

Acknowledgements
Thank OpenADMET for organizing the PXR Blind Challenge and providing the training dataset. 