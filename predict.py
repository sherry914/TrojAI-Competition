import torch
import os
import helper
import extractor
import csv

learned_parameters_dirpath = "./learned_parameters"
pred_list = []

avg = 0.52

# This directory holds the model's parameter files
scratch_dirs = ["conv", "conv", "out", "out"]
param_files = ["conv_1_4.pt", "conv_f6.pt", "m.pt", "m_f6.pt"]
input_dims = [18, 18, 2, 2]

# Process each index
for ind in range(384):
    p_list = []
    fname = f'{ind}.pt'
    
    # Process each model configuration
    for scratch_dir, param_file, input_dim in zip(scratch_dirs, param_files, input_dims):
        try:
            ensemble = torch.load(os.path.join(learned_parameters_dirpath, param_file), map_location=torch.device('cpu'))
        except FileNotFoundError:
            ensemble = torch.load(os.path.join('/', learned_parameters_dirpath, param_file), map_location=torch.device('cpu'))
        
        data_by_model = torch.load(os.path.join(scratch_dir, fname))['fvs']
        features = [data_by_model]
        trojan_probability = extractor.predict(ensemble, features, input_dim=input_dim)
        p_list.append(trojan_probability)
    
    # Calculate the average trojan probability for this index
    trojan_probability_avg = sum(p_list) / len(p_list)
    pred_list.append(trojan_probability_avg)

# Calculate distances to avg = 0.52 and find top 50 smallest distances
distances = [(i, abs(prob - avg)) for i, prob in enumerate(pred_list)]
top_50 = sorted(distances, key=lambda x: x[1])[:50]

# Print results
for ind, dist in top_50:
    print(f'Index {ind} has a trojan probability of {pred_list[ind]} with a distance of {dist} from 0.52')

print("Overall average of trojan probabilities: ", sum(pred_list) / len(pred_list))

csv_filename = "top_50_inds.csv"
with open(csv_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Index'])
    for ind, _ in top_50:
        writer.writerow([ind])

import pandas as pd
df = pd.read_csv(csv_filename)
print(df)
