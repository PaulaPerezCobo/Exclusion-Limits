# -*- coding: utf-8 -*-
"""
Created on Fri Mar 28 17:59:49 2025

@author: Asus
"""

import numpy as np
from scipy.stats import norm
import csv
import os

# UNITS
um = 1.0
cm = 10000 * um
eV = 1.0
sigma_res = 0.16

# PARAMETERS FOR THE DIFFUSION MODEL
DIFF_A = 803.25 * um * um
DIFF_b = 6.5e-4 / um
DIFF_alpha = 1
DIFF_beta = 0 / eV

# CCD GEOMETRY
PIX_THICKNESS = 15.0 * um
ROWS = (0 * um, 1600 * PIX_THICKNESS * um) #6080
COLS = (0 * um, 6144 * PIX_THICKNESS * um)
DEPTH = (0 * um, 670 * um)
BINNING_FACTOR = 100

# Define the directory where you want to save the files
save_directory = r'C:\Users\Asus\Documents\MÁSTER\TFM\QEdark\Diffusion Probabilities\Patterns'
#save_directory = r'C:\Users\escob\OneDrive\Documentos\Paula'

# Make sure the directory exists
os.makedirs(save_directory, exist_ok=True)

#CLASSIFY PATTERN
def compute_p(cdf_values, threshold=4.0):
    """Compute minimum p-value using precomputed CDF values and compare with a threshold."""
    return min(-np.log(np.prod(cdf_values_set)) for cdf_values_set in cdf_values) < threshold

def classify_pattern(complete_pattern_readen, sigma_res=0.16):
    detected_patterns = []
    
    while len(complete_pattern_readen) >= 3:
        #Sum pattern readen <= 5 for pmn
        sum_pattern_readen = sum(complete_pattern_readen[1:3]) #0 m n : m+ n
        if sum_pattern_readen > 5:
            complete_pattern_readen = complete_pattern_readen[3:]
            continue
        #Any pattern composed of charge < 3.74*sigma_res discarded    
        if complete_pattern_readen[1] < 3.74*sigma_res:
            complete_pattern_readen = complete_pattern_readen[1:]
            continue
        
        #Pattern and single event must be isolated: initial pixel in row
        if complete_pattern_readen[0] > 3.74*sigma_res:
            complete_pattern_readen = complete_pattern_readen[1:]

        #First compute all possible cdfs:
        cdf_11 = norm.cdf(complete_pattern_readen[1], loc=1, scale=sigma_res)
        cdf_12 = norm.cdf(complete_pattern_readen[1], loc=2, scale=sigma_res)
        cdf_13 = norm.cdf(complete_pattern_readen[1], loc=3, scale=sigma_res)
        cdf_14 = norm.cdf(complete_pattern_readen[1], loc=4, scale=sigma_res)

        if len(complete_pattern_readen) >= 3:   
            cdf_21 = norm.cdf(complete_pattern_readen[2], loc=1, scale=sigma_res)
            cdf_22 = norm.cdf(complete_pattern_readen[2], loc=2, scale=sigma_res)
            cdf_23 = norm.cdf(complete_pattern_readen[2], loc=3, scale=sigma_res)
            cdf_24 = norm.cdf(complete_pattern_readen[2], loc=4, scale=sigma_res)

        if len(complete_pattern_readen) >= 5:
            cdf_31 = norm.cdf(complete_pattern_readen[3], loc=1, scale=sigma_res)
            cdf_32 = norm.cdf(complete_pattern_readen[3], loc=2, scale=sigma_res)
            cdf_33 = norm.cdf(complete_pattern_readen[3], loc=3, scale=sigma_res)            
        
        #SINGLE EVENTS
        if compute_p([[cdf_11]], threshold=3.5):
            if  compute_p([[cdf_12]], threshold=3.5):
                if  compute_p([[cdf_13]], threshold=3.5):
                    #isolated single event: final pixel
                    if complete_pattern_readen[2] < 3.74*sigma_res:                                 
                        detected_patterns.append((3,))
                        complete_pattern_readen = complete_pattern_readen[2:]
                        continue
                else:
                    #isolated single event: final pixel
                    if complete_pattern_readen[2] < 3.74*sigma_res:  
                        detected_patterns.append((2,))
                        complete_pattern_readen = complete_pattern_readen[2:]
                        continue
            else:
                #isolated single event: final pixel
                if complete_pattern_readen[2] < 3.74*sigma_res:                                 
                    detected_patterns.append((1,))
                    complete_pattern_readen = complete_pattern_readen[2:]
                    continue
                
        #PATTERNS        
        if len(complete_pattern_readen) >= 4:
            #Patterns must be composed of q > 3.74*sigma_res
            if complete_pattern_readen[2] < 3.74*sigma_res:
                complete_pattern_readen = complete_pattern_readen[3:]
                continue
            
            sum_pattern_readen = sum(complete_pattern_readen[1:4]) #0 m n l m+n+l
            if sum_pattern_readen > 5:
                complete_pattern_readen = complete_pattern_readen[4:]
                continue
            
            if compute_p([[cdf_11, cdf_21]]): #11
                if compute_p([[cdf_12, cdf_21], [cdf_11, cdf_22]]): #21
                    if compute_p([[cdf_13, cdf_21], [cdf_11, cdf_23]]): #31
                        if complete_pattern_readen[3] < 3.74*sigma_res: #31 isolated
                            detected_patterns.append((3, 1))
                            complete_pattern_readen = complete_pattern_readen[3:]
                            continue
                    else:
                        if len(complete_pattern_readen) >= 5 and complete_pattern_readen[3] > 3.74*sigma_res:
                            if compute_p([[cdf_12, cdf_21, cdf_31],[cdf_11, cdf_22, cdf_31],[cdf_11, cdf_21, cdf_32]], threshold=5.5):
                                if complete_pattern_readen[4] < 3.74*sigma_res: #211 isolated
                                    detected_patterns.append((2, 1, 1))
                                    complete_pattern_readen = complete_pattern_readen[4:]
                                    continue
                        else:
                            if compute_p([[cdf_12, cdf_21], [cdf_11, cdf_22]]):
                                if compute_p([[cdf_12, cdf_22]]):
                                    if complete_pattern_readen[3] < 3.74*sigma_res:  #22 isolated 
                                        detected_patterns.append((2, 2))
                                        complete_pattern_readen = complete_pattern_readen[3:]
                                        continue
                                if complete_pattern_readen[3] < 3.74*sigma_res:    #21 isolated
                                    detected_patterns.append((2, 1))
                                    complete_pattern_readen = complete_pattern_readen[3:]
                                    continue
                else: 
                    if len(complete_pattern_readen) >= 5 and compute_p([[cdf_11, cdf_21, cdf_31]], threshold=5.5):
                        if compute_p([[cdf_12, cdf_21, cdf_31],[cdf_11, cdf_22, cdf_31],[cdf_11, cdf_21, cdf_32]],threshold=5.5):
                            if complete_pattern_readen[4] < 3.74*sigma_res: 
                                detected_patterns.append((2, 1, 1))
                                complete_pattern_readen = complete_pattern_readen[5:]
                                continue
                        else:  
                            if complete_pattern_readen[4] < 3.74*sigma_res:  
                                detected_patterns.append((1, 1, 1))
                                complete_pattern_readen = complete_pattern_readen[5:]
                                continue
                    else:
                        if complete_pattern_readen[3] < 3.74*sigma_res: 
                            detected_patterns.append((1, 1))
                            complete_pattern_readen = complete_pattern_readen[3:]
                            continue        
        
        if len(complete_pattern_readen) >= 3: 
            complete_pattern_readen = complete_pattern_readen[3:]
            
        complete_pattern_readen = complete_pattern_readen[1:]
    
    return detected_patterns

def probabilities_patterns(Ee, Niter=5000, binning=False):
    neh = np.arange(1, 11)  # neh range
    interaction_probabilities = {}
    # Patterns studied
    allowed_patterns = [(1,), (2,), (3,), (1, 1), (2, 1), (3, 1), (2, 2), (1, 1, 1), (2, 1, 1)]  

    def calculate_sigma_xy(z_positions, Ee):
        return np.sqrt(-DIFF_A * np.log(1 - DIFF_b * z_positions)) * (DIFF_alpha + DIFF_beta * Ee)
    
    results = []  # List to store results before writing to CSV
    for n in neh:
        pattern_counts = {}  
        positions = np.random.uniform(
            low=[ROWS[0], COLS[0], DEPTH[0]],
            high=[ROWS[1], COLS[1], DEPTH[1]],
            size=(Niter, 3))
        sigma_xy = calculate_sigma_xy(positions[:, 2], Ee)

        row_patterns = {}

        for i in range(Niter):
            pos_diff = np.random.normal(
                loc=positions[i, :2],
                scale=sigma_xy[i],
                size=(n, 2))

            pixel_indices = [(int(pos[0] // PIX_THICKNESS), 
                              int(pos[1] // PIX_THICKNESS)) for pos in pos_diff]
            
            unique_pixels = {}
            for px in pixel_indices:
                if binning:
                    binned_row = px[0] // BINNING_FACTOR  
                    binned_px = (binned_row, px[1])  # Binning
                else:
                    binned_px = px  # Without binning
                
                if binned_px in unique_pixels:
                    unique_pixels[binned_px] += 1
                else:
                    unique_pixels[binned_px] = 1 

            row_distribution = {}
            for (row, col), count in unique_pixels.items():
                if row not in row_distribution:
                    row_distribution[row] = {}
                row_distribution[row][col] = count
            
            # DETECTED PATTERN: WITH READOUT
            for row, cols in row_distribution.items():
                sorted_cols = sorted(cols.keys())
                min_col = sorted_cols[0]

                complete_pattern = [0]
                prev_col = min_col
                complete_pattern.append(cols[prev_col])

                for col in sorted_cols[1:]:
                    if col == prev_col + 1:
                        complete_pattern.append(cols[col])
                    else:
                        complete_pattern.extend([0] * (col - prev_col - 1))
                        complete_pattern.append(cols[col])
                    prev_col = col
                
                complete_pattern.append(0)
                
            # Readout noise
            complete_pattern_readen = np.random.normal(loc=complete_pattern, scale=sigma_res)
            # Patterns classification
            detected_patterns = classify_pattern(complete_pattern_readen, sigma_res=0.16)

            for detected_pattern in detected_patterns:
                sorted_pattern = tuple(sorted(detected_pattern, reverse=True))
                row_patterns[sorted_pattern] = row_patterns.get(sorted_pattern, 0) + 1
                
        for pattern, count in row_patterns.items():
            if pattern in allowed_patterns:
                pattern_counts[pattern] = pattern_counts.get(pattern, 0) + count
        
        for pattern, count in pattern_counts.items():
            prob_int = count / Niter
            if pattern not in interaction_probabilities:
                interaction_probabilities[pattern] = {}
            interaction_probabilities[pattern][n] = prob_int
    
    return interaction_probabilities


int_prob = probabilities_patterns(0, Niter=10000, binning=True)


output_file = os.path.join(save_directory, 'EfficienciesPatterns.csv')
with open(output_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Pattern', 'neh', 'Efficiency'])
    
    for pattern, neh_values in int_prob.items():
        for neh, efficiency in neh_values.items():
            writer.writerow([pattern, neh, efficiency])

print(f"Efficiencies saved in {output_file}")