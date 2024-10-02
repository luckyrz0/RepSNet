CoNIC Challenge Data

We supply patches that can be used for developing algorithms for the challenge. These patches are non-overlapping and are of size 256x256. Note, this will be in line with the size of all patches used for evaluation and final determination of the leaderboard. The patches are extracted from the original Lizard Dataset and therefore participants may wish to extract their own patches with overlap if they wish. Code for this can be found on the CoNIC challenge GitHub page:

https://github.com/TissueImageAnalytics/CoNIC

In total, we have extracted 4,981 patches.

Available data:
- images.npy: single numpy array of type uint8 and size 4981x256x256x3 containing all RGB image patches.
- labels.npy: single numpy array of type uint16 and size 4981x256x256x2 containing the instance segmentation map (first channel) and classification map (second channel). The instance map contains values ranging from 0 (background) to N (number of nuclei). Therefore each nucleus is uniquely labelled. The classification map contains values ranging from 0 (background) to 6 (number of classes).
- counts.csv: single csv file denoting the counts of each type of nucleus in each image patch. The order of the rows are in line with the order of patches in the numpy files.
- patch_info.csv: single csv file denoting which images the patches were extracted from in the original lizard dataset. 

When determining the counts, we only consider nuclei within the central 224x224 region. If ANY part of a nucleus is within this region, then it is considered.

The dataset provided here is for research purposes only. Commercial use is not allowed. The data is held under the following license: Attribution-NonCommercial-ShareAlike 4.0 International.