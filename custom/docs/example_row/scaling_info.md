data is vectorized
Scaling is applied after vectorization, but based on the data prior to vectorization. 
One universal scaling method for all features will not work because the data is differnt types, has outliers that will mess it up, and has wide ranges of values(0 and 1, values in billions)
After scaling, data will go through PCA. Then it will go through t-sne. Then it will be clustered by label. 
raw_feature_types is the types of the data before vectorization and raw_feature_data is the data before vectorization
feature_vector is the data after vectorization, which is float32 type.
Sample set: 30,000 train rows (21,762 after the family filter, see `custom/tsne/README.md`)
Sparseness?