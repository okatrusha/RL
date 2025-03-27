import kagglehub

# Download latest version
path = kagglehub.dataset_download("yash16jr/snp500-dataset")

print("Path to dataset files:", path)