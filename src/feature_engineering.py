# Feature engineering 

from sklearn.decomposition import PCA
import joblib

def apply_pca(data, n_components=3):
    # Applying PCA to reduce dimensionality
    pca = PCA(n_components=n_components)
    data_pca = pca.fit_transform(data)
    
    # Saving the PCA model
    joblib.dump(pca, '../models/pca_model.pkl')
    
    print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
    return data_pca
