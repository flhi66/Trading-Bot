import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import warnings

@dataclass
class PatternCluster:
    """Represents a cluster of similar patterns"""
    cluster_id: int
    center_pattern: np.ndarray
    patterns: List[np.ndarray]
    timestamps: List[pd.Timestamp]
    prices: List[float]
    confidence: float

@dataclass
class PatternMatch:
    """Represents a pattern match found in the data"""
    timestamp: pd.Timestamp
    pattern: np.ndarray
    distance: float
    cluster_id: Optional[int]
    confidence: float

class PatternClustering:
    """K-means clustering for pattern recognition"""
    
    def __init__(self, n_clusters: int = 5, random_state: int = 42):
        """
        Initialize the pattern clustering
        
        Args:
            n_clusters: Number of clusters to create
            random_state: Random seed for reproducibility
        """
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.kmeans = None
        self.scaler = StandardScaler()
        self.cluster_centers = None
        self.cluster_labels = None
        
    def _prepare_patterns(self, patterns: List[np.ndarray]) -> np.ndarray:
        """
        Prepare patterns for clustering by padding/truncating to same length
        
        Args:
            patterns: List of pattern arrays
            
        Returns:
            Array of prepared patterns
        """
        if not patterns:
            return np.array([])
        
        # Find maximum length
        max_len = max(len(p) for p in patterns)
        
        # Pad shorter patterns with their last value
        prepared_patterns = []
        for pattern in patterns:
            if len(pattern) < max_len:
                # Pad with last value
                padded = np.pad(pattern, (0, max_len - len(pattern)), 
                               mode='edge')
            else:
                padded = pattern
            prepared_patterns.append(padded)
        
        return np.array(prepared_patterns)
    
    def fit(self, patterns: List[np.ndarray], 
            timestamps: Optional[List[pd.Timestamp]] = None,
            prices: Optional[List[float]] = None) -> Dict[str, List[PatternCluster]]:
        """
        Fit K-means clustering to the patterns
        
        Args:
            patterns: List of pattern arrays
            timestamps: List of timestamps for each pattern
            prices: List of prices for each pattern
            
        Returns:
            Dictionary with clusters organized by cluster ID
        """
        if len(patterns) < self.n_clusters:
            warnings.warn(f"Number of patterns ({len(patterns)}) is less than n_clusters ({self.n_clusters})")
            self.n_clusters = min(len(patterns), 2)
        
        # Prepare patterns
        X = self._prepare_patterns(patterns)
        
        if X.size == 0:
            return {}
        
        # Scale the patterns
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit K-means
        self.kmeans = KMeans(n_clusters=self.n_clusters, 
                            random_state=self.random_state,
                            n_init=10)
        self.cluster_labels = self.kmeans.fit_predict(X_scaled)
        self.cluster_centers = self.kmeans.cluster_centers_
        
        # Create cluster objects
        clusters = {}
        for cluster_id in range(self.n_clusters):
            cluster_mask = self.cluster_labels == cluster_id
            cluster_patterns = [patterns[i] for i in range(len(patterns)) if cluster_mask[i]]
            
            # Get cluster center (unscaled)
            center_scaled = self.cluster_centers[cluster_id]
            center_unscaled = self.scaler.inverse_transform([center_scaled])[0]
            
            # Calculate confidence (silhouette score for this cluster)
            if len(cluster_patterns) > 1:
                cluster_indices = np.where(cluster_mask)[0]
                if len(cluster_indices) > 1:
                    try:
                        confidence = silhouette_score(X_scaled[cluster_indices], 
                                                   self.cluster_labels[cluster_indices])
                    except:
                        confidence = 0.5
                else:
                    confidence = 0.5
            else:
                confidence = 0.5
            
            # Get timestamps and prices for this cluster
            cluster_timestamps = []
            cluster_prices = []
            if timestamps:
                cluster_timestamps = [timestamps[i] for i in range(len(timestamps)) if cluster_mask[i]]
            if prices:
                cluster_prices = [prices[i] for i in range(len(prices)) if cluster_mask[i]]
            
            clusters[cluster_id] = PatternCluster(
                cluster_id=cluster_id,
                center_pattern=center_unscaled,
                patterns=cluster_patterns,
                timestamps=cluster_timestamps,
                prices=cluster_prices,
                confidence=confidence
            )
        
        return clusters
    
    def predict_cluster(self, pattern: np.ndarray) -> int:
        """
        Predict which cluster a pattern belongs to
        
        Args:
            pattern: Pattern array to classify
            
        Returns:
            Predicted cluster ID
        """
        if self.kmeans is None:
            raise ValueError("Model must be fitted before making predictions")
        
        # Prepare and scale the pattern
        X = self._prepare_patterns([pattern])
        X_scaled = self.scaler.transform(X)
        
        # Predict cluster
        cluster_id = self.kmeans.predict(X_scaled)[0]
        return cluster_id
    
    def get_cluster_center(self, cluster_id: int) -> np.ndarray:
        """
        Get the center pattern for a specific cluster
        
        Args:
            cluster_id: ID of the cluster
            
        Returns:
            Center pattern array
        """
        if self.cluster_centers is None:
            raise ValueError("Model must be fitted before getting cluster centers")
        
        if cluster_id >= self.n_clusters:
            raise ValueError(f"Cluster ID {cluster_id} does not exist")
        
        # Return unscaled center
        center_scaled = self.cluster_centers[cluster_id]
        center_unscaled = self.scaler.inverse_transform([center_scaled])[0]
        return center_unscaled

class PatternClassification:
    """K-Nearest Neighbors classification for pattern recognition"""
    
    def __init__(self, n_neighbors: int = 5, weights: str = 'uniform'):
        """
        Initialize the pattern classifier
        
        Args:
            n_neighbors: Number of neighbors to consider
            weights: Weight function ('uniform' or 'distance')
        """
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.knn = None
        self.scaler = StandardScaler()
        self.training_patterns = None
        self.training_labels = None
        
    def fit(self, patterns: List[np.ndarray], labels: List[str]) -> None:
        """
        Fit the KNN classifier
        
        Args:
            patterns: List of training pattern arrays
            labels: List of labels for each pattern
        """
        if len(patterns) != len(labels):
            raise ValueError("Number of patterns must equal number of labels")
        
        if len(patterns) < self.n_neighbors:
            warnings.warn(f"Number of patterns ({len(patterns)}) is less than n_neighbors ({self.n_neighbors})")
            self.n_neighbors = min(len(patterns), 2)
        
        # Prepare patterns
        X = self._prepare_patterns(patterns)
        
        if X.size == 0:
            return
        
        # Scale the patterns
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit KNN
        self.knn = KNeighborsClassifier(n_neighbors=self.n_neighbors, 
                                       weights=self.weights)
        self.knn.fit(X_scaled, labels)
        
        # Store training data
        self.training_patterns = patterns
        self.training_labels = labels
    
    def _prepare_patterns(self, patterns: List[np.ndarray]) -> np.ndarray:
        """
        Prepare patterns for classification by padding/truncating to same length
        
        Args:
            patterns: List of pattern arrays
            
        Returns:
            Array of prepared patterns
        """
        if not patterns:
            return np.array([])
        
        # Find maximum length
        max_len = max(len(p) for p in patterns)
        
        # Pad shorter patterns with their last value
        prepared_patterns = []
        for pattern in patterns:
            if len(pattern) < max_len:
                # Pad with last value
                padded = np.pad(pattern, (0, max_len - len(pattern)), 
                               mode='edge')
            else:
                padded = pattern
            prepared_patterns.append(padded)
        
        return np.array(prepared_patterns)
    
    def predict(self, pattern: np.ndarray) -> str:
        """
        Predict the label for a pattern
        
        Args:
            pattern: Pattern array to classify
            
        Returns:
            Predicted label
        """
        if self.knn is None:
            raise ValueError("Model must be fitted before making predictions")
        
        # Prepare and scale the pattern
        X = self._prepare_patterns([pattern])
        X_scaled = self.scaler.transform(X)
        
        # Predict label
        label = self.knn.predict(X_scaled)[0]
        return label
    
    def predict_proba(self, pattern: np.ndarray) -> Dict[str, float]:
        """
        Predict probability scores for each class
        
        Args:
            pattern: Pattern array to classify
            
        Returns:
            Dictionary with class probabilities
        """
        if self.knn is None:
            raise ValueError("Model must be fitted before making predictions")
        
        # Prepare and scale the pattern
        X = self._prepare_patterns([pattern])
        X_scaled = self.scaler.transform(X)
        
        # Get probability scores
        proba = self.knn.predict_proba(X_scaled)[0]
        classes = self.knn.classes_
        
        # Create dictionary
        proba_dict = {classes[i]: proba[i] for i in range(len(classes))}
        return proba_dict
    
    def find_nearest_neighbors(self, pattern: np.ndarray, 
                              n_neighbors: Optional[int] = None) -> List[Tuple[int, float, str]]:
        """
        Find the nearest neighbors for a pattern
        
        Args:
            pattern: Pattern array to find neighbors for
            n_neighbors: Number of neighbors to return (defaults to self.n_neighbors)
            
        Returns:
            List of tuples (index, distance, label)
        """
        if self.knn is None:
            raise ValueError("Model must be fitted before finding neighbors")
        
        if n_neighbors is None:
            n_neighbors = self.n_neighbors
        
        # Prepare and scale the pattern
        X = self._prepare_patterns([pattern])
        X_scaled = self.scaler.transform(X)
        
        # Find nearest neighbors
        distances, indices = self.knn.kneighbors(X_scaled, n_neighbors=n_neighbors)
        
        # Return results
        neighbors = []
        for i in range(len(indices[0])):
            idx = indices[0][i]
            distance = distances[0][i]
            label = self.training_labels[idx]
            neighbors.append((idx, distance, label))
        
        return neighbors

class PatternAnalyzer:
    """Main class that combines clustering and classification"""
    
    def __init__(self, n_clusters: int = 5, n_neighbors: int = 5):
        """
        Initialize the pattern analyzer
        
        Args:
            n_clusters: Number of clusters for K-means
            n_neighbors: Number of neighbors for KNN
        """
        self.clustering = PatternClustering(n_clusters=n_clusters)
        self.classification = PatternClassification(n_neighbors=n_neighbors)
        self.clusters = {}
        self.is_fitted = False
    
    def fit_clustering(self, patterns: List[np.ndarray], 
                      timestamps: Optional[List[pd.Timestamp]] = None,
                      prices: Optional[List[float]] = None) -> Dict[str, List[PatternCluster]]:
        """
        Fit the clustering model
        
        Args:
            patterns: List of pattern arrays
            timestamps: List of timestamps for each pattern
            prices: List of prices for each pattern
            
        Returns:
            Dictionary with clusters
        """
        self.clusters = self.clustering.fit(patterns, timestamps, prices)
        self.is_fitted = True
        return self.clusters
    
    def fit_classification(self, patterns: List[np.ndarray], labels: List[str]) -> None:
        """
        Fit the classification model
        
        Args:
            patterns: List of training pattern arrays
            labels: List of labels for each pattern
        """
        self.classification.fit(patterns, labels)
    
    def analyze_pattern(self, pattern: np.ndarray) -> Dict:
        """
        Analyze a pattern using both clustering and classification
        
        Args:
            pattern: Pattern array to analyze
            
        Returns:
            Dictionary with analysis results
        """
        results = {}
        
        # Clustering analysis
        if self.is_fitted:
            try:
                cluster_id = self.clustering.predict_cluster(pattern)
                cluster_center = self.clustering.get_cluster_center(cluster_id)
                
                results['clustering'] = {
                    'cluster_id': cluster_id,
                    'cluster_center': cluster_center.tolist(),
                    'cluster_info': self.clusters.get(cluster_id, {})
                }
            except Exception as e:
                results['clustering'] = {'error': str(e)}
        
        # Classification analysis
        try:
            predicted_label = self.classification.predict(pattern)
            probabilities = self.classification.predict_proba(pattern)
            neighbors = self.classification.find_nearest_neighbors(pattern)
            
            results['classification'] = {
                'predicted_label': predicted_label,
                'probabilities': probabilities,
                'nearest_neighbors': neighbors
            }
        except Exception as e:
            results['classification'] = {'error': str(e)}
        
        return results
    
    def get_cluster_summary(self) -> pd.DataFrame:
        """
        Get a summary of all clusters
        
        Returns:
            DataFrame with cluster information
        """
        if not self.is_fitted:
            return pd.DataFrame()
        
        summary_data = []
        for cluster_id, cluster in self.clusters.items():
            summary_data.append({
                'cluster_id': cluster_id,
                'num_patterns': len(cluster.patterns),
                'confidence': cluster.confidence,
                'avg_price': np.mean(cluster.prices) if cluster.prices else np.nan,
                'price_std': np.std(cluster.prices) if cluster.prices else np.nan
            })
        
        return pd.DataFrame(summary_data)
