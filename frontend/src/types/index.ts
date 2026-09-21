export interface ModelBenchmark {
  rank: number;
  key: string;
  model: string;
  category: string;
  n_test_samples: number;
  accuracy_pct: number;
  macro_f1: number;
  macro_precision: number;
  macro_recall: number;
  weighted_f1: number;
  label_space: string;
}

export interface ModelPerformanceResponse {
  evaluation_split: string;
  metric_protocol: string;
  num_active_classes: number;
  num_test_samples: number;
  models: ModelBenchmark[];
}

export interface ModelInfo {
  id: string;
  name: string;
  category: 'classical_ml' | 'deep_learning';
  architecture: string;
  input_representation?: string;
  configuration?: string;
  description: string;
  available: boolean;
  has_checkpoint: boolean;
  has_stored_predictions: boolean;
  label_space: string;
  benchmark?: ModelBenchmark;
}

export interface SampleItem {
  sample_id: string;
  filename: string;
  split: 'train' | 'val' | 'test';
  split_index: number;
  ground_truth_id: number;
  ground_truth_name: string;
  image_url: string;
  has_stored_prediction: boolean;
}

export interface SampleListResponse {
  total: number;
  page: number;
  page_size: number;
  total_pages: number;
  samples: SampleItem[];
}

export interface TopPrediction {
  rank: number;
  class_id: number;
  class_name: string;
  probability: number;
}

export interface ClassifyResponse {
  sample_id: string;
  model: string;
  model_name: string;
  predicted_class_id: number;
  predicted_class_name: string;
  ground_truth_id: number;
  ground_truth_name: string;
  correct: boolean;
  confidence: number | null;
  confidence_display: string;
  top_predictions: TopPrediction[];
  inference_source: string;
  inference_time_ms: number | null;
  inference_time_display: string;
  status: string;
  model_test_metrics?: Record<string, unknown> | null;
}

export interface HealthResponse {
  status: string;
  dataset_loaded: boolean;
  test_samples: number;
  val_samples: number;
  train_samples: number;
  models_available: number;
  checkpoints_loaded: number;
}

export interface ClassDistributionItem {
  class_id: string;
  class_name: string;
  train_count: number;
  val_count: number;
  test_count: number;
  total: number;
}

export interface FeatureBreakdown {
  color_statistics: number;
  rgb_histograms: number;
  glcm_texture: number;
  hog: number;
  edge_descriptors: number;
  gradient_statistics: number;
}

export interface DatasetStats {
  total_calibrated_images: number;
  total_labeled_images: number;
  active_classes: number;
  format: string;
  preprocessed_size: string;
  color_modes: {
    rgb: number;
    grayscale: number;
  };
  classical_feature_vector_dim: number;
  feature_breakdown: FeatureBreakdown;
  splits: {
    test: number;
    val: number;
    train: number;
    total: number;
  };
  classes: ClassDistributionItem[];
  total_images?: number;
}
