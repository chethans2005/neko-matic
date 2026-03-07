"use client";

import { FormEvent, useEffect, useMemo, useState } from "react";
import { uploadConfig } from "@/lib/api";

type ProblemTypeOverride = "classification" | "regression" | null;
type MissingValueStrategy = "median" | "mean" | "most_frequent" | "constant" | "drop";
type CategoricalEncoding = "onehot" | "label";
type FeatureScaling = "standard" | "minmax" | "none";
type OutlierMethod = "none" | "zscore" | "iqr" | "isolation_forest";
type FeatureSelectionMethod =
  | "variance_threshold"
  | "mutual_information"
  | "recursive_feature_elimination";
type OptimizationMethod = "optuna";
type GpuUsage = "auto" | boolean;
type PrimaryMetric =
  | "accuracy"
  | "f1"
  | "f1 score"
  | "precision"
  | "recall"
  | "roc_auc"
  | "rmse"
  | "mse"
  | "mae"
  | "r2";

type StoredDataset = {
  columns?: string[];
  target_column_guess?: string | null;
};

type AutoMLConfig = {
  dataset_settings: {
    target_column: string | null;
    problem_type_override: ProblemTypeOverride;
    train_test_split: number;
    cross_validation_folds: number;
  };
  data_cleaning: {
    missing_value_strategy: MissingValueStrategy;
    categorical_encoding: CategoricalEncoding;
    feature_scaling: FeatureScaling;
  };
  outlier_removal: {
    method: OutlierMethod;
    threshold_parameters: {
      zscore_threshold: number;
      iqr_multiplier: number;
      isolation_forest_contamination: number;
    };
  };
  feature_engineering: {
    log_transform: boolean;
    polynomial_features: boolean;
    feature_interactions: boolean;
    feature_selection: {
      enabled: boolean;
      method: FeatureSelectionMethod;
      k_features: number;
    };
  };
  model_selection: {
    list_of_models_to_train: string[] | null;
  };
  hyperparameter_optimization: {
    optimization_method: OptimizationMethod;
    number_of_trials: number;
    timeout: number | null;
    early_stopping: boolean;
  };
  training_strategy: {
    parallel_training: boolean;
    gpu_usage: GpuUsage;
    time_budget: number | null;
  };
  evaluation_metrics: {
    primary_metric: PrimaryMetric;
  };
  explainability: {
    enable_shap: boolean;
  };
};

const MODEL_OPTIONS = [
  "GaussianNB",
  "GradientBoostingClassifier",
  "GradientBoostingRegressor",
  "KNeighborsClassifier",
  "LGBMClassifier",
  "LGBMRegressor",
  "Lasso",
  "LinearRegression",
  "LogisticRegression",
  "RandomForestClassifier",
  "RandomForestRegressor",
  "Ridge",
  "SVC",
  "SVR",
  "XGBClassifier",
  "XGBRegressor",
] as const;

const defaultConfig: AutoMLConfig = {
  dataset_settings: {
    target_column: null,
    problem_type_override: null,
    train_test_split: 0.2,
    cross_validation_folds: 5,
  },
  data_cleaning: {
    missing_value_strategy: "median",
    categorical_encoding: "onehot",
    feature_scaling: "standard",
  },
  outlier_removal: {
    method: "none",
    threshold_parameters: {
      zscore_threshold: 3,
      iqr_multiplier: 1.5,
      isolation_forest_contamination: 0.05,
    },
  },
  feature_engineering: {
    log_transform: false,
    polynomial_features: false,
    feature_interactions: false,
    feature_selection: {
      enabled: false,
      method: "variance_threshold",
      k_features: 20,
    },
  },
  model_selection: {
    list_of_models_to_train: null,
  },
  hyperparameter_optimization: {
    optimization_method: "optuna",
    number_of_trials: 20,
    timeout: null,
    early_stopping: true,
  },
  training_strategy: {
    parallel_training: false,
    gpu_usage: "auto",
    time_budget: null,
  },
  evaluation_metrics: {
    primary_metric: "accuracy",
  },
  explainability: {
    enable_shap: true,
  },
};

function normalizeConfig(input: unknown): AutoMLConfig {
  const value = (input && typeof input === "object" ? input : {}) as Record<string, unknown>;

  const datasetSettings = (value.dataset_settings as Record<string, unknown>) ?? {};
  const dataCleaning = (value.data_cleaning as Record<string, unknown>) ?? {};
  const outlierRemoval = (value.outlier_removal as Record<string, unknown>) ?? {};
  const thresholdParameters =
    (outlierRemoval.threshold_parameters as Record<string, unknown>) ?? {};
  const featureEngineering = (value.feature_engineering as Record<string, unknown>) ?? {};
  const featureSelection =
    (featureEngineering.feature_selection as Record<string, unknown>) ?? {};
  const modelSelection = (value.model_selection as Record<string, unknown>) ?? {};
  const hyperparameter =
    (value.hyperparameter_optimization as Record<string, unknown>) ?? {};
  const trainingStrategy = (value.training_strategy as Record<string, unknown>) ?? {};
  const evaluation = (value.evaluation_metrics as Record<string, unknown>) ?? {};
  const explainability = (value.explainability as Record<string, unknown>) ?? {};

  const listOfModels = modelSelection.list_of_models_to_train;
  const normalizedModels =
    Array.isArray(listOfModels) && listOfModels.length > 0
      ? listOfModels.filter((item): item is string => typeof item === "string")
      : null;

  return {
    dataset_settings: {
      target_column:
        typeof datasetSettings.target_column === "string" && datasetSettings.target_column.trim()
          ? datasetSettings.target_column
          : null,
      problem_type_override:
        datasetSettings.problem_type_override === "classification" ||
        datasetSettings.problem_type_override === "regression"
          ? datasetSettings.problem_type_override
          : null,
      train_test_split:
        typeof datasetSettings.train_test_split === "number"
          ? datasetSettings.train_test_split
          : defaultConfig.dataset_settings.train_test_split,
      cross_validation_folds:
        typeof datasetSettings.cross_validation_folds === "number"
          ? datasetSettings.cross_validation_folds
          : defaultConfig.dataset_settings.cross_validation_folds,
    },
    data_cleaning: {
      missing_value_strategy:
        typeof dataCleaning.missing_value_strategy === "string"
          ? (dataCleaning.missing_value_strategy as MissingValueStrategy)
          : defaultConfig.data_cleaning.missing_value_strategy,
      categorical_encoding:
        dataCleaning.categorical_encoding === "label"
          ? "label"
          : defaultConfig.data_cleaning.categorical_encoding,
      feature_scaling:
        dataCleaning.feature_scaling === "minmax" || dataCleaning.feature_scaling === "none"
          ? (dataCleaning.feature_scaling as FeatureScaling)
          : defaultConfig.data_cleaning.feature_scaling,
    },
    outlier_removal: {
      method:
        outlierRemoval.method === "zscore" ||
        outlierRemoval.method === "iqr" ||
        outlierRemoval.method === "isolation_forest"
          ? (outlierRemoval.method as OutlierMethod)
          : defaultConfig.outlier_removal.method,
      threshold_parameters: {
        zscore_threshold:
          typeof thresholdParameters.zscore_threshold === "number"
            ? thresholdParameters.zscore_threshold
            : defaultConfig.outlier_removal.threshold_parameters.zscore_threshold,
        iqr_multiplier:
          typeof thresholdParameters.iqr_multiplier === "number"
            ? thresholdParameters.iqr_multiplier
            : defaultConfig.outlier_removal.threshold_parameters.iqr_multiplier,
        isolation_forest_contamination:
          typeof thresholdParameters.isolation_forest_contamination === "number"
            ? thresholdParameters.isolation_forest_contamination
            : defaultConfig.outlier_removal.threshold_parameters.isolation_forest_contamination,
      },
    },
    feature_engineering: {
      log_transform:
        typeof featureEngineering.log_transform === "boolean"
          ? featureEngineering.log_transform
          : defaultConfig.feature_engineering.log_transform,
      polynomial_features:
        typeof featureEngineering.polynomial_features === "boolean"
          ? featureEngineering.polynomial_features
          : defaultConfig.feature_engineering.polynomial_features,
      feature_interactions:
        typeof featureEngineering.feature_interactions === "boolean"
          ? featureEngineering.feature_interactions
          : defaultConfig.feature_engineering.feature_interactions,
      feature_selection: {
        enabled:
          typeof featureSelection.enabled === "boolean"
            ? featureSelection.enabled
            : defaultConfig.feature_engineering.feature_selection.enabled,
        method:
          featureSelection.method === "mutual_information" ||
          featureSelection.method === "recursive_feature_elimination"
            ? (featureSelection.method as FeatureSelectionMethod)
            : defaultConfig.feature_engineering.feature_selection.method,
        k_features:
          typeof featureSelection.k_features === "number"
            ? featureSelection.k_features
            : defaultConfig.feature_engineering.feature_selection.k_features,
      },
    },
    model_selection: {
      list_of_models_to_train: normalizedModels,
    },
    hyperparameter_optimization: {
      optimization_method: "optuna",
      number_of_trials:
        typeof hyperparameter.number_of_trials === "number"
          ? hyperparameter.number_of_trials
          : defaultConfig.hyperparameter_optimization.number_of_trials,
      timeout: typeof hyperparameter.timeout === "number" ? hyperparameter.timeout : null,
      early_stopping:
        typeof hyperparameter.early_stopping === "boolean"
          ? hyperparameter.early_stopping
          : defaultConfig.hyperparameter_optimization.early_stopping,
    },
    training_strategy: {
      parallel_training:
        typeof trainingStrategy.parallel_training === "boolean"
          ? trainingStrategy.parallel_training
          : defaultConfig.training_strategy.parallel_training,
      gpu_usage:
        trainingStrategy.gpu_usage === true || trainingStrategy.gpu_usage === false
          ? (trainingStrategy.gpu_usage as boolean)
          : "auto",
      time_budget: typeof trainingStrategy.time_budget === "number" ? trainingStrategy.time_budget : null,
    },
    evaluation_metrics: {
      primary_metric:
        typeof evaluation.primary_metric === "string"
          ? (evaluation.primary_metric as PrimaryMetric)
          : defaultConfig.evaluation_metrics.primary_metric,
    },
    explainability: {
      enable_shap:
        typeof explainability.enable_shap === "boolean"
          ? explainability.enable_shap
          : defaultConfig.explainability.enable_shap,
    },
  };
}

function validateConfig(config: AutoMLConfig): string | null {
  if (config.dataset_settings.train_test_split <= 0 || config.dataset_settings.train_test_split >= 1) {
    return "Train/test split must be between 0 and 1.";
  }
  if (config.dataset_settings.cross_validation_folds < 2 || config.dataset_settings.cross_validation_folds > 20) {
    return "Cross-validation folds must be between 2 and 20.";
  }
  if (config.hyperparameter_optimization.number_of_trials < 1) {
    return "Number of optimization trials must be at least 1.";
  }
  if (
    config.hyperparameter_optimization.timeout !== null &&
    config.hyperparameter_optimization.timeout <= 0
  ) {
    return "Optimization timeout must be greater than 0 when provided.";
  }
  if (config.training_strategy.time_budget !== null && config.training_strategy.time_budget <= 0) {
    return "Training time budget must be greater than 0 when provided.";
  }
  if (config.outlier_removal.method === "zscore" && config.outlier_removal.threshold_parameters.zscore_threshold <= 0) {
    return "Z-score threshold must be greater than 0.";
  }
  if (config.outlier_removal.method === "iqr" && config.outlier_removal.threshold_parameters.iqr_multiplier <= 0) {
    return "IQR multiplier must be greater than 0.";
  }
  if (
    config.outlier_removal.method === "isolation_forest" &&
    (config.outlier_removal.threshold_parameters.isolation_forest_contamination <= 0 ||
      config.outlier_removal.threshold_parameters.isolation_forest_contamination > 0.5)
  ) {
    return "Isolation Forest contamination must be in the range (0, 0.5].";
  }
  if (config.feature_engineering.feature_selection.enabled && config.feature_engineering.feature_selection.k_features < 1) {
    return "Feature selection k-features must be at least 1 when feature selection is enabled.";
  }
  return null;
}

export default function AutoMLConfigurationPage() {
  const [config, setConfig] = useState<AutoMLConfig>(defaultConfig);
  const [datasetInfo, setDatasetInfo] = useState<StoredDataset | null>(null);
  const [configId, setConfigId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [successMessage, setSuccessMessage] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    const storedConfig = localStorage.getItem("neko-matic.config");
    if (storedConfig) {
      try {
        const parsed = JSON.parse(storedConfig) as unknown;
        setConfig(normalizeConfig(parsed));
      } catch {
        setConfig(defaultConfig);
      }
    }

    const storedConfigId = localStorage.getItem("neko-matic.config_id");
    if (storedConfigId) {
      setConfigId(storedConfigId);
    }

    const storedDataset = localStorage.getItem("neko-matic.dataset");
    if (storedDataset) {
      try {
        const parsedDataset = JSON.parse(storedDataset) as StoredDataset;
        setDatasetInfo(parsedDataset);
      } catch {
        setDatasetInfo(null);
      }
    }
  }, []);

  const previewPayload = useMemo(() => JSON.stringify(config, null, 2), [config]);
  const byteSize = useMemo(() => new Blob([previewPayload]).size, [previewPayload]);

  const selectedModels = config.model_selection.list_of_models_to_train ?? [];
  const effectiveTarget =
    config.dataset_settings.target_column ?? datasetInfo?.target_column_guess ?? "(auto-detect from dataset)";

  function toggleModel(modelName: string) {
    const current = config.model_selection.list_of_models_to_train ?? [];
    const next = current.includes(modelName)
      ? current.filter((value) => value !== modelName)
      : [...current, modelName];

    setConfig((prev) => ({
      ...prev,
      model_selection: {
        list_of_models_to_train: next.length > 0 ? next : null,
      },
    }));
  }

  async function onSubmit(event: FormEvent) {
    event.preventDefault();
    setLoading(true);
    setError(null);
    setSuccessMessage(null);

    try {
      const validationError = validateConfig(config);
      if (validationError) {
        throw new Error(validationError);
      }

      const payload: AutoMLConfig = {
        ...config,
        dataset_settings: {
          ...config.dataset_settings,
          target_column: config.dataset_settings.target_column?.trim() || null,
        },
        model_selection: {
          list_of_models_to_train:
            config.model_selection.list_of_models_to_train &&
            config.model_selection.list_of_models_to_train.length > 0
              ? config.model_selection.list_of_models_to_train
              : null,
        },
      };

      const datasetRaw = localStorage.getItem("neko-matic.dataset");
      if (datasetRaw) {
        const dataset = JSON.parse(datasetRaw) as StoredDataset;
        if (!payload.dataset_settings.target_column) {
          payload.dataset_settings.target_column = dataset.target_column_guess ?? null;
        }
      }

      const result = await uploadConfig(payload);
      setConfigId(result.config_id);
      localStorage.setItem("neko-matic.config_id", result.config_id);
      localStorage.setItem("neko-matic.config", JSON.stringify(payload));
      setSuccessMessage("Configuration uploaded successfully.");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Invalid configuration");
    } finally {
      setLoading(false);
    }
  }

  return (
    <section className="card">
      <h2>AutoML Configuration</h2>
      <p>
        Configure training options with guided form controls. This keeps the same backend
        payload shape while providing validation and safer inputs.
      </p>
      <form onSubmit={onSubmit}>
        <div className="config-grid">
          <fieldset className="form-section">
            <legend>Task Settings</legend>
            <div className="field-row">
              <label htmlFor="target-column">Target Column</label>
              <select
                id="target-column"
                value={config.dataset_settings.target_column ?? ""}
                onChange={(event) =>
                  setConfig((prev) => ({
                    ...prev,
                    dataset_settings: {
                      ...prev.dataset_settings,
                      target_column: event.target.value || null,
                    },
                  }))
                }
              >
                <option value="">Auto-detect</option>
                {(datasetInfo?.columns ?? []).map((column) => (
                  <option key={column} value={column}>
                    {column}
                  </option>
                ))}
              </select>
              <p className="hint">Current effective target: {effectiveTarget}</p>
            </div>

            <div className="field-row">
              <label htmlFor="problem-type">Problem Type Override</label>
              <select
                id="problem-type"
                value={config.dataset_settings.problem_type_override ?? ""}
                onChange={(event) =>
                  setConfig((prev) => ({
                    ...prev,
                    dataset_settings: {
                      ...prev.dataset_settings,
                      problem_type_override:
                        event.target.value === "classification" || event.target.value === "regression"
                          ? (event.target.value as ProblemTypeOverride)
                          : null,
                    },
                  }))
                }
              >
                <option value="">Use profiler default</option>
                <option value="classification">Classification</option>
                <option value="regression">Regression</option>
              </select>
            </div>

            <div className="field-row">
              <label htmlFor="train-test-split">Train/Test Split</label>
              <input
                id="train-test-split"
                type="number"
                min={0.05}
                max={0.95}
                step={0.01}
                value={config.dataset_settings.train_test_split}
                onChange={(event) =>
                  setConfig((prev) => ({
                    ...prev,
                    dataset_settings: {
                      ...prev.dataset_settings,
                      train_test_split: Number(event.target.value),
                    },
                  }))
                }
                required
              />
            </div>

            <div className="field-row">
              <label htmlFor="cv-folds">Cross-Validation Folds</label>
              <input
                id="cv-folds"
                type="number"
                min={2}
                max={20}
                step={1}
                value={config.dataset_settings.cross_validation_folds}
                onChange={(event) =>
                  setConfig((prev) => ({
                    ...prev,
                    dataset_settings: {
                      ...prev.dataset_settings,
                      cross_validation_folds: Number(event.target.value),
                    },
                  }))
                }
                required
              />
            </div>
          </fieldset>

          <fieldset className="form-section">
            <legend>Preprocessing</legend>
            <div className="field-row">
              <label htmlFor="missing-strategy">Missing Value Strategy</label>
              <select
                id="missing-strategy"
                value={config.data_cleaning.missing_value_strategy}
                onChange={(event) =>
                  setConfig((prev) => ({
                    ...prev,
                    data_cleaning: {
                      ...prev.data_cleaning,
                      missing_value_strategy: event.target.value as MissingValueStrategy,
                    },
                  }))
                }
              >
                <option value="median">Median</option>
                <option value="mean">Mean</option>
                <option value="most_frequent">Most frequent</option>
                <option value="constant">Constant</option>
                <option value="drop">Drop (mapped internally)</option>
              </select>
            </div>

            <div className="field-row">
              <label htmlFor="categorical-encoding">Categorical Encoding</label>
              <select
                id="categorical-encoding"
                value={config.data_cleaning.categorical_encoding}
                onChange={(event) =>
                  setConfig((prev) => ({
                    ...prev,
                    data_cleaning: {
                      ...prev.data_cleaning,
                      categorical_encoding: event.target.value as CategoricalEncoding,
                    },
                  }))
                }
              >
                <option value="onehot">One-hot</option>
                <option value="label">Label encoding</option>
              </select>
            </div>

            <div className="field-row">
              <label htmlFor="feature-scaling">Feature Scaling</label>
              <select
                id="feature-scaling"
                value={config.data_cleaning.feature_scaling}
                onChange={(event) =>
                  setConfig((prev) => ({
                    ...prev,
                    data_cleaning: {
                      ...prev.data_cleaning,
                      feature_scaling: event.target.value as FeatureScaling,
                    },
                  }))
                }
              >
                <option value="standard">Standard scaler</option>
                <option value="minmax">Min-max scaler</option>
                <option value="none">None</option>
              </select>
            </div>

            <div className="field-row">
              <label htmlFor="outlier-method">Outlier Removal Method</label>
              <select
                id="outlier-method"
                value={config.outlier_removal.method}
                onChange={(event) =>
                  setConfig((prev) => ({
                    ...prev,
                    outlier_removal: {
                      ...prev.outlier_removal,
                      method: event.target.value as OutlierMethod,
                    },
                  }))
                }
              >
                <option value="none">None</option>
                <option value="zscore">Z-score</option>
                <option value="iqr">IQR</option>
                <option value="isolation_forest">Isolation Forest</option>
              </select>
            </div>

            <div className="field-row">
              <label htmlFor="zscore-threshold">Z-score Threshold</label>
              <input
                id="zscore-threshold"
                type="number"
                min={0.1}
                step={0.1}
                value={config.outlier_removal.threshold_parameters.zscore_threshold}
                onChange={(event) =>
                  setConfig((prev) => ({
                    ...prev,
                    outlier_removal: {
                      ...prev.outlier_removal,
                      threshold_parameters: {
                        ...prev.outlier_removal.threshold_parameters,
                        zscore_threshold: Number(event.target.value),
                      },
                    },
                  }))
                }
                required={config.outlier_removal.method === "zscore"}
              />
            </div>

            <div className="field-row">
              <label htmlFor="iqr-multiplier">IQR Multiplier</label>
              <input
                id="iqr-multiplier"
                type="number"
                min={0.1}
                step={0.1}
                value={config.outlier_removal.threshold_parameters.iqr_multiplier}
                onChange={(event) =>
                  setConfig((prev) => ({
                    ...prev,
                    outlier_removal: {
                      ...prev.outlier_removal,
                      threshold_parameters: {
                        ...prev.outlier_removal.threshold_parameters,
                        iqr_multiplier: Number(event.target.value),
                      },
                    },
                  }))
                }
                required={config.outlier_removal.method === "iqr"}
              />
            </div>

            <div className="field-row">
              <label htmlFor="iforest-contamination">Isolation Forest Contamination</label>
              <input
                id="iforest-contamination"
                type="number"
                min={0.001}
                max={0.5}
                step={0.001}
                value={config.outlier_removal.threshold_parameters.isolation_forest_contamination}
                onChange={(event) =>
                  setConfig((prev) => ({
                    ...prev,
                    outlier_removal: {
                      ...prev.outlier_removal,
                      threshold_parameters: {
                        ...prev.outlier_removal.threshold_parameters,
                        isolation_forest_contamination: Number(event.target.value),
                      },
                    },
                  }))
                }
                required={config.outlier_removal.method === "isolation_forest"}
              />
            </div>
          </fieldset>

          <fieldset className="form-section">
            <legend>Feature Engineering</legend>

            <label className="inline-toggle" htmlFor="log-transform">
              <input
                id="log-transform"
                type="checkbox"
                checked={config.feature_engineering.log_transform}
                onChange={(event) =>
                  setConfig((prev) => ({
                    ...prev,
                    feature_engineering: {
                      ...prev.feature_engineering,
                      log_transform: event.target.checked,
                    },
                  }))
                }
              />
              Enable log transform
            </label>

            <label className="inline-toggle" htmlFor="poly-features">
              <input
                id="poly-features"
                type="checkbox"
                checked={config.feature_engineering.polynomial_features}
                onChange={(event) =>
                  setConfig((prev) => ({
                    ...prev,
                    feature_engineering: {
                      ...prev.feature_engineering,
                      polynomial_features: event.target.checked,
                    },
                  }))
                }
              />
              Enable polynomial features
            </label>

            <label className="inline-toggle" htmlFor="feature-interactions">
              <input
                id="feature-interactions"
                type="checkbox"
                checked={config.feature_engineering.feature_interactions}
                onChange={(event) =>
                  setConfig((prev) => ({
                    ...prev,
                    feature_engineering: {
                      ...prev.feature_engineering,
                      feature_interactions: event.target.checked,
                    },
                  }))
                }
              />
              Enable feature interactions
            </label>

            <label className="inline-toggle" htmlFor="feature-selection-enabled">
              <input
                id="feature-selection-enabled"
                type="checkbox"
                checked={config.feature_engineering.feature_selection.enabled}
                onChange={(event) =>
                  setConfig((prev) => ({
                    ...prev,
                    feature_engineering: {
                      ...prev.feature_engineering,
                      feature_selection: {
                        ...prev.feature_engineering.feature_selection,
                        enabled: event.target.checked,
                      },
                    },
                  }))
                }
              />
              Enable feature selection
            </label>

            <div className="field-row">
              <label htmlFor="feature-selection-method">Feature Selection Method</label>
              <select
                id="feature-selection-method"
                value={config.feature_engineering.feature_selection.method}
                onChange={(event) =>
                  setConfig((prev) => ({
                    ...prev,
                    feature_engineering: {
                      ...prev.feature_engineering,
                      feature_selection: {
                        ...prev.feature_engineering.feature_selection,
                        method: event.target.value as FeatureSelectionMethod,
                      },
                    },
                  }))
                }
                disabled={!config.feature_engineering.feature_selection.enabled}
              >
                <option value="variance_threshold">Variance threshold</option>
                <option value="mutual_information">Mutual information</option>
                <option value="recursive_feature_elimination">Recursive feature elimination</option>
              </select>
            </div>

            <div className="field-row">
              <label htmlFor="k-features">Top K Features</label>
              <input
                id="k-features"
                type="number"
                min={1}
                step={1}
                value={config.feature_engineering.feature_selection.k_features}
                onChange={(event) =>
                  setConfig((prev) => ({
                    ...prev,
                    feature_engineering: {
                      ...prev.feature_engineering,
                      feature_selection: {
                        ...prev.feature_engineering.feature_selection,
                        k_features: Number(event.target.value),
                      },
                    },
                  }))
                }
                disabled={!config.feature_engineering.feature_selection.enabled}
                required={config.feature_engineering.feature_selection.enabled}
              />
            </div>
          </fieldset>

          <fieldset className="form-section">
            <legend>Model Selection</legend>
            <p className="hint">
              Select one or more models to force training. Leave all unchecked to use backend
              auto-recommendation.
            </p>
            <div className="checkbox-grid">
              {MODEL_OPTIONS.map((modelName) => (
                <label className="inline-toggle" key={modelName} htmlFor={`model-${modelName}`}>
                  <input
                    id={`model-${modelName}`}
                    type="checkbox"
                    checked={selectedModels.includes(modelName)}
                    onChange={() => toggleModel(modelName)}
                  />
                  {modelName}
                </label>
              ))}
            </div>
          </fieldset>

          <fieldset className="form-section">
            <legend>Optimization</legend>
            <div className="field-row">
              <label htmlFor="optimization-method">Optimization Method</label>
              <select
                id="optimization-method"
                value={config.hyperparameter_optimization.optimization_method}
                onChange={(event) =>
                  setConfig((prev) => ({
                    ...prev,
                    hyperparameter_optimization: {
                      ...prev.hyperparameter_optimization,
                      optimization_method: event.target.value as OptimizationMethod,
                    },
                  }))
                }
              >
                <option value="optuna">Optuna</option>
              </select>
            </div>

            <div className="field-row">
              <label htmlFor="num-trials">Number of Trials</label>
              <input
                id="num-trials"
                type="number"
                min={1}
                max={1000}
                step={1}
                value={config.hyperparameter_optimization.number_of_trials}
                onChange={(event) =>
                  setConfig((prev) => ({
                    ...prev,
                    hyperparameter_optimization: {
                      ...prev.hyperparameter_optimization,
                      number_of_trials: Number(event.target.value),
                    },
                  }))
                }
                required
              />
            </div>

            <div className="field-row">
              <label htmlFor="optimization-timeout">Optimization Timeout (seconds)</label>
              <input
                id="optimization-timeout"
                type="number"
                min={1}
                step={1}
                value={config.hyperparameter_optimization.timeout ?? ""}
                onChange={(event) =>
                  setConfig((prev) => ({
                    ...prev,
                    hyperparameter_optimization: {
                      ...prev.hyperparameter_optimization,
                      timeout: event.target.value ? Number(event.target.value) : null,
                    },
                  }))
                }
                placeholder="Optional"
              />
            </div>

            <label className="inline-toggle" htmlFor="early-stopping">
              <input
                id="early-stopping"
                type="checkbox"
                checked={config.hyperparameter_optimization.early_stopping}
                onChange={(event) =>
                  setConfig((prev) => ({
                    ...prev,
                    hyperparameter_optimization: {
                      ...prev.hyperparameter_optimization,
                      early_stopping: event.target.checked,
                    },
                  }))
                }
              />
              Enable early stopping
            </label>

            <label className="inline-toggle" htmlFor="parallel-training">
              <input
                id="parallel-training"
                type="checkbox"
                checked={config.training_strategy.parallel_training}
                onChange={(event) =>
                  setConfig((prev) => ({
                    ...prev,
                    training_strategy: {
                      ...prev.training_strategy,
                      parallel_training: event.target.checked,
                    },
                  }))
                }
              />
              Parallel training strategy
            </label>

            <div className="field-row">
              <label htmlFor="gpu-usage">GPU Usage</label>
              <select
                id="gpu-usage"
                value={String(config.training_strategy.gpu_usage)}
                onChange={(event) =>
                  setConfig((prev) => ({
                    ...prev,
                    training_strategy: {
                      ...prev.training_strategy,
                      gpu_usage:
                        event.target.value === "auto"
                          ? "auto"
                          : event.target.value === "true",
                    },
                  }))
                }
              >
                <option value="auto">Auto</option>
                <option value="true">Enabled</option>
                <option value="false">Disabled</option>
              </select>
            </div>

            <div className="field-row">
              <label htmlFor="time-budget">Training Time Budget (seconds)</label>
              <input
                id="time-budget"
                type="number"
                min={1}
                step={1}
                value={config.training_strategy.time_budget ?? ""}
                onChange={(event) =>
                  setConfig((prev) => ({
                    ...prev,
                    training_strategy: {
                      ...prev.training_strategy,
                      time_budget: event.target.value ? Number(event.target.value) : null,
                    },
                  }))
                }
                placeholder="Optional"
              />
            </div>
          </fieldset>

          <fieldset className="form-section">
            <legend>Evaluation</legend>
            <div className="field-row">
              <label htmlFor="primary-metric">Primary Metric</label>
              <select
                id="primary-metric"
                value={config.evaluation_metrics.primary_metric}
                onChange={(event) =>
                  setConfig((prev) => ({
                    ...prev,
                    evaluation_metrics: {
                      primary_metric: event.target.value as PrimaryMetric,
                    },
                  }))
                }
              >
                <option value="accuracy">Accuracy</option>
                <option value="f1">F1</option>
                <option value="f1 score">F1 score</option>
                <option value="precision">Precision</option>
                <option value="recall">Recall</option>
                <option value="roc_auc">ROC AUC</option>
                <option value="rmse">RMSE</option>
                <option value="mse">MSE</option>
                <option value="mae">MAE</option>
                <option value="r2">R2</option>
              </select>
            </div>

            <label className="inline-toggle" htmlFor="enable-shap">
              <input
                id="enable-shap"
                type="checkbox"
                checked={config.explainability.enable_shap}
                onChange={(event) =>
                  setConfig((prev) => ({
                    ...prev,
                    explainability: {
                      enable_shap: event.target.checked,
                    },
                  }))
                }
              />
              Enable SHAP explainability
            </label>
          </fieldset>
        </div>

        <div className="field-row" style={{ marginTop: "1rem" }}>
          <label htmlFor="config-preview">Payload Preview ({byteSize} bytes)</label>
          <textarea id="config-preview" rows={10} value={previewPayload} readOnly />
        </div>

        <br />
        <button type="submit" disabled={loading}>
          {loading ? "Uploading..." : "Upload Configuration"}
        </button>
      </form>
      {configId && (
        <p>
          Config ID: <code>{configId}</code>
        </p>
      )}
      {successMessage && <p className="status-ok">{successMessage}</p>}
      {error && <p className="status-error">{error}</p>}
    </section>
  );
}
