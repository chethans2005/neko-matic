"use client";

import { FormEvent, useEffect, useState, useMemo } from "react";
import {
  uploadDataset,
  uploadConfig,
  setDefaultConfig,
  startRun,
  getActiveRunStatus,
  getActiveLeaderboard,
  getActiveFeatureImportance,
  downloadActiveModelUrl,
  downloadActiveArtifactUrl,
} from "@/lib/api";
import { useDataset } from "@/lib/DatasetContext";
import { GuidedDataExplorer } from "@/components/GuidedDataExplorer";

type ProblemTypeOverride = "classification" | "regression" | null;
type MissingValueStrategy = "median" | "mean" | "most_frequent" | "constant" | "drop";
type CategoricalEncoding = "onehot" | "label";
type FeatureScaling = "standard" | "minmax" | "none";
type OutlierMethod = "none" | "zscore" | "iqr" | "isolation_forest";
type FeatureSelectionMethod = "variance_threshold" | "mutual_information" | "recursive_feature_elimination";
type OptimizationMethod = "optuna";
type GpuUsage = "auto" | boolean;
type PrimaryMetric = "accuracy" | "f1" | "f1 score" | "precision" | "recall" | "roc_auc" | "rmse" | "mse" | "mae" | "r2";

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
    threshold_parameters: { zscore_threshold: 3, iqr_multiplier: 1.5, isolation_forest_contamination: 0.05 },
  },
  feature_engineering: {
    log_transform: false,
    polynomial_features: false,
    feature_interactions: false,
    feature_selection: { enabled: false, method: "variance_threshold", k_features: 20 },
  },
  model_selection: { list_of_models_to_train: null },
  hyperparameter_optimization: { optimization_method: "optuna", number_of_trials: 20, timeout: null, early_stopping: true },
  training_strategy: { parallel_training: false, gpu_usage: "auto", time_budget: null },
  evaluation_metrics: { primary_metric: "accuracy" },
  explainability: { enable_shap: true },
};

export default function TrainingPage() {
  const { dataset, setDataset, run, setRunStatus, resetRun } = useDataset();
  
  // UI state
  const [uploadFile, setUploadFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  
  // Configuration state
  const [config, setConfig] = useState<AutoMLConfig>(defaultConfig);
  const [configError, setConfigError] = useState<string | null>(null);
  const [configSuccess, setConfigSuccess] = useState(false);
  
  // Training state
  const [launching, setLaunching] = useState(false);
  const [launchError, setLaunchError] = useState<string | null>(null);
  const [leaderboard, setLeaderboard] = useState<Array<Record<string, unknown>>>([]);
  const [featureImportance, setFeatureImportance] = useState<Record<string, unknown> | null>(null);
  
  // Tab state
  const [activeTab, setActiveTab] = useState<"upload" | "explore" | "config" | "monitor">("upload");

  // Load default config from localStorage
  useEffect(() => {
    const stored = localStorage.getItem("neko-matic.default_config");
    if (stored) {
      try {
        setConfig(JSON.parse(stored));
      } catch {
        // Ignore parse errors
      }
    }
  }, []);

  // Poll for active run status
  useEffect(() => {
    if (run.status === "none" || run.status === "completed" || run.status === "failed") {
      return;
    }
    const timer = setInterval(async () => {
      try {
        const status = await getActiveRunStatus();
        setRunStatus(status);
        if (status.status === "completed" || status.status === "failed") {
          clearInterval(timer);
        }
      } catch (err) {
        console.error("Failed to fetch status:", err);
      }
    }, 2500);
    return () => clearInterval(timer);
  }, [run.status, setRunStatus]);

  // Fetch leaderboard when run completes
  useEffect(() => {
    if (run.status === "completed") {
      getActiveLeaderboard()
        .then((data) => setLeaderboard(data.leaderboard || []))
        .catch((err) => console.error("Failed to fetch leaderboard:", err));
      
      getActiveFeatureImportance()
        .then((data) => setFeatureImportance(data))
        .catch((err) => console.error("Failed to fetch feature importance:", err));
    }
  }, [run.status]);

  // Handle dataset upload
  async function handleUpload(event: FormEvent) {
    event.preventDefault();
    if (!uploadFile) return;

    setUploading(true);
    setUploadError(null);
    try {
      const payload = await uploadDataset(uploadFile);
      setDataset(payload);
      setUploadFile(null);
      setActiveTab("explore");
    } catch (err) {
      setUploadError(err instanceof Error ? err.message : "Upload failed");
    } finally {
      setUploading(false);
    }
  }

  // Handle config submit
  async function handleConfigSubmit(event: FormEvent) {
    event.preventDefault();
    setConfigError(null);
    setConfigSuccess(false);

    try {
      const payload = {
        ...config,
        dataset_settings: {
          ...config.dataset_settings,
          target_column: config.dataset_settings.target_column?.trim() || dataset?.target_column_guess || null,
        },
      };
      await uploadConfig(payload);
      setConfigSuccess(true);
      localStorage.setItem("neko-matic.config", JSON.stringify(payload));
    } catch (err) {
      setConfigError(err instanceof Error ? err.message : "Configuration failed");
    }
  }

  // Handle save as default
  async function handleSaveDefault() {
    try {
      await setDefaultConfig(config);
      localStorage.setItem("neko-matic.default_config", JSON.stringify(config));
      alert("✓ Default configuration saved!");
    } catch (err) {
      alert(`Failed to save default: ${err instanceof Error ? err.message : "Unknown error"}`);
    }
  }

  // Handle launch training
  async function handleLaunchTraining() {
    if (!dataset) {
      setLaunchError("No dataset uploaded");
      return;
    }

    setLaunching(true);
    setLaunchError(null);
    resetRun();

    try {
      const payload = await startRun(config);
      setRunStatus({ run_id: payload.run_id, status: "queued" });
      setActiveTab("monitor");
    } catch (err) {
      setLaunchError(err instanceof Error ? err.message : "Failed to start run");
    } finally {
      setLaunching(false);
    }
  }

  // Effects for config updates
  useEffect(() => {
    if (dataset && !config.dataset_settings.target_column) {
      setConfig((prev) => ({
        ...prev,
        dataset_settings: {
          ...prev.dataset_settings,
          target_column: dataset.target_column_guess,
        },
      }));
    }
  }, [dataset]);

  const effectiveTarget = config.dataset_settings.target_column || dataset?.target_column_guess || "(auto-detect)";

  return (
    <div className="training-page">
      <h2>Unified AutoML Training</h2>
      <p>Upload data, review insights, configure pipeline, and run AutoML—all in one page.</p>

      {/* Tabs */}
      <div className="tabs">
        <button
          className={`tab ${activeTab === "upload" ? "active" : ""}`}
          onClick={() => setActiveTab("upload")}
        >
          1. Upload
        </button>
        <button
          className={`tab ${activeTab === "explore" ? "active" : ""} ${!dataset ? "disabled" : ""}`}
          onClick={() => dataset && setActiveTab("explore")}
        >
          2. Explore
        </button>
        <button
          className={`tab ${activeTab === "config" ? "active" : ""} ${!dataset ? "disabled" : ""}`}
          onClick={() => dataset && setActiveTab("config")}
        >
          3. Configure
        </button>
        <button
          className={`tab ${activeTab === "monitor" ? "active" : ""} ${run.status === "none" ? "disabled" : ""}`}
          onClick={() => run.status !== "none" && setActiveTab("monitor")}
        >
          4. Monitor
        </button>
      </div>

      {/* Tab Content */}
      <div className="tab-content">
        {/* Upload Tab */}
        {activeTab === "upload" && (
          <section className="card">
            <h3>Dataset Upload</h3>
            <p>Upload a CSV or Excel file to get started.</p>
            <form onSubmit={handleUpload}>
              <label htmlFor="dataset-file">Dataset File</label>
              <input
                id="dataset-file"
                type="file"
                accept=".csv,.xlsx,.xls"
                required
                onChange={(e) => setUploadFile(e.target.files?.[0] ?? null)}
              />
              <button type="submit" disabled={!uploadFile || uploading}>
                {uploading ? "Uploading..." : "Upload Dataset"}
              </button>
            </form>
            {uploadError && <p style={{ color: "red" }}>{uploadError}</p>}

            {dataset && (
              <div className="upload-success">
                <h4>✓ Dataset Loaded</h4>
                <p>
                  <strong>{dataset.filename}</strong> ({dataset.shape[0]} rows × {dataset.shape[1]} columns)
                </p>
                <button onClick={() => setActiveTab("explore")}>Review Insights →</button>
              </div>
            )}
          </section>
        )}

        {/* Explore Tab */}
        {activeTab === "explore" && dataset && (
          <section className="card">
            <GuidedDataExplorer
              profile={dataset.profile}
              shape={dataset.shape}
              columns={dataset.columns}
              targetColumn={dataset.target_column_guess}
            />
            <div style={{ marginTop: "2rem" }}>
              <button onClick={() => setActiveTab("config")}>Configure Pipeline →</button>
            </div>
          </section>
        )}

        {/* Config Tab */}
        {activeTab === "config" && dataset && (
          <section className="card">
            <h3>AutoML Configuration</h3>
            <form onSubmit={handleConfigSubmit}>
              <div className="config-grid">
                {/* Task Settings */}
                <fieldset className="form-section">
                  <legend>Task Settings</legend>
                  <div className="field-row">
                    <label htmlFor="target-column">Target Column</label>
                    <select
                      id="target-column"
                      value={config.dataset_settings.target_column ?? ""}
                      onChange={(e) =>
                        setConfig((prev) => ({
                          ...prev,
                          dataset_settings: {
                            ...prev.dataset_settings,
                            target_column: e.target.value || null,
                          },
                        }))
                      }
                    >
                      <option value="">Auto-detect</option>
                      {dataset.columns.map((col) => (
                        <option key={col} value={col}>
                          {col}
                        </option>
                      ))}
                    </select>
                    <p className="hint">Effective target: {effectiveTarget}</p>
                  </div>

                  <div className="field-row">
                    <label htmlFor="problem-type">Problem Type Override</label>
                    <select
                      id="problem-type"
                      value={config.dataset_settings.problem_type_override ?? ""}
                      onChange={(e) =>
                        setConfig((prev) => ({
                          ...prev,
                          dataset_settings: {
                            ...prev.dataset_settings,
                            problem_type_override:
                              (e.target.value as ProblemTypeOverride) || null,
                          },
                        }))
                      }
                    >
                      <option value="">Profiler Default</option>
                      <option value="classification">Classification</option>
                      <option value="regression">Regression</option>
                    </select>
                  </div>

                  <div className="field-row">
                    <label htmlFor="train-split">Train/Test Split</label>
                    <input
                      id="train-split"
                      type="number"
                      min={0.05}
                      max={0.95}
                      step={0.01}
                      value={config.dataset_settings.train_test_split}
                      onChange={(e) =>
                        setConfig((prev) => ({
                          ...prev,
                          dataset_settings: {
                            ...prev.dataset_settings,
                            train_test_split: Number(e.target.value),
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
                      onChange={(e) =>
                        setConfig((prev) => ({
                          ...prev,
                          dataset_settings: {
                            ...prev.dataset_settings,
                            cross_validation_folds: Number(e.target.value),
                          },
                        }))
                      }
                      required
                    />
                  </div>
                </fieldset>

                {/* Preprocessing */}
                <fieldset className="form-section">
                  <legend>Preprocessing</legend>
                  <div className="field-row">
                    <label htmlFor="missing-strategy">Missing Value Strategy</label>
                    <select
                      id="missing-strategy"
                      value={config.data_cleaning.missing_value_strategy}
                      onChange={(e) =>
                        setConfig((prev) => ({
                          ...prev,
                          data_cleaning: {
                            ...prev.data_cleaning,
                            missing_value_strategy: e.target.value as MissingValueStrategy,
                          },
                        }))
                      }
                    >
                      <option value="median">Median</option>
                      <option value="mean">Mean</option>
                      <option value="most_frequent">Most Frequent</option>
                      <option value="constant">Constant</option>
                      <option value="drop">Drop</option>
                    </select>
                  </div>

                  <div className="field-row">
                    <label htmlFor="categorical-enc">Categorical Encoding</label>
                    <select
                      id="categorical-enc"
                      value={config.data_cleaning.categorical_encoding}
                      onChange={(e) =>
                        setConfig((prev) => ({
                          ...prev,
                          data_cleaning: {
                            ...prev.data_cleaning,
                            categorical_encoding: e.target.value as CategoricalEncoding,
                          },
                        }))
                      }
                    >
                      <option value="onehot">One-hot</option>
                      <option value="label">Label</option>
                    </select>
                  </div>

                  <div className="field-row">
                    <label htmlFor="feature-scale">Feature Scaling</label>
                    <select
                      id="feature-scale"
                      value={config.data_cleaning.feature_scaling}
                      onChange={(e) =>
                        setConfig((prev) => ({
                          ...prev,
                          data_cleaning: {
                            ...prev.data_cleaning,
                            feature_scaling: e.target.value as FeatureScaling,
                          },
                        }))
                      }
                    >
                      <option value="standard">Standard</option>
                      <option value="minmax">Min-Max</option>
                      <option value="none">None</option>
                    </select>
                  </div>
                </fieldset>

                {/* Hyperparameter Optimization */}
                <fieldset className="form-section">
                  <legend>Hyperparameter Optimization</legend>
                  <div className="field-row">
                    <label htmlFor="n-trials">Number of Trials</label>
                    <input
                      id="n-trials"
                      type="number"
                      min={1}
                      max={200}
                      value={config.hyperparameter_optimization.number_of_trials}
                      onChange={(e) =>
                        setConfig((prev) => ({
                          ...prev,
                          hyperparameter_optimization: {
                            ...prev.hyperparameter_optimization,
                            number_of_trials: Number(e.target.value),
                          },
                        }))
                      }
                      required
                    />
                  </div>

                  <div className="field-row">
                    <label htmlFor="early-stop">
                      <input
                        id="early-stop"
                        type="checkbox"
                        checked={config.hyperparameter_optimization.early_stopping}
                        onChange={(e) =>
                          setConfig((prev) => ({
                            ...prev,
                            hyperparameter_optimization: {
                              ...prev.hyperparameter_optimization,
                              early_stopping: e.target.checked,
                            },
                          }))
                        }
                      />
                      Early Stopping
                    </label>
                  </div>
                </fieldset>

                {/* Explainability */}
                <fieldset className="form-section">
                  <legend>Explainability</legend>
                  <div className="field-row">
                    <label htmlFor="enable-shap">
                      <input
                        id="enable-shap"
                        type="checkbox"
                        checked={config.explainability.enable_shap}
                        onChange={(e) =>
                          setConfig((prev) => ({
                            ...prev,
                            explainability: { enable_shap: e.target.checked },
                          }))
                        }
                      />
                      Enable SHAP Explanations
                    </label>
                  </div>
                </fieldset>
              </div>

              {configError && <p style={{ color: "red" }}>{configError}</p>}
              {configSuccess && <p style={{ color: "green" }}>✓ Configuration saved</p>}

              <div style={{ display: "flex", gap: "1rem", marginTop: "1rem" }}>
                <button type="submit">Save Configuration</button>
                <button type="button" onClick={handleSaveDefault} style={{ background: "#1f7a4a" }}>
                  Save as Default
                </button>
              </div>
            </form>

            <div style={{ marginTop: "2rem" }}>
              <button disabled={!dataset || launching} onClick={handleLaunchTraining}>
                {launching ? "Starting..." : "Start Training →"}
              </button>
              {launchError && <p style={{ color: "red" }}>{launchError}</p>}
            </div>
          </section>
        )}

        {/* Monitor Tab */}
        {activeTab === "monitor" && run.run_id && (
          <section className="card">
            <h3>Training Monitor</h3>
            <div className="status-grid">
              <div>
                <strong>Status</strong>
                <p>{run.status.toUpperCase()}</p>
              </div>
              <div>
                <strong>Progress</strong>
                <p>{Number(run.progress).toFixed(1)}%</p>
              </div>
              <div>
                <strong>Message</strong>
                <p>{run.message}</p>
              </div>
              <div>
                <strong>Best Model</strong>
                <p>{run.best_model_name || "-"}</p>
              </div>
            </div>
            <progress value={run.progress} max={100} style={{ width: "100%", marginBottom: "1rem" }} />

            {run.status === "completed" && (
              <>
                <h4>Results</h4>
                <p>
                  <strong>Best Score:</strong> {run.best_score?.toFixed(4) ?? "-"}
                </p>

                {leaderboard.length > 0 && (
                  <div>
                    <h5>Leaderboard</h5>
                    <table style={{ width: "100%", marginTop: "0.5rem" }}>
                      <thead>
                        <tr>
                          <th>Model</th>
                          <th>Metric</th>
                          <th>Time (s)</th>
                        </tr>
                      </thead>
                      <tbody>
                        {leaderboard.slice(0, 10).map((row: any, idx) => (
                          <tr key={idx}>
                            <td>{row.model_name || "-"}</td>
                            <td>{Number(row.metric_value ?? 0).toFixed(4)}</td>
                            <td>{Number(row.training_time ?? 0).toFixed(2)}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}

                <div style={{ marginTop: "1rem", display: "flex", gap: "1rem" }}>
                  <a href={downloadActiveModelUrl()} download="best_model.pkl">
                    <button>Download Model</button>
                  </a>
                  <a href={downloadActiveArtifactUrl("training_report.json")} download="training_report.json">
                    <button>Download Report</button>
                  </a>
                  <button onClick={() => { resetRun(); setActiveTab("upload"); }}>
                    Start New Run
                  </button>
                </div>
              </>
            )}

            {run.status === "failed" && (
              <p style={{ color: "red" }}>
                <strong>Training Failed:</strong> {run.message}
              </p>
            )}
          </section>
        )}
      </div>

      {/* CSS-in-JSX styles for tabs and layout */}
      <style>{`
        .training-page {
          max-width: 1200px;
          margin: 0 auto;
        }

        .tabs {
          display: flex;
          gap: 0.5rem;
          margin: 2rem 0 1rem 0;
          border-bottom: 2px solid #e0e0e0;
        }

        .tab {
          padding: 0.75rem 1rem;
          border: none;
          background: transparent;
          cursor: pointer;
          font-size: 1rem;
          border-bottom: 2px solid transparent;
          transition: all 0.2s;
        }

        .tab.active {
          border-bottom-color: #1f7a4a;
          color: #1f7a4a;
          font-weight: 600;
        }

        .tab.disabled {
          cursor: not-allowed;
          opacity: 0.5;
        }

        .tab-content {
          animation: fadeIn 0.2s ease-in;
        }

        @keyframes fadeIn {
          from {
            opacity: 0;
          }
          to {
            opacity: 1;
          }
        }

        .upload-success {
          background: #f0fdf4;
          border: 1px solid #86efac;
          border-radius: 0.5rem;
          padding: 1rem;
          margin-top: 1rem;
        }

        .config-grid {
          display: grid;
          grid-template-columns: 1fr 1fr;
          gap: 1.5rem;
          margin-bottom: 1.5rem;
        }

        .form-section {
          border: 1px solid #e0e0e0;
          border-radius: 0.5rem;
          padding: 1rem;
        }

        .form-section legend {
          font-weight: 600;
          padding: 0 0.5rem;
        }

        .field-row {
          margin-bottom: 1rem;
        }

        .field-row label {
          display: block;
          margin-bottom: 0.5rem;
          font-weight: 500;
        }

        .field-row input,
        .field-row select {
          width: 100%;
          padding: 0.5rem;
          border: 1px solid #ccc;
          border-radius: 0.25rem;
        }

        .hint {
          font-size: 0.875rem;
          color: #666;
          margin-top: 0.25rem;
        }

        .status-grid {
          display: grid;
          grid-template-columns: repeat(4, 1fr);
          gap: 1rem;
          margin-bottom: 1.5rem;
          padding: 1rem;
          background: #f9f9f9;
          border-radius: 0.5rem;
        }

        .status-grid div strong {
          display: block;
          font-size: 0.875rem;
          color: #666;
          margin-bottom: 0.5rem;
        }

        .status-grid div p {
          margin: 0;
          font-size: 1.25rem;
          font-weight: 600;
        }

        .explorer-section {
          margin-bottom: 2rem;
        }

        .summary-grid {
          display: grid;
          grid-template-columns: repeat(4, 1fr);
          gap: 1rem;
          margin: 1.5rem 0;
          padding: 1rem;
          background: #f9f9f9;
          border-radius: 0.5rem;
        }

        .summary-grid div {
          text-align: center;
        }

        .summary-grid strong {
          display: block;
          font-size: 0.875rem;
          color: #666;
          margin-bottom: 0.5rem;
        }

        .summary-grid p {
          margin: 0;
          font-size: 1.1rem;
          font-weight: 600;
        }

        .guides-container {
          margin: 1.5rem 0;
        }

        .guides-container h4 {
          margin-bottom: 1rem;
        }

        .guide-card {
          background: #fff3cd;
          border: 1px solid #ffc107;
          border-radius: 0.5rem;
          padding: 1rem;
          margin-bottom: 0.75rem;
        }

        .guide-card h5 {
          margin: 0 0 0.5rem 0;
          color: #856404;
        }

        .guide-description {
          margin: 0.5rem 0;
          font-size: 0.95rem;
          color: #856404;
        }

        .guide-recommendation {
          margin: 0.5rem 0 0 0;
          font-size: 0.9rem;
          color: #856404;
        }

        .visualizations-grid {
          display: grid;
          grid-template-columns: repeat(2, 1fr);
          gap: 1.5rem;
          margin-top: 1.5rem;
        }

        .viz-card {
          border: 1px solid #e0e0e0;
          border-radius: 0.5rem;
          padding: 1rem;
          background: #fafafa;
        }

        .viz-card h4 {
          margin-top: 0;
        }

        @media (max-width: 768px) {
          .config-grid,
          .summary-grid,
          .visualizations-grid,
          .status-grid {
            grid-template-columns: 1fr;
          }
        }
      `}</style>
    </div>
  );
}
