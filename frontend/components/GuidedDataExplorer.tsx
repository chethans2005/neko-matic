"use client";

import { useMemo } from "react";
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { DatasetProfile } from "@/lib/DatasetContext";

export type ExplorationGuide = {
  title: string;
  description: string;
  recommendation: string;
};

interface GuidedDataExplorerProps {
  profile: DatasetProfile;
  shape: [number, number];
  columns: string[];
  targetColumn: string;
}

export function GuidedDataExplorer({
  profile,
  shape,
  columns,
  targetColumn,
}: GuidedDataExplorerProps) {
  // Extract and prepare data for visualizations
  const missingSeries = useMemo(() => {
    const rows = profile.missing_info ?? [];
    return rows.slice(0, 12).map((item: any) => ({
      column: item.column,
      missing: Number(item.missing_count ?? 0),
    }));
  }, [profile]);

  const classSeries = useMemo(() => {
    const distribution = profile.class_distribution ?? {};
    return Object.entries(distribution).map(([label, value]) => ({
      label,
      pct: Number(value) * 100,
    }));
  }, [profile]);

  // Generate exploration guides based on data characteristics
  const guides: ExplorationGuide[] = useMemo(() => {
    const result: ExplorationGuide[] = [];

    // Guide 1: Missing values strategy
    const totalMissing = (profile.missing_info ?? []).reduce(
      (sum, item: any) => sum + (item.missing_count ?? 0),
      0
    );
    if (totalMissing > 0) {
      result.push({
        title: "Handle Missing Values",
        description: `Found ${totalMissing} missing values across columns.`,
        recommendation:
          "Consider median imputation for numerical features or mode for categorical. Enable feature-specific strategies in configuration.",
      });
    } else {
      result.push({
        title: "No Missing Values Detected",
        description: "Your dataset is clean with no missing values.",
        recommendation:
          "You can proceed directly to model training or enable data cleaning options for robustness.",
      });
    }

    // Guide 2: Class balance (for classification)
    if (profile.problem_type === "classification" && classSeries.length > 0) {
      const minPct = Math.min(...classSeries.map((s) => s.pct));
      const maxPct = Math.max(...classSeries.map((s) => s.pct));
      if (maxPct - minPct > 50) {
        result.push({
          title: "Class Imbalance Detected",
          description: `Target class ranges from ${minPct.toFixed(1)}% to ${maxPct.toFixed(1)}%.`,
          recommendation:
            "Consider enabling stratified cross-validation or adjusting evaluation metrics to F1-score for balanced assessment.",
        });
      } else {
        result.push({
          title: "Balanced Target Classes",
          description: `Classes are evenly distributed (${minPct.toFixed(1)}% – ${maxPct.toFixed(1)}%).`,
          recommendation: "Accuracy is a suitable primary metric. Proceed with standard training.",
        });
      }
    }

    // Guide 3: Row count guidance
    const rowCount = shape[0];
    if (rowCount < 100) {
      result.push({
        title: "Small Dataset",
        description: `Dataset has only ${rowCount} rows.`,
        recommendation:
          "Use fewer CV folds (3–5) to preserve training data. Consider reducing the number of hyperparameter trials.",
      });
    } else if (rowCount > 100000) {
      result.push({
        title: "Large Dataset",
        description: `Dataset has ${rowCount.toLocaleString()} rows.`,
        recommendation:
          "Consider enabling sample-based training or parallel processing. GPU acceleration may be beneficial.",
      });
    }

    // Guide 4: Feature dimensionality
    const featureCount = shape[1] - 1; // Exclude target
    if (featureCount > 50) {
      result.push({
        title: "High Dimensionality",
        description: `Dataset has ${featureCount} features (excluding target).`,
        recommendation:
          "Enable feature selection in configuration to reduce noise and improve model interpretability.",
      });
    }

    return result;
  }, [profile, classSeries, shape]);

  return (
    <div className="explorer-section">
      <h3>Dataset Explorer</h3>

      {/* Summary */}
      <div className="summary-grid">
        <div>
          <strong>Problem Type</strong>
          <p>{profile.problem_type || "Classification"}</p>
        </div>
        <div>
          <strong>Target Column</strong>
          <p>{targetColumn}</p>
        </div>
        <div>
          <strong>Shape</strong>
          <p>
            {shape[0]} rows × {shape[1]} columns
          </p>
        </div>
        <div>
          <strong>Features</strong>
          <p>{shape[1] - 1} features</p>
        </div>
      </div>

      {/* Exploration Guides */}
      <div className="guides-container">
        <h4>✨ Smart Recommendations</h4>
        {guides.map((guide, idx) => (
          <div key={idx} className="guide-card">
            <h5>{guide.title}</h5>
            <p className="guide-description">{guide.description}</p>
            <p className="guide-recommendation">
              <strong>💡 Tip:</strong> {guide.recommendation}
            </p>
          </div>
        ))}
      </div>

      {/* Visualizations */}
      <div className="visualizations-grid">
        {/* Missing Values */}
        <div className="viz-card">
          <h4>Missing Values</h4>
          {missingSeries.length === 0 ? (
            <p>✓ No missing values detected—your data is clean!</p>
          ) : (
            <div style={{ width: "100%", height: 260 }}>
              <ResponsiveContainer>
                <BarChart data={missingSeries}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="column" hide />
                  <YAxis />
                  <Tooltip />
                  <Bar dataKey="missing" fill="#d65434" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}
        </div>

        {/* Class Balance */}
        <div className="viz-card">
          <h4>Target Distribution</h4>
          {classSeries.length === 0 ? (
            <p>Class distribution available for classification problems.</p>
          ) : (
            <div style={{ width: "100%", height: 260 }}>
              <ResponsiveContainer>
                <BarChart data={classSeries}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="label" />
                  <YAxis />
                  <Tooltip />
                  <Bar dataKey="pct" fill="#1f7a4a" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}
        </div>

        {/* Feature Types Overview */}
        <div className="viz-card">
          <h4>Feature Types</h4>
          <div>
            {profile.feature_types ? (
              Object.entries(profile.feature_types).map(([type, features]) => (
                <div key={type}>
                  <strong>{type}:</strong> {(features as string[]).length} features
                </div>
              ))
            ) : (
              <p>Feature type analysis coming soon.</p>
            )}
          </div>
        </div>

        {/* Correlation Heatmap Placeholder */}
        <div className="viz-card">
          <h4>Correlation Matrix</h4>
          <p>
            Correlation heatmap is computed by the backend and available via{" "}
            <code>profile.correlation_matrix</code>. Use a dedicated heatmap library (e.g.,
            Plotly, D3) for rich visual analysis.
          </p>
        </div>
      </div>
    </div>
  );
}
