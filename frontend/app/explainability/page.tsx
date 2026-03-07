"use client";

import { useEffect, useState } from "react";
import { getFeatureImportance } from "@/lib/api";

export default function ExplainabilityPage() {
  const [runId, setRunId] = useState("");
  const [rows, setRows] = useState<any[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const persisted = localStorage.getItem("neko-matic.run_id");
    if (persisted) {
      setRunId(persisted);
      refresh(persisted);
    }
  }, []);

  async function refresh(targetRunId: string = runId) {
    if (!targetRunId) return;
    try {
      const payload = await getFeatureImportance(targetRunId);
      setRows(payload.feature_importance ?? []);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to fetch feature importance");
      setRows([]);
    }
  }

  return (
    <section className="card">
      <h2>Explainability</h2>
      <p>Top features from SHAP analysis of the best model.</p>
      <div className="grid">
        <div>
          <label htmlFor="run-id">Run ID</label>
          <input id="run-id" value={runId} onChange={(e) => setRunId(e.target.value)} />
        </div>
        <div>
          <label>&nbsp;</label>
          <button onClick={() => refresh()}>Load Feature Importance</button>
        </div>
      </div>

      {error && <p>{error}</p>}
      {rows.length === 0 && !error && <p>No explainability output available yet.</p>}
      {rows.length > 0 && (
        <div style={{ overflowX: "auto" }}>
          <table>
            <thead>
              <tr>
                <th>Feature</th>
                <th>Importance</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((row, index) => (
                <tr key={index}>
                  <td>{String(row.feature)}</td>
                  <td>{Number(row.importance).toFixed(6)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}
