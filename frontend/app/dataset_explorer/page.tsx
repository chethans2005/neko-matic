"use client";

import { useEffect, useMemo, useState } from "react";
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

export default function DatasetExplorerPage() {
  const [payload, setPayload] = useState<any>(null);

  useEffect(() => {
    const raw = localStorage.getItem("neko-matic.dataset");
    if (raw) {
      setPayload(JSON.parse(raw));
    }
  }, []);

  const missingSeries = useMemo(() => {
    const rows = payload?.profile?.missing_info ?? [];
    return rows.slice(0, 12).map((item: any) => ({
      column: item.column,
      missing: Number(item.missing_count ?? 0),
    }));
  }, [payload]);

  const classSeries = useMemo(() => {
    const distribution = payload?.profile?.class_distribution ?? {};
    return Object.entries(distribution).map(([label, value]) => ({
      label,
      pct: Number(value) * 100,
    }));
  }, [payload]);

  return (
    <div className="grid">
      <section className="card">
        <h2>Dataset Explorer</h2>
        {!payload && <p>Upload a dataset first.</p>}
        {payload && (
          <>
            <p>
              Problem Type: <strong>{payload.profile?.problem_type}</strong>
            </p>
            <p>
              Target Guess: <code>{payload.target_column_guess}</code>
            </p>
            <p>
              Rows: {payload.shape?.[0]} | Columns: {payload.shape?.[1]}
            </p>
          </>
        )}
      </section>

      <section className="card">
        <h3>Missing Values</h3>
        {missingSeries.length === 0 && <p>No missing values detected.</p>}
        {missingSeries.length > 0 && (
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
      </section>

      <section className="card">
        <h3>Class Balance</h3>
        {classSeries.length === 0 && <p>Class distribution is only available for classification datasets.</p>}
        {classSeries.length > 0 && (
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
      </section>

      <section className="card">
        <h3>Correlation Snapshot</h3>
        <p>
          Correlation heatmap matrix is returned in the backend payload under
          <code> profile.correlation_matrix</code>. Hook this to a dedicated heatmap component
          for richer visual analysis.
        </p>
      </section>
    </div>
  );
}
