"use client";

import { useEffect, useState } from "react";
import { getLeaderboard } from "@/lib/api";

export default function LeaderboardPage() {
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
    setError(null);
    try {
      const payload = await getLeaderboard(targetRunId);
      setRows(payload.leaderboard ?? []);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to fetch leaderboard");
      setRows([]);
    }
  }

  return (
    <section className="card">
      <h2>Leaderboard</h2>
      <div className="grid">
        <div>
          <label htmlFor="run-id">Run ID</label>
          <input id="run-id" value={runId} onChange={(e) => setRunId(e.target.value)} />
        </div>
        <div>
          <label>&nbsp;</label>
          <button onClick={() => refresh()}>Load Leaderboard</button>
        </div>
      </div>
      {error && <p>{error}</p>}
      {rows.length === 0 && !error && <p>No rows available for this run yet.</p>}
      {rows.length > 0 && (
        <div style={{ overflowX: "auto" }}>
          <table>
            <thead>
              <tr>
                {Object.keys(rows[0]).map((key) => (
                  <th key={key}>{key}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {rows.map((row, index) => (
                <tr key={index}>
                  {Object.keys(rows[0]).map((key) => (
                    <td key={`${index}-${key}`}>{String(row[key] ?? "")}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}
