"use client";

import { useEffect, useState } from "react";
import { getTrainingStatus, startRun } from "@/lib/api";

export default function TrainingMonitorPage() {
  const [datasetId, setDatasetId] = useState("");
  const [configId, setConfigId] = useState("");
  const [runId, setRunId] = useState("");
  const [status, setStatus] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);
  const [launching, setLaunching] = useState(false);

  useEffect(() => {
    setDatasetId(localStorage.getItem("neko-matic.dataset_id") ?? "");
    setConfigId(localStorage.getItem("neko-matic.config_id") ?? "");
    setRunId(localStorage.getItem("neko-matic.run_id") ?? "");
  }, []);

  useEffect(() => {
    if (!runId) return;
    const timer = setInterval(async () => {
      try {
        const payload = await getTrainingStatus(runId);
        setStatus(payload);
      } catch {
        setStatus(null);
      }
    }, 2500);
    return () => clearInterval(timer);
  }, [runId]);

  async function launchRun() {
    setLaunching(true);
    setError(null);
    try {
      const payload = await startRun({
        dataset_id: datasetId,
        config_id: configId || undefined,
      });
      setRunId(payload.run_id);
      localStorage.setItem("neko-matic.run_id", payload.run_id);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to start run");
    } finally {
      setLaunching(false);
    }
  }

  return (
    <div className="grid">
      <section className="card">
        <h2>Training Monitor</h2>
        <label htmlFor="dataset-id">Dataset ID</label>
        <input id="dataset-id" value={datasetId} onChange={(e) => setDatasetId(e.target.value)} />
        <label htmlFor="config-id">Config ID</label>
        <input id="config-id" value={configId} onChange={(e) => setConfigId(e.target.value)} />
        <br />
        <button disabled={!datasetId || launching} onClick={launchRun}>
          {launching ? "Starting..." : "Start AutoML Run"}
        </button>
        {runId && (
          <p>
            Active Run ID: <code>{runId}</code>
          </p>
        )}
        {error && <p>{error}</p>}
      </section>

      <section className="card">
        <h3>Status</h3>
        {!status && <p>Start a run to see live updates.</p>}
        {status && (
          <>
            <p>
              State: <strong>{status.status}</strong>
            </p>
            <p>
              Message: {status.message}
            </p>
            <p>
              Progress: {Number(status.progress ?? 0).toFixed(1)}%
            </p>
            <progress value={Number(status.progress ?? 0)} max={100} style={{ width: "100%" }} />
            <p>
              Current Best: {status.best_model_name ?? "-"} | Score: {String(status.best_score ?? "-")}
            </p>
            <p>
              Start: {status.started_at ?? "-"}
            </p>
            <p>
              End: {status.ended_at ?? "-"}
            </p>
          </>
        )}
      </section>
    </div>
  );
}
