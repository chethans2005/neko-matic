"use client";

import { useEffect, useState } from "react";
import { downloadArtifactUrl, downloadModelUrl } from "@/lib/api";

export default function ModelExportPage() {
  const [runId, setRunId] = useState("");

  useEffect(() => {
    const persisted = localStorage.getItem("neko-matic.run_id") ?? "";
    setRunId(persisted);
  }, []);

  return (
    <section className="card">
      <h2>Model Export</h2>
      <p>Download the trained best model and run artifacts.</p>
      <label htmlFor="run-id">Run ID</label>
      <input id="run-id" value={runId} onChange={(e) => setRunId(e.target.value)} />
      <br />
      <div className="grid">
        <button onClick={() => runId && window.open(downloadModelUrl(runId), "_blank") } disabled={!runId}>
          Download best_model.pkl
        </button>
        <button onClick={() => runId && window.open(downloadArtifactUrl(runId, "pipeline.pkl"), "_blank") } disabled={!runId}>
          Download pipeline.pkl
        </button>
        <button onClick={() => runId && window.open(downloadArtifactUrl(runId, "training_report.json"), "_blank") } disabled={!runId}>
          Download training_report.json
        </button>
      </div>
    </section>
  );
}
