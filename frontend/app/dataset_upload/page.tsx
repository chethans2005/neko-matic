"use client";

import { FormEvent, useState } from "react";
import { uploadDataset } from "@/lib/api";

export default function DatasetUploadPage() {
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);

  async function onSubmit(event: FormEvent) {
    event.preventDefault();
    if (!file) return;

    setLoading(true);
    setError(null);
    try {
      const payload = await uploadDataset(file);
      setResult(payload);
      localStorage.setItem("neko-matic.dataset", JSON.stringify(payload));
      localStorage.setItem("neko-matic.dataset_id", payload.dataset_id);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Upload failed");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="grid">
      <section className="card">
        <h2>Dataset Upload</h2>
        <p>Upload CSV or Excel, then continue to Explorer and Configuration.</p>
        <form onSubmit={onSubmit}>
          <label htmlFor="dataset-file">Dataset file</label>
          <input
            id="dataset-file"
            type="file"
            accept=".csv,.xlsx,.xls"
            onChange={(event) => setFile(event.target.files?.[0] ?? null)}
          />
          <br />
          <button type="submit" disabled={!file || loading}>
            {loading ? "Uploading..." : "Upload Dataset"}
          </button>
        </form>
        {error && <p>{error}</p>}
      </section>

      <section className="card">
        <h3>Preview</h3>
        {!result && <p>Upload a file to preview dataset rows.</p>}
        {result && (
          <>
            <p>
              Dataset ID: <code>{result.dataset_id}</code>
            </p>
            <p>
              Shape: {result.shape?.[0]} rows x {result.shape?.[1]} columns
            </p>
            <div style={{ overflowX: "auto" }}>
              <table>
                <thead>
                  <tr>
                    {result.columns?.map((column: string) => (
                      <th key={column}>{column}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {(result.preview ?? []).map((row: Record<string, string>, index: number) => (
                    <tr key={index}>
                      {result.columns?.map((column: string) => (
                        <td key={`${index}-${column}`}>{String(row[column] ?? "")}</td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </>
        )}
      </section>
    </div>
  );
}
