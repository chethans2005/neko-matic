import Link from "next/link";

export default function HomePage() {
  return (
    <div className="card">
      <h2>Welcome to neko-matic</h2>
      <p>A modern, user-friendly AutoML platform for faster model development.</p>

      <h3>Getting Started</h3>
      <ol>
        <li>
          <strong>
            <Link href="/training">Go to Training (Unified Workflow)</Link>
          </strong>
          — Upload data, explore insights, configure, and monitor training all in one place.
        </li>
      </ol>

      <h3>Advanced: Individual Pages (Legacy)</h3>
      <p>
        For workflows that prefer modular interfaces, individual pages are available in the navigation bar:
      </p>
      <ul>
        <li>
          <Link href="/dataset_upload">Dataset Upload</Link> — Upload CSV or Excel
        </li>
        <li>
          <Link href="/dataset_explorer">Dataset Explorer</Link> — View data insights
        </li>
        <li>
          <Link href="/automl_configuration">Configuration</Link> — Fine-tune training settings
        </li>
        <li>
          <Link href="/training_monitor">Training Monitor</Link> — Launch and track runs
        </li>
        <li>
          <Link href="/leaderboard">Leaderboard</Link> — View model rankings
        </li>
        <li>
          <Link href="/explainability">Explainability</Link> — Inspect SHAP values
        </li>
        <li>
          <Link href="/model_export">Model Export</Link> — Download artifacts
        </li>
      </ul>

      <h3>Backend</h3>
      <p>
        Backend API is expected at <code>http://localhost:8000</code>.
      </p>
      <p>
        For help on setup, see the <code>README.md</code> in the project root.
      </p>
    </div>
  );
}
