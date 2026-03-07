export default function HomePage() {
  return (
    <div className="card">
      <h2>AutoML Workflow</h2>
      <p>
        Use the navigation bar to upload datasets, configure the pipeline, launch runs,
        monitor training, inspect leaderboard metrics, and export artifacts.
      </p>
      <p>
        Backend endpoints are expected at <code>http://localhost:8000</code>.
      </p>
    </div>
  );
}
