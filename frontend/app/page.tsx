import Link from "next/link";

export default function HomePage() {
  return (
    <div className="card">
      <h2>Welcome to neko-matic</h2>
      <p>A focused AutoML platform with one guided workflow from upload to model export.</p>

      <h3>Getting Started</h3>
      <ol>
        <li>
          <strong>
            <Link href="/training">Go to Training (Unified Workflow)</Link>
          </strong>
          — Upload data, explore insights, configure, train, and export in one place.
        </li>
      </ol>

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
