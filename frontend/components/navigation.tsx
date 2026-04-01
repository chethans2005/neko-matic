import Link from "next/link";

const routes = [
  ["Training (v2)", "/training"],
  ["Dataset Upload", "/dataset_upload"],
  ["Explorer", "/dataset_explorer"],
  ["Configuration", "/automl_configuration"],
  ["Monitor", "/training_monitor"],
  ["Leaderboard", "/leaderboard"],
  ["Explainability", "/explainability"],
  ["Export", "/model_export"],
] as const;

export function Navigation() {
  return (
    <nav className="nav">
      {routes.map(([label, href]) => (
        <Link key={href} href={href}>
          {label}
        </Link>
      ))}
    </nav>
  );
}
