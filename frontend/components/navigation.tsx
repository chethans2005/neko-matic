import Link from "next/link";

const routes = [
  ["Unified Training", "/training"],
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
