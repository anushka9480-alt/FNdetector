import { useEffect, useState } from "react";
import { NavLink, Outlet, useLocation } from "react-router";
import { Menu, X } from "lucide-react";

import { Badge } from "./ui/badge";
import { Button } from "./ui/button";
import { Card } from "./ui/card";
import { cn } from "./ui/utils";

const navLinks = [
  { name: "Overview", path: "/" },
  { name: "Fake News", path: "/fake-news" },
  { name: "Deepfake", path: "/deepfake" },
  { name: "How It Works", path: "/how-it-works" },
  { name: "About", path: "/about" },
  { name: "Contact", path: "/contact" },
];

export function Layout() {
  const [isOpen, setIsOpen] = useState(false);
  const location = useLocation();

  useEffect(() => {
    setIsOpen(false);
  }, [location.pathname]);

  return (
    <div className="min-h-screen bg-background text-foreground">
      <header className="sticky top-0 z-50 border-b border-border/80 bg-background/90 backdrop-blur">
        <div className="mx-auto flex max-w-7xl items-center justify-between gap-4 px-4 py-4 sm:px-6 lg:px-8">
          <NavLink to="/" className="flex items-center gap-3">
            <div className="flex size-10 items-center justify-center rounded-xl border border-border bg-card p-1.5">
              <img src="/logo.svg" alt="FN Detector logo" className="size-full object-contain" />
            </div>
            <div className="space-y-0.5">
              <p className="text-sm font-semibold tracking-tight">FN Detector</p>
              <p className="text-xs text-muted-foreground">Detection workspace</p>
            </div>
          </NavLink>

          <nav className="hidden items-center gap-1 md:flex">
            {navLinks.map((link) => (
              <NavLink
                key={link.path}
                to={link.path}
                className={({ isActive }) =>
                  cn(
                    "rounded-md px-3 py-2 text-sm text-muted-foreground transition-colors hover:bg-accent hover:text-accent-foreground",
                    isActive && "bg-accent text-foreground",
                  )
                }
              >
                {link.name}
              </NavLink>
            ))}
          </nav>

          <div className="hidden md:block">
            <Badge variant="secondary" className="rounded-full px-3 py-1">
              Minimal workspace
            </Badge>
          </div>

          <Button
            variant="ghost"
            size="icon"
            className="md:hidden"
            onClick={() => setIsOpen((current) => !current)}
          >
            {isOpen ? <X className="size-5" /> : <Menu className="size-5" />}
          </Button>
        </div>

        {isOpen ? (
          <div className="border-t border-border md:hidden">
            <div className="mx-auto flex max-w-7xl flex-col gap-1 px-4 py-4 sm:px-6 lg:px-8">
              {navLinks.map((link) => (
                <NavLink
                  key={link.path}
                  to={link.path}
                  className={({ isActive }) =>
                    cn(
                      "rounded-md px-3 py-2 text-sm text-muted-foreground transition-colors hover:bg-accent hover:text-accent-foreground",
                      isActive && "bg-accent text-foreground",
                    )
                  }
                >
                  {link.name}
                </NavLink>
              ))}
            </div>
          </div>
        ) : null}
      </header>

      <main>
        <Outlet />
      </main>

      <footer className="border-t border-border/80 bg-background">
        <div className="mx-auto max-w-7xl px-4 py-10 sm:px-6 lg:px-8">
          <Card className="rounded-2xl border-border/80 bg-card/60">
            <div className="flex flex-col gap-4 px-6 py-6 md:flex-row md:items-center md:justify-between">
              <div>
                <p className="text-sm font-medium">FN Detector</p>
                <p className="text-sm text-muted-foreground">
                  Local fake-news and deepfake screening workspace.
                </p>
              </div>
              <p className="text-sm text-muted-foreground">
                © {new Date().getFullYear()} Model-backed verification workflow.
              </p>
            </div>
          </Card>
        </div>
      </footer>
    </div>
  );
}
