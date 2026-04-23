import { ArrowRight, BarChart3, FileSearch, ImageIcon } from "lucide-react";
import { Link } from "react-router";

import { Badge } from "../components/ui/badge";
import { Button } from "../components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "../components/ui/card";

const tools = [
  {
    title: "Fake News Detector",
    description: "Paste article text, inspect calibrated scores, and review evaluation charts in one place.",
    href: "/fake-news",
    icon: FileSearch,
  },
  {
    title: "Deepfake Detector",
    description: "Upload an image, inspect the verdict, and compare it against saved model statistics.",
    href: "/deepfake",
    icon: ImageIcon,
  },
  {
    title: "Workflow Notes",
    description: "See how training, validation, metrics, and deployment artifacts flow through the app.",
    href: "/how-it-works",
    icon: BarChart3,
  },
];

const highlights = [
  { label: "Text test accuracy", value: "99.9%" },
  { label: "Detector surfaces", value: "2" },
  { label: "Frontend direction", value: "Shadcn-first" },
];

export function Home() {
  return (
    <div className="mx-auto flex min-h-[calc(100vh-73px)] max-w-7xl flex-col gap-10 px-4 py-10 sm:px-6 lg:px-8">
      <section className="grid gap-6 lg:grid-cols-[1.35fr_0.65fr]">
        <Card className="rounded-3xl border-border/80 bg-card/80">
          <CardHeader className="gap-4">
            <Badge variant="secondary" className="rounded-full px-3 py-1">
              Component-led interface
            </Badge>
            <div className="space-y-4">
              <CardTitle className="text-4xl font-semibold tracking-tight md:text-5xl">
                A quieter workspace for fake-news and deepfake screening.
              </CardTitle>
              <CardDescription className="max-w-2xl text-base leading-7 text-muted-foreground">
                The interface now leans on the shadcn component system and its preset color tokens,
                with less visual noise and more emphasis on cards, forms, metrics, and readable results.
              </CardDescription>
            </div>
          </CardHeader>
          <CardContent className="flex flex-col gap-6 md:flex-row md:items-center md:justify-between">
            <div className="grid gap-4 sm:grid-cols-3">
              {highlights.map((item) => (
                <div key={item.label} className="rounded-2xl border border-border bg-background px-4 py-4">
                  <p className="text-xs uppercase tracking-[0.18em] text-muted-foreground">
                    {item.label}
                  </p>
                  <p className="mt-2 text-2xl font-semibold tracking-tight">{item.value}</p>
                </div>
              ))}
            </div>
            <div className="flex flex-wrap gap-3">
              <Button asChild size="lg" className="rounded-full px-6">
                <Link to="/fake-news">
                  Open fake-news detector
                  <ArrowRight className="size-4" />
                </Link>
              </Button>
              <Button asChild variant="outline" size="lg" className="rounded-full px-6">
                <Link to="/deepfake">Open deepfake detector</Link>
              </Button>
            </div>
          </CardContent>
        </Card>

        <Card className="rounded-3xl border-border/80 bg-primary text-primary-foreground">
          <CardHeader>
            <div className="flex size-12 items-center justify-center rounded-2xl bg-primary-foreground/10 p-2">
              <img src="/logo.svg" alt="FN Detector logo" className="size-full object-contain brightness-0 invert" />
            </div>
            <CardTitle className="text-2xl font-semibold">Theme direction</CardTitle>
            <CardDescription className="text-primary-foreground/75">
              The preset introduced a fresh component palette, and the frontend now uses those
              colors as the main visual language instead of custom neon accents.
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-3 text-sm leading-6 text-primary-foreground/85">
            <p>Soft surfaces instead of full-screen hero sections.</p>
            <p>Neutral spacing with the green preset color reserved for focus and action.</p>
            <p>Cards, badges, and buttons carry the brand more than backgrounds do.</p>
          </CardContent>
        </Card>
      </section>

      <section className="grid gap-6 md:grid-cols-3">
        {tools.map((tool) => {
          const Icon = tool.icon;
          return (
            <Card key={tool.title} className="rounded-3xl border-border/80 bg-card/80">
              <CardHeader className="gap-4">
                <div className="flex size-11 items-center justify-center rounded-2xl border border-border bg-background">
                  <Icon className="size-5 text-primary" />
                </div>
                <div className="space-y-2">
                  <CardTitle className="text-xl font-semibold">{tool.title}</CardTitle>
                  <CardDescription className="text-sm leading-6">
                    {tool.description}
                  </CardDescription>
                </div>
              </CardHeader>
              <CardFooter>
                <Button asChild variant="ghost" className="px-0 text-primary hover:bg-transparent">
                  <Link to={tool.href}>
                    Open section
                    <ArrowRight className="size-4" />
                  </Link>
                </Button>
              </CardFooter>
            </Card>
          );
        })}
      </section>
    </div>
  );
}
