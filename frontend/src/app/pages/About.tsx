import { Blocks, Eye, Shield, Workflow } from "lucide-react";

import { Badge } from "../components/ui/badge";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "../components/ui/card";

const principles = [
  {
    title: "Readable results",
    description: "Predictions should be understandable before they are impressive.",
    icon: Eye,
  },
  {
    title: "Model context",
    description: "Each detector should show the metrics behind the verdict, not only the verdict itself.",
    icon: Workflow,
  },
  {
    title: "Minimal surfaces",
    description: "The interface should feel like a toolbench, not a marketing landing page.",
    icon: Blocks,
  },
  {
    title: "Measured trust",
    description: "A strong score is useful, but the app should still make evaluation limits obvious.",
    icon: Shield,
  },
];

export function About() {
  return (
    <div className="mx-auto max-w-7xl px-4 py-10 sm:px-6 lg:px-8">
      <div className="grid gap-6 lg:grid-cols-[0.9fr_1.1fr]">
        <Card className="rounded-3xl border-border/80 bg-card/80">
          <CardHeader className="gap-4">
            <Badge variant="secondary" className="rounded-full px-3 py-1">
              About this interface
            </Badge>
            <CardTitle className="text-4xl font-semibold tracking-tight">
              Built as a focused verification workspace.
            </CardTitle>
            <CardDescription className="text-base leading-7">
              FN Detector combines two separate ML surfaces, one for text and one for image
              analysis. The frontend is intentionally quieter now so the component system,
              statistics, and detector outputs stay front and center.
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4 text-sm leading-7 text-muted-foreground">
            <p>
              The fake-news detector is a text classifier with calibration and evaluation reporting.
            </p>
            <p>
              The deepfake detector is currently a lightweight image pipeline and still benefits
              from larger datasets for stronger real-world robustness.
            </p>
          </CardContent>
        </Card>

        <div className="grid gap-6 sm:grid-cols-2">
          {principles.map((item) => {
            const Icon = item.icon;
            return (
              <Card key={item.title} className="rounded-3xl border-border/80 bg-card/80">
                <CardHeader className="gap-4">
                  <div className="flex size-11 items-center justify-center rounded-2xl border border-border bg-background">
                    <Icon className="size-5 text-primary" />
                  </div>
                  <div className="space-y-2">
                    <CardTitle className="text-xl font-semibold">{item.title}</CardTitle>
                    <CardDescription className="text-sm leading-6">
                      {item.description}
                    </CardDescription>
                  </div>
                </CardHeader>
              </Card>
            );
          })}
        </div>
      </div>
    </div>
  );
}
