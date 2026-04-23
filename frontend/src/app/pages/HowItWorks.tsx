import { ArrowRight, BarChart3, Brain, Database, FileSearch, ImageIcon } from "lucide-react";

import { Badge } from "../components/ui/badge";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "../components/ui/card";
import { Separator } from "../components/ui/separator";

const steps = [
  {
    title: "Input",
    description: "Users submit article text or upload an image from the detector surfaces.",
    icon: FileSearch,
  },
  {
    title: "Inference",
    description: "The backend runs the active model, calibration logic, and prediction formatting.",
    icon: Brain,
  },
  {
    title: "Metrics lookup",
    description: "Saved test metrics, confidence, history, and confusion matrices are loaded from model artifacts.",
    icon: Database,
  },
  {
    title: "Frontend reporting",
    description: "The interface renders the verdict beside charts and supporting evaluation context.",
    icon: BarChart3,
  },
];

export function HowItWorks() {
  return (
    <div className="mx-auto max-w-7xl px-4 py-10 sm:px-6 lg:px-8">
      <div className="grid gap-6 lg:grid-cols-[0.72fr_1.28fr]">
        <Card className="rounded-3xl border-border/80 bg-card/80">
          <CardHeader className="gap-4">
            <Badge variant="secondary" className="rounded-full px-3 py-1">
              Workflow
            </Badge>
            <CardTitle className="text-4xl font-semibold tracking-tight">
              From input to evaluation context.
            </CardTitle>
            <CardDescription className="text-base leading-7">
              The app is organized around a simple loop: submit content, run inference, read saved
              artifact metrics, and present the result with enough context to interpret it.
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-5 text-sm leading-7 text-muted-foreground">
            <div className="flex items-start gap-3">
              <ImageIcon className="mt-1 size-4 text-primary" />
              <p>The deepfake path is image-based and remains intentionally lightweight for deployment.</p>
            </div>
            <div className="flex items-start gap-3">
              <FileSearch className="mt-1 size-4 text-primary" />
              <p>The fake-news path uses saved evaluation artifacts so charts can stay in sync with the model.</p>
            </div>
          </CardContent>
        </Card>

        <Card className="rounded-3xl border-border/80 bg-card/80">
          <CardContent className="px-6 py-6">
            <div className="grid gap-4">
              {steps.map((step, index) => {
                const Icon = step.icon;
                return (
                  <div key={step.title}>
                    <div className="flex items-start gap-4 rounded-2xl border border-border bg-background px-4 py-4">
                      <div className="flex size-10 items-center justify-center rounded-xl border border-border bg-card">
                        <Icon className="size-4 text-primary" />
                      </div>
                      <div className="flex-1">
                        <p className="text-sm text-muted-foreground">Step {index + 1}</p>
                        <h2 className="mt-1 text-lg font-semibold">{step.title}</h2>
                        <p className="mt-2 text-sm leading-6 text-muted-foreground">
                          {step.description}
                        </p>
                      </div>
                      {index < steps.length - 1 ? (
                        <ArrowRight className="mt-2 hidden size-4 text-muted-foreground lg:block" />
                      ) : null}
                    </div>
                    {index < steps.length - 1 ? <Separator className="my-2" /> : null}
                  </div>
                );
              })}
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
